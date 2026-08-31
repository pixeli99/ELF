# 实验 setup：怎么训，怎么测

当前口径，对应 v6（2026-08-31）。所有数字都从配置和代码里核过。

---

## 0. 这个实验在问什么

给一个条件生成任务（prompt → response），把推理过程（thinking，来自 R1 类模型的蒸馏轨迹，不是人写的）压成 K 个「计划槽」，
和回答一起做扩散去噪。问：**回答会不会用这些计划槽，用了之后会不会更好。**

对照的核心是「计划槽以什么方式参与」，除此之外四组的数据、顺序、噪声、预算完全一致。

---

## 1. 数据

`data/conditional_v1/`，manifest 有 sha 门（`200a24b6…`）。

| 划分 | 行数 | 字段 |
|---|---|---|
| train | 474460（19 个 shard） | example_id, source, prompt, thinking, response |
| validation | 2000 | prompt（参考在 `validation_references-00000.parquet`） |
| test | 1000 | prompt（参考在 `test_references-00000.parquet`） |

来源混合：Nemotron-Cascade-SFT-Stage-2 70.8%、dolphin-r1 16.0%、GeneralThoughtArchive 13.2%。
长度：prompt 中位 81 / p99 999 token，response 中位 158 / p99 588，thinking 均值 494 / 最长 1019。

### 固定日程（四组可比性的基础）

不是一个文件，是两个数：`conditional_rows=80000`、`conditional_master_seed=42`。

- 呈现顺序 = `randperm(474460, seed)[:80000]`
- 每行的噪声种子 = hash(master_seed, 位置, example_id, 流名)，流名区分
  response 噪声 / 时钟 / 分支 / plan 噪声 / plan 时钟

所以四组看到的是同一批行、同一个顺序、同一份噪声。启动时打 fingerprint
`509d1b0b0da4c1b592af3cfe45c149f48cb08c64e056f463fa0992217096068a`，日志里对得上才算同一个实验。

### 窗口布局

一个 2048 位置的窗口里放 `[prompt | response | EOS 尾巴]`：

- prompt 最多 1024 token，是**干净前缀**——不加噪、不算 loss，只作条件
- response 接在后面，是被去噪的目标；1024 + 2048 保证最长的 response（1006）永不被截
- response 之后填 EOS，**其中紧跟着的 64 个位置进 loss**（`pad_token: eos` + `eos_tail_loss_tokens: 64`）。
  这是模型学会停下来的唯一来源；早先用 `pad_token: pad` 时尾巴不进 loss，四组 EOS 率全是 0。
  只监督 64 个而不是全部 1600 多个，是避免平凡的 EOS 目标淹没约 200 个 response token。

计划那一路单独走：gold thinking 最多 1024 token（= 16 token/槽 × 64 槽），不进这个窗口。

---

## 2. 模型与底座

ELF-B，105,618,404 参数。冻结 t5-small 做 encoder（d_model 512），latent 统一除以
`latent_std=0.2`。bottleneck 128。前缀里有 4 个时间 token、4 个 self-cond-CFG token、
4 个 mode token、4 个 plan 时间 token，加 64 个计划槽。

**底座**：官方发布的 `embedded-language-flows/ELF-B-owt-torch` 的 `checkpoint_95085`（取 EMA 权重），
热启动，缺 11 个 key（全是新的 plan head），符合预期。

官方是 1024 窗口，我们用 2048 —— RoPE 直接外推，靠 Stage-B 微调适应，四组一视同仁。

> 自己用 dolma3 预训的底座（5.5B token）已弃用：只有官方算力的八分之一，去噪器塌成均值
> （t=0.5 时 token 还原率 18%，官方 97%），无条件生成全是 "the the the"。

---

## 3. 计划目标怎么来的（两阶段，VAE 先训好再冻结）

### 阶段 0：Plan-VAE

单独训，训完冻结，Stage-B 只调用不更新。

```
gold thinking → t5-small → latent ÷ 0.2 → 固定切成 64 段（每段 16 token）
              → 每段整块 16×512 进编码器 → 2 层 slot self-attn → μ, logσ² ∈ R^128
```

- **固定分段**，不是可学的 query 路由 —— 后者在这个规模下五连败，重构一直卡在位置先验地板
- **编码器吃整块，不是段均值** —— 段均值是天花板：哪怕完全不压（32768 浮点）也只保留 30% 的
  答案数字，比只有它四分之一大小的全内容 PCA-128 还差
- **null 合同**：thinking 用不到的尾部槽，μ 和 logσ² 精确为零。所以「计划有多长」是生成出来的
  内容，不是输入的 mask —— `plan_mask` 恒为全 1

损失 = 重构 MSE + β·KL（slot 级 free bits）+ λ·aux（预测 response 的池化 latent）。

当前 artifact：`artifacts/plan_vae/plan_vae_v8_flat_beta0.1.pt`（sha `fe06a247…`），
β=0.1、free bits 1.0、λ=0.5、3 万步。

**验收指标是 round-trip 答案保留率，不是重构 MSE。** gold thinking 过一遍编解码，再用 ELF
解码头读回文本，看参考答案的数字还在不在：v8 是 37.5%，上一版 v7 是 21.1%，同预算 PCA 上限 44%。
重构 MSE 分不清「保住了推理」和「只保住了骨架」。

### 阶段 1：Stage-B 每步现算

训练时每个 batch 现场跑：thinking token → T5 → VAE 后验 → `x0_plan ∈ R^{64×128}`（取 μ，不采样）。
VAE 全程冻结。

---

## 4. 四组差在哪

只差两件事：**计划槽的时钟怎么取**，以及**有没有计划损失**。其余全同。

| 组 | 计划槽 | 计划时钟 t_plan | 计划损失 | 推理轨迹 |
|---|---|---|---|---|
| **ordered** | 64 个，装 VAE 目标 | 与 t 独立均匀采样，另有 50% 的行钉在 t_plan=1 | 有 | planning_first，α=2（计划先行） |
| **diagonal** | 同上 | 恒等于 t | 有 | 与回答同步 |
| **register** | 64 个，纯噪声 | 恒为 0 | 无 | null（全程噪声） |
| **vanilla** | 没有 | — | — | — |

`plan_done_frac`（钉在 t_plan=1 的行占比）是 ordered 唯一可调的协议旋钮，必须严格落在 (0,1) 之间：
太小则回答训练时几乎见不到干净计划，学会忽略；等于 1 则计划去噪器见不到带噪时钟，推理时生成不出计划。
实测 0.5 优于 0.75，v1–v4 用的 0.15 太稀。

启动时 `assert_group_protocol` 会核对这四个开关，标着 diagonal 却把 `plan_diag_frac` 留成 0 会直接报错。

### 注意力

`plan_response_attention: prompt_causal_bottleneck`。prompt token、计划槽、计划时钟构成一个
逐层封闭的子系统：彼此可读，但读不到 response 或任何已经吸收过 response 信息的共享 token；
response 可以读计划槽。这切断了「计划槽偷看答案」的捷径——早先的双向版本实测扰动 response
会让计划输出变 36.5%。

---

## 5. 一个训练步里发生什么

每一行**独立**抛硬币决定走哪个分支（`decoder_prob=0.2`）：

| | decoder 行（20%） | denoiser 行（80%） |
|---|---|---|
| 输入 | z = λ·x0 + (1−λ)·noise·5.0，λ=sigmoid(N(0.8, 0.8)) | z = t·x0 + (1−t)·noise·2.0 |
| 时钟 t | 1 | logit-normal(−1.5, 0.8) |
| 计划时钟 | **强制为 1**（干净计划） | 按组的规则取 |
| 损失 | token 交叉熵 | v-预测 L2 |

一次前向同时算两个头，再用 mask 把两种损失切到各自的行上：

```
loss = (Σ CE·decoder行 + Σ L2·denoiser行) / 监督位置总数
     + plan_loss_weight × plan_L2
```

- 监督位置 = response token + 紧跟的 64 个 EOS，**prompt 位置永不计入**
- `plan_L2` 只在 denoiser 行上算，且在 **x 空间**而不是 token 用的 v 空间——v 空间的 MSE 会被
  1/(1−t)² 放大，而 t_plan=1 那批行会直接炸到 1/t_eps²
- 自条件：50% 的概率跑一次额外前向拿 x_pred 回填，self-cond CFG 系数 3.0
- decoder 行的计划时钟强制为 1，是为了和推理时的解码角 (t=1, t_plan=1) 对齐

---

## 6. 训练超参

| 项 | 值 | 备注 |
|---|---|---|
| micro-batch | 1 | 日程种子逐行绑定，强制为 1 |
| 梯度累积 | 8 | 有效 batch 8 |
| 优化步 | 10000 | = 80000 micro 步 = 80000 行，每行恰好过一次 |
| lr | 1e-3 显式 | 三点扫描（1e-4 / 3e-4 / 1e-3）单调选出 |
| warmup | 300 优化步 | 交接配置里是 9508，占 95%，等于没训 |
| 调度 | cosine 到 0 | |
| 优化器 | Muon（2D 参数）+ Nesterov-AdamW（其余） | weight decay 0 |
| EMA | 0.999 | 0.9999 半衰期约 7k 步，1 万步微调太慢 |
| 精度 | bf16 | 开梯度检查点 |
| checkpoint | 1000 / 2000 / 5000 / 10000 优化步 | |

单卡约 5.5 it/s，一组约 4 小时。四组四张卡并行。

**启动**：`bash outputs/run_stage_b_v6.sh`（`STAGE_B_GROUPS="ordered diagonal" GPU_BASE=0` 可跑子集）。
supervisor 崩溃自动续训重试 8 次，从 `resume` 恢复；日志 `outputs/logs/stage_b_{组}_v6.log`。

---

## 7. 怎么测

### 7.1 自由生成（主协议）

`tools/eval_conditional_dev.py`。给 prompt，让模型自己生成，不给参考的任何信息。

| 项 | 值 |
|---|---|
| 采样器 | SDE，γ=1.5，最后一步走 ODE |
| NFE | 32 |
| 时间网格 | logit-normal |
| CFG / self-cond CFG | 1.0 / 3.0 |
| 回答长度上限 | 512 token（覆盖 96.9% 的参考，看结果前定死的） |
| 权重 | raw（不是 EMA——EMA 在这个步数下滞后，会把计划近似关掉） |
| 样本 | validation 512 或 1000，test 1000 只看一次 |

计划轨迹按组走：ordered → planning_first（t_plan = min(1, 2t)，计划先行），
diagonal → t_plan = t，register → 全程 0。解码时喂 t_plan=1 的干净计划（null 组喂 0）。

四组共享同一批样本、同一个顺序、成对的噪声。

**指标**：BLEU、ROUGE-1/2/L、EOS 率、生成字符数、生成的计划预算（‖μ‖>2.0 的槽数，
训好的活跃槽范数约 8.5，null 精确为 0）。

### 7.2 可验证子集准确率

`tools/eval_boxed_accuracy.py`。test 参考里 25%（250/1000）以 `\boxed{}` 或 `#### N` 收尾。
取生成里**第一个**答案对参考里**最后一个**答案，做数值归一化后比。

坑：t5-small 词表里没有反斜杠和花括号，`\boxed{42}` 往返之后是 `boxed42`，工具按这个形式解析。

### 7.3 内容审计（判断「用没用计划的内容」）

`tools/audit_oracle_plan_response_upper_bound.py`。在 checkpoint **没见过**的日程行上
（`--start-index 80000`），固定回答的初始噪声、时间网格和 SDE 噪声，只换计划：

- `generated` —— 模型自己生成的计划
- `oracle_matched` —— 把 gold thinking 的精确 VAE 目标冻结在 t_plan=1 全程喂进去
- `oracle_shuffled` —— 换成**活跃槽数完全相同**的别人的计划（`--exact-k-shuffle`）

`oracle_matched − oracle_shuffled` 是关键差值：显著为正才说明回答在读计划的**内容**；
只有 `oracle_matched − generated` 为正，说明回答只吃「计划干不干净」这类粗信号。

### 7.4 VAE 验收

`tools/eval_plan_vae_roundtrip.py --artifact <path>`。换 VAE 之前必跑，见 §3。

### 7.5 显著性

`tools/compare_conditional_evals.py --left A.jsonl --right B.jsonl`，配对 bootstrap
10000 次，出 mean_delta 和 95% CI。所有组间比较都走这个，不看单点 ROUGE 差。

### 7.6 有参考的评测行

`tools/eval_schedule_rows.py`。validation/test 只有 prompt，没有 thinking，所以
「把 thinking 当前缀」这类诊断模型没法在上面测。这个工具在训练日程之后的行上跑，
那些行自带 thinking 和 response，同时出 ROUGE 和可验证子集准确率。

**终评**：`bash outputs/run_eval_v6.sh`，等四个 checkpoint_10000 齐了自动跑。

---

## 8. 判据（按重要性）

1. **可验证子集准确率高于 vanilla** —— 这才叫「用推理帮到了解题」
2. **oracle_matched − oracle_shuffled 显著为正** —— 回答在读计划的内容
3. **ROUGE 高于 vanilla / diagonal / register / 关掉计划** —— 最弱的一条，
   只能支持「计划状态有效应」

当前状态：第 3 条成立（ordered − vanilla 的 ROUGE-1 +1.14，CI 不跨零），
第 1、2 条都不成立。

---

## 9. 一次完整的复现

```bash
cd /cpfs01/shared/public/users/pengxiang.li/ELF
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# 0. 底座（一次性，需要代理）
#    source ../agent-persist/aliyun-proxy.sh
#    python -c "from huggingface_hub import snapshot_download; \
#      snapshot_download('embedded-language-flows/ELF-B-owt-torch', local_dir='outputs/reference_elf_b_owt')"

# 1. Plan-VAE（一次性，约 1 小时；换设置才需要重跑）
python tools/train_plan_vae.py --beta 0.1 --free-bits 1.0 --aux-weight 0.5 \
  --steps 30000 --span-input flat --out data/plan_vae_runs/flat_b0.1fb1_30k
python tools/eval_plan_vae_roundtrip.py --artifact data/plan_vae_runs/flat_b0.1fb1_30k/final.pt --rows 128
# 达标后导出到 artifacts/plan_vae/ 并把 sha 填进配置

# 2. Stage-B 四组（四卡各一组，约 4 小时）
bash outputs/run_stage_b_v6.sh

# 3. 终评（自动等训练结束）
bash outputs/run_eval_v6.sh

# 4. 内容审计单跑
python tools/audit_oracle_plan_response_upper_bound.py \
  --config src/configs/training_configs/train_conditional_common48w_ordered_v8_v6.yml \
  --checkpoint outputs/elf_b_conditional_common48w_ordered_v8_v6/checkpoint_10000 \
  --output outputs/audits/x.json --start-index 80000 --num-samples 128 --exact-k-shuffle
```

配置链：`train_conditional_common48w_ordered_v8_v6.yml` → `…_causal_eos_v5.yml`
→ `…_causal_10k_v1.yml` → `…_ordered_10k_v1.yml` → `train_conditional_ELF-B_common48w.yml`。
`--config_override k=v` 可覆盖任意字段，清空一个字段用字符串 `none`（不是 `null`）。

---

## 10. 已知的坑

- 配置覆盖的空值哨兵是 `none`，写 `null` 会被当成真值字符串
- bash 里 `GROUPS` 是保留变量，赋值被静默忽略；脚本里用 `STAGE_B_GROUPS`
- 换 VAE 必须同时改 `plan_vae_artifact` 和 `plan_vae_artifact_sha256`，否则启动即报 sha 不符
- `max_plan_slots` 必须等于 VAE 的 K_MAX=64（vanilla 除外，它没有计划流）
- 评测一律 raw 权重；用 EMA 会得到计划预算接近 0 的假象
- 内网（dlcw / git push / pip 阿里源）要 unset 代理，外网（HF / GitHub）要挂代理

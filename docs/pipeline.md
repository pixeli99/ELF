# 这棵树是什么

官方 `lillian039/ELF@b29d883`（分支 `pytorch_elf`）**逐字节原样**，加上一小撮我们自己的东西。
核对：`git diff upstream/pytorch_elf HEAD -- src scripts requirements.txt README.md LICENSE`
只应出现下面「改了官方三处」这三处。

上一条 plan-slot 线的全部代码和结论已删除，见 commit `cdd2482` / `c1456cb`。

## 改了官方三处

| 位置 | 改了什么 | 为什么官方不需要 |
|---|---|---|
| `config.init_from` + `checkpoint_utils.load_init_weights` + `train.py` | 只取权重和 EMA 热启动，优化器 / 调度 / 步数从零开始 | 官方只有 `resume`，那是续跑用的。用 `resume` 加载 95085 步的预训练权重，会让 2700 步的微调直接落在别人的 cosine 末端 |
| `encoder_utils.encode_text` | 3D 条件 mask 拆成两趟 2D | transformers 4.45 起 T5Stack 无条件写 `attention_mask[:, None, None, :]`，3D 变 5D 崩掉。官方 `requirements.txt` 钉了 `<4.45`，但集群镜像是 4.51.3，不归我们钉 |
| `train_step.py` / `generation.py` | 把手里已有的 `cond_seq_mask` 传给 `encode_text` | 配套上面那条 |

契约由 `tests/test_encoder_masks.py` 守着：条件位置的 latent 不能因为答案变了而变。

## 我们加的东西

```
tools/survey_hrm.py        HRM 语料普查：哪些源真有 cot/direct 配对
tools/build_math_data.py   造两臂数据集，直接用官方 conditional loader 的列名
tools/build_gsm8k_eval.py  GSM8K test 写成 src/eval.py 认的 {input, output} jsonl
tools/score_gsm8k.py       读 src/eval.py 的产物，算精确匹配 + Wilson 区间 + 置换地板
scripts/stage_assets.sh    把 worker 拿不到的东西落到 assets/
scripts/dlc_stage0.sh      DLC 入口：一个 pod 上并行跑多个臂，训完接评测接打分
src/configs/training_configs/math_*.yml
```

## 数据

`tools/survey_hrm.py` 的结论：HRM 九个文件里只有三个真有配对。

| 源 | cot | direct | 配对 |
|---|---:|---:|---:|
| numinamath | 442127 | 369794 | 369783 |
| math_train | 7500 | 7496 | 7496 |
| omnimath | 4428 | 4428 | 4406 |
| gsm8k_train | 0 | 7473 | 0（从原始 GSM8K 拆 `####` 重建） |
| amps_khan / no_robots | 只有 cot | | 0 |
| natural_reasoning / principia_collection / webinstruct_verified | 只有 direct | | 0 |

`tools/build_math_data.py` 只留答案是纯数字的行，剩 **114437 行**
（numinamath 100061 / gsm8k 7473 / math 4847 / omnimath 2056）。
这不是图省事：整个测量就是数字精确匹配，而 t5-small 词表里没有反斜杠和花括号，
`\boxed{\frac{1}{2}}` 既评不了分也写不出来。

两臂同一批行、同一句 `The answer is 42` 收尾，只差目标里有没有推理过程。

实测 token 长度（`data/math_v1/manifest.json`）：
问题 p95 209 / p100 1057，纯答案 p100 19，推理+答案 p50 249 / p90 665 / p99 1302。

## 窗口不能随便给大

`pad_token: eos` 会把目标之后的**每一个位置**都放进 loss。所以窗口比目标需要的宽不是免费的：
64 位窗口配 5 个 token 的答案，92% 的信号是在预测 EOS，冒烟跑 500 步就把 ce 压到 0.0002。

所以 `noreason` 窗口 32，`cot` 窗口 768。这同时也是整条线赖以成立的成本轴。
`math_b_noreason_wide` 是配套对照：只答答案，但给 cot 那么宽的画布，用来分清
差距来自「推理内容」还是「画布更大」。

## 跑

```bash
bash scripts/stage_assets.sh                      # t5-small + muon 落到 assets/（gitignore）
python tools/survey_hrm.py --out data/hrm_survey.json
python tools/build_math_data.py --out data/math_v1
python tools/build_gsm8k_eval.py --out data/gsm8k_test.jsonl
python -m pytest -q
```

DLC（UserCommand 由 dash 执行，所以只许 POSIX 的 cd / export / bash）：

```
cd /cpfs01/shared/public/users/pengxiang.li/ELF
export RUN=stage0 ARMS="math_b_noreason math_b_cot math_l_noreason math_l_cot" GPUS_PER_ARM=4 STAGE=all
bash scripts/dlc_stage0.sh
```

提交前本地过一遍：把上面原文塞进 `/bin/sh -c '...'`，`STAGE=preflight`，十秒钟省一轮排队。

## 已经踩过的坑

- **worker 没有 muon。** 镜像 `ppu-training:lpx-0805` 不带 `muon-optimizer`，而官方配置默认选它。
  错误在模型都建好之后才冒出来，第一次交任务因此白费一个 16 卡 pod。
  现在 `assets/pydeps` 里放一份，preflight 里 `import muon` 先验。
- **DLC 注入 `WORLD_SIZE` / `MASTER_PORT`。** 一个 pod 上跑多个独立单机任务时，
  每个裸 `python` 都以为自己是 rank 0，四个评测进程一起抢注入的 23456 端口，全部 EADDRINUSE。
  入口脚本开头 unset 掉，torchrun 会给自己的子进程重新设。
- **worker 上没有 HF 缓存。** `t5-small` 只在 `/root/.cache`，`/root` 会随环境重启清空，
  worker 更是完全看不到。配置里指向 `assets/t5-small`。
- **假 Succeeded。** DLC 报的是脚本最后一条命令的退出码，入口脚本必须 `exit $rc`。
  第一次失败的任务确实报了 Failed，这条是验证过的。

## 判据的地板要一起报

`tools/score_gsm8k.py` 每次都跑置换检验。这不是装饰：被替换掉的那条线报过
「同预算 PCA 保留 44%」并据此定了计划，后来查出那个判据是子串包含，
64 条评测里 49 条的答案只有 1~2 位数字，随机地板就有 27.9%。

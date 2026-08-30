# Plan 流负结果汇总（2026-08-30）

结论：**response 没在用 plan。** 两条独立线索（我的 Stage-B 重跑 + Codex 的 causal 版审计）指向同一处。

## 证据

实验1（Codex，双向注意力 v4，ckpt 2000，validation 1000）
- ordered 采样路径 vs diagonal 路径：ROUGE-1 差 0.006。ordered vs null 路径：+0.12。
- 方向探针：扰动 response，plan 输出相对变化 0.365；扰动 plan，response 变化 0.105。plan 在读 response，不是反过来。
- 文件：`outputs/audits/bidirectional_to_causal_pivot_20260829.json`，`outputs/audits/plan_attention_direction_probe_20260829.json`

实验2（Codex，causal 版 ordered，ckpt 5000/10000，训练外样本 256/128）
- oracle plan（gold thinking 的精确 VAE 目标，t_plan=1 冻结整条轨迹）vs 生成 plan：ROUGE-1 Δ −0.33（10k）、+0.02（5k），CI 跨零。
- oracle vs shuffled（同 budget、换别人的 plan）：Δ ≈ 0。
- 文件：`outputs/audits/prompt_causal_10k_oracle_plan_exactk_n128_20260830.json`，`outputs/audits/prompt_causal_5k_oracle_plan_response_upper_bound_n256_with_shuffle_20260830.json`

实验3（Codex，test 1000，causal 10k，raw 权重，NFE 32，cap 512）

| 组 | ROUGE-1 | ROUGE-L | budget |
|---|---|---|---|
| ordered, plan-CFG 3 | 17.9 | 11.2 | 26±7 |
| vanilla | 17.7 | 11.0 | — |
| ordered, plan 关掉 (null) | 17.5 | 10.9 | — |
| diagonal | 16.3 | 10.3 | 28±7 |
| register | 16.0 | 10.3 | — |

- ordered vs vanilla：Δ 0.16，CI [−0.12, 0.44]。ordered 开/关 plan 只差 0.4。
- ordered > diagonal / register 的 1.6~1.9 分，更像"多了 64 个可学的槽"，不是"用了计划"。
- 文件：`outputs/formal_eval/prompt_causal_ckpt10000_selected_test_n1000_cap512/`

实验4（我，双向 v3 ordered，ckpt 2000，EMA 权重，训练外 32 行）
- 喂干净 plan（t_plan=1）时 plan 输出仍≈0：active 槽范数 1.0，目标 8.5。plan head 零初始化 + EMA 0.999 滞后，早期 checkpoint 的 budget=0 有一半是这个原因。
- 训练端 plan loss 0.12 → 0.06（预测均值基线 0.267）：在学，只是慢。
- 文件：`outputs/plan_probe.py`

实验5（Codex，mediation 分支，5k→6k 配对训练日志）
- 干预 plan→response 通道，response loss 只差 0.04。
- 文件：`outputs/audits/prompt_causal_mediation_p50_5k_to6k_paired_training_log_20260830.json`

## 已排除的原因

- 底座：官方 ELF-B-owt checkpoint_95085。去噪探针 t=0.5 时 token 还原 97%（自训 dolma 底座 18%，已弃）。
- 采样器：官方权重用同一套采样器出通顺文本。
- 优化：lr 探针 1e-4/3e-4/1e-3 单调，定 1e-3；warmup 300，cosine，EMA 0.999。同事配置的 warmup 9508 已删。
- plan 目标截断：collator 容量 256 → 1024（commit 83f3e35）。budget 15.7 → 31.4 槽。

## 还没解决

- 所有组都不会停：eos 率≈0，生成填满 512 cap。BLEU/ROUGE 都被长度压着。

## 候选下一步（要拍板）

1. 训练时随机丢 plan（p≈0.3）+ 推理 plan-CFG。现在 pcfg3 只比 null 多 0.4，CFG 在放大一个几乎不存在的信号，得先让信号存在。
2. response 侧瓶颈：response 对 prompt 的直接注意力做 drop，逼它经 plan 取信息。
3. 先把停止学会。
4. 换评测：ROUGE 对"有没有计划"不敏感，换 GSM8K/MATH 答案准确率。

## 工作树状态

- 我的修复：commit 83f3e35（未 push）。
- Codex 的改动：model.py（prompt_causal_bottleneck、mediation mask）、train_step、generation_utils、eval 工具等 12 个文件，未提交。代码快照 `outputs/code_snapshot_v3_0829_2203/`。
- GPU 0-3 空闲，等拍板再起。

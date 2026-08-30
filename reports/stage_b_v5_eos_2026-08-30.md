# Stage-B v5：修好停止之后，四组的差异消失了（2026-08-30）

## 一句话

v4 报告里"Ordered 显著优于 Diagonal/Register、plan-state 效应可检出"这个结论，**在修好 EOS 之后不复现**。
四组现在质量相同，plan 依然没有被 response 使用。

## 改了什么

v1~v4 的条件配置用 `pad_token: pad`，response 之后的窗口不进 loss，模型学不到停止：EOS 率恒为 0，
每条都写到 512 上限（参考平均约 727 字符）。ELF 自己的条件任务（xsum、de-en）用 `pad_token: eos`。
v5 改成 EOS 填充 + 只监督 response 之后 64 个位置，其余协议与 v4 完全一致（prompt-causal 注意力、
同一日程、同一 backbone、10k 优化步、raw 权重、NFE 32、cap 512）。Ordered 另把干净 plan 行的比例
从 0.15 提到 0.5。

## 结果（validation 512 条，checkpoint 10000）

| 组 / 推理路径 | BLEU | R1 | R2 | RL | EOS 率 | 生成字符 | 答案率 | 准确率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ordered，plan-CFG 3 | 4.04 | 28.61 | 10.47 | 19.94 | 0.998 | 368 | 64% | 3/120 |
| Ordered，默认 | 3.97 | 28.44 | 10.45 | 19.86 | 0.998 | 367 | 64% | 1/120 |
| Ordered，关掉 plan | 3.93 | 28.23 | 10.34 | 19.67 | 0.996 | 369 | 65% | 1/120 |
| Diagonal | 4.14 | 28.62 | 10.28 | 19.84 | 0.998 | 374 | 56% | 0/120 |
| Register | 3.57 | 27.85 | 10.07 | 19.50 | 0.998 | 344 | 58% | 3/120 |
| Vanilla | 3.87 | 28.40 | 10.66 | 19.95 | 0.998 | 345 | 61% | 3/120 |

配对 bootstrap（Ordered plan-CFG 3 减对照，95% CI）：

| 对照 | R1 | R2 | RL |
|---|---:|---:|---:|
| Vanilla | +0.21 [-0.42, +0.84] | -0.19 [-0.63, +0.24] | -0.01 [-0.47, +0.46] |
| 同权重 null plan | +0.38 [-0.01, +0.78] | +0.13 [-0.14, +0.40] | +0.27 [-0.03, +0.58] |
| Diagonal | -0.01 [-0.62, +0.61] | +0.19 [-0.20, +0.58] | +0.10 [-0.35, +0.55] |
| Register | +0.76 [+0.15, +1.38] | +0.40 [-0.04, +0.82] | +0.44 [-0.02, +0.89] |

十二个区间里只有一个不跨零（对 Register 的 R1）。v4 的同类差值是 +2.27 / +2.32 / +0.59 / +0.55，
六项对结构对照全部显著。**差别只在 EOS**：v4 四组都在跑飞，ROUGE 差的是跑飞的程度，不是回答质量。

## plan 仍然没被使用（三条独立证据，v5 checkpoint）

1. 训练损失：四组回答端 l2 逐 bin 完全重合（0.354~0.360）。Ordered 有一半的行看得到干净 plan，
   如果 response 用它，这半数行的 l2 会低下来。
2. 教师强制探针 `outputs/tf_probe.py`（未见训练行）：喂真 plan、喂别人的 plan、喂纯噪声 plan，
   response 的 x0 预测误差三者相同（t=0.5 时 0.1595 / 0.1601 / 0.1502，噪声 plan 反而略低）。
3. Oracle 审计（`outputs/audits/v5_ordered_2k_oracle_plan_n128.json`，ckpt 2000）：
   真 plan 比模型生成的 plan 高 R1 +1.58 [0.55, 2.61]，但**和长度匹配的打乱 plan 没有差别**
   （-0.16 [-0.65, +0.35]）。response 感知的是"plan 干不干净、多长"，不是 plan 的内容。

## 信息量：plan 里确实有东西，只是没被用

冻结 T5 latent 上的岭回归，预测 response 的平均 latent（8000 行，7:1 划分，R²）：

| 特征 | resp 均值 | resp 前 4 段 | resp 词袋 2000 |
|---|---:|---:|---:|
| prompt 均值（512 维） | 0.309 | 0.111 | 0.085 |
| VAE plan（有效槽均值，128 维） | 0.439 | 0.164 | 0.116 |
| thinking 均值（512 维） | 0.532 | 0.185 | 0.135 |

plan 比 prompt 多带信息，thinking 原文更多。所以瓶颈在利用，不在 plan 表示本身。

## 主指标应该换

test 参考里 25%（250/1000）以 `\boxed{}` 或 `#### N` 收尾，是可验证子集。
`tools/eval_boxed_accuracy.py` 在这个子集上算答案率和准确率（T5 词表没有反斜杠和花括号，
`\boxed{42}` 往返成 `boxed42`，工具按此解析）。

- v4（不会停）：答案率 2~7%，准确率 0/252。
- v5（会停）：答案率 56~65%，准确率 0~3/120，即 0~2.5%，组间无差别。

模型学会了答案的格式，但算不对。在 105M 参数、5.5B token 微调预算下这不意外，但它说明
**ROUGE 上那 0.2~0.8 分的差异不代表推理能力**，而 512 条样本的 CI 宽度（±0.6）本来也容不下
这么小的效应。

## 正在跑的两个诊断

1. **上界**：vanilla + gold thinking 直接当条件前缀（`condition_includes_thinking`）。
   如果连 thinking 原文都帮不了 response，这个数据上"计划"没有可捞的收益。
   产出 `outputs/eval_v5/ceiling_sched256/`。
2. **能力上限**：ordered 且 `plan_done_frac=1`（response 训练时永远看到干净 gold plan，
   `diagnostic_run: true`）。如果连这样都不用 plan 内容，是通路本身不成立，不是训练协议稀释。
   产出 `outputs/audits/v5_oracleplan_10k_oracle_exactk_n128.json`。

## 产物

- 训练：`outputs/elf_b_conditional_common48w_{ordered,diagonal,register,vanilla}_causal_eos_v5/`
- 终评：`outputs/eval_v5/val512_ckpt10000/`（含 `accuracy.jsonl` 和配对比较 `cmp_*.json`）
- 中检：`outputs/eval_v5/mid_ckpt2000/`
- 探针：`outputs/tf_probe.py`、`outputs/info_probe.py`、`outputs/info_probe2.py`
- 代码：commit ec286e1、7d5233d、ff0991d、1ef7a19

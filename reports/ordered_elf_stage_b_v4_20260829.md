# Ordered ELF Stage-B v4 实验记录（2026-08-29）

## 结论

当前最强结果已经从旧双向 2k 推进到 prompt-aware 因果 10k，并完成唯一一次锁定协议的正式 test-1000。Ordered plan-CFG=3 得到 **BLEU/R1/R2/RL = 3.088/17.901/4.585/11.223**：

- 相对 Diagonal/Register，R1/R2/RL 六项差值全部显著为正；
- 相对同一 Ordered 权重的 null-plan，R1 +0.412 [0.292, 0.532]、R2 +0.207 [0.132, 0.282]、RL +0.298 [0.225, 0.374]，三项均显著，说明运行时 plan-state 通路在未见 test 上稳定可检出；
- 相对 matched Vanilla，RL +0.197 [0.029, 0.361]，R1 +0.162 [-0.119, 0.444]，但 R2 -0.149 [-0.288, -0.011]。因此这是质量权衡，不是全面超过 Vanilla。

当前最强且证据支持的论文表述为：

> 在相同训练样本与回答-token 预算下，prompt-aware Ordered 计划训练配合预先锁定的 plan-CFG 推理，在唯一一次 held-out test 上显著优于对角线计划与无信息寄存槽对照；关闭运行时计划会稳定损害三项 ROUGE。相对无计划 Vanilla，Ordered 提高 ROUGE-L、保持 ROUGE-1，但以较低 ROUGE-2 为代价。

这仍不是“回答稳定使用了计划语义”的结论。10k 未见池尾 exact-K 在锁定 plan-CFG=3 下方向转正但仍不显著，不能升级语义机制主张。模型的自由停止也仍未解决，plan-CFG=3 约增加 70% 采样墙钟，且当前只有一个训练 seed；三者必须与主结果一并报告。

## Prompt-aware 因果 10k 正式 test

协议在查看 test 前锁定：共同 10k checkpoint / 15,206,929 有效回答 tokens、raw weights、NFE=32、seed=42、test 1000 条、自由生成、统一最大回答长度 512 tokens；Ordered 使用 validation 阶段已锁定的 alpha=2 / plan-CFG=3。所有组使用相同样本顺序与成对随机噪声，不使用参考答案内容或逐样本参考长度。

| 组别/推理路径 | BLEU | R1 | R2 | RL | 平均生成字符 | EOS 率 |
|---|---:|---:|---:|---:|---:|---:|
| Ordered，锁定 plan-CFG=3 | 3.088 | **17.901** | 4.585 | **11.223** | 2023.1 | 0.000 |
| 同一 Ordered 权重 + null plan | 3.014 | 17.489 | 4.378 | 10.925 | 2013.7 | 0.000 |
| Vanilla | **3.215** | 17.739 | **4.734** | 11.026 | 2082.7 | 0.006 |
| Diagonal | 2.987 | 16.301 | 4.106 | 10.334 | 1912.5 | 0.001 |
| Register | 2.670 | 15.974 | 3.893 | 10.277 | 2038.2 | 0.010 |

### Ordered 的配对差值

| 对照/干预 | R1 差值 [95% CI] | R2 差值 [95% CI] | RL 差值 [95% CI] |
|---|---:|---:|---:|
| Diagonal | +1.600 [1.338, 1.861] | +0.479 [0.350, 0.608] | +0.889 [0.744, 1.034] |
| Register | +1.927 [1.671, 2.189] | +0.692 [0.561, 0.825] | +0.946 [0.797, 1.095] |
| 同权重 null plan | +0.412 [0.292, 0.532] | +0.207 [0.132, 0.282] | +0.298 [0.225, 0.374] |
| Vanilla | +0.162 [-0.119, 0.444] | -0.149 [-0.288, -0.011] | +0.197 [0.029, 0.361] |

每个区间来自 10,000 次 paired bootstrap。正式 test 复现了扩大 validation 的两个核心结论：结构对照和 null-plan 效应稳定；相对 Vanilla 则从 validation 上的“R1/R2 持平、RL 更高”收紧为“RL 更高、R1 持平、R2 略低”。因此主张应是 **Ordered 带来可检出的计划状态效应与输出质量权衡**，而不是质量全面领先。

## 历史双向 2k 正式 test（诊断基线）

该结果来自修复答案泄漏之前的双向 checkpoint，只用于说明旧训练协议与评测修复过程，不进入当前因果版本主表。协议：checkpoint 2k、raw weights、NFE=32、seed=42、test 1000 条、自由生成、统一最大回答长度 512 tokens，不使用参考答案内容或逐样本参考长度。

| 组别 | BLEU | R1 | R2 | RL | 平均生成字符 | EOS 率 | 主导 token 比例 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ordered | **2.376** | **15.500** | **3.841** | **9.790** | 2251.1 | 0.000 | **0.0871** |
| Vanilla | 1.973 | 14.551 | 3.413 | 9.411 | 2229.1 | 0.020 | 0.1273 |
| Register | 1.955 | 13.245 | 2.956 | 8.303 | 2137.7 | 0.000 | 0.1153 |
| Diagonal | 2.041 | 13.293 | 3.265 | 8.728 | 2228.8 | 0.000 | 0.1133 |

### Ordered 的配对增益

| 对照 | R1 差值 [95% CI] | R2 差值 [95% CI] | RL 差值 [95% CI] |
|---|---:|---:|---:|
| Vanilla | +0.950 [0.652, 1.246] | +0.428 [0.280, 0.574] | +0.379 [0.190, 0.566] |
| Register | +2.256 [1.954, 2.558] | +0.885 [0.743, 1.030] | +1.487 [1.312, 1.666] |
| Diagonal | +2.208 [1.907, 2.508] | +0.576 [0.435, 0.714] | +1.062 [0.885, 1.243] |

每个区间来自 10,000 次 paired bootstrap；四组使用相同 eval_id 顺序和成对随机噪声。

## 固定参考长度的机制诊断

该协议只使用参考回答的 token 数来屏蔽尾部，不使用参考内容，因此只能作为机制诊断，不能替代自由生成结果。

| 组别 | BLEU | R1 | R2 | RL | 生成计划槽均值 |
|---|---:|---:|---:|---:|---:|
| Ordered | **8.336** | **34.258** | **10.693** | **22.065** | 28.59 |
| Vanilla | 7.853 | 32.693 | 10.098 | 21.171 | N/A |
| Register | 7.894 | 30.746 | 8.426 | 19.713 | N/A |
| Diagonal | 7.709 | 31.416 | 9.149 | 20.531 | 22.42 |

Ordered 相对 Diagonal、Register 的三项 ROUGE，以及相对 Vanilla 的 R1/RL，配对区间均排除 0。它与自由生成的方向一致。

## Loss 是否正常

正常。2k 前最后 100 个日志点（microstep 15010–16000）的均值为：

| 组别 | total | token L2 | token CE | plan L2 | grad norm |
|---|---:|---:|---:|---:|---:|
| Ordered | 0.7121 | 0.5176 | 0.0645 | **0.1300** | 0.1855 |
| Diagonal | 0.7976 | 0.5172 | 0.0641 | 0.2163 | 0.1775 |

回答端 L2/CE 几乎相同，差异集中在计划损失；所有主实验和诊断支线均未出现 NaN/Inf。说明此前的问题不是数值爆炸，而是数据容量、生成协议和计划注意力方向。

Prompt-aware 因果三组在共同 1k checkpoint 前最后 100 个日志点（microstep 7010–8000）的均值进一步确认数值正常：

| 组别 | total | 回答端 L2+CE | plan L2 | grad norm |
|---|---:|---:|---:|---:|
| Ordered | 0.7582 | 0.5909 | 0.1674 | 0.2214 |
| Diagonal | 0.8135 | 0.5903 | 0.2231 | 0.2128 |
| Register | 0.5911 | 0.5911 | 0.0000 | 0.2048 |

三组回答端损失几乎完全相同，Ordered/Diagonal 的 total 只因额外包含 plan loss 而更高，不能与 Register 的 total 直接横比。三组 1k checkpoint 都恰好使用 1,524,937 个有效回答 tokens。

共同 5k 前最后 100 个日志点（microstep 39010–40000）同样正常：

| 组别 | total | 回答端 L2+CE | plan L2 | grad norm |
|---|---:|---:|---:|---:|
| Ordered | 0.6756 | 0.5479 | 0.1277 | 0.1770 |
| Diagonal | 0.7637 | 0.5455 | 0.2182 | 0.1676 |
| Register | 0.5471 | 0.5471 | 0.0000 | 0.1676 |

三组回答端损失仍严格匹配；Ordered 的额外 total 来自计划监督，不是回答建模退化或数值异常。三组 5k checkpoint 各自恰好使用 7,612,542 个有效回答 tokens。

三组全部完成 10k 后，最后 100 个日志点（microstep 79000–79990）的最终均值为：

| 组别 | 有效回答 tokens | total | 回答端 L2+CE | plan L2 | grad norm |
|---|---:|---:|---:|---:|---:|
| Ordered | 15,206,929 | 0.6288 | 0.5114 | 0.1174 | 0.1736 |
| Diagonal | 15,206,929 | 0.7164 | 0.5114 | 0.2050 | 0.1652 |
| Register | 15,206,929 | 0.5131 | 0.5131 | 0.0000 | 0.1702 |

Ordered/Diagonal/Register 的回答端 loss 几乎重合，梯度范数也同量级；Ordered/Diagonal 较高的 total 仍可完全由额外 plan loss 解释。三条日志均无 NaN/Inf，checkpoint 均在 global step 80000 正常写出。没有训练发散、数值故障或 Ordered 回答端优化独有退化。

## 已定位的三个问题

### 1. 计划文本被错误截到 256 tokens

Span-VAE 的真实容量是 16 tokens/slot × 64 slots = 1024 tokens，但旧 collator 沿用了 4 × 64 = 256：

- 80k 固定训练日程中 72,490 条（90.61%）受影响；
- 丢失 19,499,718 / 39,537,736 个 thinking tokens（49.32%）；
- 平均有效计划槽从应有的 31.36 被压到 15.70；
- 解释了旧 Ordered 2k checkpoint 的生成预算几乎恒定在 16。

v4 已改为 1024，80k 日程 thinking 截断为 0。旧 v3 Ordered/Diagonal 只保留作故障诊断，不进入主表。

### 2. 自由评测曾解码训练支持范围之外的尾部

条件训练中回答最大支持 1024 tokens；旧 pilot evaluator 却在 prompt 移除后把最多约 2048 个位置都计入回答。该尾部没有对应训练监督，会夸大超长与重复。

评测现已与正式生成路径对齐，并支持显式统一回答上限。512-token 上限在看 test 结果前由训练/验证长度分布确定：

- 覆盖 96.9% 验证回答，只截去 2.29% 的参考 tokens；
- test 分布相符：覆盖 96.5%，截去 2.14% 的参考 tokens。

无上限错误结果保存在 `outputs/pilot_eval/v4_ckpt2000/full_uncapped_bad/`，不得用于结论。

### 3. 双向注意力允许回答泄漏进计划槽

v4 主配置的 `plan_response_attention=bidirectional` 允许计划槽读取回答 token。实际 checkpoint 方向性探针显示：固定计划、只改变回答输入时，2k Ordered 的计划输出相对变化 36.50%；这说明 plan loss 可以借助回答状态，未被迫形成独立的先验计划。

同一个 Ordered 2k checkpoint、1000 条 validation、其余参数与噪声完全固定时：

| 推理干预 | R1 差值 | R2 差值 | RL 差值 |
|---|---:|---:|---:|
| Ordered 路径 - Diagonal 路径 | +0.006 [-0.071, 0.083] | +0.004 [-0.040, 0.048] | +0.016 [-0.032, 0.063] |
| Ordered 路径 - Null 计划 | +0.116 [0.029, 0.204] | +0.068 [0.014, 0.123] | +0.055 [-0.001, 0.111] |

所以运行时路径效应几乎为零，Null 效应虽在 R1/R2 上可检出，但远小于 Ordered 对训练组对照的增益。主收益目前应归因于训练辅助效应，不能归因于运行时有序规划。

第一版 `causal_bottleneck` 虽切断了回答→计划，却也把条件任务的 prompt 一并挡掉：计划只能看计划时钟和计划槽，无法知道用户问题。该版本在约 910–970 microsteps 即停止，未产出正式 checkpoint；其日志和 2-step smoke 已完整归档到 `outputs/diagnostics/no_prompt_causal_bad_20260829/`，不得进入结果表。

现已改成 `prompt_causal_bottleneck`：prompt、计划时钟和计划槽组成一个逐层封闭的子系统，能够相互读取，但不能读取回答或已经吸收回答信息的共享 token。真实 2k 权重方向探针结果为：

- 只改回答：回答→计划相对变化 **0**，回答→prompt 最大绝对变化 **0**；
- 只改 prompt：prompt→计划相对变化 **0.5653**；
- 只改计划：计划→回答相对变化 **0.1768**。

因此既消除了直接和间接答案泄漏，也保留了条件规划与计划→回答通路。最新 168 项回归测试和新的 2-step forward/backward/checkpoint smoke 均已通过；Ordered/Diagonal/Register 已用该掩码完成共同 10k。

## Prompt-aware 因果 1k 筛查（2026-08-30 更新）

协议：共同 1k optimizer steps / 1,524,937 有效回答 tokens，validation 128 条、raw weights、NFE=32、seed=42、自由生成 cap=512。该结果只用于故障筛查和 checkpoint 选择，不进入正式 test 主表。

| 组别/干预 | BLEU | R1 | R2 | RL | 平均生成字符 | EOS 率 |
|---|---:|---:|---:|---:|---:|---:|
| Ordered | 1.980 | 13.865 | 2.450 | 8.868 | 2235.0 | 0.000 |
| Diagonal | 1.312 | 12.190 | 1.958 | 7.875 | 1957.3 | 0.000 |
| Register | 2.117 | 14.684 | 2.655 | 9.238 | 2197.3 | 0.000 |
| Vanilla | **2.219** | **16.855** | **3.780** | **10.252** | 2210.9 | 0.000 |
| Ordered 权重 + null plan | 1.943 | 13.525 | 2.340 | 8.570 | 2225.3 | 0.000 |

Ordered 的配对差值为：

| 对照/干预 | R1 差值 [95% CI] | R2 差值 [95% CI] | RL 差值 [95% CI] |
|---|---:|---:|---:|
| Diagonal | +1.675 [0.845, 2.506] | +0.493 [0.165, 0.829] | +0.994 [0.478, 1.502] |
| Register | -0.819 [-1.573, -0.047] | -0.205 [-0.529, 0.106] | -0.369 [-0.819, 0.089] |
| Vanilla | -2.990 [-3.759, -2.248] | -1.330 [-1.763, -0.929] | -1.384 [-1.860, -0.926] |
| 同权重 null plan | +0.340 [0.110, 0.582] | +0.111 [-0.013, 0.237] | +0.299 [0.155, 0.451] |

这给出一个清晰的早期结论：因果 Ordered 已显著优于 Diagonal，而且关闭运行时计划会显著损害 R1/RL，说明修正后模型确实使用计划→回答通路；但 1k 时回答质量仍显著落后 Vanilla，R1 也落后 Register。因此 1k 机制通路通过、论文质量门槛未通过，不能作为主结果。

Ordered 生成计划预算均值为 40.18 槽，标准差 6.45，与 prompt 长度的相关系数为 0.742；它已不是旧容量错误造成的近常数预算。自由停止仍未学会，五组 EOS 率均接近 0。

## Prompt-aware 因果 2k 与推理协议选择

共同预算：2k optimizer steps / 3,021,202 有效回答 tokens。筛选协议仍为 validation 128 条、raw、NFE=32、seed=42、cap=512。

| 组别/推理路径 | BLEU | R1 | R2 | RL |
|---|---:|---:|---:|---:|
| Ordered，默认 plan-CFG=1 | 2.577 | 15.341 | 3.300 | 9.307 |
| Ordered，validation 选择 plan-CFG=3 | **2.727** | **15.764** | 3.476 | 9.537 |
| 同一 Ordered 权重 + null plan | 2.488 | 15.222 | 3.136 | 9.271 |
| Register | 2.090 | 12.756 | 3.177 | 8.274 |
| Diagonal | 1.873 | 13.322 | 3.103 | 8.941 |
| Vanilla | 2.140 | 15.704 | **3.706** | **9.924** |

默认 plan-CFG=1 时，Ordered 相对 Register 的 R1/RL 为 +2.585 [1.802, 3.416] / +1.033 [0.569, 1.514]，相对 Diagonal 的 R1 为 +2.019 [1.359, 2.714]；但相对 null-plan 只有 R2 +0.164 [0.026, 0.300] 显著，R1/RL 接近零。

在不改权重的前提下，validation 上的 plan-CFG sweep 为：

| plan-CFG | BLEU | R1 | R2 | RL |
|---:|---:|---:|---:|---:|
| 1.0 | 2.577 | 15.341 | 3.300 | 9.307 |
| 1.5 | 2.637 | 15.406 | 3.324 | 9.423 |
| 2.0 | 2.704 | 15.662 | 3.391 | 9.457 |
| 3.0 | **2.727** | **15.764** | **3.476** | **9.537** |

plan-CFG=3 相对默认值的 R1/RL 为 +0.423 [0.144, 0.713] / +0.230 [0.068, 0.398]。更关键的是，它相对同权重 null-plan 的 R1/R2/RL 为：

- +0.542 [0.203, 0.877]
- +0.340 [0.120, 0.564]
- +0.266 [0.078, 0.462]

三项区间全部大于 0，因此运行时 plan-state 通路在因果 mask 下可被稳定检出。相对 Vanilla 的 R1/R2/RL 差值为 +0.060 / -0.230 / -0.387，三个区间均跨 0：此处只能写“质量统计持平、plan-state 干预有效”，不能据此写“语义规划机制成立”或“全面超过 Vanilla”；后续 exact-K 内容置换给出了更严格的反证。

锁定 alpha=2 / plan-CFG=3 后，将样本扩大到 validation 512 条得到：

| 路径 | BLEU | R1 | R2 | RL |
|---|---:|---:|---:|---:|
| Ordered plan-CFG=3 | **2.432** | **15.025** | 3.254 | 9.253 |
| 同权重 null plan | 2.333 | 14.564 | 2.974 | 9.011 |
| Vanilla | 2.060 | 14.692 | **3.446** | **9.404** |

Ordered 相对 null-plan 的 R1/R2/RL 为 +0.461 [0.305, 0.618] / +0.280 [0.190, 0.377] / +0.242 [0.143, 0.340]，三项在扩大样本后仍稳定显著。相对 Vanilla 为 +0.333 [-0.070, 0.735] / -0.192 [-0.390, 0.000] / -0.152 [-0.400, 0.088]：BLEU/R1 较高，R2/RL 较低，仍属于混合胜负。

因此 2k 可以保留为“因果规划通路可检出”的机制证据，但不进入“质量全面优于 Vanilla”的最终主表；质量主结论继续由共同 5k/10k checkpoint 决定。

权重口径也已锁定为 raw：Ordered 2k EMA 的 BLEU/R1/R2/RL 为 2.070/15.048/2.945/9.142，均低于 raw，且生成计划预算均值仅 0.47 槽，说明 decay=0.999 的 EMA 在该阶段仍严重滞后并近似关闭计划。5k 不再重复 EMA 搜索。

plan-CFG 每个采样步额外增加一次 null-plan forward；本配置通过 4 个 learned self-conditioning CFG tokens 在一次调用内完成默认前向，因此 plan-CFG 使采样期模型前向数约变成 2 倍，实测吞吐约从 1.2 降到 0.7 samples/s（墙钟增加约 70%）。若最终采用，必须报告这项计算代价。

同一 1k Ordered 权重的轨迹 sweep 还给出方向性反证：alpha=0.5（规划落后）相对默认 alpha=2 的 R1 为 -0.335 [-0.569, -0.116]，RL 为 -0.212 [-0.367, -0.066]；alpha=1/1.5 与 2 近似持平，alpha=3 仅小幅上升且区间跨 0。因此锁定 alpha=2，不继续根据小样本向上搜索。

## Prompt-aware 因果 5k 验证

共同预算：5k optimizer steps / 7,612,542 有效回答 tokens。协议与选择规则均在看到该 checkpoint 前锁定：validation 128 条、raw、NFE=32、seed=42、cap=512、alpha=2；Ordered 同时报告默认 plan-CFG=1 和已由 2k validation 选择的 plan-CFG=3。

| 组别/推理路径 | BLEU | R1 | R2 | RL |
|---|---:|---:|---:|---:|
| Ordered，默认 plan-CFG=1 | 3.152 | 18.715 | 4.832 | 11.804 |
| Ordered，锁定 plan-CFG=3 | 3.247 | **19.195** | **5.124** | **11.855** |
| 同一 Ordered 权重 + null plan | 3.213 | 18.661 | 4.822 | 11.764 |
| Diagonal | 3.225 | 18.576 | 4.638 | 11.116 |
| Register | 3.301 | 18.119 | 4.603 | 11.188 |
| Vanilla | **3.892** | **20.209** | **5.459** | **12.365** |

锁定的 plan-CFG=3 相对结构对照的配对差值为：

| 对照/干预 | R1 差值 [95% CI] | R2 差值 [95% CI] | RL 差值 [95% CI] |
|---|---:|---:|---:|
| Diagonal | +0.620（跨 0） | +0.485 [0.119, 0.856] | +0.739 [0.277, 1.206] |
| Register | +1.076 [0.231, 1.977] | +0.521 [0.113, 0.922] | +0.667 [0.178, 1.201] |
| 同权重 null plan | +0.534 [0.137, 0.930] | +0.302 [0.076, 0.539] | +0.091（跨 0） |
| Vanilla | -1.014 [-1.689, -0.324] | -0.336（跨 0） | -0.510 [-0.973, -0.036] |

这把问题定位得更具体：训练没有崩，Ordered 也不是只靠额外槽位；规划监督和运行时规划确实带来可重复的局部收益。但 5k 的规划收益不足以抵消 Vanilla 主任务建模优势，尤其 R1/RL 仍显著落后。

随后对 5k raw 权重做了两项只读审计。第一项在 64 个固定日程样本上，以完全相同的噪声、时钟、分支、自条件和 dropout 分别反传回答 loss 与计划 loss；54 个 denoiser 样本的结果为：

- 共享 Transformer 上两类梯度的聚合余弦为 **-0.00002**，约一半样本为负，说明不是稳定的梯度冲突，而是几乎没有正迁移；
- 计划输入/时钟参数上的计划梯度范数是回答梯度的 **3.47 倍**，说明该接口主要被“重建计划”目标塑形，回答目标对“什么计划有用”的约束很弱；
- 配对前向的回答指标逐元素完全一致，排除了两次反传随机性不匹配。

第二项把计划目标拆成有效内容槽与尾部零槽，并复现正式的 32-NFE、planning-first alpha=2 计划生成路径。同一批未被 5k checkpoint 见过的日程样本上：

- 真实有效预算均值 33.70 槽，生成预算 34.34 槽：长度已经学准；
- 生成有效槽与配对真实计划潜变量的余弦只有 **0.043**；
- 但正确配对相对批内循环打乱配对仍有 +0.036 [0.029, 0.043] 的余弦优势，MSE 也小 -0.019 [-0.038, -0.0005]。

因此不能说生成计划完全没有内容，也不能把单一 gold thinking 当作唯一正确模式；但证据一致指向同一瓶颈：当前计划流先学会了预算和很弱的 prompt-specific 信号，却没有形成足够强、足够面向回答效用的语义计划。问题是**计划内容信噪比与 plan→response 对齐不足**，不是 loss 爆炸，也不是两项 loss 明显互相抵消。2k→5k 的生成预算 27.66→34.34、配对相对打乱余弦优势 0.024→0.036，说明计划生成能力仍在随训练改善，这也是继续看共同 10k 的直接依据。

最后用回答指标区分“计划生成不准”与“回答不会用计划”。在 checkpoint 尚未见过的后续日程样本上，保持回答初始噪声、时间网格和 SDE 噪声完全一致，比较模型生成计划与直接冻结 gold thinking 的干净 Plan-VAE 计划。扩大到 256 条后，oracle-matched 相对 generated 的 R1/R2/RL 仅为：

- +0.024 [-0.280, 0.311]
- +0.037 [-0.150, 0.217]
- +0.126 [-0.066, 0.317]

干净真实计划没有稳定优于模型生成计划，说明单纯继续提高 plan denoiser 精度不是主解。一个先前按 batch 循环打乱、但未逐样本匹配计划长度的 oracle 控制曾在 R2/RL 上显著；为了排除预算混杂，又从 1024 条候选中按真实有效槽数分组，只在 exact-K 样本对之间交换计划。此时 oracle-matched 相对 oracle-shuffled 的 R1/R2/RL 为：

- +0.149 [-0.109, 0.411]
- -0.053 [-0.228, 0.133]
- -0.061 [-0.237, 0.116]

三项均不显著，且 R2/RL 方向反转。结合 null-plan、plan-CFG 和梯度审计，最窄且最可靠的定位是：回答端会响应 plan state，并利用“是否有计划/计划预算”这类粗信号；但控制住预算后，尚无证据证明它稳定利用了计划内容。若 10k 仍失败，下一轮应直接加强**语义计划对回答 loss 的约束**，而不是继续调 plan-CFG 或只降低 plan reconstruction loss。

5k 不做 validation-512 扩大，也不看新 test：它已经在 validation-128 的质量门槛上明确失败。训练按原计划推进到共同 10k；如果 Ordered 10k 不能同时满足“超过 Register/Diagonal、相对 null-plan 变差可检出、质量不低于 matched Vanilla”，本轮因果版本就只保留机制证据，不宣称质量 SOTA。

## Prompt-aware 因果 10k 锁定验证

协议：共同 10k optimizer steps / **15,206,929 有效回答 tokens**、相同训练日程、validation 128 条、raw、NFE=32、seed=42、cap=512。Ordered 的 plan-CFG=3 与 alpha=2 均在看到 10k 前由 2k validation 锁定。

| 组别/推理路径 | BLEU | R1 | R2 | RL |
|---|---:|---:|---:|---:|
| Ordered，默认 plan-CFG=1 | 3.315 | 19.193 | 5.005 | 11.969 |
| Ordered，锁定 plan-CFG=3 | 3.494 | **19.583** | **5.248** | **12.095** |
| 同一 Ordered 权重 + null plan | 3.312 | 19.033 | 4.986 | 11.992 |
| Diagonal | 3.434 | 17.315 | 4.513 | 10.727 |
| Register | 3.185 | 17.263 | 4.600 | 11.043 |
| Vanilla | **3.666** | 18.990 | 5.128 | 11.542 |

锁定 plan-CFG=3 的配对差值：

| 对照/干预 | R1 差值 [95% CI] | R2 差值 [95% CI] | RL 差值 [95% CI] |
|---|---:|---:|---:|
| Diagonal | +2.268 [1.527, 3.037] | +0.734 [0.324, 1.153] | +1.367 [0.949, 1.801] |
| Register | +2.320 [1.508, 3.154] | +0.647 [0.221, 1.094] | +1.051 [0.574, 1.543] |
| Vanilla | +0.592 [0.014, 1.194] | +0.119 [-0.264, 0.487] | +0.552 [0.169, 0.953] |
| 同权重 null plan | +0.549 [0.175, 0.930] | +0.262 [0.025, 0.499] | +0.103 [-0.096, 0.309] |

因此 10k 通过锁定筛选门槛：结构对照全部被显著击败；相对 Vanilla 的 R1/RL 显著更高、R2 不劣；相对 null 的 R1/R2 显著。默认 plan-CFG=1 相对 null 基本相同，说明可检出的运行时计划收益依赖已锁定的 plan-CFG=3，同时带来约 70% 墙钟开销。

扩大到 validation 512 条后：

| 组别/推理路径 | BLEU | R1 | R2 | RL |
|---|---:|---:|---:|---:|
| Ordered，锁定 plan-CFG=3 | 3.109 | **18.025** | 4.422 | **11.242** |
| 同一 Ordered 权重 + null plan | 2.949 | 17.589 | 4.202 | 11.070 |
| Vanilla | **3.357** | 17.798 | **4.507** | 10.935 |

Ordered 相对 Vanilla 的 R1/R2/RL 为 +0.227 [-0.095, 0.561] / -0.085 [-0.268, 0.098] / +0.307 [0.100, 0.513]：R1/R2 统计持平，RL 显著更高。相对 null 为 +0.436 [0.269, 0.607] / +0.221 [0.118, 0.323] / +0.172 [0.072, 0.277]，三项均显著。扩大验证保持了“质量不低于 Vanilla、plan-state 干预有效”的门槛，因此按预注册规则执行了上文唯一一次 test-1000；test 没有再用于调参。

10k exact-K 进一步从同一确定性排列的 80,000 训练行之后取 512 个候选，选出 128 个精确匹配有效槽数的未见样本。审计工具同时修正了两处可能产生假证据的口径：用 checkpoint global step 强制排除已见行；在 oracle override 前保存原始计划噪声，确保 plan-CFG 的 null 分支不会被干净计划覆盖。锁定 plan-CFG=3 下 oracle-matched 相对 oracle-shuffled 的 R1/R2/RL 为：

- +0.118 [-0.240, 0.525]
- +0.096 [-0.169, 0.392]
- +0.198 [-0.077, 0.522]

三项方向为正但均不显著；oracle-matched 还显著低于模型生成计划。因而当前可以主张 plan-state/预算通路与回答质量收益，不能主张回答稳定使用了 gold thinking 的语义内容。

## Plan-mediation 探索支线

从相同 Ordered 5k checkpoint 续训到 6k（累计 **9,129,082 有效回答 tokens**）的 p=0.5 全时钟中介，在 7,990 个严格匹配的 microstep 上，相对原日程回答 loss 平均增加 **0.04238**；计划 loss 仅变化 +0.000005。高介入区间的回答惩罚为 +0.05085，低介入区间为 +0.02916，介入行数与惩罚的相关系数为 0.503，说明损失上升由中介本身造成，而非数据或学习率漂移。

| p=0.5 分支 6k 推理路径 | BLEU | R1 | R2 | RL |
|---|---:|---:|---:|---:|
| 默认 | 2.906 | 16.732 | 4.438 | 10.753 |
| plan-CFG=3 | 2.922 | 17.224 | 4.592 | 11.045 |
| null plan | 2.874 | 16.488 | 4.368 | 10.648 |

plan-CFG=3 相对 null 的 R1/R2/RL 为 +0.736 [0.333, 1.142] / +0.224 [0.001, 0.451] / +0.398 [0.124, 0.682]，说明通路更强；但相对原 Ordered 5k plan-CFG=3 分别下降 -1.971/-0.532/-0.809，三项均显著，因此质量门槛明确失败。它证明“强迫使用当前计划”并不能自动得到更好回答。

后续 `t_plan>=0.75` 修正版只在较干净的计划时钟上切断旁路。完整 7,990 个配对 microstep 的回答 loss 仍增加 **0.04100**，只比 p=0.5 的 +0.04238 少 0.00138；后半程仍为 +0.03956。它的 plan-CFG=3 验证结果为 3.012/16.897/4.174/10.802，相对原 Ordered 5k 的 R1/R2/RL 分别显著下降 -2.299/-0.950/-1.052，且 R2 还显著低于 p=0.5 分支。结论是：干净时钟筛选降低了 L2 代价，却因覆盖全部 decoder 行增大 CE 代价，没有形成净修复；该方向停止。

## Vanilla checkpoint 曲线

同一 validation-128 协议下，Vanilla 的训练曲线明显非单调：

| checkpoint | BLEU | R1 | R2 | RL |
|---:|---:|---:|---:|---:|
| 1k | 2.219 | 16.855 | 3.780 | 10.252 |
| 2k | 2.140 | 15.704 | 3.706 | 9.924 |
| 5k | **3.892** | **20.209** | **5.459** | **12.365** |
| 10k | 3.666 | 18.990 | 5.128 | 11.542 |

5k 相对 10k 的 R1 为 +1.219 [0.586, 1.845]，RL 为 +0.822 [0.403, 1.250]；相对 1k/2k 的三项 ROUGE 区间也全部大于 0。说明 1k 只能用于排除故障，不能据此提前终止结构实验。因果 Ordered 5k 已在 matched Vanilla 门槛失败，但其 plan-CFG=3 的 19.195/5.124/11.855 已接近 Vanilla 10k 的 18.990/5.128/11.542；因此继续到共同 10k 是必要的匹配比较，而不是根据 Vanilla 曲线挑更弱对手。

## 已否定的支线

从相同 Ordered checkpoint 1k 分叉，额外强调低 `t_plan` 的 boost=3 在 1.5k 相对 boost=0：

- R1 -0.759，区间跨 0；
- R2 -0.907，95% CI [-1.729, -0.173]；
- RL -0.356，区间跨 0。

因此不继续该支线，主实验保持原始未加权 plan loss。

NFE=8 的低步数假设也未形成主结果。2k Ordered 默认/plan-CFG=3/同权重 null/Vanilla 的 R1 分别为 15.038/15.080/14.832/14.289；Ordered plan-CFG=3 相对 Vanilla 的 R1 为 +0.792 [0.112, 1.442]，但 R2/RL 为 -0.246/-0.148 且区间跨 0。相对 null 也只有 R1 显著。它说明低步数下不会崩溃，可作为效率附录，但不支持“低 NFE 会放大规划优势”的主假设。

## 可比性

- 四组从同一官方 backbone checkpoint `outputs/reference_elf_b_owt/checkpoint_95085` 启动；
- 固定训练日程 fingerprint：`509d1b0b0da4c1b592af3cfe45c149f48cb08c64e056f463fa0992217096068a`；
- checkpoint 2k 时四组各自恰好训练 3,021,202 个有效回答 tokens；
- 当前 10k 主比较各组使用相同日程，并各自恰好训练 15,206,929 个有效回答 tokens；
- Register 与 Ordered 拥有相同的 64 个额外槽位，但槽位只携带噪声且无计划监督；
- checkpoint 级更新审计：Ordered/Diagonal 的 plan head 在 1k→2k 实质变化并落盘；Register 的 `plan_norm`/`plan_head` 逐元素完全不变，符合无 plan loss 的预期；
- 当前正式 test 的五条路径使用相同 1000 条样本、相同顺序、成对噪声、raw weights、NFE=32、cap=512；
- 回归测试：168 passed。

## 仍需明确的限制

- 10k exact-K 内容置换不显著，当前证据只支持 plan-state/预算通路，不支持回答稳定利用 gold thinking 的语义内容。
- Ordered 在当前 test 平均生成 2023.1 字符，而参考均值约 727 字符；EOS 率为 0。相对指标有效，但自由停止尚未解决，当前依赖统一 512-token cap。
- 当前正式结论来自一个训练 seed 的 10k checkpoint；1000 条 test 和配对 bootstrap 支持样本层面的稳定性，不等同于多训练 seed 稳定性。
- plan-CFG=3 约增加 70% 采样墙钟；其质量收益相对 Vanilla 也是 RL 上升、R2 下降的权衡，不是全面领先。
- test 只在 validation-128 门槛通过并由 validation-512 复核后查看一次，之后未再调整 cap、权重、alpha 或 plan-CFG。

## 下一步

1. 主实验闭环已完成：共同 10k、validation-128/512、未见池尾 exact-K、唯一正式 test-1000 和 paired bootstrap 均已落盘。
2. p=0.5 与 `t_plan>=0.75` 两条 plan-mediation 6k 支线均已完成且质量明确失败；停止继续搜索 mediation 比例或时钟阈值。
3. 若论文需要训练随机种子层面的稳定性，下一项高价值实验是按完全相同协议补两个训练 seed；在此之前不要把单 seed 的样本层面显著性写成训练稳定性。
4. 若继续改模型，应让回答 loss 直接约束计划的语义效用，而不是继续降低 plan reconstruction loss 或调 plan-CFG；新结构需重新预注册 validation/test 门槛。
5. 自由停止应作为正交问题单独修复和评测，不能靠改变 cap 来美化当前主表。

## 关键产物

- 正式 test：`outputs/formal_eval/v4_ckpt2000_test_cap512/`
- prompt-aware 因果 10k 正式 test：`outputs/formal_eval/prompt_causal_ckpt10000_selected_test_n1000_cap512/`
- 2k validation cap512：`outputs/pilot_eval/v4_ckpt2000_cap512/`
- 2k 固定长度机制诊断：`outputs/pilot_eval/v4_ckpt2000/`
- 计划容量审计：`outputs/audits/fullplan_capacity_audit_20260829.json`
- 计划方向性审计：`outputs/audits/plan_attention_direction_probe_20260829.json`
- 1000 条同-checkpoint 轨迹/null 干预：`outputs/formal_eval/v4_ckpt2000_validation_trajectory_intervention_cap512/`
- low-t 否定试验：`outputs/pilot_eval/lowt_fullplan_1k_to1500/`
- prompt-aware 因果方向审计：`outputs/audits/prompt_causal_direction_probe_20260829.json`
- prompt-aware checkpoint 规划参数更新审计：`outputs/audits/prompt_causal_checkpoint_plan_update_20260830.json`
- prompt-aware 2k/5k 回答-计划梯度审计：`outputs/audits/prompt_causal_2k_same_rows_plan_response_gradient_conflict_20260830.json`、`outputs/audits/prompt_causal_5k_plan_response_gradient_conflict_20260830.json`
- prompt-aware 2k/5k 计划去噪与完整 rollout 审计：`outputs/audits/prompt_causal_2k_plan_denoising_quality_20260830.json`、`outputs/audits/prompt_causal_5k_plan_denoising_quality_20260830.json`
- prompt-aware 5k oracle-plan 回答上界与 exact-K 内容置换：`outputs/audits/prompt_causal_5k_oracle_plan_response_upper_bound_n256_20260830.json`、`outputs/audits/prompt_causal_5k_oracle_plan_response_upper_bound_n256_exactk_20260830.json`
- prompt-aware 10k validation-128：`outputs/pilot_eval/prompt_causal_ckpt10000_validation_n128_cap512/`、`outputs/pilot_eval/prompt_causal_ckpt10000_selected_validation_n128_cap512/`
- prompt-aware 10k 扩大验证：`outputs/pilot_eval/prompt_causal_ckpt10000_selected_validation_n512_cap512/`
- prompt-aware 10k 未见池尾 exact-K：`outputs/audits/prompt_causal_10k_pcfg3_oracle_plan_exactk_n128_20260830.json`
- prompt-aware 因果 smoke：`outputs/smoke_prompt_causal_ordered_v1/checkpoint_2`
- plan-mediation 恢复 smoke：`outputs/diagnostics/ordered_causal_mediation_p50_resume5k_smoke_v1/`
- plan-mediation p=0.5 探索训练：`outputs/elf_b_conditional_common48w_ordered_causal_mediation_p50_from5k_to6k_v1/`
- plan-mediation p=0.5 配对训练审计：`outputs/audits/prompt_causal_mediation_p50_5k_to6k_paired_training_log_20260830.json`
- plan-mediation `t_plan>=0.75` 回退训练：`outputs/elf_b_conditional_common48w_ordered_causal_mediation_t075_from5k_to6k_v1/`
- prompt-aware 因果 1k validation：`outputs/pilot_eval/prompt_causal_ckpt1000_validation_n128_cap512/`
- prompt-aware 因果 1k alpha sweep：`outputs/pilot_eval/prompt_causal_ckpt1000_alpha_sweep_validation_n128_cap512/`
- prompt-aware 因果 2k validation：`outputs/pilot_eval/prompt_causal_ckpt2000_validation_n128_cap512/`
- prompt-aware 因果 2k plan-CFG sweep：`outputs/pilot_eval/prompt_causal_ckpt2000_pcfg_sweep_validation_n128_cap512/`
- prompt-aware 因果 2k 锁定协议扩大验证：`outputs/pilot_eval/prompt_causal_ckpt2000_selected_validation_n512_cap512/`
- prompt-aware 因果 2k NFE=8 诊断：`outputs/pilot_eval/prompt_causal_ckpt2000_nfe8_validation_n128_cap512/`
- prompt-aware 因果 5k 默认验证：`outputs/pilot_eval/prompt_causal_ckpt5000_validation_n128_cap512/`
- prompt-aware 因果 5k 锁定协议验证：`outputs/pilot_eval/prompt_causal_ckpt5000_selected_validation_n128_cap512/`
- Vanilla checkpoint 曲线：`outputs/pilot_eval/vanilla_training_curve_validation_n128_cap512/`
- 无法读取 prompt 的旧 causal 运行归档：`outputs/diagnostics/no_prompt_causal_bad_20260829/`
- 因果版本正式训练：
  - `outputs/elf_b_conditional_common48w_ordered_causal_10k_v1/`
  - `outputs/elf_b_conditional_common48w_diagonal_causal_10k_v1/`
  - `outputs/elf_b_conditional_common48w_register_causal_10k_v1/`
- 修正后主 checkpoint：
  - `outputs/elf_b_conditional_common48w_ordered_10k_v4_fullplan_r1/checkpoint_2000`
  - `outputs/elf_b_conditional_common48w_diagonal_10k_v4_fullplan_r1/checkpoint_2000`

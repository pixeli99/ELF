# Ordered ELF

在 ELF(连续 flow matching 语言模型,T5-small latent 空间,PyTorch 版)上加一条 K=16 个槽的
plan 流,和 token 流各带一个噪声时钟 (t, t_plan) 联合训练。plan 的训练目标是把干净 latent
按位置切成 K 段做 mean-pool 再白化——它是 x0 的确定函数,不含 token 流之外的任何信息,
所以如果有收益,只能来自"先解全局、再解局部"这个顺序本身,而不是额外信息或容量。

推理时同一个 checkpoint 可以走不同轨迹:t_plan = min(1, α·t)。α=1 对角,α>1 plan 领先,
α<1 plan 滞后,α=0(null)plan 恒为噪声。`num_plan_slots: 0` 时和原版 ELF 完全一致。

## 环境

```bash
pip install -r requirements.txt
```

## 数据

- 无条件:`embedded-language-flows/openwebtext-t5`(OpenWebText,1024 token,已预编码成 t5-small latent,首次运行自动从 HF 拉)
- 条件任务(之后):xsum / wmt14 de-en。source 已经提供全局信息,预期 plan 收益缩水,用来验证机制解释

## 要训的模型

| 组 | 配置 | 说明 |
| --- | --- | --- |
| vanilla | `train_owt_ELF-B.yml` | `num_plan_slots: 0`,原版 ELF 基线 |
| register | `train_owt_ELF-B_register.yml` | 槽在,但输入是纯噪声(t_plan 恒 0)、没有 plan loss。量"多 16 个计算 token"本身值多少。注意 `plan_loss_weight: 0` 不等于这一组:那样输入还是会把池化目标漏进去 |
| ordered | `train_owt_ELF-B_ordered.yml` | 主模型。t_plan 对 t 条件均匀 + 15% 概率 t_plan=1(领先轨迹有一半步数停在 t_plan=1 上,连续分布采不到,必须给原子);这样任何单调轨迹的训练覆盖密度都一样,轨迹之间的对比才公平 |
| diagonal | ordered + `plan_diag_frac: 1.0, plan_done_frac: 0.0` | 单时钟双流。用来把"有 plan"和"顺序"两个因素拆开 |

```bash
NGPU=8 bash scripts/launch.sh train src/configs/training_configs/train_owt_ELF-B_ordered.yml
```

ordered 首次启动会先在 64 个 batch 上统计 plan 目标的均值/方差做白化(不白化的话,池化目标的
std 只有 token 的一半不到,同一个 t 下 plan 的信噪比反而更低,和设计意图相反)。统计量存在
checkpoint 里,resume 不会重跑。

训练中主要看 `plan_l2`:它是白化空间里的 plan x-MSE,1.0 正好等于"只会预测均值"的水平,
必须收敛到明显低于 1,否则 plan 头没学到东西,后面的评测不用做了。

## 怎么测

一份 ordered checkpoint,四条轨迹 × 两档步数,`ordered_sampling_configs.yml` 已配好一次跑齐:
diagonal、α=2(领先)、α=0.5(滞后)、null,各 32 步和 8 步,1000 条样本,GPT-2 Large 算
gen-PPL 和熵(eval 命令与原版相同,输出目录按轨迹分开)。

判读:

- 预期 α=2 ≥ diagonal > α=0.5。如果三条打平,顺序无效。
- 8 步下的差距应该比 32 步大。理论上精确后验下任何单调轨迹采出同一个分布,顺序的收益
  只能体现在少步数的近似误差里,所以结论要看 quality-vs-NFE 曲线,不看单点。
- ordered 要赢 register。赢不了说明涨点只是多了几个计算 token,和 planning 无关。

训完再跑探针:

```bash
python src/plan_probes.py --config src/configs/training_configs/train_owt_ELF-B_ordered.yml \
    --checkpoint outputs/elf_b-owt-ordered --trajectory planning_first --alpha 2.0 --steps 32
```

- graft:把 batch 内的 plan 轨迹换成别的样本的,token 输出应该跟着对方的 plan 变
- consistency_matched 应明显小于 consistency_mismatched;两者接近说明两条流各说各话,
  测到的收益多半是 register 效应

## 之后

- plan 目标换冻结句向量(分段池化 + 白化,注意 embedding 空间的锥形偏置)。语义最强,
  是整个假设最快的检验;但混入了外部模型的知识,只能作性能对照,不进主结论
- LD4LG 式压缩 AE 目标(Perceiver 压到 k 槽、重建 x_tok、训完冻结),k ∈ {8,16,32,64} 扫描,
  画 plan 信息量 vs 顺序收益
- λ_plan ∈ {0.1, 1}、plan-CFG(`plan_cfg_scale`,以 t_plan=0 的初始噪声 plan 为 null 外插,
  不需要额外 dropout 训练)
- 相关工作正面对比 CCDD (2510.03206)、LADD (2510.18114)、CADD (2510.01329):都是 masked
  离散扩散 + 外部预训练表征通道;我们是纯连续流、零新增信息目标、同一模型上推理期可调顺序

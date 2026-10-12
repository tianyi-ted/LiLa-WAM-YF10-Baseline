# LiLa-WAM 动作生成 Flow Matching 与 OPD 实验指南

> 文档状态：动作生成研究专项 v1.0  
> 更新日期：2026-09-14  
> 研究范围：只聚焦动作 Flow 的生成过程与后训练；暂不把环境状态分布 OPD、完整世界模型和真机在线学习作为第一阶段主线。

本文以当前 LiLa-WAM 十任务 baseline 为起点，详细解释现有 Conditional Flow Matching（CFM）的训练和推理过程，分析 teacher-based Flow-OPD 与 teacher-free Self-OPD 会改变什么，并给出从诊断、接口改造、单任务验证到十任务实验的可执行路线。

总路线图见：[EG_DUAL_OPD_RESEARCH_ROADMAP.md](EG_DUAL_OPD_RESEARCH_ROADMAP.md)。本文是其中“动作生成 Flow 层”的细化版。

---

## 1. 本阶段到底研究什么

本阶段固定视觉表征、任务条件和环境观测来源，尽量只改变动作生成器的训练分布与学习目标：

```text
固定部分：DINOv3 + VTT + robot state + 数据预处理
                                |
                                v
研究部分：动作 Flow 的起点、内部轨迹、局部分支、监督目标和损失
                                |
                                v
固定评测：正常 10-step Euler -> B-Spline -> 执行前 16 步
```

短期目标不是一次性做成完整“双层 OPD”，而是依次回答三个问题：

1. 当前 CFM 在专家插值状态上训练、在学生自产生状态上推理，这个差异是否真的影响动作质量？
2. 用强教师在学生 Flow 状态上提供密集速度标签，是否优于等计算量 continued SFT？
3. 不依赖教师，仅通过学生局部分支、reward 和 pull-push，能否进一步超过 teacher-based OPD？

如果第一个问题都没有可测差异，就没有必要直接投入昂贵的 Self-OPD。

---

## 2. 当前 LiLa-WAM 的动作 Flow Matching

### 2.1 条件输入与动作输出

对每个训练样本，动作模型接收：

- 当前图像的冻结 DINOv3 多层特征。
- 归一化 robot state。
- 当前任务的 VTT 向量。
- 一个被加噪的 32×14 动作块。
- 连续时间 `t`。

模型输出与动作块同形状的速度场：

```text
v_theta(x_t, t | DINO, state, VTT) -> [B, 32, 14]
```

当前实现位置：

- 输入 token 组装与 DiT：[models/vla_model_fm.py](../models/vla_model_fm.py)
- CFM loss：`calc_flow_matching_loss()`
- DINO 与动作模型封装：[models/model_runner.py](../models/model_runner.py)
- 训练循环：[train.py](../train.py)
- Euler 推理：[robotwin_infer.py](../robotwin_infer.py)

### 2.2 训练：从噪声和专家动作构造一条直线

当前代码记号是：

- `x0`：标准高斯噪声。
- `x1`：数据集中的专家动作块。
- `t=0`：噪声端。
- `t=1`：动作数据端。

每个训练样本的计算过程是：

```text
x0 ~ N(0, I)
t  ~ LogitNormal(mu=0, sigma=1)
xt = (1-t) * x0 + t * x1
v_target = x1 - x0
v_pred = v_theta(xt, t | condition)
L_CFM = mean(||v_pred - v_target||^2)
```

直线路径的目标速度是常量 `x1-x0`。网络需要从不同 `t`、不同 `xt` 和条件信息中恢复这条速度。

这不是直接预测最终动作 `x1`，而是在学习一个连续向量场。理想情况下，对 ODE 积分便能把噪声分布输运到专家动作分布。

### 2.3 时间采样

当前配置使用：

```yaml
time_sampler: logit_normal
time_mu: 0.0
time_sigma: 1.0
```

也就是先采高斯变量再通过 sigmoid 得到 `t`。它相对均匀采样更集中于中段，极靠近 0 或 1 的样本较少。影响是：

- 中段速度场训练更充分。
- 极噪声端和极数据端可能相对欠训练。
- OPD 若特别强调早期 Flow step，需要检查 baseline 在 `t≈0` 的误差，而不能默认它训练充分。

### 2.4 Padding 与当前动作监督

动作块固定为 32 步。episode 尾部不足 32 步时，当前数据策略会保留补齐的末尾动作；这些保持动作也参与 CFM 监督。这是目前有意保留的 baseline 语义，OPD 首轮不能顺便改变 mask 或 padding，否则无法判断提升来自 OPD 还是数据目标变化。

Self-OPD 的 reward 若使用专家动作距离，也必须采用与 baseline 相同的 32 步表示；可以额外报告“有效动作区间”和“完整含保持区间”两个距离，但不能偷偷更换训练语义。

### 2.5 辅助未来特征损失

当前总损失还包含未来 DINO 特征预测：

```text
L_base = L_CFM + lambda_future * L_future_feature
```

它塑造动作模型内部的条件 token，但不直接评价生成动作是否成功。进行纯 Action-Flow OPD 时建议：

- DINO 始终冻结。
- 首轮保留 future decoder 及其原始 replay loss，避免同时改两个研究因素。
- OPD 分支数据若没有可靠 future target，不要伪造 future loss；可让 OPD batch 只计算动作 OPD，示范 replay batch 继续计算原始两项损失。

### 2.6 推理：学生自己的 10 步 ODE 轨迹

推理从新的高斯噪声出发：

```text
x_0 ~ N(0, I)
for j = 0...9:
    t_j = j / 10
    v_j = v_theta(x_j, t_j | condition)
    x_(j+1) = x_j + 0.1 * v_j
A = x_10
```

随后：

```text
A_normalized
 -> denormalize
 -> 对完整 32 步做三次 B-Spline
 -> 取前 16 步加入动作队列
 -> 环境逐步执行
```

重要的是，训练点 `xt=(1-t)x0+t*x1` 由专家动作定义；推理点 `x_j` 由学生前面所有速度预测累积产生。一次早期预测误差会改变后续所有输入，这就是动作生成过程内部的 exposure bias。

---

## 3. 当前 CFM 的优势与限制

### 3.1 优势

- 训练稳定，本质是监督回归。
- 每个样本通常只需一次带梯度的动作模型前向。
- 不需要环境 reward、critic 或教师模型。
- 可以利用完整专家数据中的所有帧。
- 能表达多模态连续动作分布，不必强制输出单个均值动作。
- ODE 推理步骤可调，当前 10 步在速度与质量之间折中。

### 3.2 限制

#### 限制 A：训练状态与推理状态不一致

训练只看专家条件插值状态，推理却看学生自身轨迹。如果学生偏离训练直线，后面的速度预测可能失真。

#### 限制 B：损失与任务成功不一致

CFM 优化速度回归误差，不直接区分：

- 抓取位置偏差 1 cm 是否会失败。
- 夹爪闭合早一帧还是晚一帧。
- 平滑后的动作是否进入接触区域。
- 整个动作块在环境中是否推进任务。

#### 限制 C：所有 Flow 时间点只受同一种离线目标约束

早期 step 决定动作的全局模式，晚期 step 修正精细位置，但基础 CFM 并不利用当前模型在哪些 step 最容易犯部署错误的信息。

#### 限制 D：多任务数据不等于多任务最优教师

十任务共享参数，某些任务可能欠拟合。继续 SFT 只能重复平均离线监督，未必能修复学生特定 Flow 轨迹上的错误。

---

## 4. Teacher-based Action-Flow OPD

### 4.1 核心区别

普通 CFM 在“专家定义的 `xt`”上学习专家直线速度；Flow-OPD 在“学生自己访问的 `x_j`”上学习教师速度：

```text
普通 CFM：
expert action -> construct x_t -> student predicts -> match x1-x0

Flow-OPD：
student SDE rollout -> obtain student x_j
                    -> frozen teacher labels v_teacher(x_j, t_j)
                    -> student matches teacher at its own x_j
```

其关键不是把教师输出预先保存成固定数据，而是学生参数变化后重新采样自己的 Flow 轨迹。教师必须在学生真正访问的中间状态上给标签。

[Flow-OPD](https://arxiv.org/abs/2605.08063) 将确定性 ODE 转换为带随机探索的等价 SDE；学生和教师的局部转移核共享协方差时，两者 KL 可化为带时间权重的速度场均方误差，因此能直接回传梯度，不需要 PPO 的高方差 score-function estimator。其官方实现位于 [CostaliyA/Flow-OPD](https://github.com/CostaliyA/Flow-OPD)。

### 4.2 概念损失

令学生在自己的 SDE 轨迹上访问 `x_j`，教师在完全相同的条件、时间和 `x_j` 上输出目标：

```text
v_s = v_theta(x_j, t_j | c)
v_T = stopgrad(v_phi(x_j, t_j | c))

L_flow_opd = sum_j w_j * ||v_s - v_T||^2
```

再增加 baseline 锚定：

```text
L_total = L_flow_opd
        + lambda_replay * L_CFM(expert replay)
        + lambda_future * L_future(expert replay)
```

`w_j` 不能随意照抄图像论文数值。它取决于 SDE 方差、时间方向和离散步长；应先完成 LiLa 时间坐标下的推导和数值检查。

### 4.3 教师从哪里来

这是机器人版本最关键的现实问题。

#### 方案 A：单任务 specialist teachers

从十任务 baseline 分别对每个任务低学习率微调，得到单任务教师。OPD 时由 VTT/task name 路由到对应教师。

优点：

- 最接近 Flow-OPD 的多教师思想。
- 教师容易在对应任务上超过共享 baseline。
- 可以研究多任务共享学生是否吸收 specialist 能力。

缺点：

- 十个完整教师存储和推理成本高。
- 单任务教师可能遗忘其他任务，但硬路由下不构成直接问题。
- 必须先证明每个教师确实优于学生。

建议先选 3 个中低成功率任务训练 specialist，不要一开始训练十个。

#### 方案 B：更强架构教师

训练更深/更宽的动作模型作为教师，再蒸馏回当前模型。

优点是典型的容量蒸馏；缺点是训练成本较高，而且研究重点容易变成模型压缩。

#### 方案 C：教师 ensemble

用多个 seed/checkpoint 的速度平均或根据验证表现路由。它容易获得更稳定标签，但推理成本最高，并可能把多模态动作平均成不合理方向。

#### 无效方案：同权重自我蒸馏

如果把当前 baseline 冻结成教师，学生又从完全相同权重初始化，在同一个 `x_j,t,c` 上有：

```text
v_student == v_teacher
```

初始 OPD loss 接近 0，没有改善信号。只有学生结构、采样步数、能力或初始化与教师真正不同，这个实验才有意义。

### 4.4 Flow-OPD 预期优势

- 在学生自己产生的 Flow 状态上纠错，减少内部 exposure bias。
- 每一步都有教师速度标签，比只用最终成功率更密集、更稳定。
- 可把多个单任务 specialist 合并到一个十任务 student。
- 与当前动作模型接口匹配，只需输入同一个 `x_j,t,c`。
- 可以保留 CFM replay，减少多任务遗忘。

### 4.5 Flow-OPD 的局限

- 上限强依赖教师；弱教师不会自动产生强学生。
- 教师在学生 OOD Flow 状态上的速度也可能不可靠。
- 训练一次需要学生 rollout、教师标注和学生梯度前向，远慢于一次 CFM。
- 多个 specialist 的训练成本与存储成本较高。
- 只修正动作生成轨迹，不修正机器人进入的 OOD 环境状态。

---

## 5. Teacher-free Action Self-OPD

### 5.1 核心区别

Self-OPD 不询问外部教师，而是在学生每个 Flow step 附近产生若干局部分支，把最终 reward 较好的方向作为正目标、较差方向作为负目标。

```text
parent x_j
   |
   +-> deterministic next state -> complete ODE -> reference reward
   |
   +-> stochastic branch 1 -> complete ODE -> reward r1
   +-> stochastic branch 2 -> complete ODE -> reward r2
   +-> ...
   +-> stochastic branch K -> complete ODE -> reward rK

rewards -> relative advantages -> pull good velocities / push bad velocities
```

[Self-OPD](https://arxiv.org/abs/2608.26872) 的关键组件是：学生局部 SDE 分支、从同一父状态出发的确定性 self-reference、标准化相对优势、全分支 pull-push、负分支的方向衰减和按 SDE 方差归一化的 step-wise loss。

### 5.2 分支与 self-reference

在第 `j` 个学生状态，从确定性预测附近采样 K 个分支：

```text
x_(j+1)^k = mu_theta(x_j, t_j) + sigma_j * sqrt(dt) * z_k
z_k ~ N(0, I)
```

每个分支用确定性 ODE 完成到动作块 `A_k`。从同一父状态完成的原始确定性轨迹得到 `A_ref`。经过 reward 后：

```text
Adv_k = (R(A_k) - R(A_ref)) / (std(R(A_1...A_K)) + eps)
```

使用相对 reference 而不是全局固定阈值的好处是：它回答“这次局部扰动是否比学生默认选择更好”，不同任务和不同难度状态更容易比较。

### 5.3 Pull-push

- `Adv_k > 0`：将当前学生速度向候选有效速度拉近。
- `Adv_k < 0`：将当前学生速度从该候选方向推开。
- 如果负分支方向与最佳正分支方向接近，则削弱 push，避免把有用方向一起排斥。
- 按局部 SDE 转移方差归一化，使不同时间步的更新尺度可比较。

首个机器人实现不建议立刻启用完整负号 loss。更稳妥的递进是：

1. Best-positive only。
2. All-positive advantage-weighted pull。
3. 加入裁剪后的 negative push。
4. 最后加入 direction-aware attenuation 和论文式时间权重。

每一步都必须与前一步独立比较，否则出现发散时无法定位原因。

### 5.4 LiLa-WAM 的时间方向必须转换

Self-OPD 论文的符号把一端记作数据、另一端记作噪声，并沿反向时间采样；当前 LiLa 代码则是：

```text
LiLa: t=0 noise -> t=1 action
```

因此不能把论文中的 `sigma_t` 和 early/late 权重直接复制到代码。若论文时间为 `s`，LiLa 时间为 `t`，通常需要先做：

```text
s = 1 - t
```

再推导 SDE drift、方差和权重。端点附近还要使用 `eps` 或跳过端点，避免 `sqrt(t/(1-t))` 一类调度产生除零。正式实现前应写数值单元测试验证：

- `eta=0` 时严格退化为当前 Euler ODE。
- 固定噪声时结果可复现。
- 变换后的方差随时间符合预期。
- branch effective velocity 与实际 next-state 重构一致。

### 5.5 机器人动作 reward 怎么设计

Self-OPD 的成败很大程度取决于 reward。建议分三级验证。

#### R0：专家动作距离，仅用于代码正确性

在离线数据的条件观察上：

```text
R_bc(A) = - weighted_distance(A, A_expert)
```

可以按双臂位置、姿态、夹爪分别归一化。它便宜、确定、适合验证 advantage 和 pull 是否能降低动作误差。

但它存在明显局限：专家动作不是唯一正确动作，L2 更近也不保证任务成功。所以 R0 不能作为主要论文结果。

#### R1：动作质量组合 reward

在 R0 上加入：

- 动作越界惩罚。
- 速度和加速度平滑性。
- 双臂碰撞或不合理夹爪切换惩罚。
- B-Spline 后轨迹与原始轨迹的偏移。

R1 适合检查生成动作是否更可执行，但仍不是任务成功 reward。

#### R2：仿真结果 reward

从同一个 RoboTwin 状态执行不同候选的前缀，根据任务进度、成功、碰撞和超时给分。这是最有意义但最昂贵的版本。

如果本阶段要严格只研究动作 Flow，可以固定环境状态数据包，不迭代研究环境 occupancy；每次只比较相同状态下不同 Flow 分支。这样环境只是黑盒动作评分器，研究变量仍集中在动作生成过程。

### 5.6 动作分支的特殊问题

图像 latent 各维通常可统一加噪，机器人动作不一定适合：

- 双臂关节/末端位姿与夹爪维度具有不同物理意义。
- 接触阶段很小扰动就可能改变成败。
- 夹爪信号若接近离散开合，不适合与位置维度使用相同噪声尺度。
- 动作已在 normalized space 中生成，但各维归一化后的可执行敏感度仍可能不同。

建议至少比较：

1. 所有 14 维统一 `eta`。
2. arm 与 gripper 分组 `eta_arm/eta_gripper`。
3. 只探索手臂连续维度，夹爪沿用 reference。
4. 按训练统计或执行灵敏度进行 per-dimension noise scaling。

第一轮优先方案 3，可降低“随机夹爪翻转导致所有分支失败”的风险。

### 5.7 Self-OPD 预期优势

- 不需要训练和存储多个 specialist teacher。
- 可以使用不可微分的仿真成功率、接触检测和规则 reward。
- 在学生自己的 Flow 邻域中搜索，不被固定教师能力完全限制。
- 相对 self-reference 可降低不同任务 reward 尺度差异。
- 中间 step 的局部分支比只在最终动作加噪更容易定位哪一步需要修正。

### 5.8 Self-OPD 的局限

- 计算昂贵：每个分支都要完成剩余 ODE，R2 还要执行环境。
- 稀疏成功信号可能让所有候选同分，advantage 为零。
- 负的 MSE 项若实现不当会导致 loss 无下界或梯度爆炸。
- reward 容易被利用，动作看似平滑却不完成任务。
- 局部探索半径过小找不到更好候选，过大会离开动作流形。
- 它仍不自动解决视觉定位错误或环境状态 distribution shift。

---

## 6. 三种训练方式的直接对比

| 项目 | 当前 CFM | Teacher Flow-OPD | Self-OPD |
|---|---|---|---|
| Flow 状态来源 | 专家动作插值 | 学生 SDE 轨迹 | 学生 SDE 轨迹及局部分支 |
| 监督来源 | 数据集专家动作 | 冻结强教师速度场 | 分支最终 reward |
| 监督密度 | 随机单个连续时间点 | 学生轨迹每个选中 step | 每个被分支 step 的相对优势 |
| 是否需要教师 | 否 | 是 | 否 |
| 是否需要 reward | 否 | 否或只用于选教师 | 是 |
| 是否可突破教师 | 不适用 | 通常受教师限制 | 理论上可以 |
| 训练稳定性 | 高 | 较高 | 中低，需要分阶段验证 |
| 计算量 | 最低 | 中高 | 最高，取决于 K 和 reward |
| 主要解决问题 | 拟合离线动作分布 | 学生 Flow 状态纠错 | 奖励驱动的局部 Flow 改善 |
| 最适合作用 | baseline | 机制验证、专家合并 | 后续核心研究方向 |

推荐判断：

- **最先实现**：teacher-based Action-Flow OPD，因为目标清晰、损失稳定，适合验证生成轨迹 exposure bias。
- **最有研究潜力**：Action Self-OPD，但要围绕机器人动作特有的 reward、分组探索和接触敏感信用分配形成差异。
- **不建议**：一开始就把十任务、全 step、K=8、仿真 reward 和负分支 push 全部放进同一个实验。

---

## 7. 计算量估算

### 7.1 当前 CFM

每个 batch 大致需要：

- 当前帧 DINO 前向。
- 若启用 future loss，再做未来帧 DINO 前向。
- 1 次带梯度动作模型前向。

### 7.2 Flow-OPD

若对完整 10-step 轨迹做监督，粗略需要：

- 约 10 次学生无梯度 rollout 前向。
- 约 10 次教师标签前向。
- 约 10 个选中状态的学生带梯度前向，或通过实现优化减少重复。

动作模型计算可能达到基础 CFM 的十几至数十倍，但同一观察的 DINO 特征可缓存，不能重复计算 10 次。

首轮建议每条轨迹随机选 1～3 个 step 做 loss，而不是所有 10 个 step。

### 7.3 Self-OPD

如果在每个 step 都分 K 条并完成剩余轨迹，额外动作前向数量近似：

```text
K * sum_(j=0)^(S-1) (S-j-1)
= K * S * (S-1) / 2
```

当 `S=10, K=4` 时，仅分支续推理就约 180 次动作前向，远高于 baseline 的 10 次推理。虽然 K 条可批处理，但显存和仿真执行量仍很大。

降低成本的顺序：

1. 每个训练样本只随机一个 branch step。
2. 诊断只取 `{2,5,8}`。
3. `K=2` 验证代码，`K=4` 做主实验，最后才做 `K=8` 消融。
4. K 条共享 condition 和 DINO 特征。
5. 先 R0 离线 reward，再 R2 simulator reward。
6. 先只用 positive pull，避免为不稳定 push 反复采集昂贵数据。

---

## 8. 详细实验步骤

### E0：冻结 baseline 协议

目的：确保所有 OPD 方法都从同一模型开始。

- 固定 checkpoint、norm stats、VTT、配置和评测 seeds。
- 完成十任务 clean/randomized 逐任务结果。
- 记录正常推理的 10 个 Flow 状态和最终动作作为回归基准。
- 后续默认推理仍为 10-step Euler、B-Spline、horizon 16。

输出：`baseline_manifest.json`、逐任务成功率和 checkpoint 哈希。

### E1：不训练的 Flow 诊断

对离线条件样本和少量仿真状态记录：

- 每一步 `||v_j||`。
- 每一步更新量 `||x_(j+1)-x_j||`。
- 轨迹曲率或相邻速度余弦相似度。
- 最终动作随初始噪声的方差。
- 1/2/4/6/10/20 NFE 的动作差异和成功率。
- 在专家插值状态与学生轨迹状态上的速度误差。

推荐指标：

```text
CFM-state error:
  ||v_student(x_t^expert) - v_target||

On-policy teacher error:
  ||v_student(x_j^student) - v_teacher(x_j^student)||

Trajectory curvature:
  1 - cosine(v_j, v_(j+1))

NFE sensitivity:
  ||A_NFE=k - A_NFE=20||
```

如果学生状态上的 teacher error 不高于专家插值状态，或 NFE 增加完全无改善，Flow exposure bias 可能不是当前主要瓶颈。

### E2：推理接口最小改造

新增功能必须默认关闭：

```text
sample_flow(..., return_trace=False)
sample_flow(..., initial_noise=None)
resume_flow(x_j, start_step=j, ...)
```

`return_trace=True` 保存：

```text
times: [S+1]
states: [S+1, B, 32, 14]
velocities: [S, B, 32, 14]
initial_noise
condition identifiers
```

必须通过：

- 默认模式与改造前固定随机种子的输出逐元素一致。
- `resume_flow(trace[j])` 与原轨迹尾部一致。
- `eta=0` 的 SDE sampler 与原 Euler sampler 一致。
- CPU 保存的 trace 不保留无用计算图。

### E3：单任务 specialist teacher

从 baseline 选择一个中等成功率任务：

1. 用该任务数据从 baseline `--init_from` 低 LR 微调。
2. 每 1～2 epoch 评测，得到成功率明显高于 baseline 的教师。
3. 同时保留 baseline 作为 student initialization。
4. 如果 specialist 没有显著更强，不进入 Flow-OPD；先解决教师训练。

教师至少应满足：

- 相同任务、相同评测 seeds 上显著优于共享 baseline。
- 动作不依赖评测 seed 过拟合。
- 在学生 Flow 中间状态上输出数值稳定，无 NaN 和异常大速度。

### E4：单任务 Teacher Action-Flow OPD

逐级实验：

#### E4-A：固定学生 Flow 包

- 用 baseline student 采集固定 `x_j` 包。
- 教师为每个 `x_j` 生成速度标签。
- 用这些固定包训练学生。

这不是严格持续 on-policy，但适合验证数据格式和 loss。

#### E4-B：每个 epoch 刷新 Flow 包

- 当前学生重新采集轨迹。
- 教师在线或离线批量标注。
- 更新学生。

它更接近真正的动作生成层 OPD。

#### E4-C：加入 CFM replay anchor

比较：

```text
OPD only
OPD + 0.1 * CFM
OPD + 0.5 * CFM
OPD + 1.0 * CFM
continued SFT with matched updates
```

先从 `lambda_replay=0.5` 附近开始，但以梯度范数而非只看标量 loss 决定最终比例。

### E5：三任务 Multi-Teacher Action-Flow OPD

- 选择三个确实更强的 specialist。
- 依据 task name/VTT 硬路由教师。
- 每任务相同数量的 Flow 包。
- 每任务独立报告成功率和 teacher-student velocity gap。
- 检查学生是否接近对应 teacher，同时保留其他七任务能力。

关键对照：

- continued SFT。
- 合并三任务 specialist 数据再 SFT。
- 离线 teacher distillation：教师只在专家插值状态标注。
- on-policy teacher distillation：教师在学生 Flow 状态标注。

“离线 vs on-policy teacher distillation”是证明 Flow-OPD 价值最直接的消融。

### E6：离线 R0 Self-OPD 烟雾测试

固定一个训练 batch：

- branch step 随机选择。
- `K=2`。
- 只扰动手臂连续维度。
- reward 使用负的专家动作距离。
- 只启用 positive pull。

验收：

- 高 reward 分支 advantage 为正。
- 一次小更新后，固定 batch 的高 reward 分支距离降低。
- `eta=0` 或所有 reward 相同时，OPD 梯度为 0 或接近 0。
- stop-gradient 正确，未通过完整 rollout 反传巨大计算图。
- 无 NaN、梯度爆炸和负 loss 无限下降。

### E7：完整离线 Action Self-OPD

逐步增加：

1. `K=4`。
2. all-positive pull。
3. clipped negative push。
4. direction-aware attenuation。
5. timestep/SDE variance normalization。
6. CFM replay anchor。

这里主要验证算法正确性和动作误差，不把它当作最终成功率贡献。

### E8：固定环境状态的 simulator-reward Self-OPD

为保持研究集中于动作 Flow：

- 用 baseline rollout 收集固定环境快照。
- 从同一快照对不同 Flow 分支生成动作。
- 执行相同长度前缀。
- 计算任务进度、成功和安全 reward。
- 只更新动作 Flow，不改变 DINO 和 VTT。

先做分支搜索诊断：如果 `Oracle@K` 不高于 reference，不开始蒸馏。

### E9：十任务扩展

只有 E5 或 E8 在三任务上稳定提升后才执行：

- 十任务 task-balanced branch packages。
- 按任务归一化 reward/advantage。
- 高成功率任务做 retention anchor。
- 低成功率任务分配更多诊断预算，但训练 loss 仍报告任务权重。
- 三个独立随机种子。

---

## 9. 建议的实验矩阵

### 9.1 最小机制实验

| ID | 方法 | Flow 状态 | 标签/reward | 目的 |
|---|---|---|---|---|
| M0 | Baseline | CFM 插值 | 专家速度 | 当前模型 |
| M1 | Continued SFT | CFM 插值 | 专家速度 | 排除额外训练步 |
| M2 | Offline teacher distill | CFM 插值 | specialist 速度 | 排除教师本身贡献 |
| M3 | Action-Flow OPD | 学生 Flow | specialist 速度 | 验证 on-policy Flow 状态 |
| M4 | Best-of-K self-train | 学生终点 | 最佳 reward 动作 | 排除普通候选蒸馏 |
| M5 | Self-OPD pull only | 学生中间 Flow | 相对 reward | 稳定版自探索 |
| M6 | Self-OPD pull-push | 学生中间 Flow | 全分支 reward | 完整版 |

### 9.2 必要消融

- Flow step：`{1,2}`、`{4,5}`、`{7,8}`、随机单步、全步。
- `K=2/4/8`。
- `eta` 小/中/大；不要直接照搬图像论文的 0.6。
- 统一动作噪声 / arm-only / arm-gripper 分组。
- fixed packages / epoch refresh / update refresh。
- teacher OPD 无锚定 / CFM anchor。
- Self-OPD best-only / positive-all / pull-push。
- raw action reward / B-Spline 后动作 reward / simulator reward。
- 10-step 训练、10-step测试 / 多 step 训练、不同 NFE 测试。

---

## 10. 评价指标

### 10.1 最终任务指标

- 十任务逐任务成功率和 macro average。
- clean 与 randomized 独立结果。
- 95% 置信区间。
- 高成功率任务 retention。

### 10.2 Flow 内部指标

- 教师—学生速度差：专家插值状态 vs 学生状态。
- 轨迹曲率和速度范数分布。
- 不同 NFE 的终点一致性。
- 初始噪声到动作的多样性。
- 每个 step 的 OPD loss、advantage 方差和有效正/负分支比例。
- 分支 reward 的 `Oracle@K`。
- 更新前后 reference reward 的变化。
- Student Flow state staleness：数据由多少次更新前的策略采集。

### 10.3 效率指标

- 每个 optimizer step 的动作模型前向次数。
- 每个 epoch 的 wall time。
- 峰值显存。
- 每提升 1 个百分点成功率所需 GPU 小时。
- Self-OPD 每提升 1 个百分点所需候选和 simulator steps。

---

## 11. 实现时最容易犯的错误

1. **时间方向照抄**：论文从噪声到数据的符号可能与当前代码相反，直接复制 `sigma(t)` 会在端点除零或让探索强度倒置。
2. **把固定旧数据称为 on-policy**：固定包只能做 smoke test；学生更新后必须刷新才能保持严格 on-policy。
3. **教师和学生相同**：相同结构、相同权重、相同输入会产生零蒸馏信号。
4. **teacher target 没有 stop-gradient**：会同时更新或反传进教师，改变目标。
5. **沿完整采样链反传**：内存急剧增加；首轮应 detach 学生访问状态，做局部 step regression。
6. **重复计算 DINO**：同一观察的 K 条分支应共享缓存特征。
7. **分支噪声破坏夹爪语义**：所有分支都因随机开合失败，无法形成有效排序。
8. **负 push 无边界**：带负号的距离损失可能无下界，必须裁剪 advantage、梯度和 repulsion。
9. **只看训练 loss**：OPD loss 降低可能只是模仿教师，必须看 success 和 retention。
10. **改变 baseline 推理协议**：若同时改 B-Spline、horizon 或 mask，结论无法归因。
11. **用最终测试 seed 调 reward/eta**：这会污染最终评测。
12. **把 R0 动作 L2 当任务 reward**：它适合调试，不足以支持机器人性能结论。

---

## 12. 推荐的实际决策路线

### 路线 A：稳健、低风险

```text
Baseline
 -> Flow trace/NFE diagnosis
 -> 1-task specialist
 -> offline teacher distill
 -> on-policy Action-Flow OPD
 -> 3-task multi-teacher
 -> 10-task
```

适合先证明 Flow 轨迹分布修正有效，工程风险较低。但论文创新性主要来自机器人动作域适配、学生轨迹诊断和多任务 specialist 合并，单独创新强度有限。

### 路线 B：高风险、研究潜力更高

```text
Baseline
 -> Flow trace/NFE diagnosis
 -> R0 Self-OPD smoke test
 -> action-aware branching
 -> fixed-state simulator reward
 -> pull-push + CFM anchor
 -> 3-task
 -> 10-task
```

适合形成自己的方法，但必须比直接 Self-OPD 迁移多一个机器人特有且经验证的贡献，例如：

- arm/gripper 结构化 SDE exploration。
- 接触阶段自适应 branch timestep。
- B-Spline/执行空间感知的候选去重和 reward。
- 基于动作敏感性的 per-dimension transition covariance。
- 在固定计算预算下的 uncertainty-guided branch allocation。

### 推荐组合

先走路线 A 到单任务，得到稳定的 Action-Flow OPD 参照；随后走路线 B，把 Self-OPD 作为主方法。这样最终论文有三层清楚对照：

```text
offline CFM
    < teacher-based on-policy flow distillation
    < robot-aware teacher-free Self-OPD
```

如果路线 A 都不能超过 continued SFT，应暂停 Self-OPD，先检查是不是视觉定位、任务条件或执行 horizon 才是主要瓶颈。

---

## 13. 第一轮建议参数

这些数值是工程起点，不是最终超参数：

```yaml
action_flow_opd:
  inference_steps: 10
  supervised_steps_per_trajectory: 1
  refresh_flow_packages_each_epoch: true
  cfm_replay_ratio: 0.5
  freeze_dino: true
  update_future_decoder_on_opd_batch: false
  grad_clip_norm: 1.0

self_opd:
  branches: 2                 # smoke test，稳定后改 4
  branch_step: random
  explore_arm_dims_only: true
  positive_pull_only: true    # 稳定后再启用 push
  advantage_clip: 3.0
  eta: 0.1                    # 需通过 normalized-action 扫描确定
  reference: deterministic_ode
```

学习率建议从 baseline 后训练 LR 的 `0.1～0.25` 倍开始，并做 `1e-5 / 2e-5 / 4e-5` 小范围扫描。不要沿用 baseline 最初的 `2e-4` 直接进行 Self-OPD；分支目标噪声更大，过高 LR 容易破坏已有动作流形。

---

## 14. 相关论文

1. [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)：CFM 和连续速度场基础。
2. [LiLa-WAM](https://arxiv.org/abs/2608.03701)：当前动作 Flow、VTT 和未来特征 baseline。
3. [Flow-OPD](https://arxiv.org/abs/2605.08063)：学生 SDE 轨迹、多教师密集速度监督和流形锚定。
4. [Flow-OPD 官方代码](https://github.com/CostaliyA/Flow-OPD)：重点参考 on-policy SDE 采样、教师标签和时间权重实现，不直接复制图像数据管线。
5. [Self-OPD](https://arxiv.org/abs/2608.26872)：局部 SDE 分支、自参考 advantage 和 all-branch pull-push。
6. [Self-OPD 官方代码](https://github.com/Shiy-Zhang/Self-OPD)：后续实现时核对符号、噪声调度和 stop-gradient。
7. [Flow Matching Policy Gradients](https://arxiv.org/abs/2507.21053)：flow policy 的 on-policy policy-gradient 对照。
8. [Q-VGM](https://arxiv.org/abs/2606.08015)：价值梯度如何分配到 flow VLA 的去噪时间，是 Self-OPD 机器人版本最重要的邻近工作之一。
9. [Diffusion Policy](https://arxiv.org/abs/2303.04137)：动作扩散与 receding-horizon 控制基础。
10. [FlowPolicy](https://arxiv.org/abs/2412.04987)：一致性 Flow Matching 与快速机器人动作生成。

阅读时对每篇统一回答：训练 Flow 状态由谁产生、监督来自谁、是否需要 reward、信用分到哪一步、是否在机器人环境执行候选、是否保持原始能力。

---

## 15. Baseline 完成后的立即执行清单

- [ ] 归档 baseline checkpoint、配置、norm stats 和评测结果。
- [ ] 做 `NFE={1,2,4,6,10,20}` 的动作与成功率扫描。
- [ ] 记录成功/失败回合的 10-step Flow trace。
- [ ] 实现默认关闭的 `return_trace` 和 `resume_flow`。
- [ ] 完成固定随机种子的输出一致性测试。
- [ ] 选一个中等成功率任务训练 specialist teacher。
- [ ] 比较专家插值状态与学生 Flow 状态上的 teacher-student velocity gap。
- [ ] 先完成 offline teacher distillation，再完成 on-policy teacher distillation。
- [ ] 完成 `K=2`、R0、positive-only 的 Self-OPD smoke test。
- [ ] 只有上述测试稳定后，才接入 simulator reward 和 `K=4`。

第一阶段最重要的成果不是立即提高十任务平均成功率，而是得到一组可信诊断：当前动作 Flow 的错误主要发生在哪些时间步、学生轨迹是否偏离离线 CFM 区域、教师能否在这些状态提供更好的速度方向，以及局部随机分支中是否存在稳定的高 reward 方向。它们决定后续应该继续 Flow-OPD、转向 Self-OPD，还是回到视觉/空间与执行控制问题。


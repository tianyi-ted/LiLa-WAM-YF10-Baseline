# 基于 LiLa-WAM 的执行后果驱动双层 On-Policy 研究路线

> 动作 Flow Matching、Flow-OPD 与 Self-OPD 的专项实验说明见：[ACTION_FLOW_OPD_EXPERIMENT_GUIDE.md](ACTION_FLOW_OPD_EXPERIMENT_GUIDE.md)。

> 独立的 JEPA、执行一致性与轻量融合路线见：[JEPA_SMOOTHRL_TURBOVLA_EXPERIMENT_ROADMAP.md](JEPA_SMOOTHRL_TURBOVLA_EXPERIMENT_ROADMAP.md)。

> 文档状态：研究设计稿 v1.0  
> 更新日期：2026-09-14  
> 基线项目：LiLa-WAM  
> 暂定方法名：Execution-Grounded Dual On-Policy Distillation（EG-DOPD，执行后果驱动双层在线策略蒸馏）

本文整理当前研究方向、核心假设、与已有工作的区别、最小可行实现、完整实验矩阵、失败判据和推荐阅读。它是一份后续开题、代码开发、实验记录和论文写作的统一路线图，而不是对最终创新性或论文录用级别的保证。OPD、VLA 和世界模型方向更新很快，正式投稿前仍需重新检索同期工作。

---

## 1. 当前基础与研究目标

### 1.1 当前基线

后续实验以当前 LiLa-WAM 十任务模型为 baseline。现有配置的主要特征如下：

- 训练数据：十个 RoboTwin 任务，每个任务约为 clean 50 + randomized 200；数据清洗、异常轨迹处理、归一化统计和 VTT 预计算已完成。
- 输入：单路 `head_camera` RGB、16 维机器人状态和任务级 VTT。
- 视觉编码器：冻结的 DINOv3 ViT-L/16，抽取 `[-12, -8, -4]` 三层特征。
- 动作生成器：12 层、隐藏维度 768 的 flow-matching action expert。
- 动作维度：14；动作块长度：32。
- 辅助目标：未来 DINO 特征预测。
- 推理：10 步 Euler 积分，动作反归一化后对完整 32 步做 B-Spline 平滑，再缓存并执行前 16 步。
- 测试：RoboTwin 仿真平台，按任务统计成功率。

基线的代码入口主要是：

- 训练：[train.py](../train.py)
- CFM 损失和动作模型：[models/vla_model_fm.py](../models/vla_model_fm.py)
- 推理、Euler 采样与 B-Spline：[robotwin_infer.py](../robotwin_infer.py)
- 十任务配置：[configs/yf10_base.yaml](../configs/yf10_base.yaml)
- 十任务评测：[eval_yf10.sh](../eval_yf10.sh)
- 基线流程：[BASELINE_YF10.md](BASELINE_YF10.md)

注意：当前 `yf10_base.yaml` 中记录的是 `batch_size=64`、`grad_accum_steps=1`、`epochs=50`。它与早期文档中建议的“有效 batch 128、约 12 epoch”不是同一个实际训练计划。论文实验必须以检查点随附配置和训练日志为准，保存配置副本、代码提交号和 checkpoint 哈希，不能事后将 50-epoch 实验标记成另一套计划。

### 1.2 目标问题

离线行为克隆只在专家数据分布上学习。部署时，学生模型自身的小误差会使机器人进入训练数据较少覆盖的状态，误差随后累积。对 flow-matching 动作头而言还多一层偏移：

1. **环境层偏移**：模型访问的是自身策略产生的状态，而非专家轨迹状态。
2. **动作生成层偏移**：训练使用“噪声—专家动作”的人工插值点，推理使用学生速度场自己产生的中间状态。
3. **执行层偏移**：生成的动作块还要经过反归一化、B-Spline、截取执行窗口和控制器约束，模型内部差异不一定等价于机器人真正执行的差异。

因此，研究问题不是简单地“把 Self-OPD 用到机器人动作上”，而是：

> 能否在学生自己访问的环境状态上，对学生自己的动作 flow 生成轨迹进行分支探索，并用候选动作真正执行后的物理后果，把信用分配回关键生成步骤？

这构成本文档建议的主研究线。

---

## 2. 当前 LiLa-WAM 中的 flow matching

### 2.1 离线训练路径

令专家动作块为 `x1`，高斯噪声为 `x0`，训练随机采样时间 `t`：

```text
x0 ~ N(0, I)
xt = (1 - t) x0 + t x1
target_v = x1 - x0
```

模型以当前图像的 DINO 特征、机器人状态和 VTT 为条件，预测 `xt` 上的速度场：

```text
v_theta(xt, t | observation, state, task)
```

基础动作损失为预测速度与 `target_v` 的均方误差。若未来特征预测开启，总损失近似为：

```text
L_base = L_CFM + lambda_future * L_future_feature
```

这条训练路径始终由专家终点 `x1` 定义，中间状态不来自模型自己的 Euler 采样轨迹。

### 2.2 部署路径

当前推理从高斯噪声开始，用 10 步 Euler 积分生成动作块：

```text
x_(j+1) = x_j + dt * v_theta(x_j, t_j | h_t)
```

其中 `h_t` 表示当前视觉、状态与任务条件。之后执行映射为：

```text
normalized 32-step chunk
    -> denormalization
    -> B-Spline over all 32 steps
    -> retain first 16 steps
    -> simulator / low-level controller
```

这里有两个值得验证的现象：

- 学生在第 2、5、8 等生成步到达的 `x_j`，可能不属于离线 CFM 训练常见的插值状态。
- 两个原始动作块可能差别很大，但经过 B-Spline 和执行窗口以后几乎相同；也可能原始差别很小，却在接触阶段导致完全不同结果。

### 2.3 两条 on-policy 时间轴

本方向同时处理两条时间轴：

```text
环境时间轴：h_t -> action chunk -> h_(t+H) -> ...

生成时间轴：x_0 -> x_1 -> ... -> x_j -> ... -> x_S = action chunk
```

- 环境层 on-policy：`h_t` 必须由当前学生实际 rollout 得到。
- 生成层 on-policy：`x_j` 必须由当前学生速度场采样得到。
- 执行反馈：评价对象必须是部署管线真正执行的动作 `E(A)` 及其环境后果，而不只是生成器内部的 `A`。

如果只满足第二点，这更接近把 Self-OPD 直接迁移到动作 flow；如果只满足第一点，这更接近 VLA-OPD。两者同时成立才是这里的“双层 on-policy”。

---

## 3. 与已有工作的边界

### 3.1 为什么“动作去噪过程 on-policy”单独不够强

已有工作已经覆盖了多个邻近区域：

| 工作 | 主要 on-policy 或优化对象 | 与本方向的关系 |
|---|---|---|
| Flow-OPD | flow 模型自己的生成轨迹 | 提供 flow OPD 和流形锚定思路，但主要面向图像生成 |
| Self-OPD | 对生成中间状态作 SDE 分支，以自参考奖励做 pull/push | 是“动作生成过程分支探索”的直接方法来源 |
| VLA-OPD | 学生策略诱导的机器人环境状态，由教师密集标注 | 覆盖环境层分布偏移，但不等同于内部 flow 分支和执行后果信用分配 |
| WAM-OPD | 学生环境历史、视频计划和动作分支联合蒸馏 | 与 LiLa-WAM 最接近，说明仅做一般 WAM on-policy 很难构成充分差异 |
| FPO | flow policy 的 on-policy RL / PPO 式更新 | 说明 flow 动作头的在线优化已有直接先例 |
| Q-VGM | 用 Q 值梯度在去噪时间上指导 flow VLA | 已经把价值信息分配到 flow 生成过程，是必须正面对比的工作 |
| NORA-1.5 | 动作条件世界模型奖励与偏好优化 | 说明“世界模型打分 + VLA 后训练”本身也不是空白 |
| PTR | 用动作后的状态后果重加权离线监督 | 支持“后果归因”有价值，但它是离线、保守重加权 |

所以不能把论文核心贡献写成“首次将 on-policy 用于机器人动作去噪”。更稳妥的差异化表述应围绕以下组合展开，并通过实验逐项成立：

1. **双重 occupancy matching**：同时校正环境状态分布与动作 flow 中间状态分布。
2. **执行等价性感知**：候选在真正的部署变换 `E` 后再去重、加权和评价。
3. **物理后果到生成步骤的信用分配**：把接触、抓取、搬运和放置结果反馈到产生差异的 flow step。
4. **不确定性感知的分支查询**：只在可能改变结果或评价器不确定的状态、flow step 上消耗仿真分支预算。

这些是待验证的贡献候选，不应在完成实验与投稿前检索之前写成已证实的“首创”。

### 3.2 暂定方法定义

对学生访问到的环境历史 `h_t`，运行当前策略得到第 `j` 个 flow 状态 `x_j`，从该点产生 `K` 个局部分支并完成生成：

```text
x_j^k = Branch(x_j, sigma_j),  k = 1...K
A^k   = CompleteFlow(x_j^k | h_t)
U^k   = E(A^k; controller, smoothing, horizon)
R^k   = EnvironmentOutcome(h_t, U^k)
```

其中 `E` 至少包含当前代码中的动作反归一化、32 步 B-Spline 和前 16 步截取。候选优势可先采用自参考标准化：

```text
Adv_k = (R_k - R_ref) / (std(R_1...R_K) + eps)
```

执行敏感度可以定义为：

```text
g_exec(k) = clip(
    ||E(A_k) - E(A_ref)|| / (||A_k - A_ref|| + eps),
    g_min, g_max
)
```

最终对正优势分支施加“拉近”监督，对负优势分支施加受限“推离”监督，并保留原始 CFM/未来特征损失作为锚点：

```text
L_total = L_OPD_pull_push
        + lambda_cfm * L_CFM_replay
        + lambda_anchor * L_reference_anchor
        + lambda_future * L_future_feature
```

这里的公式是实现框架，不是最终论文公式。正式版本需要根据 flow 参数化、分支 SDE 方差和梯度稳定性推导，并与 Self-OPD、FPO、Q-VGM 的目标严格区分。

---

## 4. 论文可以围绕的核心假设

建议把研究拆成四个可证伪假设：

### H1：双层分布偏移确实存在

学生在自身 rollout 状态上的 flow 中间轨迹，与专家状态和 CFM 插值轨迹存在显著偏移；失败状态上的偏移大于成功状态。

### H2：内部多样性不等于可执行多样性

动作块在归一化空间的距离与 B-Spline、截取后的实际动作距离、环境结果差异相关性有限。执行映射感知能提高候选预算的有效利用率。

### H3：中间 flow 分支包含可恢复动作

对 baseline 失败状态，在固定计算预算下，中间步骤分支的 `Oracle@K` 明显高于直接从头重采样或仅在终点加噪。

### H4：执行后果信用能改善策略，而非只改善搜索

把分支结果用于 post-training 后，单次正常推理的成功率提升，且优于等量 continued SFT、成功轨迹筛选 SFT 和只做终点 best-of-K 蒸馏。

如果 H3 不成立，不应直接开始大规模 post-training；如果只有搜索有效但蒸馏无效，论文方向应转为推理时规划或 verifier-guided sampling，而不是继续宣称 OPD 学习有效。

---

## 5. 分阶段实验路线

### 阶段 0：冻结并归档 baseline

目标是建立不可变、可复现的参照点。需要归档：

- 最终 checkpoint、训练日志和 norm stats。
- 实际运行时的 `yf10_base.yaml` 副本。
- 代码 commit/diff、RoboTwin 版本、DINO checkpoint 路径和哈希。
- VTT manifest、数据 subset manifest 和异常数据删除报告。
- 每个任务固定测试 seed 列表。

至少测试 `demo_clean` 和 `demo_randomized`。若要称为官方 C2C/C2R，必须严格遵守官方对应的训练域、测试域、语言设置和 seed 协议。当前 `data_mode: both` 同时用了 clean 与 randomized，不能把结果直接命名为严格 C2C 或 C2R。

示例评测入口：

```bash
cd /home/lab347-no10/Tianyi/LiLa-WAM

CKPT_SETTING='<相对 checkpoints_vla 的运行目录>' \
CHECKPOINT_EP='<epoch>' \
TASK_CONFIG=demo_clean \
INSTRUCTION_TYPE=unseen \
TEST_NUM=50 \
NUM_WORKERS=1 \
bash eval_yf10.sh
```

然后只替换 `TASK_CONFIG=demo_randomized` 运行第二组。实验初期保持 `NUM_WORKERS=1` 更容易排查仿真显存、Vulkan 和日志问题。

阶段 0 输出：

- 十任务逐任务成功率、宏平均和微平均。
- 至少三组评测 seed 的均值和 95% 置信区间。
- 失败阶段标注：接近、预抓取、闭合夹爪、抬升、搬运、放置/释放。
- 推理耗时、平均完成步数和超时比例。

### 阶段 1：只做诊断，不训练模型

这是整个路线中性价比最高的一步。先证明“分支探索中存在可学习信号”。

#### 1.1 任务选择

根据 baseline 结果选三个任务：

- 一个高成功率任务：检查是否会破坏已有能力。
- 一个中成功率任务：最适合观察改善空间。
- 一个低成功率但并非感知彻底失败的任务：检查恢复能力。

可暂以 `adjust_bottle`、`press_stapler`、`stack_blocks_two` 为候选，最终应由 baseline 数据决定。

#### 1.2 分支设置

建议首轮设置：

- 每任务采集 50～100 个学生状态。
- 10 个 Euler step 中只选择 `j={2,5,8}`。
- 每个选中 step 分 `K=4` 个候选。
- 确定性原轨迹作为 self-reference。
- 每个环境快照使用相同候选噪声库，确保方法间成对比较。
- DINO 特征对同一观察只计算一次并缓存。

必须先验证 RoboTwin 环境快照恢复：从同一快照、执行完全相同动作多次，结果差异应足够小。若接触仿真恢复不确定，就要保存方差、重复执行并对分支分数做置信度修正。

#### 1.3 每个分支包至少保存

- 策略 checkpoint ID、任务、seed、环境步和快照 ID。
- 当前观察、状态、VTT 和 DINO 特征索引。
- 初始噪声、flow step、`x_j`、分支扰动和完成后的 32 步动作块。
- 原始归一化动作、反归一化动作、B-Spline 后动作、实际执行前缀。
- 稀疏成功信号、任务进度、碰撞/越界、完成用时。
- 与 reference 的原始距离、执行后距离。

#### 1.4 关键诊断指标

| 指标 | 说明 |
|---|---|
| `Oracle@K` | K 个候选中至少一个成功的比例；判断是否存在可恢复分支 |
| Raw diversity | 原始动作块的候选差异 |
| Executed diversity | 经过 `E` 后的真实执行差异 |
| Execution-equivalent rate | 原始不同但执行后近似相同的候选比例 |
| Reward variance | 同一状态不同分支结果的区分度 |
| Branch-step utility | 在不同 flow step 分支带来的收益 |
| Rank correlation | 代理评分与仿真真实结果的 Spearman/Kendall 相关性 |
| Success recall | 排名前 K 的候选是否包含真实成功候选 |

#### 1.5 Go / No-Go 判据

- 若 `Oracle@K` 相对 baseline 没有明显提升：优先改分支噪声、分支位置和候选完成方式，不进入训练。
- 若 Oracle 明显提升但代理评分无相关性：先研究 verifier，不做伪标签蒸馏。
- 若执行前后候选排序几乎完全一致：执行等价性不应作为论文主贡献，只作为工程细节。
- 若中间分支不优于从头重采样：不要声称“flow-step 信用分配”有效，可转向终点候选选择。

### 阶段 2：搭建最小分支基础设施

建议新增独立文件，避免直接破坏 baseline：

```text
tools/collect_flow_branches.py      # 采集学生环境状态与 flow trace
tools/analyze_flow_branches.py      # Oracle@K、相关性和执行等价分析
posttrain_opd.py                    # OPD/自蒸馏后训练入口
eval_opd.py                         # 与 baseline 成对评测
opd/execution_transform.py          # 纯函数形式的部署动作映射
opd/branch_sampler.py               # 中间状态分支与续推理
opd/reward.py                       # 稀疏成功、进度和安全评分
opd/replay.py                       # 分支包与 baseline replay
```

对现有推理模块的首轮改动应是可选接口：

- `return_flow_trace=False`：默认行为与 baseline 完全相同。
- `resume_from_flow_state(x_j, j, ...)`：从指定 flow step 完成剩余积分。
- 将反归一化、B-Spline 和 horizon 截取抽成无状态纯函数，并为“改造前后正常推理输出一致”写单元测试。

阶段 2 只增加观测和分支能力，不改变 baseline 默认输出。

### 阶段 3：Oracle 反馈的最小 EG-DOPD

先不用学习世界模型或 verifier，直接使用仿真结果作为 oracle，以验证核心学习机制。

建议训练策略：

1. 从归档 baseline checkpoint 初始化。
2. 用当前学生采集新状态和 flow 分支，而不是一直复用 baseline 的固定包。
3. 计算相对 self-reference 的标准化优势。
4. 先只对正优势候选做 pull，再逐步加入有上限的负优势 push。
5. 混合原始示范 replay 的 `L_CFM`，防止灾难性遗忘。
6. DINO 保持冻结；先更新动作 expert 的末端若干层或参数高效适配器，资源允许后再比较全动作头更新。
7. 每轮更新后重新采集少量学生 rollout，避免“数据是旧策略生成却仍称 on-policy”。

单卡 4070 Ti 的建议是先做 1～3 个任务的机制验证：少量 flow step、`K=4`、候选批量完成、采集阶段 `no_grad`、同一观察缓存 DINO 特征。不要一开始在十任务上对每个环境步进行大量分支。

### 阶段 4：学习执行后果评价器

Oracle 版本成立后，再减少昂贵的仿真分支查询。

评价器输入建议包括：

- 当前 DINO 多层视觉特征。
- 当前 robot state 与 VTT。
- 候选实际执行前缀 `E(A)`，不能只输入原始归一化动作。
- 可选的 LiLa-WAM 未来特征预测结果。

输出可采用成功概率、任务进度增量、成对偏好/候选排序和不确定性。当前 future-feature decoder 只受离线未来帧特征监督，不能未经验证就当作可靠的 counterfactual world model。应先测试它对不同候选动作是否敏感、是否能正确排序分支结果。若不能，需要显式注入候选动作并训练 action-conditioned consequence model。

推荐主动查询策略：评价器高置信度时直接给分，低置信度或候选排序接近时才请求 simulator oracle。论文应报告“每提升一个成功率点所需的环境交互量”。

### 阶段 5：扩展到十任务

- 对每个任务分配相近的 rollout 包数量，避免简单任务产生更多短轨迹并主导更新。
- 每个任务分别标准化 advantage，另报告全局标准化消融。
- 同时保留 clean 与 randomized 的能力，检查对高成功率任务的 retention。
- 对失败阶段分层抽样，避免数据全部集中在超时末端。
- 每次更新后用固定开发 seeds 做短评测，最终 checkpoint 用未参与在线反馈的 seeds 做完整评测。

尽管离线示范可以全部用于 baseline 训练，在线后训练与最终测试仍必须分开 seed：否则 simulator 反馈已经暴露了测试情形，最终数字不能再解释为未见环境上的泛化结果。这与是否划分离线 demonstration validation set 是两个不同问题。

### 阶段 6：真实机器人验证

- 先让 LiLa-WAM 适配迁移后的 `robot/edulite_a3_vla` 接口。
- 将 `E` 扩展为真实部署管线：限位、速度/加速度限制、B-Spline、急停和低层控制器。
- 在真机上先只做候选离线预览和人类选择，不直接自动探索危险动作。
- 用仿真训练的评价器做排序，记录 sim-to-real 排序一致性。
- 真机阶段优先证明执行映射感知和安全查询有效，不追求大量在线试错。

---

## 6. 对照实验与消融设计

### 6.1 必须保留的 baseline

| 编号 | 方法 | 目的 |
|---|---|---|
| B0 | 原始 LiLa-WAM | 主 baseline |
| B1 | Compute-matched continued SFT | 排除“只是多训练了一些步” |
| B2 | 成功 rollout 筛选后 SFT | 排除简单自训练即可取得相同收益 |
| B3 | Terminal best-of-K + distillation | 检验中间 flow 分支是否必要 |
| B4 | 仅动作生成层 Self-OPD | 检验环境 on-policy 是否必要 |
| B5 | 仅学生环境状态 + 教师/专家监督 | 对应 VLA-OPD 式外层修正 |
| B6 | 双层 on-policy，但不使用执行映射 | 检验 execution grounding |
| B7 | EG-DOPD 完整方法 | 主方法 |

若资源允许，应复现或对接最接近的公开实现，如 FPO、Q-VGM、VLA-OPD/WAM-OPD。若无法公平复现，论文必须明确实现和算力限制，不能只引用数字作直接横向比较。

### 6.2 关键消融

- 环境层 only / flow 层 only / 双层。
- 原始动作评分 / 反归一化后评分 / 完整执行映射后评分。
- flow step：早期、中期、晚期、动态选择。
- 分支数 `K={2,4,8}`。
- ODE 从头重采样 / 中间 SDE 分支 / 终点局部加噪。
- 只 pull / pull+push。
- 无 CFM anchor / 固定权重 anchor / 自适应 anchor。
- 稀疏成功奖励 / 进度奖励 / 安全约束组合。
- 当前 future-feature head / 显式 action-conditioned verifier。
- 固定分支包 / 每轮刷新 / 每若干轮刷新。
- 全局 advantage 标准化 / 按任务标准化。
- 固定执行 horizon / 可变 horizon（作为扩展，不要与第一篇主线同时堆叠过多模块）。

### 6.3 公平性要求

- 所有方法使用同一 baseline checkpoint、seed、任务配置和归一化统计。
- 使用相同的在线环境步预算、梯度更新数或同时报告两者。
- 共享候选初始噪声和环境快照，做 paired comparison。
- 报告均值、置信区间和逐任务结果，不只报告挑选后的最好平均值。
- 超参数选择不能使用最终测试 seeds。
- 报告训练 GPU 小时、仿真小时、峰值显存和推理延迟。

---

## 7. 评价指标

### 7.1 主指标

- 每任务成功率。
- 十任务 macro average；每个任务权重相等。
- micro average；作为补充，不代替 macro average。
- 三个或更多随机种子的均值与 95% 置信区间。
- clean/randomized 或严格 C2C/C2R 协议下的独立结果。

### 7.2 学习与诊断指标

- `Oracle@K`、Top-1 verifier success、successful-candidate recall。
- flow step 与最终回报之间的信用归因稳定性。
- 原始多样性、执行多样性和 execution-equivalent rate。
- verifier 的 AUROC、AUPRC、Brier score、ECE 和排序相关系数。
- 收集策略与当前策略之间的 staleness。
- baseline 高成功率任务的能力保持率。
- 成功/失败的阶段分布。

### 7.3 效率指标

- 每 1,000 个环境 step 的成功率增益。
- 达到目标成功率所需的 simulator queries。
- 每次动作块生成耗时和整回合 wall-clock 时间。
- 分支推理相对 baseline 的计算倍率。
- 学习评价器后减少的 oracle 查询比例。

---

## 8. 风险、失败模式与转向条件

| 风险 | 表现 | 对策或转向 |
|---|---|---|
| 环境快照不可稳定恢复 | 同一动作重复执行结果波动大 | 重复分支、保存方差；无法解决时用更粗粒度 episode-level 比较 |
| 分支没有有效多样性 | `Oracle@K` 接近 baseline | 调整 step、SDE 方差或从头重采样；仍无效则终止 OPD 路线 |
| B-Spline 抹掉候选差异 | raw diversity 高而 executed diversity 低 | 在执行空间去重和自适应分配候选预算 |
| 接触对微小差异极敏感 | 评价器排序不稳定 | 多次 rollout、风险敏感评分、预测分布而非单值 |
| 奖励 hacking | 进度高但任务失败或动作危险 | 成功约束、安全惩罚、reference anchor 和视频审查 |
| 灾难性遗忘 | 难任务提升、原有高分任务下降 | 混合示范 replay、参数高效更新、按任务 retention 早停 |
| 任务不平衡 | 某些短任务支配更新 | 按任务分配 rollout 包和按任务标准化 advantage |
| future head 不具备反事实能力 | 对动作变化不敏感 | 改为 action-conditioned consequence verifier |
| 方法只在搜索时有效 | Oracle 高但蒸馏后不提升 | 转为 verifier-guided inference / MPC，而不是强行维持 OPD 叙事 |
| 同期工作覆盖创新点 | 投稿前出现近似 dual OPD | 强化 execution equivalence、信用分配与真实机器人验证，并重新做文献检索 |

---

## 9. 建议的最小论文故事

一篇聚焦且可检验的论文可以按以下逻辑展开：

1. **观察**：flow-matching 机器人策略同时承受环境状态分布偏移和内部生成轨迹偏移。
2. **诊断**：模型空间的动作差异会被真实部署变换放大或消除，因此内部 reward 不完全代表执行后果。
3. **方法**：在学生访问的状态上进行中间 flow 分支，用真实执行后的后果产生相对优势，并进行带 CFM 锚定的 step-wise pull/push 更新。
4. **效率**：用 action-conditioned consequence verifier 与不确定性查询降低仿真分支成本。
5. **结果**：在十任务 RoboTwin 上超过原始 LiLa-WAM、continued SFT、成功筛选 SFT、terminal best-of-K 和单层 OPD，并保持原有任务能力。

可以争取验证的论文结论是：

- 双层 on-policy 优于只修正环境层或只修正生成层。
- execution-grounded 候选选择优于原始动作空间选择。
- 中间 flow step 信用分配优于只使用终点标签。
- 不确定性查询在相近成功率下显著减少 oracle 交互。

上述四条都应先写成假设；只有统计结果成立后才能成为摘要中的贡献。

对于“最高中科院二区”的目标，更现实的最低证据组合是：完整十任务、严格协议、至少三次重复、强对照、核心消融、交互效率分析和可解释诊断。若能增加少量真机结果，会显著增强完整性，但不能以真机演示替代仿真统计。分区会随年份与学科口径变化，不能由方法设计预先保证。

---

## 10. 推荐论文与阅读顺序

以下链接优先使用论文原始页面。2026 年工作很新，部分结论仍应结合代码、附录和后续版本审慎判断。

### A. 基线与 flow 基础

1. [LiLa-WAM: Lightweight Latent Reasoning World-Action Model for Robotic Manipulation](https://arxiv.org/abs/2608.03701)  
   当前 baseline；重点阅读 VTT、未来 latent 预测、动作 flow head 和 RoboTwin 评测协议。

2. [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)  
   理解 CFM 直线路径、速度场学习和 ODE 采样的理论基础。

3. [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)  
   动作序列生成、receding-horizon control 与多模态动作分布的基础参考。

### B. OPD 主线

4. [Flow-OPD: On-Policy Distillation for Flow Matching Models](https://arxiv.org/abs/2605.08063)  
   重点看 on-policy flow trajectory supervision 和 manifold anchor，而不是直接照搬其图像奖励设计。

5. [Self-OPD: On-Policy Distillation for Flow Matching Models without Teacher](https://arxiv.org/abs/2608.26872)  
   重点看中间状态 SDE 分支、自参考 advantage、pull/push 与方差归一化。

6. [VLA-OPD: Bridging Offline SFT and Online RL for Vision-Language-Action Models via On-Policy Distillation](https://arxiv.org/abs/2603.26666)  
   重点看学生诱导环境状态、教师密集监督和能力保持。

7. [WAM-OPD: On-Policy Distillation for World Action Models](https://arxiv.org/abs/2608.22364)  
   与 LiLa-WAM 结构最邻近；必须详细比较其视频计划条件下的动作训练与本文拟做的 execution-grounded flow-step credit。

### C. flow / diffusion 动作策略后训练

8. [Flow Matching Policy Gradients](https://arxiv.org/abs/2507.21053)  
   flow policy 的 on-policy policy-gradient 基础。

9. [Reinforcement Fine-Tuning of Flow-Matching Policies for Vision-Language-Action Models](https://arxiv.org/abs/2510.09976)  
   重点看 VLA 规模下的 FPO、结构化信用分配、latent exploration 和 Q ensemble。

10. [Q-VGM: Q-Guided Value-Gradient Matching for Flow-Matching VLA Policies](https://arxiv.org/abs/2606.08015)  
    与“把物理价值反馈到 flow step”高度相关，是最重要的强对照和差异分析对象。

11. [FlowPolicy: Enabling Fast and Robust 3D Flow-based Policy via Consistency Flow Matching](https://arxiv.org/abs/2412.04987)  
    可用于理解机器人 flow policy 的一致性约束、快速采样和 3D 空间条件。

### D. 世界模型、后果评价与偏好学习

12. [NORA-1.5: A VLA Model Trained using World Model- and Action-based Preference Rewards](https://arxiv.org/abs/2511.14659)  
    重点看动作条件世界模型评分与 DPO；用于避免把“WM reward”错误地当作无人覆盖的创新点。

13. [Conservative Offline Robot Policy Learning via Posterior-Transition Reweighting](https://arxiv.org/abs/2603.16542)  
    重点看动作后果的归因、后验转移评分和保守加权，可启发 execution consequence weighting。

14. [Value Gradient weighted Model-Based Reinforcement Learning](https://arxiv.org/abs/2204.01464)  
    重点看为什么世界模型误差应按对控制价值的影响加权，而不是平均预测所有视觉细节。

### E. 动作块与执行层参考

15. [BAKU: An Efficient Transformer for Multi-Task Policy Learning](https://arxiv.org/abs/2406.07539)  
    可参考 action chunk、temporal ensembling、多相机输入和真实控制平滑。

阅读优先级建议：LiLa-WAM → Self-OPD → VLA-OPD → WAM-OPD → Q-VGM/FPO → NORA-1.5/PTR → flow 基础论文。阅读每篇时统一记录四项：它的 on-policy 分布是什么、监督信号来自哪里、信用分配到哪里、评价的是原始生成还是实际执行结果。

---

## 11. 建议的项目里程碑

### M0：Baseline 完结

- [ ] 冻结 checkpoint、配置、代码和数据 manifest。
- [ ] 完成 clean/randomized 固定协议评测。
- [ ] 逐任务整理失败阶段。

### M1：分支诊断成立

- [ ] 推理接口可返回全部 10 个 flow 状态。
- [ ] 环境快照重复性通过测试。
- [ ] 三任务分支包采集完成。
- [ ] `Oracle@K`、执行等价率和 step utility 分析完成。
- [ ] 根据 Go/No-Go 判据决定继续、改探索或转 verifier-guided inference。

### M2：Oracle EG-DOPD 成立

- [ ] 1～3 任务上超过 compute-matched continued SFT。
- [ ] 双层优于单层消融。
- [ ] CFM anchor 能保持原有成功任务能力。

### M3：评价器降低查询成本

- [ ] action-conditioned verifier 的排序能力显著高于随机和当前 future head。
- [ ] 在近似成功率下减少 simulator oracle 查询。
- [ ] 完成校准和不确定性消融。

### M4：十任务论文级实验

- [ ] 十任务、三 seeds、逐任务成功率与置信区间。
- [ ] 完整 baseline、强对照和核心消融。
- [ ] 环境步、GPU 小时、推理延迟和查询效率。
- [ ] clean/randomized 或严格官方协议下的结果。

### M5：真机扩展

- [ ] LiLa-WAM 真机适配器完成。
- [ ] 安全执行映射与离线预览通过。
- [ ] 少量真实机器人任务验证 sim-to-real 排序与安全性。

---

## 12. 接下来最应该做的事情

当前不应立即修改训练目标。最合理的顺序是：

1. 完成并归档十任务 baseline 的逐任务测试。
2. 根据成功率选择高、中、低三个诊断任务。
3. 给 `robotwin_infer.py` 增加默认关闭的 flow trace 输出，验证关闭时输出逐元素一致。
4. 把动作执行映射抽成纯函数并写一致性测试。
5. 验证 RoboTwin 同一状态快照可重复分支。
6. 采集 `j={2,5,8}, K=4` 的小规模分支数据。
7. 只分析 `Oracle@K`、执行等价率和排序相关性。
8. 诊断证据成立后，再实现 Oracle 反馈的最小 post-training。

这样做可以用很小的训练成本回答最重要的问题：LiLa-WAM 的失败中，是否真的存在能够通过动作 flow 中间分支发现并蒸馏的更优行为。如果答案是否定的，应尽早转向空间感知、world-model/verifier 或推理时规划，而不是在缺少信号的情况下堆叠 OPD 模块。

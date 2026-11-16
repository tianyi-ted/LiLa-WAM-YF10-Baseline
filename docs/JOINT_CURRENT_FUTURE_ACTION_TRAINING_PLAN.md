# 当前—未来双路径共享动作网络：单阶段联合训练方案

日期：2026-11-12。

**状态：方案记录，尚未实现、训练或验证。本次仅新增本文及文档索引，不修改模型、训练脚本、配置或测评代码。**

目的：在一次从头训练中，让真实未来视觉信息参与塑造动作网络，同时保留完全不访问未来的部署路径。暂不使用EMA第二动作网络、latent KD、响应蒸馏、兼容初始化、action-prior或空间残差。

本文是本项目的实验设计，不是对某篇论文的完整复现，也不构成创新性已确认或性能必然提升的结论。

## 1. 先明确结论

可行。使用两个不同的意图编码器，分别产生当前条件latent和未来条件latent，再分别调用**同一个动作网络对象**，将两个动作loss相加，只执行一次联合参数更新。

这里的“两条路径”意味着同一batch内两次独立forward，不是两个动作模型，也不是先将两个latent混在一起再做一次forward。

训练时两个分支都预测**从当前时刻t开始的同一段专家动作**。未来帧仅提供额外训练条件，不是未来图像预测的监督目标，也不是把动作标签移到t+H。

部署时只使用当前分支，训练专用posterior不参与推理。

## 2. 与现有两阶段方案的区别

| 项目 | 已有teacher→student方案 | 本文计划的联合训练 |
| --- | --- | --- |
| 训练顺序 | 先训练未来teacher，再独立训练student | 一个训练阶段内交替计算两个分支 |
| 动作网络 | teacher/student各自持有一套 | 全程只有一套共享可训练动作网络 |
| 因果G2 | student阶段训练 | 从第一个epoch开始训练 |
| 未来posterior | teacher阶段训练，student阶段冻结 | 全程可训练，只接收未来分支动作梯度 |
| 未来信息影响动作网络的路径 | teacher训练、兼容初始化；原latent KD并不直接更新student DiT | 未来分支动作loss直接更新共享DiT及相关动作模块 |
| latent对齐 | student可以显式拟合teacher latent | 第一轮不添加任何latent KD |
| 权重继承 | C/D等实验继承兼容动作参数 | 第一轮从头初始化，不读取teacher checkpoint |
| 部署 | 因果student | 因果G2＋共享动作网络 |

本方案不是把已有蒸馏loss权重改成0：它要求**两个活跃分支同时训练同一套动作参数**。也不能通过复用现有`stage: teacher`直接得到。

## 3. 完整数据通路

记号：

- `F_t`：当前帧DINO patch，供G2/posterior读取，当前设置为-2层。
- `V_t`：当前帧多层DINO特征列表，供原视觉融合/适配器读取，当前为[-7,-4,-2]。
- `F_f`：同一episode内t+H帧的DINO patch。
- `s_t`、`q`：当前状态与已有VTT向量。关闭直接VTT token不等于不读取VTT。
- `G_phi`：因果G2，第一轮保持现有单query task-attention＋意图MLP。
- `P_psi`：可训练的未来posterior，可复用`FutureDrivenPosterior`定义。
- `D_theta`：共享动作路径，包含latent到token的投影、视觉融合/适配器、DiT、动作/状态投影及动作输出等实际参与模块。

```text
数据集同一anchor t
  ├─ 当前RGB I_t ── 冻结DINO ──┬─ 当前patch F_t
  │                             └─ 多层特征 V_t ───────────────┐
  ├─ 当前状态 s_t、VTT q、跨度H                               │
  ├─ 未来RGB I_(t+H) ── 冻结DINO ── 未来patch F_f              │
  └─ 当前专家动作块 A_t，形状[B,8,4]                            │
                                                               │
  当前分支：F_t、s_t、q、H → G_phi → z_C [B,768]                │
                                        ↓                      │
                         D_theta(x_tau,tau,V_t,s_t,z_C) ←───────┤
                                        ↓                      │
                                  v_C → L_C                    │
                                                               │
  未来分支：F_t、F_f、s_t、q、H → P_psi → z_P [B,768]           │
                                          ↓                    │
                         D_theta(x_tau,tau,V_t,s_t,z_P) ←───────┘
                                          ↓
                                    v_P → L_P

                  L_total = L_C + alpha × L_P
                                  ↓
               一个optimizer，联合更新theta、phi、psi
```

图中两个`D_theta`是同一对象的两次调用，参数地址相同。两次调用都只把**当前**多层特征`V_t`交给视觉适配器；真实未来只经posterior压缩成`z_P`进入未来分支。

`z_C/z_P`各自经过同一套`intent_to_token`（LN＋Linear及type embedding），变成一个512维意图token，再与原动作、状态、register、当前视觉token一起进入DiT。不是在原始RGB上叠加，也不对两个latent直接相加。

### 3.1 当前分支

沿用当前G2：VTT＋state形成query，读取当前196个patch，汇聚后经过意图MLP得到768维latent。第一轮不同时切换到LatePool-Q4，以免把预测器变化与训练机制变化混在一起。

该分支禁止访问真实未来、专家动作或Flow噪声/时间来生成latent。它的未来相关语义来自联合训练后的动作需求，而非第一轮中一个额外的固定未来变化标签。

### 3.2 未来分支

可复用现有posterior的结构：未来patch投影＋位置/类型/跨度编码，读取由当前patch、state、VTT构成的context，再经残差FFN和summary query压缩为一个768维latent。

此处posterior**不是冻结teacher**。它与动作网络从头共同学习；动作loss决定保留什么信息，但不保证它自动只保留最有用或最可预测的未来信息。posterior本身不接收专家动作、noisy actions或Flow时间。

### 3.3 部署

```text
当前RGB、当前state、VTT
    → 冻结DINO当前特征
    → 因果G2得到z_C（每次动作块生成计算一次）
    → 共享动作网络执行10步Flow积分
    → 生成8步动作，执行前4步
```

不载入或调用posterior，不读取未来RGB，不存在教师查询。保持相同因果网络结构时，部署模型规模和计算通路应与对应action-only模型一致；这不等于数值输出或成功率不变。

## 4. 两个loss究竟监督什么

当前归一化专家动作块记为`A_t`，标准高斯噪声为`epsilon`。一个batch内只采样一份噪声和一份Flow时间，两分支共享：

```text
x_tau = (1 - tau) × epsilon + tau × A_t
v_target = A_t - epsilon

v_C = D_theta(x_tau, tau, V_t, s_t, z_C)
v_P = D_theta(x_tau, tau, V_t, s_t, z_P)

L_C = mean((v_C - v_target)^2)
L_P = mean((v_P - v_target)^2)
L_total = L_C + alpha × L_P
```

**L_P是未来条件下的当前动作Flow Matching loss。** 既不是未来RGB loss，也不是下一动作块loss；其中P表示privileged（特权条件），不表示把动作标签向未来移动。

两分支同噪声、同tau有利于比较其预测差异；只是训练时拟合插值点的速度，不需要每个batch运行完整10步动作采样。

第一轮不添加：

- `MSE(z_C,z_P)`或归一化latent KD。
- `MSE(v_C,stopgrad(v_P))`等速度场蒸馏。
- 固定mean变化意图loss、稠密FutureFeat或未来RGB重建。
- OPD、EMA参数网络、下一动作块塑形、动作先验及空间残差。

### 4.1 alpha与总loss尺度

建议第一轮预先固定`alpha=0.5`，保留当前分支权重1；这只是优先保证因果路径训练的保守起点，**尚未验证最优**。先不做epoch调度或权重搜索。

本文约定使用上面的**加权和**，不能训练中途悄悄换成`(L_C+alpha*L_P)/(1+alpha)`。两种写法虽然相对权重一样，整体梯度尺度不同，在梯度裁剪、AdamW及有限步优化下不保证等价。

需要记录两个原始loss、加权未来loss、总loss、裁剪前梯度范数及裁剪情况。不能直接将新`Total`与旧单分支`Action`数值比较。未来换成加权平均时，应作为另一个显式协议，不覆盖旧记录。

## 5. 一个优化器怎样更新两条路径

将参数分为互不重叠的三组：

| 参数 | 来自当前分支的梯度 | 来自未来分支的梯度 |
| --- | --- | --- |
| phi：因果池化器与G2意图MLP | 有，由L_C提供 | 无 |
| psi：未来posterior | 无 | 有，由alpha×L_P提供 |
| theta：共享动作路径 | 有 | 有 |
| 冻结DINO与预计算VTT | 无 | 无 |

梯度对应：

```text
grad_phi   = d(L_C) / d(phi)
grad_psi   = alpha × d(L_P) / d(psi)
grad_theta = d(L_C) / d(theta) + alpha × d(L_P) / d(theta)
```

一批数据的更新顺序：

1. 只清空一次梯度。
2. 提取冻结的当前/未来特征，采样共享noise/tau。
3. 当前分支forward/backward；未来分支forward/backward。
4. 两个梯度累积完成后，对所有活跃可训练参数统一裁剪一次。
5. `optimizer.step()`一次，`scheduler.step()`一次。

也可以保留两个forward的图后，对总loss做一次backward，数学目标相同，通常激活峰值更大。不能先对当前分支step，再对未来分支step：那就变成两个先后更新，不再是本文的一次联合更新。

优化器收集一次完整`action_model`的可训练参数，再加一次posterior参数，并校验参数对象不重复。因果G2已经属于`action_model`，不能再追加一次G2参数。

**不要对posterior或未来分支动作forward使用`no_grad()`/detach。** 可以对冻结DINO输出使用detach；它不会阻断posterior和共享动作网络自己的梯度。

分开backward省显存时，两条路径需要独立构建可训练部分的计算图：可以共享冻结DINO特征，不能误复用已经backward释放的可训练视觉adapter/token投影图。也不应通过`retain_graph=True`掩盖不必要的图耦合。

## 6. 为什么可能有效，以及不能保证什么

期望机制：未来分支用真实未来信息学习动作相关表示，并通过动作loss直接影响共享动作读出；当前分支在同一训练过程中始终要求这套读出适用于当前可获得的信息。

但是共享参数不是蒸馏的同义词。这里没有显式要求`z_C≈z_P`或`v_C≈v_P`，共享DiT也可能学会区分两种latent分布，形成两套不同的内部行为。未来信息并不会自动完整转移到因果G2。

主要风险：

- **特权捷径**：L_P很低但L_C/闭环性能不改善；动作模型只在有未来条件时受益。
- **梯度冲突**：同一共享参数收到两分支方向相反的梯度；L_P降低不等于因果控制更好。
- **不可预测的未来**：真实未来含当前单帧难以推断的信息，训练收益不一定能部署兑现。
- **latent/读出配套适应**：同维度和共享投影不代表两个latent自然对齐，不能仅凭cosine下结论。
- **额外更新量混淆**：多一个loss和更多forward可能本身改善优化，与真实未来信息无关。

如果联合训练无效，先通过对照定位原因，再考虑小权重响应约束、表示约束或课程调度；不预先把所有机制叠加。

## 7. 首轮实验与固定规则

### 7.1 最小三组

| 组 | 条件与训练目标 | 要回答的问题 |
| --- | --- | --- |
| J0：因果action-only | 从头训练G2＋动作网络，只用L_C | 单分支基线 |
| J1：真实未来联合 | 从头训练；L_C＋0.5L_P；posterior读取真实F_f | 单阶段特权条件是否改善因果控制 |
| J2：当前替代未来联合 | 与J1同结构、loss和更新次数，但posterior的F_f替换为F_t | J1收益是否来自真实时间未来，而非仅额外分支/监督/参数 |

J2仍有两种编码器，不等于J0重复backward两次。它是“去除真实未来内容”的关键控制，不是所有因素都完全匹配的单分支训练。为更接近I/O/视觉计算预算，后续实现应支持仍读取/编码真实未来但不让它进入posterior；该冗余计算必须明确记录。

J0第一阶段可以用已有严格匹配的随机初始化单query action-only作参考；正式因果归因时，建议三组使用同一新入口和随机数协议重跑。不能拿继承teacher参数的C/D直接充当J0，因为初始化和额外预训练预算不同。

### 7.2 第一轮不改的设置

- 平台：先在MetaWorld MT50 corner2落地，不同时迁移RoboTwin/真机。
- 数据：现有2500 episodes、190451 anchors；所有现有训练数据继续用于训练，不新划分验证集。
- 归一化：`stats/metaworld_mt50_corner2_480_50ep_4d.json`。
- DINO-B：`pretrained/dinov3-vitb16`，冻结，当前层[-7,-4,-2]，posterior未来层-2。
- 当前480→224图像预处理、180°方向、VTT目录与版本保持不变。
- C8/E4/H8，4维状态/动作，DiT depth12/hidden512；因果预测器保持单query task-attention。
- 无gate、无直接VTT token、无dense FutureFeat、无latent KD、无动作参数继承。
- batch96、训练seed0、AdamW、原梯度裁剪规则，统一LR 2e-4→5e-5，50轮cosine，每5轮保存。
- 主比较epoch40，**不是把总训练周期改成40轮**。
- 数据shuffle、drop_last、动作尾部padding、未来帧越界截取继续沿用现有MetaWorld规则，不混入mask修复。

VTT仍用于当前query和posterior；“不直接注入VTT token”不等于彻底去掉任务条件。

### 7.3 测评规则

首轮保持cold、不使用action-reuse：50任务×每任务50次、workers8、horizon500、env seed1000、policy seed456、NFE10。配置使用每次训练保存的自身配置。

有潜力后再做不重叠环境seed组及多个训练seed；tail重用单独作为正交推理实验，不能与cold混合宣称提升。

训练seed、数据采集seed、评测env seed、policy noise seed分别记录。相同训练seed不自动保证新增posterior后所有随机序列不变；实现时应隔离/显式管理初始化、shuffle、Flow噪声与tau的随机数流，记录其状态并保存供resume恢复。

## 8. 开销与省显存方式

相对因果action-only，新增一个小posterior、一次未来DINO编码和一次共享动作网络forward/backward。**不新增第二套DiT权重、Adam状态或EMA shadow。**

不过共享权重并不意味着第二次forward免费：若同时保存两套动作激活，峰值显存仍可能明显增加。优先顺序建议：

1. 冻结DINO并复用当前特征，未来只提取posterior所需层。
2. 当前分支完成backward释放激活后，再独立构图计算未来分支；最后统一clip/step。
3. 保持相同有效batch，必要时用微batch梯度累积；梯度缩放与scheduler计数按optimizer step定义。
4. 有需要再考虑DINO特征缓存，并校验图像预处理、层和数据版本。

实际每epoch时间取决于DINO、动作网络和数据读取的占比，不能提前承诺“仅增加很少”或“固定翻倍”。也不能声称必然比原teacher＋student两阶段更省总GPU时间；需要比较完整训练预算。

基准记录至少包括每epoch秒数、峰值allocated/reserved显存、每样本/每step用时和总GPU时长。首轮batch96若无法承受，先明确等效batch方案，不悄悄与旧batch不同的实验比较。

## 9. 不影响现有实验的实现边界（计划，不是已完成）

### 9.1 已核对的可复用接口

- `models/planwam_g2.py::FutureDrivenPosterior`：已有未来压缩器定义。
- `models/vla_model_fm.py::VLAModel.predict_global_intent`：已有因果G2入口。
- `models/vla_model_fm.py::VLAModel.forward(global_intent=...)`：允许显式提供latent后复用原动作路径。
- `metaworld_vla/train_planwam.py::compute_batch`：可参考已有数据归一化、当前/未来DINO提取规则。

**当前代码不具备本文联合训练loss与训练协议。仅修改YAML不能实现。**

尤其现有`stage: teacher`会冻结`action.intent_pool`、`action.intent_predictor`，每轮还将其设为eval；`stage: student`则加载冻结posterior。它们都不是本方案需要的“双编码器同时训练”，不应修改其原有语义来凑出联合训练。

### 9.2 推荐新增的隔离文件

以下路径只是后续实现命名建议，本文不创建这些文件：

- `models/joint_current_future.py`：独立联合loss/梯度组织，不改已有`planwam_flow_loss`。
- `metaworld_vla/train_joint_current_future.py`：独立训练入口，不复用`teacher/student` stage身份。
- `metaworld_vla/configs/metaworld_mt50_easy_joint_cf.yaml`：真实未来组。
- `metaworld_vla/configs/metaworld_mt50_easy_joint_cc.yaml`：当前替代未来控制。
- `metaworld_vla/configs/metaworld_mt50_easy_joint_causal.yaml`：同入口单分支控制。
- `utils/test_joint_current_future.py`：梯度、兼容、部署与resume测试。

给新实验独立协议版本，例如`joint_current_future_v1`。保存一个`model_state`（原因果G2和共享动作网络）及独立`posterior_state`，不把新posterior注册进旧`VLAModel`导致旧state key变化。

训练checkpoint还需保存配置、数据/预处理/统计/VTT合同、posterior规格、alpha与loss缩放方式、未来输入模式、optimizer/scheduler、epoch/global_step、随机数状态。resume核对完全相同的新协议，不能将旧teacher/student checkpoint当新联合训练resume。

保存独立训练目录、日志目录和测评输出，不能覆盖原C/D、Q4、RoboTwin或真机文件。既有训练入口应明确拒绝新协议配置，避免它们忽略新字段后误跑成旧实验。

### 9.3 部署兼容的验收方式

目标是复用原因果`Policy`网络计算：按保存配置构造原action模型，严格加载`model_state`，忽略训练专用`posterior_state`。新协议须正确标记“可因果部署”，不得把它伪装为旧teacher/student来绕过保护。

目前测评代码能读取`model_state`并有teacher防泄漏保护，但这**不代表未实现的新协议已经可测**。后续需添加新协议校验或独立导出/加载适配，验证旧checkpoint照常加载、未来posterior不被调用、新模型完整严格加载。

架构由模型配置与代码共同决定；训练目标、未来信息可见性和可恢复性还由训练入口、checkpoint与协议决定，不能把“同一YAML”理解成对这些事项的全部保证。

## 10. 实现后必须先通过的验证

1. **旧路径回归**：旧配置初始化和固定输入输出保持一致，原C/D及Q4严格加载、测试不受影响。
2. **同一个动作网络**：两个分支确实引用相同参数对象；优化器无重复对象，无第二套DiT。
3. **梯度分流**：分别只反传L_C或L_P，验证第5节的梯度表；尤其L_P必须真正到达共享动作参数，不能只训练训练后被丢弃的posterior。
4. **因果性**：在eval或固定随机状态下，保持当前输入不变，替换未来，当前latent/当前分支forward保持不变；未来posterior一般应对未来变化有响应。更换当前输入则因果路径应有响应。不能把训练态dropout的随机差异误判为未来泄漏。
5. **同监督点**：两分支动作anchor、noise、tau及归一化标签一致，没有意外移动到未来动作块。
6. **部署无未来**：推理接口只有当前RGB/state/VTT，未来posterior设为“调用即报错”仍能完成生成与环境step。
7. **数值与更新**：小batch完整forward/backward/step，loss、梯度有限，活跃模块可以更新。零初始化输出层可能使上游首step梯度为0，应跨多个step检查，不能误判。
8. **连续与恢复**：同一设备/精度条件下，对比连续若干step与保存后恢复，验证权重、optimizer、scheduler和随机数序列一致；不承诺跨设备bitwise一致。
9. **顺序反传**：独立计算图的两次backward与加权总loss的一次backward在容差内梯度一致，先在无dropout确定性小模型验证，再检查真实路径。
10. **未来边界**：episode末尾future clipping与padding沿用旧规则，记录clip比例；不把这些样本误称为完整H步未来。

通过这些测试后才提供真实可执行的训练/测评命令。目前不提供指向尚不存在入口的可执行命令。

## 11. 日志与结果如何解释

建议日志：`CurrentFM`、`PrivilegedFM`、`WeightedPrivilegedFM`、`Total`、`CurrentLatentNorm`、`PrivilegedLatentNorm`、各自latent跨batch标准差、`GradNorm`、LR、时间、峰值显存。

本方案没有固定teacher target，因此不要照搬旧日志的`KD/TargetNorm`含义。latent范数不同也不必然异常，因为后续有归一化投影；重点是数值稳定、非平凡使用和闭环表现。

| 结果 | 支持的判断与下一步 |
| --- | --- |
| J1稳定优于J0，也优于J2 | 支持真实未来条件对单阶段塑形有增益，再做训练seed和公平预算验证 |
| J1≈J2，二者均优于J0 | 收益可能主要来自额外分支或训练正则，不能归因于真实未来 |
| L_P下降很多，但J1因果性能不提升 | 存在特权条件收益，尚未转成部署收益；先检查梯度/依赖，不直接加大alpha |
| J1差于J0 | 先排除泄漏、尺度/裁剪与预算问题，再诊断冲突；不立即叠加Q4/空间残差 |
| J1只在训练动作MSE上变好 | 不足以说明控制提升；以同协议闭环成功率与逐任务结果为准 |

必要时低频测共享模块上`grad(L_C)`与`grad(L_P)`的范数和cosine；负cosine只说明该batch/参数子集的冲突，不是全局机制定论。两个分支loss大小也不能直接代表梯度大小。

同一checkpoint禁用/打乱因果latent的推理干预可检查模型是否使用这条路径，但这是分布外干预；性能下降不能单独证明“利用了未来知识”。

## 12. 后续顺序与本次完成边界

建议顺序：**确认本设计 → 隔离实现＋回归测试 → J0/J1/J2从头训练 → epoch40同规则cold比较 → 多seed与预算核验 → 依据诊断决定是否需要显式对齐。**

暂不并行加入EMA、响应蒸馏、历史输入、LatePool-Q4或region gate；如首轮支持单阶段塑形，再逐项验证。

本次已完成：读取现有接口、冻结行为和测评加载逻辑，记录此设计及隔离要求。

本次未完成且未执行：新loss、新训练入口、新YAML、checkpoint协议适配、任何回归/训练/性能测试。旧方案仍按原配置、原checkpoint和原入口复现。

相关历史记录：[两阶段蒸馏机制与结果](PLANWAM_DISTILLATION_MECHANISM_AND_RESULTS.md)、[轻量PlanWAM路线](PLANWAM_LIGHTWEIGHT_G2_EXPERIMENT_ROADMAP.md)、[多Query迟汇聚对照](PLANWAM_LATEPOOL_Q4_EXPERIMENT.md)。

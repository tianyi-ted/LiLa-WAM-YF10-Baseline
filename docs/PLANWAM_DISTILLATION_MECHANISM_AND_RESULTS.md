# 教师—学生蒸馏机制、PlanWAM核心对比与三优先级实验记录

更新日期：2026-11-11（以下早期章节保留原实验时间语境）。适用范围：本项目的Meta-World MT50 corner2轻量PlanWAM实验。

本文记录代码机制、已核验结果，以及后续实验。**2026-11-11状态：兼容初始化C/D、C-uniform/D-uniform均已完成；新的4query迟汇聚学生及无KD/有KD配置已实现并通过CPU验证，正式训练待执行；响应蒸馏仍未实现。** 新实验结构、旧方案保护、验证记录及完整命令见[多Query迟汇聚实验](PLANWAM_LATEPOOL_Q4_EXPERIMENT.md)。下文涉及“未实现/待训练”的早期记录不代表最新状态。虽然对话时IDE打开的是真机配置，本轮结果并不来自真机实验，不能直接外推到EDULITE部署或RoboTwin。

相关文档：[实验路线与实现入口](PLANWAM_LIGHTWEIGHT_G2_EXPERIMENT_ROADMAP.md)、[G2 task-attention实验](G2_TASK_ATTENTION_EXPERIMENT.md)。

2026-11-12新增[当前—未来双路径共享动作网络联合训练方案](JOINT_CURRENT_FUTURE_ACTION_TRAINING_PLAN.md)：记录单阶段、双条件、同一动作网络的联合FM实验设计。**该方案尚未实现；本次仅记录文档，不改变下述两阶段训练及已有实验。**

2026-11-10另新增[RoboTwin十任务训练/测试命令及C-uniform迁移](ROBOTWIN_EASY_G2_PRIOR_EXPERIMENT.md)。该分支使用14维动作、16维状态和原RoboTwin图像方向，必须单独训练教师；本文下方的MetaWorld教师、4维输入和结果不跨平台复用。

2026-11-11平台入口已隔离：本文MetaWorld继续使用`metaworld_vla/train_planwam.py`，RoboTwin教师及C/D-uniform改用根目录`train_planwam_robotwin.py`与独立`robotwin_planwam_transfer.py`。两套训练脚本互不导入，误传另一平台配置会拒绝；模型定义与FM loss仍为现有公共模块。本次不改变MetaWorld配置、训练目标或已有结果。

## 1. 当前结论与完成状态

当前判断：**教师完整组合具有离线动作误差优势，但当前latent-only蒸馏尚未把这一优势转化为学生的闭环提升。双向替换latent均退化，支持表示与动作读出配套适应、动作相关方向未充分对齐等解释；不能唯一确定机制。**

| 工作 | 截至本次核验的状态 |
| --- | --- |
| 未来驱动posterior教师 | 已实现，已完成50轮训练；本组使用epoch40 |
| 从头初始化的latent蒸馏学生 | 已实现，已完成50轮训练；本组使用epoch40 |
| 同协议baseline、原G2、蒸馏学生闭环比较 | 已完成并核验，见第7节 |
| S/S与S/T离线诊断 | 已完成，400个anchor |
| 完整S/S、S/T、T/S、T/T离线诊断 | 已完成，400个anchor，见第8节 |
| action-only同结构对照 | 已完成50轮；epoch40为2192/2500，即87.68%，已核验 |
| 动作响应/速度场蒸馏 | 仅方案，训练loss及配置分支尚未实现 |
| 教师兼容动作权重初始化迁移 | 2026-11-08新增可选分支及C/D/统一LR对照配置，CPU验证完成；正式训练待用户执行 |

当前执行顺序更新为：**2×2诊断（完成）→随机初始化action-only（完成）→C：兼容初始化＋action-only → D：同初始化与LR＋latent KD →按结果决定响应蒸馏或其他路线。** 原三个优先级编号保留作为历史方案索引，响应蒸馏不再被理解为下一项必须立即执行的训练。

## 2. 固定的数据、网络与训练设置

| 项目 | 当前设置 |
| --- | --- |
| 数据 | 自采Meta-World MT50 corner2，2500 episodes、190451 anchors，50任务 |
| 数据目录 | `data/metaworld_mt50_corner2` |
| 归一化 | `stats/metaworld_mt50_corner2_480_50ep_4d.json` |
| VTT | `embeddings/metaworld_mt50_corner2_480_50ep_input224_vtt_dinov3b` |
| 视觉编码器 | 冻结DINOv3-B：`pretrained/dinov3-vitb16` |
| 图像 | 原始480×480，模型输入224×224，既有180°方向协议 |
| 视觉层 | `[-7,-4,-2]`；意图使用-2层patch |
| 状态/动作 | 各4维，状态历史长度1 |
| 时间跨度 | chunk 8步、执行前4步、未来帧偏移8步，即C8/E4/H8 |
| 学生池化器 | VTT＋当前状态生成单query，attention宽256、4 heads |
| 动作网络 | DiT depth12、hidden512、8 heads；视觉适配器64 queries、depth4 |
| 关闭项 | 直接VTT token、稠密FutureFeat、gate、action-prior、action-reuse |
| 保留项 | VTT间接影响学生query/预测器，意图token实际注入DiT |
| 训练 | seed0、batch96、AdamW、梯度裁剪；50轮cosine，LR 2e-4→5e-5 |
| 主比较checkpoint | epoch40，保留原50轮学习率规划 |

当前数据均可用于训练；这里没有新增训练/验证划分。离线诊断从训练数据抽样，最终控制性能由单独仿真闭环评测判断。

不要将`--epochs`直接从50改成40来做本组对照：这样会改变前40轮的cosine学习率曲线。训练seed、采集环境seed、评测环境seed和离线诊断seed也不是同一概念。

## 3. 必须区分的三类特征

以下形状省略batch维。

| 对象 | 形状 | 含义 |
| --- | --- | --- |
| 当前/未来DINO patch `F_t/F_future` | 196×768 | 冻结视觉编码器输出，不是可训练的动作意图标签本身 |
| 教师/学生latent `z_T/z_S` | 768 | posterior或因果预测器的输出；当前KD在这里计算 |
| 实际注入DiT的意图token `u_T/u_S` | 1×512 | latent经过各自动作网络的LayerNorm、Linear及type embedding |

对应代码：[意图投影及注入](../models/vla_model_fm.py)、[教师posterior与蒸馏loss](../models/planwam_g2.py)。

教师latent并不是未来DINO patch的平均值，也不是原G2的固定全局变化标签。它是经过动作任务训练的posterior输出。配置中保留的`target_pooling=mean`等兼容字段，不代表独立PlanWAM入口仍在使用原mean意图loss。

本方案不预测未来RGB图像；也不将真实未来帧直接送给部署学生。需要特别区分“训练教师时可见未来”和“学生推理时可见未来”。

## 4. 阶段A：教师如何学习动作相关的未来表示

### 4.1 教师数据通路

```text
当前图像 I_t ── 冻结DINO ── 当前patch F_t ─────────────→ 教师视觉适配器 ─────┐
                                │                                        │
当前状态 s_t、VTT、未来跨度 H ──┼────→ 当前context                        │
                                │          ↑                             │
未来图像 I_(t+H) → 冻结DINO → 未来patch queries                            │
                                           ↓                             │
                         cross-attention＋残差FFN                         │
                                           ↓                             │
                         可学习summary query汇总                          │
                                           ↓                             │
                                      z_T：768维                         │
                                           ↓                             │
                       教师自己的LN＋Linear＋type embedding               │
                                           ↓                             │
                                      u_T：1×512                         │
                                           └────────→ 教师DiT ←───────────┘
                                                        ↑
                                              x_tau、tau、状态token
                                                        ↓
                                               预测速度 v_T
                                                        ↓
                                              专家动作FM监督
```

posterior内部的196个future queries不是196个最终意图token；其输出最终压缩为一个768维latent。summary query是可训练参数，不是由VTT动态生成的query。

### 4.2 教师训练目标和反向传播

记当前因果条件为`c=(F_t,s_t,VTT,H)`，动作块为归一化专家动作`a`，高斯源为`epsilon`：

```text
z_T   = T_phi(c, F_future)
u_T   = P_teacher(z_T)
x_tau = (1-tau) * epsilon + tau * a
v_T   = D_teacher(x_tau, tau, 当前视觉条件, 状态, u_T)
L_T   = MSE(v_T, a - epsilon)
```

梯度路径：

```text
FM动作loss → 教师动作头/DiT → 意图投影 → z_T → posterior
                         └→ 当前视觉适配器等可训练动作条件模块
```

posterior没有直接输入专家动作、FM噪声或去噪时间；专家动作通过loss塑造它。DINO本体不更新。教师保留但冻结旧的`intent_pool`和`intent_predictor`，实际使用新posterior。

这使教师有机会保留有助于动作的未来信息，但不保证只保留重要信息，不保证一定使用未来，更不保证这些信息都能从当前观测预测出来。之前的教师未来干预诊断支持其确实利用未来；它本身不等于教师强于学生的证据。

注意：该latent不是严格的物理状态，也没有被训练成可接受任意候选动作并预测反事实未来的动力学模型。

## 5. 阶段B：当前latent蒸馏究竟更新哪些模块

### 5.1 当前真实执行的流程

```text
教师侧：当前patch＋真实未来patch＋状态＋VTT＋H
                          ↓
                    冻结posterior
                          ↓
                  z_T → fixed LN → 停止梯度目标 ─────┐
                                                    ↓
学生侧：当前patch＋状态＋VTT＋H → G_theta → z_S → fixed LN → L_latent
                                               │
                                               ↓
                               学生自己的LN＋Linear＋type embedding
                                               ↓
                                         u_S：1×512
                                               ↓
                   当前视觉适配器、状态、x_tau、tau → 学生DiT
                                               ↓
                                         v_S → L_FM
```

已完成的旧学生训练阶段**只加载教师posterior生成标签，不加载教师DiT权重，也不运行教师动作网络**。旧学生的task-attention、MLP、意图投影、DiT等从头初始化。例外是固定复用的预训练DINO、数据、VTT和stats。2026-11-08新增初始化分支仅由新配置显式开启，不能反过来解释旧结果。

这不是先单独学完latent再训练DiT，而是同一训练步骤内联合计算：

```text
z_S      = G_theta(c)
z_T      = stop_gradient(T_phi(c, F_future))
L_latent = MSE(fixed_LN(z_S), stop_gradient(fixed_LN(z_T)))
u_S      = P_psi(z_S)
v_S      = D_omega(x_tau, tau, 当前条件, u_S)
L_FM     = MSE(v_S, a - epsilon)
L_total  = L_FM + 0.1 * L_latent
```

### 5.2 直接梯度分工

| 模块 | FM动作loss直接更新 | latent KD直接更新 |
| --- | --- | --- |
| 学生task/state-attention池化器 | 是 | 是 |
| 学生latent预测MLP | 是 | 是 |
| 768→512意图投影与type embedding | 是 | 否 |
| 当前视觉融合、适配器 | 是 | 否 |
| 学生DiT及动作头 | 是 | 否 |
| 教师posterior、DINO、预计算VTT | 否 | 否 |

若`theta`为学生池化/预测参数，`psi`为token投影参数，`omega`为其余动作网络参数，则：

```text
grad_theta(L) = grad_theta(L_FM) + 0.1 * grad_theta(L_latent)
grad_psi(L)   = grad_psi(L_FM)
grad_omega(L) = grad_omega(L_FM)
```

因此，当前方案的准确表述是：**教师标签直接约束学生latent；专家动作监督同时塑造latent、投影及DiT。教师通过改变学生latent间接影响DiT，而不是用教师动作输出直接监督DiT。**

代码依据：[独立训练入口](../metaworld_vla/train_planwam.py)、[planwam_flow_loss](../models/planwam_g2.py)、[VLAModel](../models/vla_model_fm.py)。

### 5.3 为什么latent接近仍可能不能替换

一种可能情况：学生无法完全恢复教师latent，但学生DiT学会了使用这个有偏近似；二者形成相互补偿，直接替换z_T反而破坏了这种配合。另一种可能是剩余误差集中在动作敏感方向，均匀latent MSE没有反映功能差异。当前观测的不可预测性、教师表示公共成分较强等也仍是候选原因。

不能把高cosine解释为“已理解全部动作语义”。当前fixed LN后的MSE与cosine近似满足`MSE ≈ 2*(1-cosine)`，二者不是独立证据；单纯再加cosine loss可能主要改变约束强度。

## 6. 与PlanWAM原文的核心对比

本项目应称为**PlanWAM启发的轻量动作蒸馏变体**，不是严格复现。下表原文架构依据[官方方法页](https://jinchan9.github.io/PlanWAM_project_page/)，训练细节依据[论文v1第3.5节及附录B](https://arxiv.org/html/2610.11382v1)。

| 维度 | PlanWAM原文 | 当前项目 |
| --- | --- | --- |
| 因果输入 | 多帧、多视角历史及压缩history | 单帧corner2＋当前状态＋VTT |
| 未来表示 | 256个结构对齐的future tokens | 一个768维latent，投影为1×512 token |
| posterior/prior | 未来观测queries/learnable queries，兼容结构 | 未来驱动posterior/task-state attention＋MLP，不同结构 |
| 教师任务 | 轨迹生成＋评分 | Gaussian Flow Matching动作监督 |
| 学生初始化 | 继承第一阶段兼容参数 | 学生动作网络、预测器均从头初始化 |
| 第二阶段LR | 低于第一阶段；峰值1e-4→1e-5 | 两阶段各用2e-4→5e-5规划 |
| latent对齐 | 未逐token fixed-LN的MSE，权重0.05 | fixed-LN后MSE，权重0.1 |
| 其他约束 | 范数/slot多样性正则 | 当前无此额外正则 |

原文附录D.4报告：仅prior、latent loss时，相似度0.96、PDMS 93.0；扩大联合训练模块后，相似度0.93、PDMS 93.8。相似性不能替代任务指标。[原文附录D.4](https://arxiv.org/html/2610.11382v1)

### 6.1 对本项目的启示：需要验证，不是已经证明

- 优先检查表示—动作读出的兼容性，而不是仅增加预测器容量。教师动作权重迁移是一个可单独测试的初始化变量。
- 继续保留专家动作loss，不默认只训练预测器、长期冻结DiT就能获得控制收益。
- 0.05与0.1不能按字面比较约束强弱：latent归一化、目标尺度、任务loss均不同。
- 若教师优势难以迁移，需检查学生当前输入是否包含足够时序信息；增加少量历史是一条后备路线，不是本轮已实施改动。
- 当前单latent没有slot间多样性，不能直接照搬多slot diversity loss。以后若扩展多个token，应明确槽位对应关系，先从少量token开始，不同时更改多个因素。
- 教师latent由未来示范轨迹产生，不一定可从当前观测唯一恢复；更大教师并非必然更容易蒸馏。

特别注意：教师checkpoint中的旧`intent_pool/intent_predictor`是冻结未训练分支，不能误当作学好的因果模块迁移。新增初始化实现只复制实际训练过且语义/结构兼容的模块，并明确排除不兼容项。

这种两阶段初始化也不是恢复旧学生优化器：它是新的学生初始化实验。必须单独记录权重来源、复制/排除参数、LR和训练成本，不与当前随机初始化协议混称。

### 6.2 两个预测器对照：已记录，暂不实施

教师没有学生的`TaskConditionedIntentPool`，不等于教师不使用任务信息：其context仍含状态和VTT。蒸馏对齐输出latent，而不是要求中间attention相同。学生池化器也会受到KD与动作loss的梯度更新，不是固定裁剪器。

| 待做对照 | 保持项 | 主要问题 |
| --- | --- | --- |
| P1：task-attention与mean pooling | VTT/状态/H仍输入预测MLP；保持同一输出维度、初始化、loss与评测协议 | 动态任务query对当前特定监督是否有益？ |
| P2：当前单query提前池化与延迟汇总预测器 | 最终仍输出一个768维latent；VTT/状态仍保留；不同时加历史/改teacher | 提前压缩是否丢失多物体、接触区域等关系信息？ |

P1不能简单关闭`use_task_cond`或`inject_task_cond`，否则会同时改变任务输入路径；mean并不自动保留更多信息。P2可考虑少量可学习queries先读当前patch并交互，最后汇总，具体结构/槽位数尚未冻结，尚无配置或训练结果。

若比较蒸馏监督的影响，池化器对照最好与有/无KD配对；控制变量后才可区分结构收益与监督收益。当前`train_planwam.py`和teacher输入合同绑定task-attention，P1/P2需要新的明确协议，不能仅改yaml或绕过checkpoint校验。

**本轮不改变学生task-attention。** 先执行兼容初始化对照；这些预测器方案不是同时加入C/D的新结构。

## 7. 已完成的同协议闭环结果

统一协议：epoch40；MT50每任务50次，总2500次；horizon500；env seed1000（每任务episode seeds 1000–1049）；policy seed456；workers8；10步推理；C8/E4；stop-on-success；无action-reuse。

| 模型 | 成功次数 | 成功率 |
| --- | --- | --- |
| easy_noffvtt | 2184/2500 | 87.36% |
| 原G2 task-attention | 2195/2500 | 87.80% |
| 随机初始化action-only | 2192/2500 | 87.68% |
| 当前PlanWAM latent蒸馏学生 | 2145/2500 | 85.80% |

数据来源：

- [baseline同协议补测](../eval_result/planwam_comparison/easy_noffvtt_ep40_cold_env1000_policy456.json)。
- [原G2 cold结果](../eval_result/action_prior_g2_base/g2_ep40_cold_confirm_env1000_policy456.json)。
- [蒸馏学生cold结果](../eval_result/planwam_student01_seed1000/student_ep40_cold_env1000_policy456.json)。
- [action-only cold结果](../eval_result/planwam_comparison/action_only_ep40_cold_env1000_policy456.json)。

学生比baseline低1.56个百分点，比原G2低2.00个百分点。G2这份结果打开了generation计时，另两份未打开；该选项用于计时同步，不是动作reuse开关，不应将这些并行运行的耗时直接当作隔离的模型延迟对比。记录字段一致不代表已经审计了全部环境版本差异。

本组为各自当前训练实例和一组环境seed区间，不是多训练seed的显著性结论。此前policy seed未固定的历史baseline不能替代此补测。

2026-11-08新增结论：action-only比蒸馏学生高1.88个百分点，净增47次成功；任务均值16个改善、11个下降、23个持平。同环境seed的episode配对中，action-only独有成功164次，蒸馏学生独有成功117次。训练配置和输入协议一致，配置差异仅stage及lambda_kd。支持当前KD可能存在负作用，但不是普遍负迁移的证明。与原G2仅差3次成功、与baseline仅差8次，不能据此宣称固定未来监督或task-attention通路有稳定额外收益。

## 8. 已完成的离线S/S、S/T与完整2×2结果

### 8.1 定义与可比性

| 记号 | 动作网络 | latent来源 |
| --- | --- | --- |
| S/S | 学生 | 学生当前条件预测器 |
| S/T | 学生 | 冻结教师posterior |
| T/S | 教师 | **学生**预测器，不调用教师未训练的旧因果分支 |
| T/T | 教师 | 冻结教师posterior |

“动作网络”包括该checkpoint自己的视觉适配器、意图投影、DiT和动作头。跨行比较不能仅归因于某一层。跨列替换的是768维原始latent，不是把教师512维token塞到学生内部。

已核验的主结果：[完整2×2 JSON](../eval_result/planwam_offline/student40_teacher40_2x2_seed0_n8_chunk10.json)。旧结果：[S/S、S/T JSON](../eval_result/planwam_offline/student40_ss_st_seed0_n8_chunk10.json)。

核验项目：

- 学生与教师均为epoch40，权重文件SHA256与JSON一致。
- CUDA/bf16，离线seed0、batch8，每任务8个不同anchor，共400个，非400个完整episode。
- 四组同次运行共用样本、当前视觉条件、状态、VTT、高斯噪声；tau为0.1/0.5/0.9。
- 完整生成从同一个高斯出发，10步Euler，无reuse；两组latent各只预测一次。
- 所有指标有限，整体均值重新计算一致；新旧结果的样本身份及已有S/S、S/T指标逐项完全相同。
- 这些样本来自训练数据，每anchor一份噪声；相邻anchor可能相关，不作独立试验显著性推断。

### 8.2 整体矩阵：以下都是误差，不是成功率

| 指标 | S/S | S/T | T/S | T/T |
| --- | ---: | ---: | ---: | ---: |
| 平均去噪MSE | 0.01931990 | 0.03721858 | 0.03243930 | 0.01544002 |
| 完整8步动作块MSE | 0.01128443 | 0.02397841 | 0.01764947 | 0.00584091 |
| 前4步执行前缀MSE | 0.01311895 | 0.02468872 | 0.02114036 | 0.00575410 |
| 仅真实记录步MSE | 0.01105246 | 0.02365800 | 0.01661696 | 0.00543273 |
| 反归一化xyz指令MSE | 0.01470923 | 0.02135539 | 0.02229085 | 0.00749743 |
| 反归一化夹爪指令MSE | 0.00102078 | 0.03186427 | 0.00373790 | 0.00088686 |

平均去噪为三个tau的算术平均，不是训练logit-normal分布加权期望。xyz为控制指令，不是测量到的末端位姿误差；表中指令误差是未clip值，JSON另有按[-1,1] clip的指标。

### 8.3 四项配对结论

| 比较 | 动作块均值变化 | 每任务均值情况 | 已验证事实与解释边界 |
| --- | --- | --- | --- |
| T/T 对 S/S | 降低48.24% | 37任务更低、13更高 | 教师完整组合在所抽离线数据上有优势，不等于可部署上界 |
| S/T 对 S/S | 增加112.49% | 39任务变差、11改善 | 教师latent不能直接等价替换到学生动作网络 |
| T/S 对 T/T | 增加202.17%，约3.02倍 | 41任务变差、9改善 | 学生latent尚未达到教师动作网络可等价使用的程度 |
| T/S 对 S/S | 增加56.41% | 27任务更高、23更低 | 直接把学生预测器接到教师动作网络也不是已解决方案 |

T/T相对S/S的平均去噪误差降低20.08%，前4步误差降低56.14%。T/T在56.75%的anchor上动作块误差更低；均值降幅不能解释为所有样本均改善约一半。

### 8.4 尾部、padding与任务差异

共有354个anchor可取得完整H8未来跨度，46个未来帧被截断到episode末尾；39个动作块含人工padding。未来截断与动作padding不是同一判定。

| 完整未来跨度的354个anchor | S/S | S/T | T/S | T/T |
| --- | ---: | ---: | ---: | ---: |
| 完整动作块MSE | 0.01114262 | 0.02517011 | 0.01770085 | 0.00526637 |

因此当前问题不由末尾截断/padding主导。总MSE仍包含原训练padding目标，另报recorded-only供诊断，不改变训练语义。

尾部稳健性检查：对每一对比较，按两组误差的较大值排序，共同去掉最高5%的anchor，再计算均值（事后描述性检查，不是主指标）：

- T/T相较S/S仍降低约36.17%。
- S/T相较S/S仍增加约14.99%。
- T/S相较T/T仍增加约88.85%。

说明少数大误差会放大均值差异，但主要趋势不只来自极端值。以原始夹爪MSE>1计，S/T有5个anchor，其他三组均为0；但教师并非所有夹爪统计都更好，不能只看整体均值宣称全面改善。

任务举例（每任务仅8个anchor，不能当作稳定排名）：

| 任务 | S/S动作块MSE | T/T动作块MSE | 趋势 |
| --- | ---: | ---: | --- |
| handle-pull-v3 | 0.05230 | 0.00681 | 教师较好 |
| stick-pull-v3 | 0.06424 | 0.02053 | 教师较好 |
| disassemble-v3 | 0.03311 | 0.00014 | 教师较好 |
| door-lock-v3 | 0.00142 | 0.02574 | 教师较差 |
| button-press-v3 | 0.00428 | 0.02003 | 教师较差 |
| coffee-pull-v3 | 0.00036 | 0.00613 | 教师较差 |

### 8.5 latent指标该怎样解释

| 指标 | 本组均值 |
| --- | ---: |
| normalized latent MSE | 0.04120044 |
| normalized latent cosine | 0.97939961 |
| 学生raw norm | 38.93083 |
| 教师raw norm | 33.05514 |
| 经学生共享意图投影后的token MSE | 0.21470504 |

raw norm不能与日志中归一化teacher target的`~sqrt(768)=27.713`直接比较。投影包含LayerNorm，原始范数差异本身不证明注入过强。768维与512维MSE也不能直接比较大小来断言“投影放大了误差”。

### 8.6 结论分级

**已验证：** 教师完整组合存在离线优势；当前学生闭环未超过本组baseline；双向latent替换退化；高整体相似性未带来功能可替换性。

**有证据支持、尚未唯一确定：** 学生latent与学生动作网络形成配套适应；剩余latent误差可能落在动作敏感方向；当前蒸馏没有有效传递教师的全部可利用能力。

**尚未证明：** KD必然负迁移、教师只学了不可预测信息、低成功率只由随机初始化造成、增加KD/增加token一定有效、教师将有更高在线成功率。

特别是T/T用了专家真实未来，该未来不一定是生成动作本身会导致的未来，故不是严格反事实预测或性能上界。控制泛化不能用训练数据MSE替代。

## 9. 三个优先级实验：状态、命令与决策

### 优先级1：完整2×2交叉诊断——已完成

目的：先验证教师完整组合是否更好，以及表示替换/动作读出是否存在兼容问题。结果及限制见第8节；当前不必重复同一400样本测试。

脚本：[diagnose_planwam_student.py](../metaworld_vla/diagnose_planwam_student.py)。不加`--include-teacher-dit`仍是旧两组模式；加上后才加载完整教师动作网络。正式命令如下，已有输出不可覆盖；重复时必须改输出名。

```bash
cd /home/lab347-no10/Tianyi/LiLa-WAM
conda activate MetaWorldVLA
RUN="checkpoints/metaworld_mt50_easy_planwam_student_seed0"
TEACHER="checkpoints/metaworld_mt50_easy_planwam_teacher_seed0/checkpoint_epoch_40.pt"

CUDA_VISIBLE_DEVICES=0 python metaworld_vla/diagnose_planwam_student.py \
  --config "$RUN/config.yaml" \
  --stats stats/metaworld_mt50_corner2_480_50ep_4d.json \
  --checkpoint "$RUN/checkpoint_epoch_40.pt" \
  --teacher-checkpoint "$TEACHER" \
  --samples-per-task 8 --batch-size 8 --seed 0 \
  --generate-chunks --num-inference-steps 10 \
  --include-teacher-dit \
  --output eval_result/planwam_offline/student40_teacher40_2x2_seed0_n8_chunk10.json
```

`overall.comparisons`及各任务下的`st_vs_ss/ts_vs_tt/tt_vs_ss/ts_vs_ss`给出带方向的配对统计，`delta_candidate_minus_reference<0`表示candidate误差更低。离线seed不等于env seed或训练seed。

资源：无反向、DINO共享，但多一份冻结动作网络和瞬时激活，四组动作前向约为两组的两倍；不需要启动仿真。

### 优先级2：action-only同结构从头训练——已完成，结果见第7节

目的：确认当前教师KD到底帮助还是妨碍了**同一个学生结构**。它保留task-attention、MLP、意图投影与DiT，只去掉教师及KD，不是关闭意图注入。

配置：[metaworld_mt50_easy_planwam_action_only.yaml](../metaworld_vla/configs/metaworld_mt50_easy_planwam_action_only.yaml)。与已训练学生run配置逐项核验，差异只有：

```yaml
planwam:
  stage: action_only  # student → action_only
  lambda_kd: 0.0     # 0.1 → 0.0
```

原`lambda_intent=0`、`lambda_future_feat=0`保持不变；训练入口的action_only分支不读取未来帧。动作监督仍可经DiT回传到预测器，但该latent不受未来标签约束，不能直接称为已学会未来预测。

训练命令（新目录，不传teacher/resume/init-from）：

```bash
CUDA_VISIBLE_DEVICES=0 python metaworld_vla/train_planwam.py \
  --config metaworld_vla/configs/metaworld_mt50_easy_planwam_action_only.yaml \
  --stats stats/metaworld_mt50_corner2_480_50ep_4d.json \
  --save-dir checkpoints/metaworld_mt50_easy_planwam_action_only_seed0
```

保持50轮scheduler，主评epoch40。生成epoch40后可测试，不需要先看epoch50或改变协议：

```bash
RUN="checkpoints/metaworld_mt50_easy_planwam_action_only_seed0"
CUDA_VISIBLE_DEVICES=0 python metaworld_vla/eval_parallel.py \
  --config "$RUN/config.yaml" \
  --stats stats/metaworld_mt50_corner2_480_50ep_4d.json \
  --checkpoint "$RUN/checkpoint_epoch_40.pt" \
  --workers 8 --test-num 50 --horizon 500 \
  --seed 1000 --policy-seed 456 --num-inference-steps 10 \
  --output eval_result/planwam_comparison/action_only_ep40_cold_env1000_policy456.json
```

决策标准：

| action-only结果 | 下一步解释 |
| --- | --- |
| 明显优于蒸馏学生 | 当前KD可能有负迁移，优先检查监督/权重，不盲目增加对齐 |
| 蒸馏学生明显更好 | KD可能有益，但尚未充分迁移教师能力，可继续探索响应蒸馏 |
| 二者接近 | 尚无清楚的KD收益证据，需考虑训练seed和任务差异 |

这里的“明显”不能预设为某个单seed下的微小百分点阈值。需要结合配对任务表现、重复训练和环境seed覆盖判断。保持原G2作参考，不删除已有结果，不同时更改网络容量/图像/未来跨度。

### 优先级3：动作响应／速度场蒸馏——待实现，条件式推进

目的：对齐具有共同动作含义的8×4速度输出，而不是只要求内部768维表示逐维接近。2×2已显示教师离线优势，但还需要action-only补充KD作用证据。

候选流程：

```text
同一expert anchor、同一epsilon/tau → 同一个x_tau

当前＋未来 → 冻结posterior → z_T → 冻结教师意图投影/DiT → v_T（停止梯度）
当前条件   → 学生预测器   → z_S → 学生意图投影/DiT     → v_S

L_FM       = MSE(v_S, a - epsilon)
L_response = MSE(v_S, stop_gradient(v_T))
L_total    = L_FM + beta * L_response + alpha * L_latent
```

响应loss将直接更新学生预测器、意图投影、视觉适配器及DiT；教师所有参数冻结，并在no_grad下前向。第一版可考虑`alpha=0`隔离响应蒸馏的贡献，beta需提前明确并固定，不能根据最终测试集反复挑选最佳值。

**这不是当前lambda_kd换个数字即可实现的功能**；需要新增独立loss与配置分支，并做梯度、输入一致性和旧模型回归测试。本文不提供不存在的响应训练入口命令。

相比latent-only学生训练，可复用当前/未来DINO与posterior计算，多一份冻结教师动作网络及每batch一次动作前向；不复制DINO，不做教师10步积分，不采环境交互。权重和瞬时激活仍增加显存，必须实测，不能保证免费。

这是expert插值上的离线响应蒸馏，不是沿学生自身生成轨迹查询教师，不能称为OPD。教师不一定等于专家目标：若一致，额外项主要增强相同监督；若有偏差，则引入有偏软目标，因此保留L_FM。

### 当前下一步：C/D兼容初始化与分组LR——2026-11-08已实现

以action-only 87.68%高于原蒸馏学生85.80%的结果为依据，先做C，再做D；不立即叠加响应loss。下述新配置不修改旧结果或旧模型结构。

| 实验 | 初始化 | 监督 | LR | 状态 |
| --- | --- | --- | --- | --- |
| 原action-only | 随机 | 仅FM | 原统一LR | 已完成87.68% |
| 原student | 随机 | FM＋0.1 latent KD | 原统一LR | 已完成85.80% |
| C | 教师兼容动作模块＋新因果预测器 | 仅FM | 分组LR | 配置/CPU smoke完成，正式训练待执行 |
| D | 与C相同 | FM＋0.1 latent KD | 与C相同 | 已准备，建议C之后执行 |
| C-uniform（可选） | 与C相同 | 仅FM | 原统一LR | 用于进一步拆分初始化与分组LR的贡献，不要求立即跑 |

实现：[planwam_transfer.py](../metaworld_vla/planwam_transfer.py)，由[train_planwam.py](../metaworld_vla/train_planwam.py)的显式新开关启用。旧配置不含开关，保持原单组AdamW、CosineAnnealingLR及旧resume协议。

复制白名单：teacher实际使用的time/action/state投影、类型/位置嵌入、register、视觉融合/适配器、意图投影、DiT及动作头。保留新初始化：`intent_pool`与`intent_predictor`；不复制未使用的直接VTT分支及gate。兼容性包含teacher阶段/版本、完整epoch、模型键/形状、输入合同、stats/VTT及posterior规格，并记录源文件SHA256。不会恢复teacher优化器、scheduler、epoch。

真实模型审计为复制226个state tensor（含2个位置buffer），其中224个可训练参数tensor、77,189,892个参数进入继承组；新预测器32个参数tensor、3,111,440个参数进入新组。未使用兼容分支保留在state_dict，分组优化器不包含它们。推理参数键不变，沿用现有eval入口。

分组LR是候选设置，不是已调优最优值：

| 参数组 | 初始LR | 50轮末端LR |
| --- | ---: | ---: |
| inherited_action | 2e-5 | 5e-6 |
| new_predictor | 2e-4 | 5e-5 |

两组同步cosine、分别设置floor，不新增warmup、不冻结DiT。旧`training.learning_rate/lr_min`保留为兼容字段，分组开启后实际LR以`planwam.grouped_lr`为准。尤其不能把旧5e-5下限套在初始2e-5的继承组上。日志同时打印两组LR，不能只看主行的第一个LR。

#### C训练与评测（先执行）

```bash
cd /home/lab347-no10/Tianyi/LiLa-WAM
conda activate MetaWorldVLA
TEACHER="checkpoints/metaworld_mt50_easy_planwam_teacher_seed0/checkpoint_epoch_40.pt"

CUDA_VISIBLE_DEVICES=0 python metaworld_vla/train_planwam.py \
  --config metaworld_vla/configs/metaworld_mt50_easy_planwam_action_init_action_only.yaml \
  --stats stats/metaworld_mt50_corner2_480_50ep_4d.json \
  --init-teacher-action "$TEACHER" \
  --save-dir checkpoints/metaworld_mt50_easy_planwam_action_init_action_only_seed0
```

此处的`--init-teacher-action`仅初始化，不能换成`--teacher-checkpoint`或`--resume`。C不读取未来帧、不常驻教师posterior/教师动作副本，因此不增加一个教师动作前向；复制后的动作参数仍属于学生并参与训练。

```bash
RUN="checkpoints/metaworld_mt50_easy_planwam_action_init_action_only_seed0"
CUDA_VISIBLE_DEVICES=0 python metaworld_vla/eval_parallel.py \
  --config "$RUN/config.yaml" \
  --stats stats/metaworld_mt50_corner2_480_50ep_4d.json \
  --checkpoint "$RUN/checkpoint_epoch_40.pt" \
  --workers 8 --test-num 50 --horizon 500 \
  --seed 1000 --policy-seed 456 --num-inference-steps 10 \
  --output eval_result/planwam_comparison/action_init_action_only_ep40_cold_env1000_policy456.json
```

#### D训练（C完成后再决定是否执行）

```bash
TEACHER="checkpoints/metaworld_mt50_easy_planwam_teacher_seed0/checkpoint_epoch_40.pt"
CUDA_VISIBLE_DEVICES=0 python metaworld_vla/train_planwam.py \
  --config metaworld_vla/configs/metaworld_mt50_easy_planwam_action_init_student.yaml \
  --stats stats/metaworld_mt50_corner2_480_50ep_4d.json \
  --init-teacher-action "$TEACHER" \
  --teacher-checkpoint "$TEACHER" \
  --save-dir checkpoints/metaworld_mt50_easy_planwam_action_init_student_seed0
```

D与C同样从原teacher epoch40开始，**不是从C训练结果继续**。D的初始化教师和KD教师必须是同一文件身份。D评测沿用C命令，将RUN改为D目录，输出改为`action_init_student_ep40_cold_env1000_policy456.json`。

C优于原action-only，仅支持“初始化＋优化协议”组合有效，不是KD收益；D优于C才支持本组KD有额外作用。若需要单独隔离初始化，使用[统一LR控制配置](../metaworld_vla/configs/metaworld_mt50_easy_planwam_action_init_action_only_uniformlr.yaml)，其余命令相同但必须换新save-dir。开启该控制时两组都回到原2e-4→5e-5，不是只关闭某一组训练。

每次保存`action_init_report.json`和扩展的`planwam_protocol.json`，checkpoint也记录初始化及优化协议。同阶段续训使用相同配置、原始初始化源文件及`--resume <本次学生checkpoint>`；只恢复学生，不重新覆盖teacher权重。修改KD、LR、初始化源或组成员会被协议检查拒绝。部署只需学生checkpoint，不需要初始化源文件。

#### 验证边界

- 新增7项测试，覆盖选择性加载、因果预测器保持新值、源协议保护、分组覆盖/无重复、各组floor、梯度及优化器/scheduler恢复；原PlanWAM 6项、离线诊断8项及无gate回归检查通过。
- 真实2500episode索引、DINO-B、完整h512模型：C/D各一个CPU优化步骤并保存，FM和KD均有限；现有Policy严格加载两者并在没有未来输入时生成有限8×4动作。
- 小型单anchor/epoch的真实CPU入口恢复测试：恢复后的权重、scheduler、协议和global step与不间断运行完全一致，不代表完整数据训练收益。
- 当前工具环境无法访问NVIDIA驱动；正式GPU显存/速度、长训练效果仍待用户验证。所有开发产物在`/tmp`，不作为正式结果。

## 10. 成本、公平性与复现边界

- 当前G2约50轮训练；教师50＋学生50包含额外教师成本，不能仅按学生50轮宣称等计算预算。记录总GPU小时、峰值显存和训练阶段。
- 教师初始化迁移改变初始化协议；响应蒸馏改变目标与教师前向成本，必须作为不同实验标记。
- 本轮未更改padding、数据划分、gate或action-reuse；所有实验默认不影响旧G2与真机模型。
- 复现需要配置＋对应checkpoint＋stats/VTT/视觉权重＋代码版本；不能仅凭yaml恢复已训练权重，也不能把同样维度等同于同样语义。
- 诊断的后处理裁剪不改变主结果JSON；正式结果文件应保留原始数据，重跑使用新名字。
- 训练模块smoke通过不等于完整GPU性能验证；用户生成的正式GPU结果已单独核验，不能再写成“完整训练/闭环尚未执行”。

## 11. 关键实现与产物索引

| 内容 | 路径 |
| --- | --- |
| 教师posterior、当前latent KD | [models/planwam_g2.py](../models/planwam_g2.py) |
| 独立teacher/student/action_only入口 | [metaworld_vla/train_planwam.py](../metaworld_vla/train_planwam.py) |
| 视觉提取与归一化 | [models/model_runner.py](../models/model_runner.py) |
| 意图token、学生池化器及DiT | [models/vla_model_fm.py](../models/vla_model_fm.py) |
| 教师未来干预诊断 | [metaworld_vla/diagnose_planwam.py](../metaworld_vla/diagnose_planwam.py) |
| 两组及完整2×2诊断 | [metaworld_vla/diagnose_planwam_student.py](../metaworld_vla/diagnose_planwam_student.py) |
| PlanWAM模块测试 | [utils/test_planwam_g2.py](../utils/test_planwam_g2.py) |
| 配对诊断测试 | [utils/test_planwam_student_diagnostic.py](../utils/test_planwam_student_diagnostic.py) |
| 教师主checkpoint | `checkpoints/metaworld_mt50_easy_planwam_teacher_seed0/checkpoint_epoch_40.pt` |
| 学生主checkpoint | `checkpoints/metaworld_mt50_easy_planwam_student_seed0/checkpoint_epoch_40.pt` |
| 学生日志 | `log/metaworld_planwam_student/train.log` |

2026-11-08修订新增了可选初始化/分组LR训练分支及新配置，但未改网络forward、模型state键、旧配置、已训练权重或原始测评结果。响应蒸馏及P1/P2预测器对照仍未实施。

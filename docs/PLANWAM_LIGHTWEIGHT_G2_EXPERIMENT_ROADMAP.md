# 轻量 Action-Shaped G2：Posterior 教师与因果学生实验路线

日期：2026-11-04。

同日修订：主教师改为**未来特征驱动（future-driven）Posterior**；先让多个未来patch与当前上下文交互，再压缩为一个latent。原task-query双池化教师降为备选对照。学生仍保留当前有效的task-attention G2。

状态更新（2026-11-08）：**教师、蒸馏学生和随机action-only均完成训练；epoch40同协议闭环分别确认学生85.80%、action-only87.68%，完整2×2诊断已完成。当前新增可选“教师兼容动作初始化＋分组LR”的C/D配置，CPU管线验证完成、正式训练待执行；响应蒸馏与两个预测器结构对照仍未实现。** 使用独立入口 `metaworld_vla/train_planwam.py`，不是旧 `train.py`。原有G2默认行为保持不变。

详细机制、梯度通路、PlanWAM核心对比、已完成结果及三个优先级的统一记录见：[教师—学生蒸馏机制与结果分析](PLANWAM_DISTILLATION_MECHANISM_AND_RESULTS.md)。本文保留早期设计和开发smoke记录；其中“未执行/未验证”应按当时语境理解，当前状态以上述更新及新汇总文档为准。

平台入口更新（2026-11-11）：本文件的MetaWorld实验继续使用`metaworld_vla/train_planwam.py`；RoboTwin已拆成独立`train_planwam_robotwin.py`，不能再用本文训练入口传RoboTwin配置。详见[RoboTwin实验命令](ROBOTWIN_EASY_G2_PRIOR_EXPERIMENT.md)。公共模型/loss未在本次拆分中修改。

## 1. 本轮范围与决定

借鉴PlanWAM的“任务塑形未来表示，再蒸馏到可部署预测器”思想，不完整复制其网络或训练规模。

- 学生保留现有有效的VTT＋当前状态task-attention单query、一个意图token和Flow Matching DiT；教师不受该入口单query池化限制。
- 无gate；直接VTT token关闭，VTT仍供query与预测器使用。
- 末端稠密FutureFeat Decoder关闭。
- 不引入action-prior R、tail cache、动作复用或其训练源分布；使用原标准高斯Flow Matching源。
- 不引入attention-target、mean/std目标、多意图token/学生多query、空间残差、region gate、EMA、RL或在线OPD。教师内部保留多个未来patch queries，不等于新增多个部署意图token。
- 不新增多帧历史、多路摄像头或多个未来时间。
- **不使用已有G2 checkpoint短微调作为可行性判断。教师、学生的主实验都从头初始化训练。** 只复用冻结的预训练DINO权重及现有数据/VTT/stats。

主要问题：与固定DINO全局变化目标相比，由动作任务训练的未来posterior表示，是否能成为更有用且可从当前观测预测的G2监督？

## 2. 学生还需要task-attention query吗？

需要保留，作为本轮实验选择；它不是posterior蒸馏的数学必要条件。

理由：现有实验已报告task-attention优于mean输入汇聚，本轮应只改变监督来源，不同时回退输入结构。学生的query仍由当前VTT和状态生成，读取当前DINO patch。query不是未来图像，不输入真实未来动作，不依赖训练期特权信息。

教师不再先用VTT/state query读取未来并压成单向量；主方案改为未来patch提供queries，当前视觉、state、VTT提供context，最后通过可学习summary query压缩。summary query是可训练参数，不由VTT/state动态生成；不是冻结随机query。

教师和学生没有必须共享的attention权重：阶段B教师完全冻结，学生拥有独立可训练参数。保留学生task query是实验选择，不是蒸馏的硬性要求；未来如研究mean学生，应作为另一个容量/汇聚对照。

### 2.1 为什么输入不同仍能蒸馏？

蒸馏的对象是同一条episode、同一anchor的教师输出，而不是教师输入本身：

```text
教师输入：F_t、F_(t+H)、s_t、VTT、H → 产生z_post标签
学生输入：F_t、          s_t、VTT、H → 预测z_prior
```

两者的episode/anchor、状态归一化、VTT、未来跨度、DINO特征层和图像方向必须对应，但输入信息量和内部query结构不必相同。真实未来只出现在训练期教师端，这正是特权信息蒸馏的意义；若学生也输入真实未来，正式部署时就无法执行。

接口规定两者各输出一个768维latent，在fixed LayerNorm后计算逐维蒸馏误差。**相同维度并不自动意味着语义对齐**：冻结教师定义稳定坐标，学生靠蒸馏梯度学习预测这些坐标，不要求其attention图等同于教师，也不直接复制教师权重。学生动作DiT与其token投影同时训练，学习使用预测而非真实教师latent。

当前观测不一定包含教师输出的全部信息，因此不能保证精确拟合。教师过度编码未来随机性/多解细节时，学生可能只学到条件折中；单token瓶颈和归一化不能自动消除这一问题，要同时看蒸馏误差、动作误差和部署成功率。

### 2.2 三种教师与PlanWAM的对应关系

| 方案 | 未来信息怎样进入 | 何时压缩 | 地位/局限 |
|---|---|---|---|
| 原双池化教师 | VTT/state query分别读两帧，摘要送MLP | 在入口处先压成摘要 | 保留作备选；沿用任务条件选择，但可能过早丢失空间信息 |
| 去掉task query、改mean教师 | 两帧平均池化后送MLP | 同样在入口压缩 | 最简单控制；取消任务query不等于保留更多未来信息 |
| **本轮主教师** | 未来patch作为query，读取当前context | 交互后用summary query压成1个latent | 延迟压缩，保持学生接口；仍存在最终单latent瓶颈 |
| PlanWAM原posterior | 真实未来scene registers作query、历史作K/V | 保留多个future latent供规划读取 | 更接近本轮主教师，但不包含我们最终单token汇聚的折中 |

原文reported设置保留256个未来latent（4未来时刻×4相机×16 slots），学生也预测对应的256 slots。我们不将其256-token教师直接对齐到单token学生；本方案教师**输出也只有一个latent**，内部的196个future patch queries仅用于教师计算。这是轻量迁移，不是完整复现。

## 3. 冻结本轮基准配置

以 `metaworld_vla/configs/metaworld_mt50_corner2_easy_g2.yaml` 的归档副本为基础，不使用 `metaworld_mt50_easy_prior_r_g2_scratch50.yaml`。

| 项目 | 本轮固定设置 |
|---|---|
| 数据 | 自采MT50 corner2，当前已有50任务列表 |
| stats | `stats/metaworld_mt50_corner2_480_50ep_4d.json` |
| VTT | `embeddings/metaworld_mt50_corner2_480_50ep_input224_vtt_dinov3b` |
| 编码器 | 冻结DINOv3-B，沿用相同checkpoint |
| 图像 | 原生corner2，已有180度方向协议；渲染480×480，输入224×224 |
| 当前视觉层 | `[-7,-4,-2]`；意图使用与原G2相同的-2层 |
| patch | 196×768；不改变层号与视觉融合 |
| 状态/动作 | 4D/4D，历史长度1 |
| chunk/执行/未来 | C8/E4/H8 |
| task-attention | 单query，宽度256、4 heads |
| 动作模型 | depth12、hidden512、8 heads、适配器64 queries、depth4 |
| 直接VTT/FutureFeat/gate | 全部关闭，VTT间接输入保留 |
| 训练 | batch96、seed0起步、AdamW及梯度裁剪沿用原G2 |
| Flow源/推理 | 标准高斯源、10步积分，不复用上一动作chunk |

后续如使用tiny或-4单层，应建立另一组完全匹配的教师/学生/基线，不能混入本组结果。

## 4. 阶段A：从头训练Posterior教师

### 4.1 输入与结构

只新增一个未来端视觉输入，不输入未来state、未来动作标签、奖励或物体真值。示范动作仅作为原FM训练监督。

```text
当前RGB → 冻结DINO → 当前patch F_t ─────────────────→ 当前视觉适配器 ─┐
                          ↓                                          │
                   投影＋空间/类型编码 ──┐                            │
当前state、VTT → context tokens ────────┴→ 当前context C               │
                                              ↑ cross-attention      │
未来RGB → 冻结DINO → F_f → 投影＋位置/H编码 → future queries U         │
                                              ↓                      │
                              未来残差＋上下文交互＋FFN                │
                                              ↓                      │
                              可学习summary query读取未来tokens       │
                                              ↓                      │
                          z_post [B,768] → 原意图投影 → 单token ──────┤
加噪动作x_tau、时间tau、当前state ────────────────────────────────────┤
                                                                     ↓
                                                               原动作DiT
                                                                     ↓
                                                               FM动作loss
```

第一版候选规格：教师内部宽度256、4 heads、1个future-context block（cross-attention＋FFN，均带残差），一个256维可学习summary query，汇聚后投影为768维。当前和未来各196个patch；context额外加入state/VTT token。先不加入future self-attention、多层金字塔和多个输出token；这些均非PlanWAM原模型的完整设置。具体参数量和开销在实现时统计。

示意：`U0=Proj(F_f)+Pos+H_emb`；`U1=U0+CrossAttn(LN(U0),LN(C),LN(C))`；`U2=U1+FFN(LN(U1))`；`z_post=OutProj(SummaryAttn(q_summary,LN(U2),LN(U2)))`。

必须保留U0未来内容的残差，不将交互输出写成仅对当前context求和后就丢弃未来输入。context仅含当前视觉/状态/VTT，不含加噪动作、真实动作、未来state。summary query可学习但不使用task/state生成，不硬编码关注区域。保留位置/未来跨度信息。

未来patch是教师**输入**，不是重建标签；不对z_post施加原mean-DINO意图回归loss。posterior分支不读取x_tau或真实动作，即使DiT按原FM协议使用x_tau。相比原双池化方案，读取未来的结构改变了，训练监督仍是同一个动作FM loss。

### 4.2 训练信号和梯度

令a为归一化专家动作chunk：

```text
x0 ~ N(0,I)
x_tau = (1-tau)*x0 + tau*a
target_v = a-x0
L_teacher = MSE(DiT(x_tau,tau,current_obs,z_post), target_v)
```

- 冻结DINO的前向放在no_grad中；不等于将可学习posterior放在no_grad中。
- posterior投影、context编码、cross-attention/FFN、summary汇聚、intent_to_token、当前视觉适配器和动作DiT全部训练。
- 动作loss必须通过意图token回传到posterior。
- 直接VTT token仍关闭，避免顺手改变现有任务条件路径。
- 第一版不附加复杂正则。记录跨样本方差/有效秩等退化指标；若确有退化再加入可解释正则作为独立实验。
- 单token没有slot间多样性，不能直接照搬多token的diversity loss；非零范数也不能证明表示未坍缩。

### 4.3 训练预算与checkpoint规则

- 教师从随机动作模型/posterior初始化开始，训练50 epochs；LR2e-4→5e-5的原50轮cosine规划作为首版候选，不声称已调优。
- 保存每5轮。预先规定epoch40教师为阶段B的主教师，epoch50作为敏感性结果；不能看所有学生成功率后倒选教师。
- 若选epoch40，训练路径与原50轮scheduler的epoch40一致；报告实际使用40轮权重，并记录完整run训练到50轮产生的实际开销。
- 不需要中途短微调已有G2。开发时的随机tensor/单步smoke test只验证代码，不用于判断结构收益。

### 4.4 必做诊断（非部署成功率）

对同一离线batch固定x0和tau，比较正确未来、future=current、同任务其他episode的未来。记录每样本动作误差、z_post变化及其有效秩，按低/中/高tau分桶，避免高tau下动作本身泄露大部分标签信息掩盖未来作用。

错误未来不一定让所有样本误差都增大，替换本身也是分布外干预；它只能与匹配的当前输入教师对照一起提供证据。若未来替换几乎无影响，先判断教师是否忽略未来，不能仅凭训练loss更低进入方法结论。

离线教师诊断不等于正式闭环成功率，不将专家未来帧接到任意学生状态上计算可部署成绩。

## 5. 阶段B：冻结教师，从头训练因果学生

### 5.1 采用哪一种初始化？

**本轮主方案：学生task-attention、G2 MLP、intent投影和DiT均从头初始化。** 不加载已有G2，也不从教师复制动作权重；这样与原G2的学生训练预算和初始化协议最清楚。

PlanWAM式教师兼容参数初始化可以以后单独消融，但不是本轮必做，也不和从头学生结果混称。教师产出监督标签不等于学生加载教师权重。

### 5.2 蒸馏目标与动作学习

```text
当前/未来DINO特征 → frozen posterior → z_post → fixed LN → detach
                                                     ↑ MSE
当前特征＋VTT＋state＋H → 原task-attention G2 → z_prior → fixed LN
                                                     ↓
                                 原intent_to_token → 学生DiT → FM动作loss
```

教师返回的是**投影为DiT token之前**的768维latent。第一版固定采用：

```text
L_KD = mean((fixed_LN(z_prior.float())
             - stop_gradient(fixed_LN(z_post.float())))**2)
L_student = L_FM(student_latent) + lambda_KD * L_KD
```

- lambda_KD=0.1作为初始候选，不复用“0.1必然合适”的假设；与原SmoothL1目标不同，记录梯度/量级，数值失衡时单独设计权重消融。
- 原固定mean意图loss关闭；不同时叠加attention-target。
- 原intent_to_token中已有LayerNorm；明确以归一化latent对齐是减少尺度歧义，不是保证信息正确/防坍缩。
- 学生loss始终使用学生z_prior，不以教师token替换输入。
- 教师requires_grad=False、eval、no_grad；学生池化器/预测器/投影/视觉适配器/DiT正常优化。
- 教师只需保留冻结posterior及必要预处理，不需要阶段B驻留/运行完整教师DiT。
- 同一batch的当前/未来冻结DINO结果可供教师/学生共用，但不能混用不同图像变换、层号、state归一化或VTT协议。

### 5.2.1 学生训练的逐batch流程

1. 读取当前/未来RGB、当前state、VTT、H及专家动作；所有信息属于同一episode/anchor。
2. 冻结DINO各提取一次当前和未来特征。
3. frozen posterior接收两帧特征和当前条件，产生z_post；归一化后detach为监督标签。
4. 学生G2只接收当前特征、当前state、VTT、H，产生z_prior。未来特征不作为学生attention的K/V或query，不放入学生DiT。
5. 学生DiT只使用自己的z_prior，与当前观测/加噪动作产生速度场预测，计算动作FM loss。
6. 比较学生和教师归一化latent，计算KD loss；两项相加，只更新学生参数。
7. 评测时省去第2步的未来编码、第3步和KD loss，当前G2调用方式保持因果。

教师并不是在训练时给学生附送未来输入；它像一个使用更多证据生成标签的标注器。学生学习的是从可获得的当前输入到该标签的映射。不是匹配两者query、attention权重或中间patch序列，也不是多步动作去噪轨迹蒸馏。

### 5.3 预算和部署

- 学生独立训练50 epochs，沿用原G2的LR和50轮cosine；以epoch40为主评测，其他checkpoint提前规定用途。
- 学生依然以task-attention单query读取当前图像，预测一个latent，不读取未来。
- 部署只加载学生。未来图像加载、posterior和KD loss全部移除；保留VTT经G2的间接任务条件。
- 推理结构/参数规模应与匹配G2相同；测量每chunk延迟、峰值显存、成功率，不预先承诺零延迟差异。

## 6. 从头训练的对照矩阵和优先级

| ID | 学生/模型 | 监督 | 目的 |
|---|---|---|---|
| G | 现有效的task-attention G2 | 固定mean-DINO变化＋动作 | 主要参照，结构与数据保持一致 |
| T | Posterior＋动作DiT | 动作loss；可见未来 | 训练教师及特权信息诊断，不作为部署主成绩 |
| S | 相同因果G2学生 | 冻结T的归一化latent＋动作 | 本轮核心方法 |
| N | 相同因果G2学生 | 仅动作，意图监督权重0 | 区分任务条件通道/容量与未来监督的价值 |
| B | 标准G2延长从头训练 | 固定mean目标 | 总训练预算敏感性，不是旧模型短微调 |

执行顺序：G配置冻结和已完成结果整理 → 实现/单步测试 → T完整从头训练与诊断 → S完整从头训练 → N从头训练 → 训练seed和预算控制扩展。

N非常重要，但可与T/S并行排期，不要求先通过旧模型短微调。零监督模式需要代码显式支持（不能只忽略训练入口对lambda_intent的校验而不检查实际梯度路径）。

教师训练是额外成本；S50轮不能简单称为与G50轮等训练预算。报告T+S总GPU小时，并补一个从头100轮或按GPU小时匹配的G2对照。G100轮若改用100轮cosine，属于另一种scheduler，需要明确记录；总GPU小时仍未必等于T50+S50，应实测。另可在冻结基线50轮scheduler结束后预先定义后50轮LR，作为独立预算控制，不从旧checkpoint临时续训挑结果。

## 7. 评测与成功条件

- 沿用相同50任务、每任务50 episodes、horizon500、seed0/1/2起步的现有评测协议，不改goal/reset/RNG协议。
- 评测seed不等于训练seed：确定候选后追加独立训练seed，报告均值和波动，不只报告最好run。
- 已有直接VTT关闭时，移除意图token也移除了唯一显式任务通道，不能单靠这一消融证明未来语义。
- 报告每任务成功率/失败类型、总成功率、低数据/视觉扰动表现（若另有协议），KD误差和latent范数只作诊断。
- S优于G和N，且教师确实利用未来、不同训练seed方向一致，才支持“动作塑形的未来监督有效”。
- 教师有效但S无收益：优先检查不可预测未来信息、latent对齐尺度、学生学习不足；不立即加入region gate或OPD。
- 教师不使用未来：先做同容量当前输入教师对照，不能把一般额外训练收益归因于未来表示。

## 8. 资源控制

- 每样本仍只用当前和一个未来图像，冻结DINO不增加第三次视觉编码。
- 教师训练增加future-context cross-attention和summary汇聚；学生阶段多小型冻结posterior前向，无教师动作采样。教师内部token更多带来额外训练计算，但部署仍只有学生单意图token。
- 两阶段累计训练量增加，不能承诺比当前G2更省总训练时间；推理只保留学生。
- 可缓存教师latent，但第一版先在线计算避免坐标/augmentation/version错配；若缓存，必须绑定teacher SHA、图像方向、层、H、VTT、state归一化、episode与anchor索引。

## 9. 已实现接口与验证（2026-11-04）

1. `models/planwam_g2.py`：future-driven posterior、归一化latent蒸馏和独立标准高斯FM loss，不修改旧 `VLAModel` 或旧loss。
2. `metaworld_vla/train_planwam.py`：`planwam.stage=teacher/student/action_only`，教师/学生均从头初始化。学生只加载posterior标签，不加载教师动作权重。
3. `metaworld_vla/diagnose_planwam.py`：离线未来替换诊断，固定样本/噪声/tau，记录动作MSE与latent变化，不是闭环成功率。
4. 新配置保留原easy-G2结构，只关闭原lambda_intent并增加独立planwam协议。旧 `train.py` 遇到新协议会明确拒绝并提示正确入口；旧配置不受此检查影响。
5. 教师checkpoint包含posterior权重；为保持现有action参数键布局，保留但冻结未使用的原因果G2池化器/预测器。教师中实际训练的是新posterior、intent投影、适配器和动作DiT。
6. 学生 `model_state` 不增加posterior参数，可以由现有 `eval.py` strict加载；教师禁止作为因果G2评测，避免误用未经训练的因果分支。
7. 教师/学生协议校验包括数据和模型配置、stats/VTT hash、posterior规格、checkpoint版本；学生续训还绑定教师文件SHA。未来换配置/统计量需新建实验，不应绕过校验。
8. 支持同阶段完整epoch checkpoint的 `--resume`，恢复优化器/scheduler和主进程、DataLoader RNG。image_aug固定false；不承诺跨硬件/库版本逐位一致。单步smoke的partial checkpoint不能用于正式续训或正式学生训练，只有另一次显式smoke可读取它。

### 9.1 已运行验证

- `utils/test_planwam_g2.py`：6项CPU测试通过，包括教师未来敏感性、动作loss梯度路由、学生动作分支因果性、冻结教师无梯度、action_only无未来输入、CPU bf16 autocast、checkpoint重载和配置对照。梯度路由测试主动激活了原DiT的zero-init门控，不能将第一步暂时没有posterior梯度误认为分支断开。
- 原 `test_g2_task_pool.py`、`test_g2_nogate.py`、`test_vtt_injection_switch.py`、`test_g2_attention_target.py` 回归测试通过。
- 使用真实2500个episode数据索引、真实DINO-B、完整depth12/h512，batch1/CPU：教师和学生各完成一个优化步骤及保存。仅为管线验证，结果不用于判断研究收益。
- 真实学生smoke checkpoint由现有Policy strict加载，不读取未来生成8×4动作chunk；离线诊断脚本真实数据smoke通过。
- 当前环境无法访问NVIDIA驱动，未验证GPU显存/速度、完整训练、仿真成功率和完整epoch断点恢复。完整epoch恢复有接口和协议保护，但尚未做真实长训练恢复实验。

### 9.2 完整教师训练

```bash
cd /home/lab347-no10/Tianyi/LiLa-WAM
conda activate MetaWorldVLA
mkdir -p log/metaworld_planwam_teacher
set -o pipefail
CUDA_VISIBLE_DEVICES=0 python metaworld_vla/train_planwam.py \
  --config metaworld_vla/configs/metaworld_mt50_easy_planwam_teacher.yaml \
  --stats stats/metaworld_mt50_corner2_480_50ep_4d.json \
  --save-dir checkpoints/metaworld_mt50_easy_planwam_teacher_seed0 \
  2>&1 | tee log/metaworld_planwam_teacher/train.log
```

50轮scheduler，保存每5轮；主教师取epoch40。新训练入口不接受init-from，且拒绝覆盖已有run，必须使用新目录。日志teacher的KD=0正常；LatentStd是batch内归一化latent的跨样本标准差，batch1为0属于定义结果，不代表坍缩；batch96也不能仅凭该指标证明语义正确。

### 9.3 教师离线未来利用诊断

```bash
CUDA_VISIBLE_DEVICES=0 python metaworld_vla/diagnose_planwam.py \
  --config metaworld_vla/configs/metaworld_mt50_easy_planwam_teacher.yaml \
  --stats stats/metaworld_mt50_corner2_480_50ep_4d.json \
  --checkpoint checkpoints/metaworld_mt50_easy_planwam_teacher_seed0/checkpoint_epoch_40.pt \
  --batch-size 8 --batches 32 --seed 0 \
  --output eval_result/planwam_teacher_ep40_intervention_seed0.json
```

在相同anchor与高斯噪声下比较tau=0.1/0.5/0.9；错误未来来自同任务不同episode随机anchor。JSON保存采样身份、checkpoint SHA、每batch数据。该诊断默认从训练数据采样，不是独立验证集，不是部署评分；配对差异和外推结论需审慎解释。

### 9.4 从头训练蒸馏学生

```bash
mkdir -p log/metaworld_planwam_student
CUDA_VISIBLE_DEVICES=0 python metaworld_vla/train_planwam.py \
  --config metaworld_vla/configs/metaworld_mt50_easy_planwam_student.yaml \
  --stats stats/metaworld_mt50_corner2_480_50ep_4d.json \
  --teacher-checkpoint checkpoints/metaworld_mt50_easy_planwam_teacher_seed0/checkpoint_epoch_40.pt \
  --save-dir checkpoints/metaworld_mt50_easy_planwam_student_seed0 \
  2>&1 | tee log/metaworld_planwam_student/train.log
```

这里的teacher-checkpoint仅提供冻结标签，不是初始化学生。KD权重在 `planwam.lambda_kd: 0.1`；`training.lambda_intent: 0.0` 表示不再计算原mean-DINO监督。共享ModelFactory日志可能仍显示target_pooling=mean，这是兼容结构字段，独立入口不会调用原mean意图loss。归一化教师TargetNorm约为sqrt(768)，不能与原差值目标TargetNorm直接比较。

### 9.5 正式学生评测

```bash
for SEED in 0 1 2; do
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python metaworld_vla/eval.py \
    --config metaworld_vla/configs/metaworld_mt50_easy_planwam_student.yaml \
    --stats stats/metaworld_mt50_corner2_480_50ep_4d.json \
    --checkpoint checkpoints/metaworld_mt50_easy_planwam_student_seed0/checkpoint_epoch_40.pt \
    --test-num 50 --horizon 500 --seed "$SEED" \
    --output "eval_result/metaworld_planwam_student_ep40_seed${SEED}_n50.json"
done
```

只评测学生，不传action-reuse参数。学生stage不在评测时加载教师，即使训练配置里记录有teacher路径也无需在线读取未来。

### 9.6 无未来监督的同结构对照

```bash
mkdir -p log/metaworld_planwam_action_only
CUDA_VISIBLE_DEVICES=0 python metaworld_vla/train_planwam.py \
  --config metaworld_vla/configs/metaworld_mt50_easy_planwam_action_only.yaml \
  --stats stats/metaworld_mt50_corner2_480_50ep_4d.json \
  --save-dir checkpoints/metaworld_mt50_easy_planwam_action_only_seed0 \
  2>&1 | tee log/metaworld_planwam_action_only/train.log
```

它保留G2 token和task-attention，仅通过动作loss训练，不读取未来、不计算任何意图目标；因此速度更快属于正常现象。评测时将9.5的配置、checkpoint和输出目录替换为对应action_only版本即可。

### 9.7 同阶段断点恢复

使用相同配置和同一save-dir，加入 `--resume <该阶段checkpoint>`；学生还必须提供原teacher-checkpoint。不要把教师checkpoint传入resume来切换阶段。

最初开发单步产物位于 `/tmp/planwam_smoke_teacher_20261104` 和 `/tmp/planwam_smoke_student_20261104`，不能用作正式教师。2026-11-06检查时，上述正式教师/学生目录已有完整训练checkpoint；不要重新执行非resume命令覆盖这些目录。

### 9.8 学生 S/S 与 S/T 离线配对诊断（2026-11-06 新增）

入口：`metaworld_vla/diagnose_planwam_student.py`。这是独立只读诊断，不修改训练、仿真评测、已有配置和checkpoint。

| 分支 | 动作网络与意图投影 | 输入的意图 latent |
| --- | --- | --- |
| S/S | 当前学生 DiT、适配器、`intent_to_token` | 学生只用当前图像、状态、VTT 预测 |
| S/T | **完全相同的学生网络** | 冻结教师用离线数据的当前/真实未来帧产生 |

替换的是768维原始latent，不是教师投影后的512维token，不加载教师DiT。两分支共享同一anchor、当前视觉特征、状态、VTT、高斯噪声和tau。教师latent和学生latent都交给**学生自己的**归一化/投影层。这里的教师未来来自专家示范的 `I[min(t+8,T-1)]`，不是仿真在线获取未来，不是学生生成动作导致的真实未来。

脚本检查学生stage、完整epoch、配置/stats/VTT协议，并校验教师SHA必须与该学生训练时使用的教师一致；不能随意将教师40换为50。正式命令不应添加开发专用的 `--allow-smoke-checkpoint`。

#### A. 推荐命令：同时测去噪和完整动作块

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
  --samples-per-task 8 \
  --batch-size 8 \
  --seed 0 \
  --generate-chunks \
  --num-inference-steps 10 \
  --output eval_result/planwam_offline/student40_ss_st_seed0_n8_chunk10.json
```

默认覆盖配置内50任务，各抽8个不同anchor，共400个anchor。不是每个任务8个完整episode。同一任务内按帧均匀抽样，长episode可能占更高比例；任务之间样本数相同。每个anchor只抽一份高斯噪声，两分支共用。脚本包括最后不足一个batch的样本，不丢尾样本。

如果只想快速测去噪误差，删掉 `--generate-chunks`（`--num-inference-steps`此时不参与计算），并换一个新输出文件名。默认tau为0.1、0.5、0.9，每批6次学生DiT前向；完整10步动作块再增加20次DiT前向。当前/未来DINO各编码一次，latent在Euler过程不重新预测。无action-reuse，无动作先验，无环境仿真。

仅检查少量任务时可加 `--tasks coffee-pull-v3 lever-pull-v3 door-lock-v3`；它只限制诊断抽样，不修改模型协议。扩大样本量可改 `--samples-per-task 32`。保持样本数、batch、seed、设备和精度一致，便于重复比较。脚本的seed是**离线抽样/噪声seed**，不是环境测试seed，也不会覆盖训练seed。

#### B. 如何读结果

JSON保留 `overall`、`per_task`、`samples` 和 `strata`。终端打印整体配对指标，逐批打印进度。已有输出文件会拒绝覆盖。

重点看 `overall.paired` 下：

| 指标 | 含义 |
| --- | --- |
| `velocity_mse_tau_0.1/0.5/0.9` | 固定专家插值 `x_tau=(1-tau)x0+tau*a`，速度预测与 `a-x0` 的归一化MSE |
| `velocity_mse_tau_mean` | 三个tau误差的算术平均，不是训练logit-normal分布加权loss |
| `chunk_mse_normalized` | 纯高斯出发，经10次Euler完整生成8步动作，对专家动作块的归一化MSE |
| `prefix_mse_normalized` | 同上，但只看在线实际执行的前4步 |
| `recorded_steps_mse_normalized` | 仅看真实记录动作，不计episode末尾人工padding，供诊断用 |
| `step_1...step_8_mse_normalized` | 各时间步动作MSE，帮助检查动作块末尾 |
| `command_raw/clipped_xyz/gripper_mse` | 反归一化控制指令的xyz/夹爪MSE；clipped按在线规则限制到[-1,1]，不是末端位姿误差 |

每项含 `ss_mean`、`st_mean`、`delta_st_minus_ss`、`relative_mean_reduction` 和 `st_better_fraction`。

- `delta_st_minus_ss < 0`：该误差上教师latent更好。
- `relative_mean_reduction > 0`：教师latent降低误差；例如0.1表示误差均值下降10%，**不是成功率提升10个百分点**。
- `st_better_fraction`：在所抽anchor中S/T误差更小的比例，不是任务成功率。
- `latent/normalized_mse` 对应未乘0.1权重的KD误差；`normalized_cosine`看方向；两项raw norm仅描述原始latent尺度。`student_projected_token_mse_vs_teacher`检查共享学生投影后的差距，不能单独据此判断控制好坏。

总动作MSE保留训练时的末尾padding目标，不改padding行为。JSON同时记录有效动作步数、future index、实际未来跨度和是否截断；`strata.full_horizon`与`clipped_future`分开汇总，避免末尾重复未来帧掩盖问题。

#### C. 判断后续实验，而不是把离线MSE当闭环结论

1. **S/T在去噪和完整生成均更好**：支持继续检查学生对教师latent的可预测性/蒸馏差距，先看各任务差异，再考虑新的KD目标；不等于增大KD权重必然更好。
2. **去噪更好，但完整生成相近或更差**：专家插值路径上的改善没有转化到生成轨迹；关注低tau、积分误差和学生自己生成的中间状态，不能只凭训练loss选择模型。
3. **S/T与S/S接近**：可能是可利用信息已经接近，也可能是动作读出对差异不敏感。需结合同结构action_only和闭环结果。
4. **S/T更差**：可能是学生DiT适配了学生latent分布，换入教师latent造成分布错配；不能据此单独认定教师无用。

这些anchor来自训练数据，指标只是训练分布的配对诊断，不是泛化验证，更不是教师在线成功率或严格性能上界。离线单条专家动作只是多种合理动作之一；最终仍以固定协议的闭环成功率为准。相邻anchor可能相关，不能把这些样本比例直接作为独立试验的统计显著性证据。

CPU单元测试：`python utils/test_planwam_student_diagnostic.py`。已验证配对噪声、Euler积分、原FM loss对齐、无参数更新、任务均衡抽样、padding/前缀指标和统计方向；另用真实数据、DINO-B、完整学生结构与开发smoke checkpoint完成CPU端到端测试。正式checkpoint亦做少量anchor的CPU管线检查；这些小样本不是正式效果结论。未运行完整400anchor GPU实验。

### 9.9 第一优先级：完整2×2交叉诊断（2026-11-07已实现）

沿用9.8入口，新增可选 `--include-teacher-dit`，未指定时仍只测S/S、S/T，不加载教师动作网络。这里的“DiT”实际上包含该checkpoint自己的视觉适配器、意图投影、类型嵌入及动作头，不能只替换几个DiT block。两网络共享同一冻结DINO特征和统计量，完整teacher动作权重由同一teacher checkpoint严格加载。

| 分支 | 动作网络 | latent来源 |
| --- | --- | --- |
| ss | 学生 | 学生因果预测器 |
| st | 学生 | 教师特权posterior |
| ts | 教师 | **学生**因果预测器，不使用教师保留但未训练的旧G2预测器 |
| tt | 教师 | 教师特权posterior |

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

保持与9.8相同的任务列表、抽样数、seed、batch、设备/精度；本次四组在**同一次运行中**共用噪声和输入。不需要另行提供teacher配置，因为学生训练绑定的teacher输入协议与网络结构已被验证。新增一个完整冻结动作网络，显存多一份动作权重及瞬时激活；DINO不复制。DiT调用从两组变四组，计算量会上升，但没有训练反向。

JSON `overall.metrics`保存四组原始均值；`overall.paired`仍为原S/T对S/S；新增`overall.comparisons`（per_task和strata同样提供）：

- `st_vs_ss`：保持学生网络，替换教师latent。
- `ts_vs_tt`：保持教师网络，将教师latent替换为学生latent。
- `tt_vs_ss`：教师完整组合与学生完整组合的离线误差比较，不是在线性能上界。
- `ts_vs_ss`：保持学生latent，改用教师动作网络；包括适配器和投影的差异，不能仅归因于某一个层。

每项含`candidate_mean`、`reference_mean`、`delta_candidate_minus_reference`，负数表示candidate误差更小。先看`chunk_mse_normalized`与`prefix_mse_normalized`，再看三个tau和夹爪/xyz误差，不只看一个总均值。

若T/T明显好而交叉组合退化，支持表示与读出配套适配的解释；若T/T没有优势，则本次离线测试尚未证明教师更值得蒸馏。单种组合/小样本不能唯一确定机制。原教师替换未来诊断只证明其对未来输入敏感，不证明其优于学生。

验证：8项诊断CPU单元测试通过；真实epoch40权重、DINO-B、3任务6anchor、10步Euler完成四组，S/S、S/T与扩展前相同CPU样本的逐项误差完全一致。只做管线检查，未代跑400样本GPU测试。

### 9.10 第二优先级：action-only同结构对照（已具备实现）

配置和命令见9.6。它保留学生的task/state attention、768维预测器及512维意图token、视觉适配器和DiT；只去除教师及KD监督。`planwam.stage=action_only`、`planwam.lambda_kd=0`、`training.lambda_intent=0`、`lambda_future_feat=0`，并非只关闭推理token。

目标：判断蒸馏是否帮助**同一个结构**，而不是将“多一个task条件通路”的收益误当作未来监督收益。action-only的latent没有未来标签，只能称为动作监督下的任务条件瓶颈，不能据此宣称已学到未来预测。

保持50轮scheduler训练预算、seed0、batch96和LR2e-4→5e-5，主比较取epoch40；不加载教师、不给resume/init-from，不把epochs改为40（那会改变前40轮学习率曲线）。两版本训练入口/数据遍历等尽量一致，但不保证不同计算路径逐位复现。

- action-only高于蒸馏学生：当前KD在该训练实例可能有负迁移；不能只凭KD loss下降认定有效。
- 蒸馏学生高于action-only但低于原G2：教师监督可能有益，但未超过固定mean变化监督方案。
- 两者接近：本实验没有明确证明KD额外收益，需要重复训练seed或更细的任务分析。

主闭环评测仍固定env seed1000、policy seed456、workers8、每任务50次、horizon500、NFE10、C8/E4、无reuse。无需为了本实验修改数据集或padding。

### 9.11 第三优先级：动作响应/速度场蒸馏（待实现，非OPD）

该方案不是目前的`lambda_kd`直接改配置就能启用，需要新增独立loss/配置分支，尚未修改训练代码。

当前：`L = L_FM + 0.1 * MSE(LN(z_student), LN(z_teacher))`，对768维归一化表示均匀对齐。候选方案是在同一样本、同一个高斯x0与tau下，用专家动作构造`x_tau=(1-tau)*x0+tau*a`，冻结教师生成`v_teacher=TeacherAction(x_tau,tau,current,z_teacher)`；学生从当前输入产生z_student，计算`v_student=StudentAction(x_tau,tau,current,z_student)`。

候选目标：

```text
L_FM     = MSE(v_student, a - x0)
L_vKD    = MSE(v_student, stop_gradient(v_teacher))
L_total  = L_FM + beta * L_vKD + alpha * L_latent
```

第一版可先比较`alpha=0`的纯响应蒸馏与action-only，beta需另行明确预设；再考虑弱latent约束，不能将新参数误当作现有配置已支持。所有学生仍从头训练，保留专家FM主目标，教师所有参数冻结，教师前向用no_grad；学生未来不可见。

它对齐的是同一物理含义的8×4速度场输出，弱化任意latent坐标系与读出适配造成的问题；不保证解决未来不可预测、多模态和教师动作错误。若teacher恰好等于专家速度目标，额外项主要相当于增强同一监督；若teacher有偏差，则是引入有偏软目标，不能预设必然有益。

当前版本学生训练已有当前/未来DINO及posterior计算；新方案可复用这些特征，再增加**每batch一次冻结教师动作前向**，不复制DINO，不做教师10步积分，不采环境轨迹，不对教师反向。相比latent-only蒸馏会增加冻结动作权重和前向开销，但不是self-OPD那种沿学生去噪轨迹反复查询教师。本版x_tau来自专家插值，因此不能称为on-policy蒸馏。

仅当2×2显示值得利用的教师动作能力，并完成action-only对照后再投入这个训练分支；若教师没有优势，先重新审视教师目标而不是继续堆叠蒸馏loss。

## 10. 参考与创新边界

- [PlanWAM论文](https://arxiv.org/html/2610.11382v1)：借鉴动作/规划塑形的特权未来表示与因果蒸馏，不复制其全部历史金字塔、多视角、多latent和轨迹评分结构。
- [现有G2实验文档](G2_TASK_ATTENTION_EXPERIMENT.md)：task-attention、gate、直接VTT和attention-target的既有实现/消融。

这是向机器人操作的轻量迁移方案，不应把“posterior教师→prior学生”本身宣称为首次提出。当前G2正向结果和attention-target略降来自用户报告，本文未重新审计数值或训练seed覆盖情况。

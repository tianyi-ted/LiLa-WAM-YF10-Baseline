# LiLa-WAM 全部模型版本与测试结果汇总

更新时间：2026-11-10。

本文汇总当前仓库中已经找到正式评测 JSON 的 RoboTwin YF10 与 Meta-World
MT50 模型。数值优先取 `eval_result` 下原始 JSON 的成功次数，而不是训练 loss、
聊天记录或旧文档中的手工转录值。

需要先明确：两个平台、不同环境 seed、不同 action chunk、不同推理步数和不同
训练预算之间不能按绝对成功率直接排序。本文把严格同协议结果和历史筛选结果分开。

## 1. 一页结论

### RoboTwin

- 历史 DINOv3-Large＋h512 的 `small_ep40` 为当前记录中的最高总体结果：
  **75.2%**；它计算量明显高于后续主用的 DINOv3-Base＋h512。
- 资源受限主基线 DINOv3-Base＋h512 epoch40 为 **67.4%**。
- 从该基线继续微调4轮的 C32 为 **69.1%**，因此后续4轮结构消融必须与
  C32比较，不能只与原epoch40比较。
- 4轮空间残差中，S-residual v2 为 **70.5%**，是该组最高；但从头训练版本
  只有 **65.5%**，说明短微调收益没有转化为同预算scratch收益。
- G2 scratch40 为 **67.8%**：Clean提高到75.4%，Randomized降到60.2%，没有
  形成稳定总体收益。
- 当前所有RoboTwin表格均为单训练run、评测seed0。它们适合开发筛选，尚不是
  多seed统计结论。

### Meta-World

- 固定C8/E4/H8、epoch40、env seed1000、policy seed456、2500 episodes的
  cold严格组中：baseline为 **87.36%**，原G2为 **87.80%**，随机初始化
  action-only为 **87.68%**，旧PlanWAM latent蒸馏学生为 **85.80%**。
- 动作先验完整匹配训练中，R（旧动作尾部＋噪声）为 **88.84%**，比对应C
  的87.36%高1.48个百分点；但R+G2只有87.64%，没有出现简单叠加收益。
- 同一个原G2 checkpoint在开启推理尾部复用后达到 **89.48%**，cold为
  87.80%；这是单训练run和一个seed区间的推理消融，仍需多seed验证。
- PlanWAM教师的离线2×2诊断显示教师网络配教师latent的动作块MSE最低，但旧
  latent-only蒸馏没有转化为学生闭环提升。
- 兼容动作初始化＋uniform LR的action-only达到 **88.84%**；对应latent KD
  学生的现有结果为84.68%，说明当前蒸馏设置仍有明显负差，但两个结果是否为
  最终完全配对协议应继续以保存配置核验。

## 2. RoboTwin YF10 固定评测协议

共同任务为十任务YF10；每种场景每任务50次，共500次，评测seed0，
`instruction_type=unseen`。训练集为clean＋randomized混合数据，因此这些结果
不是官方严格C2C/C2R。当前本地RoboTwin步数限制也曾修改，不能直接当作官方
排行榜复现。

后续主模型的共同结构通常为单路head camera、DINOv3-B、Action Expert
depth12/h512、动作chunk32、执行前16步、10次Flow积分。较早的`screen`和
`small`系列使用DINOv3-Large，详见第4节。

## 3. RoboTwin 全部已找到的成对结果

| 结果ID | Clean | Randomized | 两场景合计 |
|---|---:|---:|---:|
| `dinov3b_h512_ep30` | 344/500 (68.8%) | 304/500 (60.8%) | 648/1000 (64.8%) |
| **`dinov3b_h512_ep40`** | 351/500 (70.2%) | 323/500 (64.6%) | **674/1000 (67.4%)** |
| `g2_scratch_ep40` | 377/500 (75.4%) | 301/500 (60.2%) | 678/1000 (67.8%) |
| **`phase2_c32_ep4`** | 376/500 (75.2%) | 315/500 (63.0%) | **691/1000 (69.1%)** |
| `phase2_f16_ep4` | 370/500 (74.0%) | 317/500 (63.4%) | 687/1000 (68.7%) |
| `phase3_g2_ep4` | 366/500 (73.2%) | 319/500 (63.8%) | 685/1000 (68.5%) |
| `phase3_g2_gate01_ep4` | 363/500 (72.6%) | 316/500 (63.2%) | 679/1000 (67.9%) |
| `phase3_gaux_ep4` | 364/500 (72.8%) | 314/500 (62.8%) | 678/1000 (67.8%) |
| `screen_ep30` | 364/500 (72.8%) | 328/500 (65.6%) | 692/1000 (69.2%) |
| `screen_ep40` | 391/500 (78.2%) | 335/500 (67.0%) | 726/1000 (72.6%) |
| `screen_ep50` | 389/500 (77.8%) | 343/500 (68.6%) | 732/1000 (73.2%) |
| `small_ep30` | 361/500 (72.2%) | 329/500 (65.8%) | 690/1000 (69.0%) |
| **`small_ep40`** | 406/500 (81.2%) | 346/500 (69.2%) | **752/1000 (75.2%)** |
| `spatial_s2_ep4` | 370/500 (74.0%) | 317/500 (63.4%) | 687/1000 (68.7%) |
| `spatial_saux_ep4` | 373/500 (74.6%) | 318/500 (63.6%) | 691/1000 (69.1%) |
| `spatial_sres_ep4` | 379/500 (75.8%) | 324/500 (64.8%) | 703/1000 (70.3%) |
| `spatial_sres_scratch_ep40` | 362/500 (72.4%) | 305/500 (61.0%) | 667/1000 (66.7%) |
| **`spatial_sres_v2_ft4_fp32gate`** | 386/500 (77.2%) | 319/500 (63.8%) | **705/1000 (70.5%)** |
| `spatial_sres_v2_scratch_ep40` | 354/500 (70.8%) | 301/500 (60.2%) | 655/1000 (65.5%) |
| `v2_gate0_ep40_residual_off` | 368/500 (73.6%) | 302/500 (60.4%) | 670/1000 (67.0%) |
| `v2_gate0_scratch_ep40` | 365/500 (73.0%) | 306/500 (61.2%) | 671/1000 (67.1%) |
| `v2_gate0_scratch_ep40_residual_off` | 367/500 (73.4%) | 306/500 (61.2%) | 673/1000 (67.3%) |

原始数据入口：[RoboTwin评测目录](../eval_result/yf10/)。

### 3.1 RoboTwin版本简要说明

| 系列 | 简要结构/目的 | 当前结论 |
|---|---|---|
| `screen` | 历史DINOv3-Large＋h768动作专家，比较epoch30/40/50 | epoch40的Clean最好，epoch50总体略高；计算最重，不是后续低成本主线 |
| `small` | DINOv3-Large＋h512动作专家 | `small_ep40`为RoboTwin记录最高，说明h512本身不是主要性能瓶颈 |
| `dinov3b_h512` | DINOv3-Base＋h512资源基线 | epoch40总体67.4%，换取约一半训练时间，是后续主要控制组 |
| C32/F16 | 从同一Base epoch40继续训练4轮；未来监督偏移32/16 | C32仅高0.4个百分点，F16没有稳定优势，后续固定H=32 |
| G-aux | 预测全局未来意图但不注入DiT | 67.8%，低于同预算C32 |
| G2 | 预测全局未来意图并作为token注入DiT | 4轮68.5%，scratch40为67.8%；Randomized没有改善 |
| G2 gate01 | G2可训练gate初值0.1 | 67.9%，未优于默认G2或C32 |
| S-aux | 2×4固定区域空间意图，只做辅助监督 | 69.1%，与C32持平，不能证明空间目标改善控制 |
| S2 | 8个空间意图token追加到DiT | 68.7%，未优于C32 |
| S-residual v1 | fusion前用预测空间变化修改视觉patch | 4轮70.3%，scratch40退化到66.7% |
| S-residual v2 | fusion后注入空间残差；FP32 gate修复 | 4轮70.5%，但scratch40只有65.5%，说明联合从头学习不稳定 |
| v2 gate0/on-off | gate0训练及同checkpoint residual-off推理消融 | on/off总体只差约0.2个百分点，当前残差没有稳定净贡献 |

注意：`spatial_sres_v2_scratch_ep40` 的原始JSON为655/1000，即65.5%；旧版
`NEXT_SPATIAL_EXPERIMENT_PROTOCOL.md`曾写65.0%，本文以JSON为准。

## 4. RoboTwin当前推荐引用方式

论文开发表至少保留三条：

1. DINOv3-Large＋h512 epoch40：75.2%，代表较强历史上限；
2. DINOv3-Base＋h512 epoch40：67.4%，代表资源主基线；
3. C32：69.1%，代表与4轮结构筛选匹配的continued-SFT控制。

当前空间和G2实验都没有在scratch同预算下稳定超过Base。因而不能只引用最佳的
4轮微调数值宣称新结构成立；应继续保留scratch和多seed结果。

## 5. Meta-World数据与共同设置

当前主要数据为自采MT50 corner2：2500 episodes、190451 anchors、50任务。
视觉为冻结DINOv3-B，输入224×224；状态/动作均4维。后期严格组统一采用：

```text
epoch40
C8 / E4 / H8
horizon=500
每任务50 episodes，共2500
10次Flow积分
env seed=1000（每任务1000--1049）
policy seed=456
stop-on-success
```

Meta-World和RoboTwin的成功率不可直接比较。Meta-World的旧seed0结果与后期
env1000/policy456结果也应分表解读。

## 6. Meta-World历史架构与消融结果（seed0）

以下均为epoch40、50任务×50次，但部分早期JSON没有固定独立policy seed，主要
用于结构开发历史，不作为后期最严格主表。

| 模型/结果ID | 结构简述 | 成功率 |
|---|---|---:|
| `metaworld_mt50_base` | C32/E4/H32、depth12、future feature、直接VTT | 2164/2500 = 86.56% |
| `metaworld_mt50_short` | C8/E4/H8、depth12、future feature、直接VTT | **2236/2500 = 89.44%** |
| `metaworld_mt50_tiny` | C8/E4/H8、depth8、future feature、直接VTT | 2134/2500 = 85.36% |
| `metaworld_mt50_easy_noff` | depth12，关闭dense future feature，保留直接VTT | 2202/2500 = 88.08% |
| `metaworld_mt50_easy_noffvtt` | 再关闭直接VTT token | 2166/2500 = 86.64% |
| `metaworld_mt50_g2` | mean-G2，保留原future/VTT路径，gate init0 | 2166/2500 = 86.64% |
| `metaworld_mt50_g2_taskattn` | task/state attention替代mean输入池化 | 2197/2500 = 87.88% |
| `metaworld_mt50_g2_taskattn_nogate` | task-attention，取消标量gate衰减 | 2199/2500 = 87.96% |
| `metaworld_mt50_easy_g2` | easy协议：无dense future、无直接VTT，G2为唯一任务条件注入 | 2205/2500 = 88.20% |
| `metaworld_mt50_easy_g2_attntarget` | G2目标改为当前attention加权的未来变化 | 2183/2500 = 87.32% |
| `metaworld_mt50_tiny_easy` | depth8 easy、无G2 | 2157/2500 = 86.28% |
| `metaworld_mt50_tiny_g2` | depth8 easy＋G2 | 2103/2500 = 84.12% |

同一easy checkpoint的重复评测还得到：seed0新入口88.00%，env seed1000为
88.40%，seed10000为86.28%；easy_noffvtt对应为86.60%、86.80%、86.52%。
这些差异说明单个环境seed区间可产生约1--2个百分点波动。

`easy_g2_noinject=12.56%`是把唯一任务条件注入关闭后的推理消融，不是正常可部署
模型，也不是重新训练的无G2基线；它证明该checkpoint依赖G2任务通路，不能用于
普通排行榜。

原始数据入口：[Meta-World历史结果](../eval_result/)。

## 7. Meta-World动作尾部复用：5任务筛选

这组只测试5个任务、每任务20次，共100次，env seed0、policy seed123。B表示
cold生成，W表示复用上一chunk未执行尾部，NFE为Flow积分次数。

| 设置 | 成功率 | 说明 |
|---|---:|---|
| B10 | 87/100 = 87% | 10 NFE cold基线 |
| W10-S1 | 86/100 = 86% | 尾部＋sigma1.0噪声 |
| W10-S0.5 | 87/100 = 87% | 尾部＋sigma0.5噪声 |
| B5 | 88/100 = 88% | 5 NFE cold |
| W5-S0.5 | 85/100 = 85% | 5 NFE尾部复用 |
| strength0，sigma0.5 | 84/100 = 84% | 只有噪声尺度变化，不含旧尾部 |
| strength0.25，sigma0.5 | 83/100 = 83% | 弱尾部先验 |

筛选结论：纯推理复用没有在该5任务小样本筛选中提升成功率；B5也没有低于B10，
但100次筛选不足以证明5 NFE全面等价。

## 8. Meta-World完整action-prior训练与推理匹配

下表使用后期严格协议：env seed1000、policy seed456、2500 episodes。C是cold
source，N是仅改变噪声尺度/零旧尾部控制，R是旧动作尾部先验；tail表示推理也
开启尾部回传。

| 训练模型与推理 | 成功率 | 相对匹配C |
|---|---:|---:|
| C训练＋cold推理 | 2184/2500 = 87.36% | 0.00 pp |
| C训练＋tail推理 | 2200/2500 = 88.00% | +0.64 pp |
| N训练＋matched推理 | 2157/2500 = 86.28% | -1.08 pp |
| R训练＋cold推理 | 2188/2500 = 87.52% | +0.16 pp |
| **R训练＋tail推理** | **2221/2500 = 88.84%** | **+1.48 pp** |
| R+G2训练＋cold推理 | 2146/2500 = 85.84% | -1.52 pp |
| R+G2训练＋tail推理 | 2191/2500 = 87.64% | +0.28 pp |
| 原G2＋cold推理 | 2195/2500 = 87.80% | +0.44 pp |
| **原G2＋tail推理** | **2237/2500 = 89.48%** | **+2.12 pp** |

当前证据支持“推理尾部复用对原G2和R模型有正向信号”，但不支持“R与G2训练
简单组合会叠加”。正式结论还需要多个训练seed、环境seed和延迟/连续性指标。

## 9. PlanWAM启发的教师—学生蒸馏

### 9.1 严格闭环结果

| 模型 | 初始化/监督 | 成功率 |
|---|---|---:|
| easy_noffvtt | 部署baseline | 2184/2500 = 87.36% |
| 原G2 task-attention | 当前观测预测未来意图 | 2195/2500 = 87.80% |
| action-only | 与学生同结构、从头训练、无latent KD | 2192/2500 = 87.68% |
| 旧PlanWAM学生 | 从头训练、latent KD=0.1 | 2145/2500 = 85.80% |
| 旧PlanWAM学生＋tail | 同checkpoint推理尾部复用 | 2168/2500 = 86.72% |
| action-init action-only | 迁移教师兼容动作权重，分组LR | 2157/2500 = 86.28% |
| action-init action-only uniform LR | 同初始化、统一LR | **2221/2500 = 88.84%** |
| action-init latent-KD学生 | 迁移动作权重＋latent KD，分组LR | 2117/2500 = 84.68% |

`action_init_student_uniformlr`配置已经存在，但当前没有发现对应正式闭环JSON，
因此不填结果。教师在训练和诊断时使用真实未来图像，不能作为正常在线策略直接
报告闭环成功率。

当前最稳妥结论：旧latent-only蒸馏比其action-only控制低1.88个百分点；兼容
初始化的uniform-LR action-only表现最好，而已有KD学生仍退化。说明问题不只是
“学生没继承教师动作权重”，当前latent目标、读出兼容性或优化方式仍需改进。

### 9.2 离线2×2诊断（不是成功率）

400个训练anchor、同样本/噪声下的旧学生结果：

| 动作网络 / latent | 平均速度MSE | 8步动作块MSE | 前4步MSE |
|---|---:|---:|---:|
| 学生 / 学生（S/S） | 0.01932 | 0.01128 | 0.01312 |
| 学生 / 教师（S/T） | 0.03722 | 0.02398 | 0.02469 |
| 教师 / 学生（T/S） | 0.03244 | 0.01765 | 0.02114 |
| **教师 / 教师（T/T）** | **0.01544** | **0.00584** | **0.00575** |

教师配教师latent的离线动作误差最低，但交换latent后两边都恶化，支持“latent与
动作读出共同适应、不能直接互换”的解释。学生/教师latent余弦约0.979也没有转化
为更好的闭环控制，因此不能把高latent相似度当作任务成功。

兼容初始化学生的离线诊断中，S/T动作块MSE由0.01927降到0.01312，但其闭环KD
学生仍只有84.68%；再次说明训练分布动作MSE不等同于闭环成功率。

## 10. 尚未形成正式测试结果的版本

- RoboTwin easy/G2/R/R+G2四组：配置、CPU真实样本smoke和推理回传检查已完成，
  但当前没有发现对应完整50轮＋SAPIEN评测JSON。
- PlanWAM posterior教师：完成训练和离线未来干预诊断，但因部署时不可见未来，
  没有合法在线成功率。
- `action_init_student_uniformlr`：已有训练配置/目录，当前未发现正式闭环结果。
- S-aux-shared、region gate、interaction query、OPD、RL residual：属于方案或接口，
  当前没有可列入主表的正式成功率。
- 真机EDULITE：已有采集、部署和诊断代码，但本文未发现标准化闭环测试汇总，
  因此不和仿真结果混列。

## 11. 当前可以写入论文的结论边界

1. RoboTwin DINOv3-Base主线上，4轮微调S-residual v2达到70.5%，但scratch版本
   没有复现收益；只能称为候选正向信号，不能称稳定改进。
2. Meta-World G2/task-attention相对对应控制有约1个百分点量级信号，但不同seed
   区间本身也有约1--2个百分点变化，需要多训练seed。
3. 动作尾部先验在匹配训练与原G2推理中出现正向结果，R匹配为88.84%，原G2＋
   tail为89.48%；组合R+G2没有超过各自最好结果。
4. 当前PlanWAM latent-only KD没有改善学生闭环；教师未来表示的离线优势尚未被
   成功迁移。
5. 任何跨平台“谁更高”的比较都无意义；主张应分别建立在各平台内部严格控制组上。

## 12. 关键结果文件索引

- [RoboTwin全部summary](../eval_result/yf10/)
- [Meta-World easy baseline严格补测](../eval_result/planwam_comparison/easy_noffvtt_ep40_cold_env1000_policy456.json)
- [原G2 cold](../eval_result/action_prior_g2_base/g2_ep40_cold_confirm_env1000_policy456.json)
- [原G2 tail](../eval_result/action_prior_g2_base/g2_ep40_tail_env1000_policy456.json)
- [R匹配训练](../eval_result/action_prior_full_r/r_ep40_matched_env1000_policy456.json)
- [R+G2 tail](../eval_result/action_prior_g2/r_g2_ep40_tail_env1000_policy456.json)
- [旧PlanWAM学生](../eval_result/planwam_student01_seed1000/student_ep40_cold_env1000_policy456.json)
- [action-only控制](../eval_result/planwam_comparison/action_only_ep40_cold_env1000_policy456.json)
- [兼容初始化uniform-LR action-only](../eval_result/planwam_comparison/action_init_action_only_uniformlr_ep40_cold_env1000_policy456.json)
- [旧学生/教师2×2离线诊断](../eval_result/planwam_offline/student40_teacher40_2x2_seed0_n8_chunk10.json)
- [兼容初始化学生2×2离线诊断](../eval_result/planwam_offline/action_init_student40_teacher40_2x2_seed0_n8_chunk10.json)
- [PlanWAM机制与结果详解](PLANWAM_DISTILLATION_MECHANISM_AND_RESULTS.md)
- [RoboTwin版本和命令](MODEL_VERSION_AND_COMMANDS.md)


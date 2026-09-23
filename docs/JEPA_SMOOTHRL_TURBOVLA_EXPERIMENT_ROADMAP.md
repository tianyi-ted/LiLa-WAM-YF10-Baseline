# LiLa-WAM 的 JEPA、执行一致性与轻量融合研究路线

> 文档状态：研究设计稿 v1.2（2026-09-22 可行性复核；资源受限、与 OPD 解耦）  
> 文献初次检索：2026-09-15；本次更新重点为本地评测证据和实施约束，不代表完成全面的新颖性查重。  
> 主基线：DINOv3-Base＋Action Expert `depth=12, hidden_size=512`，batch=96；核心方法有效后在 DINOv3-Large＋同一 `12×512` 专家上复核。

本文分析如何从 INTACT-JEPA、SmoothRL 和 TurboVLA 吸收可验证的设计，形成一条独立于 OPD 的研究路线。当前路线只使用现有离线示范，不训练教师、不收集 on-policy rollout、不引入 critic，也不运行环境分支。本文中的“创新候选”表示经当前公开论文和官方实现检索后仍有研究空间，不代表已经完成全球范围的首创证明。投稿前还需要按最终方法名称、公式和模块重新检索同期工作。

相关现有文档：

- [YF10 Baseline 冻结协议](YF10_BASELINE_FREEZE_PROTOCOL.md)
- [DINOv3-Base＋h512 实际基线冻结记录](YF10_DINOV3B_H512_BASELINE_FREEZE.md)

OPD 的两份已有路线文档继续独立保留，不属于本文实验依赖。

### 本次复核结论与使用入口

可以开始 C32/F16 配对微调；补齐训练随机种子后，再实现全局预测意图。空间意图是核心创新候选，但目标侧匹配的质量与学习路径仍需验证。本文件是实验设计，新增意图模块、训练 seed 接口和配对实验配置尚未实现。

2026-09-22 核对的结果（epoch 40、评测 seed 0、每任务每场景50个有效 episode）：

| 视觉骨干＋动作专家 | Clean | Randomized | 两场景平均 | 约每 epoch 时间 |
|---|---:|---:|---:|---:|
| DINO-L＋h768 | 78.2% | 67.0% | 72.6% | 97.6分钟 |
| DINO-L＋h512 | 81.2% | 69.2% | 75.2% | 84.1分钟 |
| **DINO-B＋h512（后续主基线）** | **70.2%** | **64.6%** | **67.4%** | **40.0分钟** |

证据：[Base clean](../eval_result/yf10/dinov3b_h512_ep40_demo_clean_seed0_n50/summary.json)、[Base randomized](../eval_result/yf10/dinov3b_h512_ep40_demo_randomized_seed0_n50/summary.json)、[Large h512 clean](../eval_result/yf10/small_ep40_demo_clean_seed0_n50/summary.json)、[Large h512 randomized](../eval_result/yf10/small_ep40_demo_randomized_seed0_n50/summary.json)。耗时取训练日志或相邻 checkpoint 时间间隔，是近似 wall-clock，不是受控硬件基准。

Base 相对 Large h512 的成功率下降7.8个百分点、单轮时间减少约52%。选择 Base 是资源取舍，不能写成两者精度相当。现有结果仅为单训练 run、单开发评测 seed，不证明统计显著性；h768与h512之间还存在batch 64/96差异。

主基线起点：

```text
checkpoint: checkpoints_vla/yf10_dinov3b_small/sft_2026-09-20_15-07-28/checkpoint_epoch_40.pt
saved config: checkpoints_vla/yf10_dinov3b_small/sft_2026-09-20_15-07-28/config.yaml
reference config: configs/yf10_small_dinov3b.yaml
dataset: data/yf10                 # 2441 episodes / 392736 frames
normalization: artifacts/yf10/norm_stats.json
VTT: embeddings/yf10_vtt_dinov3b  # 10 × 768
```

后续增益以“原始 Base vs 改进 Base”为主对照，并增加相同额外训练预算的 continued-SFT 对照。Large复核也必须使用自己的原始checkpoint和匹配训练预算，不能直接跨视觉维度加载Base权重。冻结文件清单、哈希和环境版本；旧的基线冻结文档中的历史架构建议不能替代这里明确的运行配置。

---

## 1. 先给出研究判断

以下方案已经被近期工作直接覆盖，单独实现时不宜作为论文核心贡献：

- 增加一个未来 DINO 特征辅助损失；当前 LiLa-WAM 已经实现。
- 联合预测动作和未来特征；JEPA Policy 已经系统研究。
- 用动作预测未来特征、再用当前与未来特征恢复动作；SLIM 已经结合 IDM、FDM 和 Flow Matching。
- 预测下一任务阶段，再将阶段特征注入 WAM；JEPA-WAM 已经在 RoboTwin 2.0 上实现。
- 将普通未来 latent 用作 Flow Matching 条件；VLA-JEPA、SLIM、JEPA-WAM 等工作已经覆盖邻近设计。
- 仅把视觉和语言改成双向 cross-attention；这是 TurboVLA 的核心模块之一。
- 仅对动作块增加执行区 mask；SmoothRL 已经针对异步在线 RL 给出严格的执行区梯度定义。

当前最值得研究的中心问题是：

> **动作模型学习的未来意图，能否与机器人下一次真正执行的动作窗口对齐，并以空间交互变化的形式直接引导动作 Flow？**

本文将候选方法暂称为：

```text
EASI-Flow
Execution-Aligned Spatial-Intent Flow Policy
执行对齐的空间意图动作 Flow
```

它由三个互相支持的部分构成：

1. 比较 `t+16` 与 `t+32` 的监督跨度，验证它们与实际执行窗口的关系；“执行未来/计划未来”是待验证解释。
2. 用局部空间交互 token 表示物体、夹爪和接触区域的变化。
3. 把可部署的预测意图注入动作 Flow 的生成过程，并使监督重点与实际执行区间一致。

这比“给 LiLa-WAM 增加 JEPA 模块”更容易形成清楚的科学问题，也能与 SmoothRL、TurboVLA 和 INTACT 建立明确联系。

### 1.1 当前资源边界

本路线冻结以下约束：

```text
只使用 data/yf10 离线示范
DINOv3 始终冻结
默认每个样本读取一个未来帧；双未来仅作为 Phase 2B 独立实验
每个 batch 默认只运行一次 action Flow 前向
不加载额外 teacher checkpoint
不训练 reward model / critic
不进行仿真 on-policy 数据采集
```

当前 baseline 每个样本已经需要一次当前图像 DINO 和一次未来图像 DINO。资源受限版本采用随机未来跨度，而不是同时编码两个未来帧：

```text
H ~ {16, 32}
future target = o[t+H]
```

这样仍然是：

```text
2 次 frozen DINO forward + 1 次 action Flow forward
```

应避免的高开销实现：

| 实现 | 额外开销来源 | 当前决定 |
|---|---|---|
| 同时读取 `t+16`、`t+32` | DINO 从2次增加到3次 | 纳入 Phase 2B，不作为默认训练配置 |
| physical/predicted intent 双 action call | action DiT 前向接近翻倍 | 仅保留为后期可选消融 |
| OPD teacher labeling | 额外教师前向与 checkpoint 显存 | 排除 |
| 环境分支和 critic | 仿真 rollout、存储和价值学习 | 排除 |

如果未来DINO仍是瓶颈，可以离线缓存紧凑target。按392,736帧估算，一个H的`8×256`个FP16数约1.5 GiB；两个H约3 GiB，另计索引和元数据。只缓存由固定编码器、投影、匹配及预处理得到的目标；若目标模块学习或EMA更新，缓存会过时，不能直接复用。

### 1.2 相对当前 baseline 的计算量

当前代码中的 current/future DINO 都由 `@torch.no_grad()` 包裹，因此不会保存 DINO 的反向图。以当前 Base `batch=96`、300个 patch、DINO维度768为例，一个 BF16 future target 本身约42.2 MiB；实际峰值增加还来自 DINO 前向的临时张量和 future decoder 激活。

| 方案 | Frozen DINO / batch | Action Flow / batch | 预期训练变化 |
|---|---:|---:|---|
| 当前 F32 baseline | 2次 | 1次 | 1.00× |
| F16 替换 F32 | 2次 | 1次 | 基本不变 |
| Fmix，单样本随机 H | 2次 | 1次 | 基本不变，只有很小的 horizon embedding 开销 |
| 单调用 global intent | 2次 | 1次 | 原future loss与intent target共用同一H；增加轻量predictor |
| 8-query spatial intent | 2次 | 1次 | 还包含匹配与注入开销，需实测，不能仅按token数推断 |
| Fdual：同时编码 F16+F32，共享 decoder trunk | 3次 | 1次 | 编码图像数量增加50%；总wall-clock和显存需实测 |
| Fdual：两个完整 decoder call | 3次 | 1次 | decoder 激活和反向也增加，显存风险进一步上升，暂不采用 |

Base＋`12×512` 当前 action model 约94.83M参数，其中 future decoder 约17.37M。Fdual 第一版采用一个共享 decoder trunk 和两个轻量输出 head：trunk只运行一次，分别预测 `t+16` 与 `t+32`；额外 `Linear(512,768)` head约增加0.394M参数。若在12 GB显卡上已经接近显存上限，额外未来DINO前向仍可能需要降低batch。

以上2次DINO的前提是原future decoder和intent target复用同一未来帧。若分别监督`t+32`和`t+16`，则实际需要当前＋两个未来共3份图像编码，即使每个分支各自只调用一次。默认实现禁止无意引入这份额外开销。

上述百分比是按当前结构得到的工程估计。正式实验应使用相同 batch 运行固定100至200个 optimizer steps，记录 `samples/s`、epoch时间和 `torch.cuda.max_memory_allocated()` 后再决定是否扩大实验。

---

## 2. 当前 LiLa-WAM 的关键事实

### 2.1 当前输入和动作路径

当前模型使用：

```text
head-camera RGB
    -> frozen DINOv3 ViT-B, layers [-7, -4, -2], feature_dim=768
    -> concat fusion + visual adapter

task name
    -> precomputed task-level VTT vector

robot state + VTT + DINO tokens + noised 32-step action
    -> action DiT
    -> 32×14 Flow velocity
```

当前 VTT 并不是在线语言编码。根据
[`embeddings/yf10_vtt_dinov3b/meta.json`](../embeddings/yf10_vtt_dinov3b/meta.json)，每个任务只有一个768维向量：

```text
mean_episode[DINO_CLS(last_frame) - DINO_CLS(first_frame)]
```

因此：

- 同一任务的所有状态使用同一个条件。
- `seen/unseen instruction` 的文字变化不会进入模型。
- VTT 能表示任务级整体视觉变化，但不提供当前阶段或目标实例的位置。
- 当前图像的 DINO patch token 仍然包含空间信息，只是任务条件没有与这些 patch 做显式、动态的空间交互。

### 2.2 当前未来预测路径

当前 action DiT 输出两部分：

```text
action tokens -> Flow velocity
condition tokens -> FutureFeatureDecoder -> future DINO patch features
```

未来特征 cosine loss 会更新 adapter、DiT 和 future decoder，所以它确实能塑造动作主干。但预测出的未来特征没有作为显式意图再次输入动作 Flow。

### 2.3 监督跨度与执行边界：待验证的关系

当前配置为：

```yaml
action_chunk_size: 32
action_execution_horizon: 16
future_frame_offset: null
```

数据集在 `future_frame_offset=null` 时使用动作块长度作为默认偏移，因此未来监督目标是：

```text
future target = observation at t + 32
```

推理时却只把前16步加入执行队列：

```text
predicted chunk: a[t : t+32]
executed now:    a[t : t+16]
then observe again and replan
```

这里训练端的`t+32`指处理后episode的索引，推理端的16指执行队列中的16次动作调用，两者不能直接等同于物理时间。数据删除过静止片段（见[预处理说明](BASELINE_YF10.md)），采集保存间隔与`take_action(..., action_type='qpos')`内部执行也会影响对应关系。先把F16/F32称为监督跨度消融，不能直接宣称F16就是精确执行对齐。

需要核对采集间隔、静止清理的索引映射、图像与动作时序及环境动作执行方式。若处理后数据缺少原始时间戳，不补造物理时间解释。后期固定其它因素，对比监督跨度H∈{8,16,32}与执行窗口E∈{8,16,32}，检查匹配组合是否更好；若只使用同一训练模型改变E，只能称为执行窗口敏感性实验。执行次数会影响step limit的含义，还需记录实际仿真步数/耗时和控制频率。

这正是 EASI-Flow 的第一项实验依据。

---

## 3. INTACT-JEPA 可以吸收什么

[INTACT-JEPA](https://arxiv.org/abs/2607.26056) 的核心不是普通未来预测，而是让同一个动作算子解释两类运动意图：

```text
local intent = z[t+1] - z[t]
goal intent  = stopgrad(z[goal]) - z[t]
```

两种条件都对应示范中的正确动作，并通过共享 predictor 建立“意图族到动作律族”的关系。它不要求两种 latent displacement 逐点相等。

### 3.1 可以直接吸收的原则

- **物理意图与部署意图分开构造。** 真实未来适合训练监督，但部署时只能使用从当前观测预测的意图。
- **共享动作生成器。** 物理意图分支和部署意图分支应调用同一套动作 Flow 参数。
- **目标侧停止梯度。** 避免预测器通过移动目标特征得到虚假的低损失。
- **比较动作结果，而不只比较 latent。** 两种意图可以在 latent 中不同，只要它们诱导出相容的动作分布。

### 3.2 不能直接照搬的地方

- INTACT 的目标条件在部署时可以从目标观测获得；当前 RoboTwin 接口只有当前图像、任务名和文字指令。
- INTACT 使用直接的分布动作读出；LiLa-WAM 使用10步 Flow ODE生成动作块。
- 你的 DINO 编码器当前冻结，动作梯度不能更新 DINO 主干，只能塑造后续 adapter 与 action DiT。

因此需要一个因果意图预测器：

```text
predicted intent = P(current DINO tokens, robot state, task condition)
```

训练时真实未来只作为 target；推理时不读取未来帧。

---

## 4. SmoothRL 可以吸收什么

[SmoothRL](https://arxiv.org/abs/2608.29768) 研究异步执行下的在线 RL。它把动作块分为：

```text
committed region | execution region | discarded region
```

并只让 value gradient 通过实际执行区间。它还让 actor 读取正在执行的动作前缀，以减少新旧动作块连接处的不连续。

### 4.1 对当前仿真 baseline 的正确解释

当前 LiLa-WAM 是队列耗尽后同步生成新 chunk，再执行前16步。此时没有 SmoothRL 所定义的“推理期间已由旧 chunk 执行”的 committed region：

```text
current LiLa-WAM:
execution = [0, 16)
look-ahead / currently unused = [16, 32)
```

因此不能直接把 SmoothRL 的 `[n,2n)` mask 原样搬过来。

### 4.2 当前阶段值得吸收的内容

1. **监督未来与执行边界对齐。** 增加 `t+16` target，保留 `t+32` 作为计划级辅助 target。
2. **区分动作损失的执行区与展望区。** 第一版可用：

   ```text
   L_action = L_FM[0:16] + beta * L_FM[16:32]
   ```

   `beta` 先比较 `1.0 / 0.5 / 0.25`，不建议第一版直接设为0，因为后半段仍参与动作轨迹结构、未来预测和末尾保持学习。
3. **记录动作块交界连续性。** 报告位置跳变、速度、加速度和 jerk，而不只看成功率。
4. **真实异步部署时再引入 committed prefix。** 到真机阶段，应让模型读取旧 chunk 尚未执行完的尾部，并根据实测延迟定义训练区间。

### 4.3 与 padding mask 的区别

执行窗口权重描述“生成的哪部分会实际送入环境”；padding mask 描述“episode 尾部哪些位置来自补齐”。二者是两个独立维度。当前保留末尾保持动作的 padding 语义，不应在执行对齐实验中顺便改变。

---

## 5. TurboVLA 可以吸收什么

[TurboVLA](https://arxiv.org/abs/2607.27205) 将常见的 `V -> L -> A` 改为轻量的 `V + L -> A`：视觉和语言分别编码，通过双向视觉语言交互后预测动作。其官方消融显示，双向交互优于直接拼接和单向 cross-attention；在 RoboTwin 2.0 中使用 DINOv3 ViT-L、三路相机、14维绝对关节动作和50步 ACT chunk。[官方代码](https://github.com/H-EmbodVis/TurboVLA)

### 5.1 你的模型与 TurboVLA 的关系

LiLa-WAM 当前已经是轻量的：

```text
DINO visual tokens + one VTT task vector -> compact action model
```

但这个 VTT 是离线任务均值，不是语言 token，也没有动态视觉语言交互。因此 TurboVLA 最值得借鉴的是“任务条件如何修改空间视觉特征”，而不是单纯删除大语言模型。

### 5.2 两条可选改进

#### A. 任务条件与视觉 patch 双向交互

```text
task / instruction tokens query visual patches
visual patches query task / instruction tokens
                 ↓
task-conditioned spatial tokens
```

第一版可以继续使用 VTT，将一个全局向量展开为4至8个任务 query；后续再替换为 BERT 文本 token。这样可以先验证融合结构，避免同时改变条件来源。

#### B. 真正使用 RoboTwin instruction

当前 `robotwin_infer.py` 明确忽略 `instruction` 参数。如果论文希望报告语言泛化，需要：

- 训练数据为每条样本提供 instruction 或从任务描述库采样 instruction。
- 使用冻结的轻量文本编码器，训练时和推理时采用同一处理。
- 将 `seen/unseen instruction` 作为真实输入差异。
- 增加同任务改写、目标属性、左右臂指令等语言敏感测试。

如果研究目标只是当前固定十任务成功率，这一改动的优先级低于执行未来和空间意图；任务名已经足以区分十类任务，语言 encoder 未必显著提高成功率。

---

## 6. EASI-Flow 的建议结构

实施顺序是第6.3节全局预测器（Phase 3），再升级到第6.2节空间交互版本（Phase 4）。二者共用第6.1节的未来跨度与第6.4节的动作 Flow 注入接口。

### 6.1 随机时间尺度目标

每个样本随机选择一个未来跨度：

```text
H ~ Categorical({16, 32})
o[t]       current observation
o[t+H]     one sampled future observation
A[t:t+32]  expert action chunk
```

DINO 仍然冻结：

```text
z0 = E(o[t])
zH = stopgrad(E(o[t+H]))
```

加入一个 horizon embedding，让预测器知道当前目标是执行未来还是计划未来：

```text
m_H = zH - z0
m_hat_H = IntentPredictor(z0, state, task_condition, Embedding(H))
```

第一轮先按Phase 2选定单个H，训练与部署保持一致。混合跨度是后续实验：有证据支持H=16时，可比较`P(H=16)=1.0`与0.75，推理固定已训练且验证过的H。执行窗口16只提供候选跨度的动机，不证明处理后第16帧等于实际执行边界。

#### 可选 Fdual：同时监督两个时间尺度

单未来实验确认两个跨度具有互补性后，可以让同一样本同时读取 `o[t+16]` 和 `o[t+32]`：

```text
zexec = stopgrad(E(o[t+16]))
zplan = stopgrad(E(o[t+32]))

h = SharedFutureDecoderTrunk(cond_tokens)
pred_exec = Head16(h)
pred_plan = Head32(h)
```

共享 decoder trunk 只运行一次，避免两个完整 decoder call。两个损失的总权重保持与 baseline 的 `lambda_future_feat=0.5` 一致：

```text
L_future_dual = lambda_future * [rho * L16 + (1-rho) * L32]
```

先测试 `rho=0.5`；如果 F16 的单未来结果明显更好，再测试 `rho=0.75`。不要分别给两个 future 各自0.5权重，否则辅助损失总强度相对 baseline 翻倍，实验无法只归因于双时间尺度。

### 6.2 空间交互意图（Phase 4，可在全局版后阅读）

空间版本针对“全局平均可能淹没局部接触变化”提出，不需要更换DINO。全局版正向结果可以支持进入本阶段；全局版失败也不自动否定局部表示，但必须先完成目标质量诊断。当前图像为 `320×240`，ViT-B/16对应 `20×15=300` 个768维patch；64个视觉token、32个动作token、1个state token、2个register token、1个VTT token共同进入DiT。原64个视觉token继续保留，空间意图是额外8个token。

不能直接逐位置相减 DINO patch：物体移动后，同一个图像网格位置可能对应不同物体。建议通过当前视觉生成8个 query，再让同一组 query 在未来视觉里寻找相应证据：

```text
current DINO patches [B,300,768] -> projection [B,300,256]
learnable queries [8,256] -> cross-attend current patches -> r0 [B,8,256]
r0 queries -> cross-attend future patches -> rH [B,8,256]
target spatial change = stopgrad(rH - r0)

current patches + state + VTT + H -> spatial predictor -> m_hat_H [B,8,256]
m_hat_H -> projection + small gate -> intent tokens [B,8,512]
```

其中未来 patch 只参加监督目标计算；`spatial predictor` **看不到未来图像**。同一组 query 与两个时刻交互，是为了得到比“第17号网格减第17号网格”更合理的软对应。可对注意力分数加入2D位置偏置，避免 query 无约束地跳到另一个同类物体；位置偏置强度需要消融。

建议第一版配置范围：

```yaml
intent_num_queries: 8
intent_hidden_size: 256
intent_layers: 1  # 成功后再试2
```

训练目标可采用 `SmoothL1(m_hat_H, stopgrad(rH-r0))`，仅在target范数足够大时叠加cosine；接近静止的目标上单独使用cosine会不稳定。上述learnable-query图是后续候选结构，不能直接作为已经解决的目标生成方案：只存在于stop-gradient目标分支的可学习query没有训练梯度；冻结随机投影不保证语义对应；EMA也不能单独保证不坍缩。

第一版先采用可检查的固定目标构造，再决定是否学习query：

1. 在冻结DINO的768维patch上建立固定的8区域聚合对照，固定区域覆盖和分配规则；暂不把区域自动称为物体。
2. 在冻结DINO特征上计算当前—未来余弦相似度，比较同网格聚合与带位置约束的软对应。匹配权重只生成监督target，不能作为推理输入。先可视化固定示范中的目标、夹爪和背景对应，记录置信度、覆盖与匹配熵。
3. 对应确定后再聚合变化。若需要768→256降维，第一版使用两组共用的固定、可复现映射；随机映射只用于压缩，不能充当语义匹配器。物体移动后特征可能近似不变，需检查特征差对位移的敏感性；坐标位移作为独立诊断/后续消融，不能直接称为3D几何监督。
4. 固定目标表现合理后才设计可学习query：明确哪些损失训练当前侧、目标侧如何冻结或EMA更新，记录梯度、query方差及注意力覆盖。没有有效学习信号的目标模块不能仅靠EMA获得物体语义。

目标匹配失败、背景主导或query坍缩时，先修复目标构造，暂不投入完整策略训练。目标质量好也不保证策略提升，仍需S1/S2消融。

期望这些 token 表示：

- 夹爪与目标物体相对位置；
- 接触建立或解除；
- 被操作物体的位置和姿态变化；
- 双臂协作关系。

这些都是待检验的解释，不是 query 自动具备的语义。必须通过 attention 可视化、遮挡实验、不同物体位置下的 query 一致性和空间 probe 验证。尤其要检查 query 是否只盯着机器人手臂或背景。

### 6.3 全局因果意图预测器（Phase 3，优先实施）

先实施低成本的**全局版本**。Base从 `target_layer=-2` 的DINO特征去掉CLS/register后，对300个patch做平均池化，得到当前和未来的768维向量。Large复核时才对应1024维与其固定层选择；实现从模型配置读取维度，避免硬编码。用相同的固定预处理得到：

```text
u0 = fixed_LayerNorm(mean(patch(z0)))  [B,768]
uH = fixed_LayerNorm(mean(patch(zH)))  [B,768]
m_target = stopgrad(uH - u0)     [B,768]
```

这里的 `fixed_LayerNorm` 指不带可训练 affine 参数的标准化；目标表征保持固定，避免预测器和目标投影一起缩成常量。平均池化只是**最便宜的因果意图诊断**，并不声称它能保留抓取位置。未来图像只用于 `m_target`。部署预测器输入仅包含当前 DINO、state、VTT 与跨度 H：

预测意图是当前输入的确定性变换，没有增加新的推理观测；可能的收益来自未来监督塑造表示以及条件注入方式。这里“因果”仅指不读取未来的时间因果性，不表示完成了因果识别。单帧存在动作多解时，回归预测可能平均不同未来，需要观察目标/预测方差和闭环表现。

```text
u0 [768] ----\
normalized state [16] -+--> separate LayerNorm/Linear --> concatenate
VTT [768] ---/
H embedding [32] ------------------------------> 2-layer MLP --> m_hat_H [768]

m_hat_H -> Linear(768,512) -> gate -> 1 intent token [B,1,512]
```

关键数据路径是：

| 阶段 | 预测器输入 | 未来图像 | Action DiT 接收 |
|---|---|---|---|
| 训练 | 当前图像、state、VTT、H | 只用于构造监督 target | 预测的 `m_hat_H` |
| 推理 | 当前图像、state、VTT、`H=16` | 不可获得，也不读取 | 预测的 `m_hat_16` |

**预测器放在 Action DiT 之前**。不能从当前 `FutureFeatureDecoder` 所读的 `cond_tokens` 直接生成意图：这些 token 已经与 `x_tau` 加噪动作混合，随 Flow 时间变化。新预测器应在一次 chunk 生成开始时算一次，再供10个 Euler 步重复使用；这既保证训练/推理条件一致，也避免预测器重复执行10次。

建议将 `m_hat_H` 的监督写成：

```text
L_intent = SmoothL1(m_hat_H, m_target)
         + eta * I(||m_target|| > eps) * [1 - cosine(m_hat_H, m_target)]
```

例如先从 `eta=0` 开始，再尝试 `eta=0.1`。需要分别记录 `L_intent`、目标/预测范数和 target 范数接近零的比例，避免模型对大量静止帧输出零向量却得到看似很好的损失。

目标并不是把示范未来直接交给动作模型。训练动作损失时也应使用 `m_hat_H`；如果把 `m_target` 直接送入主 Action DiT，测试时会产生未来信息缺失。只有明确标为 oracle 上界的独立诊断，才允许把真实未来输入动作分支。

### 6.4 在动作 Flow 中注入意图

现有 Action DiT 大致处理100个 token：32动作、1状态、2 register、1 VTT和64视觉 token。全局意图加入1个 token 后为101个，空间意图加入8个后为108个。原有视觉 token 不删除，意图仅表达“希望这次动作导致的变化”。

| 项目 | 当前 LiLa-WAM | 全局意图版本 | 空间意图版本 |
|---|---|---|---|
| 因果输入 | 当前DINO + state + 固定VTT | 相同输入 | 相同输入 |
| 未来训练目标 | `t+32` DINO patch | `t+16`/随机H的池化差值 | 对应的8个交互变化 token |
| 未来是否进入推理 | 否 | 否 | 否 |
| DiT额外条件 | 无 | 1个预测意图 token | 8个预测意图 token |
| DiT计算 | 约100 token | 约101 token | 约108 token |
| 输出 | 32×14 Flow速度 | 同上 | 同上 |
| 一次chunk生成 | 10次DiT | 预测意图1次 + 10次DiT | 预测意图1次 + 10次DiT |

推荐两级注入实现：

```text
Version 1: intent tokens 加类型embedding并与现有条件 tokens 拼接
Version 2: 在指定 DiT block 中以 gated cross-attention / AdaLN 注入
```

Version 1 用来验证信息是否有效；Version 2 才研究意图在不同 Flow 时间的作用。需要注意：**把新 token 的值初始化为零也不等于原模型**，因为它仍会改变 softmax 注意力的分母。若要求加载旧 checkpoint 后严格保持初始输出，应该使用零初始化的 gated residual cross-attention，或在 gate 为零时屏蔽新 token 并验证新旧输出一致。直接拼接方案更简单，但必须接受初始输出轻微变化并通过独立短热身稳定训练。记录门控值和意图分支梯度，防止模型一直忽略新条件。

部署调用：

```text
v_theta(x_tau, tau | z0, state, task, m_hat_H, H)
```

推理固定Phase 2选定且训练过的H。混合跨度实验中H=32是否提供任务进度信息需要验证。先让意图影响全部动作token，再独立比较执行区位置权重。

当前 `train.py --init_from` 以 `strict=True` 加载完整 action model。新增意图预测器、投影或 type embedding 后，它会因缺少新键而失败；即使改为 `strict=False`，也必须检查所有同名键的形状。实施时应提供显式的“旧键精确加载 + 新模块单独初始化”入口，打印缺失/新增键和形状，再用相同入口初始化所有对照实验。修改前不要直接对新结构运行现有 `--init_from`。

### 6.5 可选的物理意图调用

借鉴 INTACT，可以构造物理意图与部署意图两种调用：

```text
physical call:   v_phys = Flow(x_tau, tau | ..., m_H)
deployable call: v_pred = Flow(x_tau, tau | ..., m_hat_H)
```

两者可以共享 Flow 参数并学习同一专家速度场。完整双调用会增加接近一次 action DiT 前向，因此不进入主路线。只有单调用版本已经稳定提升且需要验证 INTACT 式动作律时，才在25%的 batch 上计算物理分支：

```text
L_law = mean(||v_pred - stopgrad(v_phys)||²)
```

主路线使用单次 deployable call，并用 `L_intent` 监督意图预测器。后期若 `L_law` 有明确增益，再测试 `0 / 0.1`；不运行完整双分支超参数搜索。

### 6.6 总损失

第一版建议：

```text
L = (L_FM_exec + beta * L_FM_lookahead) / (1 + beta)
  + lambda_intent * L_intent
  + lambda_future * L_original_future
```

建议分阶段加入，每次只增加一个因素。初始搜索范围：

上式假设前后各16步的损失分别取均值；除以`1+beta`使`beta=1`精确恢复原32步均值，避免改变执行区权重时意外改变FM与辅助损失的相对尺度。第一版固定`beta=1`。

```yaml
beta: [1.0, 0.5]
lambda_intent: [0.1, 0.5]
p_exec_horizon: [1.0, 0.75]
```

原 `lambda_future=0.5` 先保留。意图分支稳定后，再比较原始 future decoder 是否仍有独立收益；这一消融不与结构开发同时进行。

### 6.7 代码改动入口与先后依赖

当前只完成研究设计，以下接口尚未写入代码。实施时按顺序改造，避免训练与推理数据流不一致：

| 文件 | 改动 | 验证点 |
|---|---|---|
| `dataloader/dataset.py` | F16沿用已有单 future offset；Fmix/Fdual 才扩展多跨度读取，并返回 `future_horizon` | `t+H` 是否越界 clamp；同一个 episode 内的对应关系 |
| `models/vla_model_fm.py` | 新增全局/空间 intent predictor 与 DiT注入接口 | 预测器输入不含未来帧和加噪动作；输出维度正确 |
| `models/model_runner.py` | 当前DINO一次提特征，未来DINO只构造 target；把预测意图传给 action model | 训练仍只调用一次 Action DiT；DINO保持冻结 |
| `train.py` | 为结构变化增加可审计的匹配权重初始化和独立实验配置 | 旧键形状完全匹配；新增键明确列出；optimizer覆盖新模块 |
| `robotwin_infer.py` | 每次生成chunk前计算一次因果意图，并在10次Euler迭代中复用 | 推理不读取 future；每16步重规划时才重算 |

第一批自动检查至少覆盖：训练/推理的意图输入相同、目标未来图像不流入 Action DiT、冻结DINO参数没有梯度、意图预测器参数有梯度、结构初始化键集合正确、同一 observation 与固定采样噪声下旧模型/新模型的差异可解释。完整方法评测再使用原有 RoboTwin clean/randomized 流程。

---

## 7. 与现有工作的边界

| 工作 | 已经覆盖的内容 | EASI-Flow 必须体现的区别 |
|---|---|---|
| 当前 LiLa-WAM | `t+32` 未来 DINO patch 辅助预测 | 区分执行未来与计划未来，并显式条件化动作 Flow |
| VLA-JEPA | leakage-free future latent prediction | 研究实际执行边界和空间交互意图，而不是普通 future latent pretraining |
| JEPA Policy | 动作与未来表示共享 Transformer、两次 refinement | 保留 Flow 生成，研究意图在 Flow 时间和动作窗口内的作用 |
| SLIM | IDM/FDM masked prediction + Flow policy | 双执行时间尺度、空间对应和执行窗口信用分配 |
| INTACT | local/goal 双意图共享动作律 | 将双意图原则用于多步 Flow，并使用可部署的交互意图预测器 |
| JEPA-WAM | 下一阶段 latent 注入视频与动作 WAM | 关注局部执行边界和空间接触；不依赖昂贵64帧 V-JEPA2 stage predictor |
| TurboVLA | 轻量 V+L→A 与双向视觉语言交互 | 用交互模块构造空间动作意图；语言融合只作为可选扩展 |
| SmoothRL | 异步在线 RL 的执行区 value gradient | 在离线 Flow/JEPA 训练中对齐执行后果，未来扩展到真实异步执行 |

### 当前创新性判断

| 候选表述 | 风险 |
|---|---|
| “首次为VLA增加JEPA未来预测” | 极高，已有多项工作 |
| “未来latent指导Flow动作” | 高，与SLIM、VLA-JEPA、JEPA-WAM邻近 |
| “把INTACT移植到Flow Matching” | 中高，容易被认为方法组合 |
| “随机双时间尺度空间意图Flow，并按实际执行区分配监督” | 中等，当前检索中未发现完全相同定义，但必须通过完整消融证明各组件作用 |

论文不能只依赖组件组合。核心论证应围绕一个可测假设：

> 假设：相对于当前`t+32`的密集patch辅助预测，针对经验证的监督跨度构造局部交互变化，并作为可部署条件注入Flow，可能改善接触与放置；监督跨度与执行窗口的交互关系及实际训练成本需要实验验证。

---

## 8. 推荐实验步骤

### Phase 0：冻结 baseline

冻结：

- 文首明确的DINO-B＋h512 epoch40 checkpoint，作为全部开发实验的主起点；
- DINO-L＋h512的epoch40 checkpoint，作为后期跨骨干复核起点；
- 数据、统计、VTT、推理 horizon、B-Spline、RoboTwin 配置和 seed；
- 每任务 clean/randomized 成功率、吞吐、显存和延迟。

所有结构实验优先在Base＋h512上开发。得到稳定提升后，在Large＋h512上复核核心配置；h768只作为历史参考。

**实施前补齐训练随机种子。** 2026-09-22核对的`train.py`未发现显式设置Python/NumPy/PyTorch训练seed。评测`SEED=0`不等于训练seed。增加并保存统一的训练seed、DataLoader generator和worker初始化；Fmix等新增随机操作也纳入管理。配对组分别从同一初始权重、同一数据顺序启动，避免新模块初始化消耗随机数后改变数据顺序；记录确定性选项及硬件/库版本，不承诺跨平台逐位一致。

所有2441条示范仍可用于训练，不要求划分示范验证集。seed 0仿真用于开发选择，最终用未参与选择的仿真seeds复核，区分训练随机种子、环境seed、动作采样随机性。需要重复训练实验才能报告训练方差，多个评测seed不能替代多个训练run。

### Phase 1：低成本诊断

1. 统计原 future prediction error 与 episode 成功/失败的相关性。
2. 分任务统计 `||z[t+16]-z[t]||` 和 `||z[t+32]-z[t]||`。
3. 检查 episode 尾部 clamp 到末帧的比例，避免未来 target 大量成为相同静态帧。
4. 比较抓取失败、放置失败和超时失败的未来误差。

判断标准：如果 future error 与结果完全无关，先检查 future target 与特征选择，再投入完整 JEPA 模块。

这些诊断仍需读取数据、提取特征或记录rollout，不能称为零成本。离线示范几乎都是成功轨迹，单独读取示范不能计算失败相关性；该分析需记录闭环轨迹及之后真正出现的图像。现有decoder读取含噪动作参与更新的cond tokens，应固定Flow时间、噪声和评估方式；示范未来误差与策略rollout未来误差不能直接混为同一指标。相关性只作诊断，不作为方法必然有效/无效的证明。

### Phase 2：未来偏移消融

仅改变未来监督偏移：

| 实验 | Future target |
|---|---|
| F32 | `t+32`，当前 baseline |
| F16 | `t+16` |
| Fmix | 每个样本随机选择 `t+16` 或 `t+32`，每个样本仍只有一个 target |
| Fdual | 每个样本同时使用 `t+16` 和 `t+32`，共享 decoder trunk、两个输出 head |

F32、F16和Fmix都只执行一次 future DINO，训练计算量基本一致。Fdual执行两次 future DINO，放在单未来结果之后。这一阶段仍不把预测 future 注入动作模型，先确认执行时间尺度是否重要。

资源受限时先做配对短微调，而不是把两个模型都从头训练50轮：

```text
同一个固定 Base+h512 epoch40 checkpoint（固定起点，不是冻结可训练参数）
├── C32: 保持 t+32，低学习率继续训练4轮
└── F16: 改为 t+16，低学习率继续训练4轮
```

两组都用 `--init_from` 仅加载相同模型权重，并重新建立相同的 optimizer 与 scheduler；训练数据顺序、batch、LR、步数和 seed 必须一致。C32 是必要的 continued-SFT 对照，否则无法判断提升来自 `t+16` 还是额外训练。

首轮固定以下参数；这是待创建独立YAML的参数说明，不是已经实现的启动入口：

```yaml
dataset:
  future_frame_offset: 32  # C32；F16只将此项改为16
training:
  batch_size: 96
  grad_accum_steps: 1
  learning_rate: 4.0e-5
  lr_min: 1.0e-5
  epochs: 4
  save_interval_epoch: 1
  lambda_future_feat: 0.5
```

其它参数继承Base保存的完整配置；chunk=32、执行窗口=16、NFE=10、B-Spline处理保持一致。DINO冻结，现有动作模型可训练参数正常微调。每轮4091次更新，4轮16364次；预先比较两组第4轮结果，不分别挑选各自最佳epoch。先对两组最终checkpoint做相同seed 0的clean/randomized筛选，候选再完成每任务50次开发评测。按40分钟/epoch，两组约5小时20分钟训练，评测另计。

F16没有提升也不否定全局意图：跨度选择和显式条件注入是两个问题。没有稳定差异时可固定H=32继续全局意图，避免无依据扩大多跨度搜索。

`F16` 只需把 YAML 的 `future_frame_offset` 改为16，不需要改变模型结构。`Fmix` 需要给 future decoder 或 intent predictor 增加 horizon embedding，放在 F16 出现明确正向结果之后实现。

#### Phase 2B：Fdual 进入条件

满足下列任一条件时运行 Fdual：

- F16 的总体成功率高于 F32，但部分长任务在 F32 更好；
- F16 改善抓取/接触，F32 改善任务推进或完成步数；
- 两者 future error 分别能解释不同阶段的失败。

如果 F16 与 F32 在逐任务结果、未来误差和失败类型上都没有稳定差异，先停止 Fdual，转向空间意图表示。

Fdual 使用与 F16/F32 相同的初始 checkpoint、optimizer steps 和随机种子。由于它消耗更多算力，论文中还应增加一个 compute-matched C32：让 C32 训练到与 Fdual 相近的 GPU 时间，排除收益只来自更多计算。

### Phase 3：全局因果意图

Phase 2 先确定未来跨度；Phase 3 固定这个跨度、相同的训练数据、action expert、执行 horizon=16 和推理 NFE=10，再研究“未来变化是否需要显式进入动作 Flow”。建议优先使用 `12×512` checkpoint，在 Phase 2 表现较好的单未来版本上开发。全局版本的数据流为：

```text
current DINO patches -> mean pool -> u0
future DINO patches  -> mean pool -> uH   [仅训练监督]
current u0 + state + VTT + H -> IntentPredictor -> m_hat_H
current visual tokens + state + VTT + m_hat_H + noisy actions -> Action DiT
```

固定Phase 2选定的H（有依据时选16，否则可保留32）。原future decoder与意图监督统一使用这个H，复用同一未来DINO特征，所有组保留相同原future loss及其权重。不默认使用“intent监督16＋原decoder监督32”，否则会增加一次未来编码。建议按以下顺序做短程训练和消融：

| 实验 | 意图预测器/新token | DiT输入 | 目的 |
|---|---|---|---|
| G0 | 无 | 原条件 | 同初始权重的 continued-SFT 对照 |
| G-aux | 有预测器及意图监督，不注入意图 | 原条件 | 分离额外辅助监督与动作条件注入的收益 |
| G1 | 有 | 固定零/常量 token | 控制新增参数和 token 数带来的影响 |
| G2 | 有 | 当前条件预测的 `m_hat_H` | 主实验：可部署的因果意图 |
| G3 | 同G2权重 | 评测时在同任务、同H样本间打乱或置零 `m_hat_H` | 检查模型是否真的使用意图；不重新训练 |
| G-oracle | 同类结构，训练与离线诊断均用真实 `m_target` | 真实未来意图 | 仅看已有轨迹上的动作损失上界，不能用于正常在线成功率 |

`G1` 要匹配 G2 的训练步数和可训练参数；如果单纯零 token 会让意图预测器收不到动作梯度，应让它继续接受相同的意图监督，只在输入 DiT 前将其替换为常量。这样能检验动作性能提升是否来自动态意图，而非多一个辅助任务。G-oracle 需要单独训练，只能在已记录示范的离线样本上比较 Flow 损失；正常闭环评测没有预先知道的真实未来帧。不能把未来只在评测时突然塞给 G2。

每个训练 batch 的逻辑为：

```text
1. 读取 o[t], o[t+H], state, VTT, expert action[32]
2. no_grad DINO 当前特征与未来 target
3. 从当前信息预测 m_hat_H；用 stopgrad(m_target) 监督
4. 将 m_hat_H 投影为意图条件；动作 DiT 只前向一次
5. 计算 L_FM + lambda_intent * L_intent + 原定的 future loss
```

推理时只需要步骤2中的当前DINO，算一次固定H对应的预测意图，供10次Euler调用复用。训练H=32时不得未经验证在部署切成16；H=16实验才使用`m_hat_16`。检查范围、范数和BF16数值稳定性。

先做100至200 optimizer steps稳定性试验：确认loss有限、意图分支梯度非零、输入一致、旧权重成功加载。先比较G0/G-aux/G2；有信号后补G1/G3和完整评测。G2优于G0但不优于G-aux，不能证明条件注入带来额外收益。不要仅凭单个seed的10次快速筛选断言提升。

判读：若 G2≈G1，可能新信息无效或被忽略；若打乱意图不影响 G2，优先查注入门控；若离线 G-oracle 的动作误差更低、但 G2 闭环成功率无提升，优先查意图预测误差或当前单帧观测不足；若 G-oracle 也没有降低离线动作误差，检查未来表示与动作的相关性。离线动作误差和闭环成功率是不同指标，不能直接比较数值。

### Phase 4：空间交互意图

先按第6.2节核验空间目标，再实施空间版本。全局版失败并不是空间版必然失败的证据，G-oracle无收益也不能单独证明瓶颈一定在全局池化。保持已选H、future loss、训练预算和评测配置。第一版优先使用固定区域及冻结特征匹配，随后再单独研究可学习query：

| 实验 | 表示/匹配 | 目的 |
|---|---|---|
| S0 | G2全局意图 | 现有可部署对照 |
| S1 | 8个固定区域目标，未来按相同网格位置聚合 | 与S2匹配token数、预测器和注入结构 |
| S2 | 同样8个区域，经冻结DINO当前—未来软对应聚合 | 主空间对应实验 |
| S3 | S2权重，打乱空间意图槽或遮挡目标/夹爪区域 | 检查模型对条件及区域的依赖 |

S0与S2 token 数不同，必须保留 S1 或等容量8-token的全局重复版本，避免把“更多 token/参数”误读为空间对应收益。S2 比 S1 更好且注意力稳定地跟随物体移动，才支持“软匹配有作用”。如果不同 query 全部聚焦同一背景区，应检查 collapse、目标侧投影和位置偏置。

可优先按任务分析 `grab_roller`、`stack_blocks_two`、`hanging_mug` 等需准确抓取/对齐的任务，以及 `press_stapler`、`turn_switch` 等短接触任务；最终主指标仍是十任务 macro success，不能只挑有利任务。建议另外记录：抓取触碰是否发生、首次接触步数、接触后掉落、最终放置误差，以及成功 rollouts 的平均执行步数。自动化失败标签若不可得，可以对固定数量视频做盲法人工归类。

可视化当前 query 在300个 DINO patch 上的注意力和对应的未来匹配热图，但 attention 不是定位精度证明。进一步测试：遮挡目标物体、遮挡背景、改变目标物体位置时，动作/意图变化是否符合预期。推理每16环境步重新提取一次当前图像并重算空间意图，而不是每一步都重算或把上次意图跨 chunk 无条件复用。

### Phase 5：梯度路由消融

比较：

```text
future auxiliary only
predicted intent condition only
stop-gradient / attached predictor path
```

主方法保持单 action call。物理意图双调用只作为资源允许时的附加机制实验，不影响本文主结论。

### Phase 6：执行区损失与连续性

比较 `beta=1.0/0.5/0.25`，并报告：

- 前16步动作误差；
- 后16步动作误差；
- chunk 边界位置/速度跳变；
- acceleration RMS 和 jerk RMS；
- 成功 episode 的平均执行步数。

### Phase 7：TurboVLA 式交互

在 EASI-Flow 已经有效后，再比较：

```text
VTT concat
VTT -> visual one-way cross-attention
visual <-> VTT bidirectional interaction
frozen BERT instruction <-> visual bidirectional interaction
```

只有使用真实 instruction 输入后，才能讨论语言改写泛化。

### Phase 8：真机同步验证

在仿真结果稳定后，将相同的同步16步执行协议迁移到真机，先验证推理延迟、chunk 交界连续性和安全限幅。异步 committed-prefix 建模留作独立后续工作，不纳入当前论文主实验。

---

## 9. 最小实验矩阵

### 快速筛选

每个配置先运行：

```text
10 tasks × clean × seed 0 × 10 valid episodes
```

通过条件建议：

- macro SR 至少提升3个百分点；或
- 对当前弱任务提升明显，同时强任务没有系统下降；或
- 相同成功率下，执行步数、jerk或推理成本明显降低。

### 开发评测

筛选后的配置运行：

```text
clean + randomized
10 tasks
seed 0
50 valid episodes per task
```

### 正式评测

最终 baseline 与方法运行预先固定的多个 seeds，报告：

- per-task success；
- macro success 与 bootstrap 95% CI；
- 相对 baseline 的 paired success flip；
- 平均成功步数；
- 推理延迟、参数量、峰值显存；
- 速度、加速度与 jerk；
- 训练 GPU hours。

同一批初始环境seeds上配对比较；记录实际有效seed列表，不只记录起始seed，因为无效场景会被跳过。可预先保留环境seed 1/2用于最终复核，seed 0继续开发使用。按任务报告结果及合适的分层/配对置信区间，不把不同任务与训练run的波动混为一谈。当前策略忽略instruction输入，`unseen`只是评测脚本选项，不构成语言泛化证据。

---

## 10. 失败模式与止损规则

### 10.1 未来预测准确，但成功率不变

说明 future loss 学到了易预测的背景或机器人整体运动。转向空间 query、遮挡实验和执行边界目标。

### 10.2 Oracle intent 有效，predicted intent 无效

意图预测器容量、任务条件或当前观测历史不足。优先增加两帧历史或语言条件，暂不扩大 action DiT。

### 10.3 Predicted intent 与 shuffled intent 表现相同

动作模型忽略了意图。检查门控值、attention 权重和梯度范数；尝试逐层 gated injection。

### 10.4 F16 提升短动作、损害长任务

保留 `t+16` 执行意图，同时用低权重 `t+32` 计划意图维持长程进度。

### 10.5 平滑度提高但成功率降低

检查夹爪通道是否被连续平滑、接触动作是否被过度平滑。对位置关节和夹爪分别处理，并报告接触阶段误差。

### 10.6 训练成本增长过快

真实未来需要一次额外 DINO 前向。可离线缓存紧凑未来 target；第一版不解冻 DINO，也不计算双未来或 physical Flow call。

---

## 11. 建议论文贡献结构

如果实验成立，可以将论文贡献组织为：

1. 揭示 action chunk 的计划跨度与 receding-horizon 实际执行边界之间的 future-supervision mismatch。
2. 提出双时间尺度空间交互意图，以执行未来指导局部动作，以计划未来维持任务进度。
3. 用训练可见的物理未来监督部署可用的因果意图预测器，并按真实执行窗口分配动作学习信号。
4. 在 RoboTwin clean/randomized 多任务和真机上验证成功率、接触精度、平滑度及效率。

建议避免的表述：

- “首个JEPA机器人策略”；
- “首个未来感知VLA”；
- “首个Flow Matching世界模型”；
- “首次把意图用于动作生成”。

这些宽泛表述已经被已有工作覆盖或很难成立。

---

## 12. 推荐阅读与应吸收的内容

### 核心相关工作

- [INTACT-JEPA](https://arxiv.org/abs/2607.26056)：双意图、共享动作律、非对称梯度、部署可用的直接控制接口。
- [SmoothRL](https://arxiv.org/abs/2608.29768)：generated chunk 与 executed chunk 的区别，执行区梯度和异步动作前缀。
- [TurboVLA](https://arxiv.org/abs/2607.27205)：轻量 `V+L→A`、双向视觉语言交互、模型深度与 action horizon 消融。
- [VLA-JEPA](https://arxiv.org/abs/2602.10098)：训练时未来信息只作为 target 的 leakage-free 设计。
- [SLIM-0.5B](https://arxiv.org/abs/2608.09771)：IDM/FDM 双向动作落地表示、masked trajectory prediction 与 Flow policy。
- [JEPA-WAM](https://arxiv.org/abs/2608.10780)：短期物理未来与阶段语义未来、gated future latent injection、RoboTwin 2.0 评测。
- [JEPA Policy](https://arxiv.org/abs/2609.09630)：动作与未来表示配对预测、共享 Transformer、未来误差作为失败信号。

### 邻近方法

- [Affordance-based Robot Manipulation with Flow Matching](https://arxiv.org/abs/2409.01083)：空间 affordance 如何作为 Flow Matching 动作条件。
- [OpenVLA](https://arxiv.org/abs/2406.09246)：DINO 与语义视觉特征的互补，以及多任务 VLA 评测规范。
- [V-JEPA](https://github.com/facebookresearch/jepa)：target encoder、latent prediction 和避免像素重建的基本原则。

---

## 13. 当前推荐优先级

```text
P0  固定Base+h512 epoch40基线，补齐训练seed与实验记录
P1  C32/F16配对微调4轮；先称为监督跨度消融
P2  全局意图：G0/G-aux/G2，再补常量和打乱对照
P3  先核验固定空间目标，再做S1/S2和可学习query扩展
P4  有证据后做双未来、执行窗口×监督跨度、归一化执行区权重
P5  Large+h512复核；多seed仿真和真机验证
P6  资源允许时研究TurboVLA式交互，作为独立附加因素
```

P1或普通future injection的单独提升不足以确认论文创新性。主线需要围绕空间变化表示、监督跨度与执行窗口关系建立可重复证据，并与近邻方法比较；完成这些实验也不保证首次提出或录用。优先推进可验证的最小实验，根据结果决定后续模块，而非一次叠加整个路线。OPD继续独立保留。

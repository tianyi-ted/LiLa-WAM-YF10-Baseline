# `yf10_base.yaml` 参数与资源影响参考

本文解释 [`configs/yf10_base.yaml`](../configs/yf10_base.yaml) 中每个参数在当前 LiLa-WAM 代码里的实际作用，重点记录它们对训练速度、GPU 显存、CPU/内存、磁盘占用、推理速度和实验可比性的影响。

本文以当前项目实现为准，不只依据参数名称推测。代码调整后，应同步更新本文。

当前另有一份仅缩小 action expert 的独立配置：
[`configs/yf10_small_d8_h512.yaml`](../configs/yf10_small_d8_h512.yaml)。它将
`depth: 12 -> 8`、`hidden_size: 768 -> 512`，其余训练、数据和推理字段与
`yf10_base.yaml` 保持一致。该配置是新的小模型 baseline，不能直接加载
大模型 checkpoint。checkpoint 选择和实验归档规则见
[`YF10_BASELINE_FREEZE_PROTOCOL.md`](./YF10_BASELINE_FREEZE_PROTOCOL.md)。

## 1. 当前配置快照与重要提醒

当前十任务数据包含：

| 项目 | 数量 |
|---|---:|
| 有效 episode | 2441 |
| 有效训练帧 | 392736 |
| clean | 485 episode / 76853 帧 |
| randomized | 1956 episode / 315883 帧 |

当前 `yf10_base.yaml` 的训练字段为：

```yaml
batch_size: 64
grad_accum_steps: 1
epochs: 50
```

因此当前有效 batch 为 `64 × 1 = 64`，不是配置注释所写的 `8 × 16 = 128`。当前设置在 4070 Ti 12 GB 上较激进，而且 50 epochs 远大于既定 baseline 的 11–12 epochs。正式复现 baseline 时建议改回：

```yaml
batch_size: 8
grad_accum_steps: 16
epochs: 12
save_interval_epoch: 1
```

若显存允许，可以使用 `16 × 8`；若 OOM，使用 `4 × 32`。三者的有效 batch 都是 128，但物理 batch 越大通常吞吐越高。

按当前结构实例化后，可训练动作模型约有 211.95M 参数：

| 模块 | 参数量 |
|---|---:|
| 12 层 DiT 主干 | 127.54M |
| DINO 视觉适配器 | 38.65M |
| 未来特征预测器 | 38.83M |
| 多层视觉特征融合 | 3.15M |
| 时间嵌入 | 2.36M |
| VTT 任务条件投影 | 1.38M |
| 合计 | **211.95M** |

DINOv3 ViT-L/16 是冻结模型，不产生梯度和优化器状态，但训练时仍占用显存并执行前向计算。启用未来特征预测后，每个样本还需要对当前图像和未来图像分别运行一次 DINO。

## 2. `common`：动作、状态与推理时序

### `action_dim: 14`

每一步动作的维度。当前对应双臂共 14 维动作。

- 必须与 HDF5 动作、归一化统计和 RoboTwin 控制接口一致。
- 修改后会改变输入/输出层形状，旧 checkpoint 通常不兼容。
- 对速度和显存影响很小，不应作为资源调节参数。

### `state_dim: 16`

机器人本体状态维度，当前由左右臂末端状态和夹爪状态拼接得到。

- 必须与数据集和归一化统计一致。
- 修改对资源影响很小，但会破坏数据和 checkpoint 兼容性。

### `action_chunk_size: 32`

模型一次并行预测 32 步动作。它必须等于 `len(action_indices)`。

- 增大：预测范围更长，但任务更难，动作 token 和 DiT 注意力计算增加。
- 减小：单次预测更容易，但推理时通常需要更频繁重规划。
- 当前 DiT 约有 `32 action + 1 state + 2 register + 1 task + 64 vision = 100` 个 token。

baseline 建议保持 32。

### `proprio_len: 1`

输入模型的机器人状态时间长度，必须与 `len(state_indices)` 一致。

当前：

```yaml
state_indices: [0]
proprio_len: 1
```

若以后改成状态历史，例如 `[-3,-2,-1,0]`，则 `proprio_len` 也应改为 4。增加少量状态历史对算力影响很小，但训练和推理必须使用一致的历史构造。

### `num_inference_steps: 10`

Flow Matching 推理阶段的 Euler 积分步数；不影响训练。

- 5 步：动作模型推理耗时约减半，但采样质量可能下降。
- 10 步：当前 baseline。
- 20 步：动作模型推理耗时约翻倍，质量不保证同比提高。

它主要影响仿真测试速度，不明显改变单次推理峰值显存。

### `action_execution_horizon: 16`

模型生成 32 步，但只把前 16 步放入动作队列；队列耗尽后再使用新观测生成下一段。

- 改为 8：更频繁闭环纠错，模型调用次数约增加一倍，可能改善抓取定位。
- 改为 32：推理更快，但更接近开环执行，误差更难及时纠正。
- 不影响训练速度和训练显存。

## 3. `dataset`：数据读取与输入构造

### `type: robotwin`

当前主要是配置语义标记。现有 `create_dataset()` 直接建立 `RobotWinTaskDataset`，没有根据该字段切换数据集类，因此修改它目前基本不影响运行或资源占用。

### `dataset_dir`

预处理后十任务数据根目录。路径本身不改变模型计算，但存储介质影响吞吐：

- 本地 NVMe/SSD 最合适。
- 机械盘、网络盘或繁忙磁盘可能让 GPU 等待数据。
- HDF5 压缩图像解码会消耗 CPU。

当前正式训练目录是：

```text
/home/lab347-no10/Tianyi/LiLa-WAM/data/yf10
```

### `data_mode: both`

可用值为 `clean`、`randomized`、`both`。

| 模式 | 当前帧数 | 每轮时间 | 典型用途 |
|---|---:|---:|---|
| clean | 76853 | 最短 | C2C baseline |
| randomized | 315883 | 较长 | 随机域训练 |
| both | 392736 | 最长 | 综合训练 |

当前 DataLoader 对全部帧普通 `shuffle=True`，没有任务均衡，也没有 clean/randomized 1:1 均衡。使用 `both` 时约 80.4% 的帧来自 randomized。

### `camera_names: ["head_camera"]`

需要读取的摄像头名称。当前实现最终只将 `camera_names[0]` 作为 `pixel_values` 输入模型。

直接增加第二路摄像头会增加：

- HDF5 读取量；
- JPEG 解码和 CPU 时间；
- 主机内存占用。

但当前模型仍只使用第一路，因此不能仅靠 YAML 实现多摄像头融合。baseline 应保持单路 `head_camera`。

### `image_size: [320, 240]`

顺序是 `[宽, 高]`。DINOv3 patch size 为 16，宽高必须能被 16 整除。

当前产生：

```text
20 × 15 = 300 patch tokens
1 CLS + 4 DINO registers + 300 patches = 305 tokens
```

这是主要资源参数之一：

- 更高分辨率有利于小物体和抓取点定位；
- DINO 的注意力开销随 patch token 数近似平方增长；
- 未来预测器 query 数也等于 patch 数；
- 数据解码、传输和显存都会增加。

改成 224×224 会降到 196 个 patch，但未来预测器的可学习 query 形状也会变化，已有 checkpoint 可能无法严格加载。

### `task_cond_dir`

VTT 任务条件向量目录。每个任务加载一个预计算的 1024 维 `task_cond.npy`，训练时作为一个任务 token。

- 不会每步运行语言模型，只是内存查表。
- 资源开销很小。
- 对十任务辨别很重要，关闭后模型只能从图像和状态猜测任务。

### `image_aug: false`

启用后，每张当前图像随机使用 1–2 个轻量 ColorJitter 操作；未来监督图像不增强。

- 增加 CPU/PIL 图像处理，可能降低数据加载速度。
- GPU 显存基本不变。
- 可能提升光照和颜色鲁棒性。
- baseline 建议保持 `false`，之后作为单独消融。

### `future_frame_offset: null`

`null` 会自动取 `len(action_indices)`，当前等于 32，即用 `t+32` 图像的 DINO patch 特征作为未来预测监督。

- 改变 offset 几乎不影响资源，因为仍只读取一张未来图像。
- offset 小：目标容易，但长期动态信息较弱。
- offset 大：目标更长期，但预测难度更高。
- episode 尾部越界时钳制到最后一帧。

### `state_indices: [0]`

相对当前锚点需要读取的状态帧。负数可表示历史，但必须同步调整 `proprio_len`。多读几个低维状态对资源影响很小。

### `camera_indices: [0]`

相对当前锚点需要读取的图像帧。当前模型期望 `(B,C,H,W)`，因此不要仅在 YAML 中改成多帧；多帧会产生 `(B,T,C,H,W)`，需要额外实现时间图像融合。

### `action_indices: [0, ..., 31]`

相对锚点读取的动作监督。长度必须等于 `action_chunk_size`。

episode 尾部不足 32 步时，越界位置复制最后一个动作。Dataset 会生成 `action_mask`，但当前 Flow Matching loss 没有使用它，因此复制的尾部“保持动作”仍参与训练。这是当前有意保留的训练语义。

## 4. `model`：视觉融合和动作模型

### `num_registers: 2`

动作模型内部的可学习 register token，与 DINO 自带的 4 个 register token 不同。它们为多模态信息提供额外工作空间，参数和显存开销很小。

### `use_task_cond: true`

启用 VTT 任务条件。它增加约 1.38M 投影参数和一个 DiT token，资源开销很小，但对多任务消歧很重要。十任务 baseline 应保持开启。

### `vision_encoder.checkpoint_path`

冻结 DINOv3 ViT-L/16 的本地路径。当前编码器为 24 层、hidden size 1024、16 heads、patch size 16、4 register tokens。

DINO 冻结意味着：

- 无梯度；
- 无优化器状态；
- 仍有权重显存和前向计算成本。

### `feat_layers: [-12, -8, -4]`

从 DINO 的中后部取三层特征，兼顾局部结构和高层语义。

当前调用使用 `output_hidden_states=True`，所以减少为单层不会跳过 DINO 其他层的计算。主要节省的是：

- 融合输入和投影参数；
- 特征保存/传递显存；
- 后续融合计算。

它不会让 DINO 主干前向缩短为一层。

### `include_cls_register: true`

决定送入视觉融合器的特征是否保留 1 个 CLS 和 4 个 DINO register token。

- `true`：305 tokens。
- `false`：300 patch tokens。

资源差别约 1.7%，主要区别是是否保留全局语义 token，而不是训练速度。

### `fusion_mode: concat`

三层 DINO 特征沿通道拼接为 `3×1024=3072` 维，投影后只使用一个视觉适配器，最终产生 64 个视觉 token。

代码还支持 `per_layer`：每个 DINO 层使用一个独立适配器，三层会产生 192 个视觉 token。它会显著增加适配器参数、显存和 DiT 注意力计算。4070 Ti baseline 建议保持 `concat`。

### `concat.proj_type: linear`

使用一个 Linear 将 3072 维融合特征投影到输出维度。改成 `mlp` 会增加 GELU 和第二个 Linear，表达能力更强但参数和计算更多。

### `concat.pre_norm: true`

拼接前分别归一化各层 DINO 特征，防止不同层数值尺度差异导致某层主导融合。计算开销很小，建议保持开启。

### `concat.out_dim: null`

`null` 自动等于 DINO hidden size 1024。减到 512 可减少融合输出和视觉适配器输入成本，但会更强地压缩视觉信息，并改变模型/checkpoint 结构。

## 5. `future_feat`：未来视觉特征预测

### `enabled: true`

开启后，每个样本执行：

```text
当前图像 → DINO → 动作条件
未来图像 → DINO → 未来特征监督
DiT 条件 token → FutureFeatureDecoder → 预测未来特征
```

它是当前最主要的训练资源开关之一：

- 图像读取和解码接近增加一倍；
- DINO 前向从每样本一张图增加为两张图；
- 增加约 38.83M 可训练参数；
- 增加 300-query TransformerDecoder 的反向传播。

关闭可明显提速，但会移除 LiLa-WAM 的核心未来表征预测目标，只适合作为消融，不适合作为完整 baseline。

### `target_layer: -4`

选择未来图像 DINO 的倒数第 4 层 patch 特征作为监督。改变它几乎不影响资源，但会改变监督偏向：浅层更偏局部结构，深层更偏语义。

### `num_decoder_layers: 4`

未来特征预测器深度，参数和计算随层数近似线性变化。改为 2 层可显著减少该模块开销，但属于结构消融。

### `num_heads: 4`

未来预测器注意力头数。hidden size 固定时，头数对总投影参数影响不大，主要改变每头维度和 kernel 行为。hidden size 必须能被头数整除。

## 6. `action_expert`：动作 DiT 与视觉适配器

### `depth: 12`

DiT 主干深度。主干当前约 127.54M 参数，层数对参数、反向计算和激活显存影响近似线性。12→8 可减少约三分之一主干层，但会改变模型容量和 baseline。

### `hidden_size: 768`

动作模型隐藏维度。Transformer 大量矩阵参数随其平方增长。若从 768 降到 512，核心矩阵参数理论比例约为 `512²/768²≈44%`，能显著降资源，但也明显降低模型容量。

必须满足 `hidden_size % num_heads == 0`。

### `num_heads: 8`

DiT 注意力头数。hidden size 固定时，改变头数通常不会大幅改变参数量；它不是首选提速参数。

### `vlm_adapter_num_queries: 64`

视觉适配器把约 305 个 DINO token 压缩成 64 个视觉 token。

- query 越多，可能保留更多空间细节；
- 视觉适配器计算增加；
- DiT token 数增加，注意力成本上升。

改成 32 后 DiT 总 token 约从 100 降到 68，能节省注意力计算，但可能损失抓取点等空间信息。

### `adapter_depth: 4`

视觉 Perceiver Adapter 的层数。当前视觉适配器约 38.65M 参数；4→2 会明显减少适配器参数和反向计算，但可能降低 DINO 特征适配能力。

## 7. `training`：优化和训练总量

### `batch_size`

一次真正送入 GPU 的物理 batch，是峰值显存最直接的控制量。

- 显存大致随 batch 线性增长；
- batch 增大通常提高 GPU 吞吐，直到显存或计算饱和；
- 当前结构同时处理当前/未来图像，`64` 很可能不适合 12 GB 显存。

### `grad_accum_steps`

累计若干 micro-batch 后才执行一次 optimizer step：

```text
effective_batch = batch_size × grad_accum_steps
```

梯度累积不会降低一个 micro-batch 的显存，只允许使用较小物理 batch 模拟较大有效 batch。相同有效 batch 下，物理 batch 越大通常墙钟吞吐越好。

### 当前训练量换算

当前 `64×1×50 epochs`：

```text
392736 // 64 = 6136 micro-batches/epoch
6136 optimizer steps/epoch
50 epochs = 306800 optimizer steps
```

建议 baseline `8×16×12 epochs`：

```text
392736 // 8 = 49092 micro-batches/epoch
ceil(49092/16) = 3069 optimizer steps/epoch
12 epochs = 36828 optimizer steps
```

因此当前 50 轮设置的 optimizer update 数约为建议 baseline 的 8.3 倍。

### `time_sampler: logit_normal`

Flow Matching 训练时的噪声时间采样方式。`mu=0,sigma=1` 时更集中在中间时间区域；`uniform` 则均匀覆盖 `[0,1]`。它对资源几乎没有影响，主要影响学习到的速度场。

### `time_mu: 0.0`

控制 logit-normal 中心。增大使采样更靠近真实动作端 `t=1`，减小则更靠近噪声端 `t=0`。不影响资源。

### `time_sigma: 1.0`

控制时间分布宽度。增大后更容易接近 0/1，减小后更集中在 `sigmoid(mu)` 附近。不影响资源。

### `learning_rate: 2e-4`

AdamW 初始学习率。几乎不改变速度和显存，但决定收敛稳定性。当前代码没有 warmup，会直接从该值开始训练。

### `lr_min: 5e-5`

CosineAnnealingLR 的最终学习率。调度在全部 optimizer steps 上从 `learning_rate` 下降到 `lr_min`。改变 epochs 或有效 batch 也会改变调度经历的实际更新次数。

`lr_min` 不应大于初始学习率，否则余弦调度会把学习率向上拉，代码会输出警告。

### `weight_decay: 1e-2`

AdamW 权重衰减，主要缓解过拟合；对运行资源影响很小。

### `betas: [0.9, 0.99]`

AdamW 一阶、二阶动量系数。改变优化动态，不改变优化器状态张量数量，因此资源差异很小。

### `grad_clip_norm: 1.0`

optimizer step 前将总梯度范数裁剪到 1.0，防止偶发梯度爆炸。增加一次梯度范数遍历，开销相对较小，建议保留。

### `epochs: 50`

完整遍历训练集的次数。总时间、总数据读取量和 checkpoint 数量近似随 epochs 线性增长。既定两阶段 baseline 建议：

1. Base：11–12 epochs，LR 2e-4；
2. Fine-tune：3–4 epochs，LR 4e-5，使用 `--init_from` 只加载模型权重并重建优化器/调度器。

### `save_interval_epoch: 1`

每轮保存一次模型、AdamW 优化器、scheduler、epoch 和 global step。DINO 不进入动作 checkpoint，但 211.95M 动作模型及优化器状态仍可能让单个文件达到约 GB 量级。

- 12 轮 baseline：每轮保存，便于选 checkpoint。
- 50 轮长实验：建议每 5 轮保存，减少磁盘和 I/O 暂停。

### `lambda_future_feat: 0.5`

总损失为：

```text
loss = action_flow_matching_loss + 0.5 × future_feature_cosine_loss
```

调节它几乎不改变资源。设为 0 也不会停止未来图像读取、DINO 前向和 decoder 计算；真正节省资源必须将 `future_feat.enabled` 设为 `false`。

### `use_vel_weight: false`

是否按动作 chunk 的平均变化量进行样本级 loss 加权。当前公式使低速 chunk 权重接近 1，高速 chunk 权重逐渐下降到 `alpha`。

开启后的计算开销很小，但需要注意当前实现使用 `x1[:,:,:-1]`，只排除最后一维，并没有显式同时排除双臂两个夹爪维度。baseline 建议保持关闭。

### `vel_weight_alpha: 0.2`

仅在 `use_vel_weight=true` 时生效，是高速 chunk 的权重下限。当前关闭加权时修改它没有效果。

### `vel_weight_sigma: 0.01`

仅在速度加权开启时生效，控制权重对动作变化量的敏感度。sigma 越小，稍有运动就越容易被降权。不影响资源。

## 8. `inference`：动作后处理

### `smooth_actions: true`

推理生成完整动作 chunk 后执行 B-Spline 平滑。当前固定使用三次 B-Spline、8 个控制点。

- 只影响推理，不参与训练。
- CPU 开销很小。
- 可以减少抖动，但也可能弱化接触点附近的快速动作或夹爪变化。

可在不重训的情况下比较 `true/false`。

### `smooth_sigma: 1.0`

当前实现只读取该字段，没有将其传给 B-Spline 函数。因此该参数目前无实际效果，改成 0.1 或 10 都不会改变输出。若需要可调平滑强度，应修改推理实现，而不只是 YAML。

## 9. `system`：DataLoader 与主机资源

### `num_workers: 4`

DataLoader 并行读取 HDF5、解码压缩图像和预处理的子进程数量。

- 增加 workers 可减少 GPU 等待数据；
- 会增加 CPU 占用、主存占用和磁盘并发读取；
- 数量过多可能使磁盘竞争加剧，反而变慢；
- 基本不增加模型显存。

本地 SSD 可从 4 试到 8。若 GPU 利用率频繁跌到 0 且 CPU/磁盘未满，可以增加；若内存不足、HDF5 worker 异常或磁盘满载，应降低。

### `pin_memory: true`

使用锁页内存以帮助 CPU→GPU 传输。会增加少量主机锁页内存，基本不增加 GPU 显存。当前代码没有显式 `non_blocking=True`，收益可能没有完全发挥，但 CUDA 训练通常仍建议保持开启。

## 10. 参数对资源影响的优先级

| 参数 | 训练速度/显存影响 | 是否改变模型或实验定义 |
|---|---|---|
| `batch_size` | 极大 | 否，适合先调 |
| `future_feat.enabled` | 极大 | 是，移除核心辅助任务 |
| `image_size` | 极大 | 是，改变视觉输入和预测器形状 |
| `hidden_size` | 极大 | 是，改变模型容量 |
| `depth` | 很大 | 是，改变模型容量 |
| `future_feat.num_decoder_layers` | 较大 | 是 |
| `vlm_adapter_num_queries` | 中等 | 是，影响空间压缩 |
| `adapter_depth` | 中等 | 是 |
| `num_workers` | 取决于数据瓶颈 | 否，适合调优吞吐 |
| `grad_accum_steps` | 不降低单 micro-batch 显存 | 否 |
| `feat_layers` | 中等偏小 | 是 |
| LR、betas、weight decay | 几乎无资源影响 | 改变优化过程 |
| `lambda_future_feat` | 几乎无资源影响 | 改变多目标权重 |
| `num_inference_steps` | 只影响推理时间 | 否 |
| `action_execution_horizon` | 只影响重规划频率 | 否 |

## 11. 推荐的实验调整顺序

为了让论文实验可解释，建议一次只改变一个因素：

1. 固定模型结构和数据，先寻找能运行的最大物理 batch；保持有效 batch 不变。
2. 完成 12 轮 base + 3–4 轮低学习率 fine-tune baseline。
3. 固定 checkpoint，比较 `action_execution_horizon=8/16` 和是否 B-Spline。
4. 再依次做 future feature、VTT、DINO 层融合、query 数等结构消融。
5. 改变图像尺寸、摄像头或状态历史时，必须重新确认训练—推理一致性和 checkpoint 兼容性。

不要在同一实验中同时修改模型结构、训练数据、学习率、有效 batch 和推理 horizon，否则成功率变化无法归因。

## 12. 相关代码入口

- 配置文件：[`configs/yf10_base.yaml`](../configs/yf10_base.yaml)
- 数据读取：[`dataloader/dataset.py`](../dataloader/dataset.py)
- 模型与 Flow Matching loss：[`models/vla_model_fm.py`](../models/vla_model_fm.py)
- DINO 包装与未来特征目标：[`models/model_runner.py`](../models/model_runner.py)
- 训练循环与余弦调度：[`train.py`](../train.py)
- 推理、动作队列与 B-Spline：[`robotwin_infer.py`](../robotwin_infer.py)
- 十任务完整流程：[`docs/BASELINE_YF10.md`](./BASELINE_YF10.md)

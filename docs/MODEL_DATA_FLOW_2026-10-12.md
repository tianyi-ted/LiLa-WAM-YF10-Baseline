# LiLa-WAM 模型完整数据通路

本文档说明 YF10 版本从 HDF5 episode、DINOv3、VTT、状态和动作数据，到 Flow Matching Action DiT 的完整数据通路，并区分训练和推理阶段。

## 1. 总体流程

~~~mermaid
flowchart TD
    A[HDF5 episode] --> B[Dataset 当前帧 t]
    B --> C[当前 RGB]
    B --> D[qpos/state]
    B --> E[动作 chunk t 到 t+31]
    B --> F[未来 RGB t+32 仅训练]
    B --> G[VTT 任务条件]
    C --> H[冻结 DINOv3-B]
    F --> I[冻结 DINOv3-B 未来目标]
    D --> J[Proprio 投影]
    E --> K[动作归一化与 Flow Matching]
    G --> L[任务条件投影]
    H --> M[当前视觉条件]
    I --> N[未来 patch target]
    J --> O[状态 token]
    L --> P[任务 token]
    K --> Q[noisy action]
    M --> R[Action DiT]
    O --> R
    P --> R
    Q --> R
    R --> S[动作速度场]
    R --> T[条件 token]
    T --> U[Future Feature Decoder]
    N --> V[未来特征 loss]
    S --> W[Flow Matching loss]
    S --> X[总 loss]
    V --> X
~~~

## 2. 固定数据和模型路径

~~~text
项目根目录：/home/lab347-no10/Tianyi/LiLa-WAM
数据集：    /home/lab347-no10/Tianyi/LiLa-WAM/data/yf10
VTT 条件：  /home/lab347-no10/Tianyi/LiLa-WAM/embeddings/yf10_vtt_dinov3b
DINOv3-B：  /home/lab347-no10/Tianyi/LiLa-WAM/pretrained/dinov3-vitb16
统计文件：  /home/lab347-no10/Tianyi/LiLa-WAM/artifacts/yf10/norm_stats.json
~~~

共同视觉配置：

~~~yaml
camera_names: ["head_camera"]
image_size: [320, 240]
feat_layers: [-7, -4, -2]
include_cls_register: true
fusion_mode: concat
~~~

共同 Action Expert：

~~~yaml
depth: 12
hidden_size: 512
num_heads: 8
vlm_adapter_num_queries: 64
adapter_depth: 4
~~~

DINOv3-B 在320×240图像上产生300个patch token，另外有1个CLS token和4个register token，所以完整视觉序列长度为305。

动作配置：

~~~yaml
action_chunk_size: 32
action_execution_horizon: 16
num_inference_steps: 10
~~~

## 3. Dataset 读取与样本构造

每个 episode 会被扫描并建立 frame index。一个训练样本围绕当前帧 t 构造：

~~~mermaid
flowchart LR
    A[episode hdf5] --> B[当前索引 t]
    B --> C[current RGB]
    B --> D[current qpos]
    B --> E[action t 到 t+31]
    B --> F[future RGB t+32]
    E --> G[action normalization]
    D --> H[state normalization]
    C --> I[pixel preprocessing]
    F --> J[pixel preprocessing]
    I --> K[batch]
    J --> K
    G --> K
    H --> K
    K --> L[VTT task condition]
~~~

典型张量：

| 张量 | 形状 | 用途 |
|---|---:|---|
| current pixel values | B×3×240×320 | 当前视觉 |
| future pixel values | B×3×240×320 | 未来监督 |
| qpos history | B×1×16 | 当前状态 |
| action chunk | B×32×14 | 目标动作 |
| task condition | B×768 | VTT任务条件 |
| 当前DINO多层特征 | 3个 B×305×768 | 视觉条件 |
| future patch target | B×300×768 | 未来视觉监督 |

未来图像只在训练阶段被读取。推理阶段只使用当前RGB、当前qpos和任务条件。

## 4. DINOv3 与 VTT 通路

~~~mermaid
flowchart TD
    A[current RGB] --> B[DINOv3-B]
    B --> C[layer -7 B×305×768]
    B --> D[layer -4 B×305×768]
    B --> E[layer -2 B×305×768]
    C --> F[逐层 LayerNorm]
    D --> F
    E --> F
    F --> G[Concat]
    G --> H[Linear projection]
    H --> I[fused visual B×305×768]
    I --> J[Visual Feature Adapter]
    J --> K[visual condition queries B×64×512]
    L[任务名称] --> M[VTT B×768]
    M --> N[task condition projection]
    N --> O[task token B×1×512]
~~~

DINOv3 backbone 是冻结的。训练的主要参数包括视觉 adapter、fusion projection、Action DiT、未来特征 decoder，以及启用时的 intent predictor 和 residual 分支。

## 5. Flow Matching 动作生成

设目标动作为 x1，随机噪声为 x0。训练时采样 tau：

~~~text
x_tau = (1 - tau) * x0 + tau * x1
target_velocity = x1 - x0
~~~

~~~mermaid
flowchart LR
    A[目标动作 x1 B×32×14] --> B[归一化]
    B --> C[随机噪声 x0]
    C --> D[采样 tau]
    B --> E[线性插值]
    D --> E
    C --> E
    E --> F[noisy action x_tau]
    F --> G[Action DiT]
    G --> H[predicted velocity]
    C --> I[target velocity]
    H --> J[MSE Flow Matching loss]
    I --> J
~~~

Action DiT 的 token 由以下部分组成：

~~~text
action tokens
+ proprio tokens
+ DiT register tokens
+ task condition token
+ DINO visual condition tokens
+ 可选 global intent token
+ 可选 spatial intent tokens
~~~

DiT 同时输出动作速度场和条件 token。条件 token送入 Future Feature Decoder，动作 token用于 Flow Matching loss。

## 6. Future Feature Prediction

训练时，未来图像经过同一个冻结的DINOv3得到 target layer=-2 的300个patch特征。当前图像的条件token经过 Future Feature Decoder，预测未来patch特征：

~~~mermaid
flowchart TD
    A[current RGB] --> B[current DINO features]
    C[future RGB] --> D[future DINO features]
    D --> E[去掉 CLS 和 register]
    E --> F[future target B×300×768]
    B --> G[Action DiT]
    G --> H[condition tokens]
    H --> I[Future Feature Decoder]
    I --> J[predicted future B×300×768]
    F --> K[Future Feature cosine loss]
    J --> K
~~~

真实 future target 不会在推理时进入动作模型。

## 7. Baseline 数据路径

~~~mermaid
flowchart TD
    A[current RGB] --> B[DINO多层融合]
    B --> C[Visual adapter]
    C --> D[visual tokens]
    E[qpos] --> F[proprio token]
    G[VTT] --> H[task token]
    I[noisy action] --> J[action token]
    D --> K[Action DiT]
    F --> K
    H --> K
    J --> K
    K --> L[action velocity]
~~~

baseline 使用配置：

~~~text
configs/yf10_small_dinov3b.yaml
~~~

baseline 不启用 global intent、spatial intent 或 spatial residual。

## 8. Global Intent 数据路径

Global Intent 对300个当前patch做全局平均，预测一个768维未来变化向量：

~~~mermaid
flowchart TD
    A[current patch B×300×768] --> B[global mean pooling]
    B --> C[current global B×768]
    D[qpos] --> E[state projection]
    F[VTT] --> G[task projection]
    H[horizon H=32] --> I[horizon embedding]
    C --> J[Global Intent Predictor]
    E --> J
    G --> J
    I --> J
    J --> K[predicted global intent B×768]
    K --> L[global intent loss]
    K --> M[optional token projection]
    M --> N[global token]
    N --> O[Action DiT]
~~~

G-aux只使用意图loss。G2同时把global token注入Action DiT。

当前G2 scratch epoch40在randomized下低于C32，因此Global Intent暂时保持为独立实验路线。

## 9. Spatial Intent 数据路径

固定2×4网格将300个patch变为8个区域：

~~~mermaid
flowchart TD
    A[current patch B×300×768] --> B[2×4 spatial pooling]
    B --> C[current regions B×8×768]
    D[qpos] --> E[state projection]
    F[VTT] --> G[task projection]
    H[horizon] --> I[horizon embedding]
    C --> J[Spatial Intent Predictor]
    E --> J
    G --> J
    I --> J
    J --> K[predicted spatial B×8×768]
    L[future patch target B×300×768] --> M[2×4 future pooling]
    M --> N[future-current region target]
    K --> O[spatial loss]
    N --> O
~~~

S-aux路径：

~~~text
Spatial Predictor → spatial loss → total loss
~~~

S2路径：

~~~text
Spatial Predictor → spatial_to_tokens → 8个token → Action DiT
~~~

## 10. S-residual 与 S-residual v2

旧版 S-residual：

~~~text
DINO patch → 残差注入 → MultiLayerConcatFusion + LayerNorm → Visual adapter
~~~

v2：

~~~text
DINO多层特征 → MultiLayerConcatFusion + LayerNorm
                 ↓
          spatial residual injection
                 ↓
           Visual adapter → Action DiT
~~~

空间预测结果 B×8×768 通过双线性插值恢复为20×15网格，再展平为B×300×768。只修改patch token，不修改CLS和register token。

v2关键配置：

~~~yaml
spatial_intent:
  enabled: true
  inject_into_dit: false
  inject_as_residual: true
  residual_injection: after_fusion
  region_grid: [2, 4]
  gate_init: 0.1
  gate_trainable: true
  residual_scale: 1.0
~~~

## 11. Gate 计算和精度

有效 gate 计算为：

~~~text
effective_gate = tanh(raw_gate)
~~~

空间残差注入为：

~~~text
fused_visual = fused_visual
             + effective_gate
             × residual_scale
             × spatial_residual
~~~

此前 gate 会被 bfloat16量化，导致日志长期显示约0.0996。当前 intent_gate 和 spatial_gate 保持float32，在注入时再转换为特征dtype。v2微调训练中 gate 从约0.099765下降到约0.063，说明 gate已经参与优化。

## 12. 总训练损失

~~~text
L_total =
L_flow_matching
+ lambda_future_feat * L_future_feature
+ lambda_intent * L_global_intent
+ lambda_spatial * L_spatial_intent
~~~

不同版本的loss和注入关系：

| 模型 | Flow Matching | Future Feature | Global Intent | Spatial Intent | 注入方式 |
|---|---|---|---|---|---|
| baseline | 有 | 有 | 无 | 无 | 无 |
| C32 | 有 | 有 | 无 | 无 | 无 |
| G-aux | 有 | 有 | 有 | 无 | 无 |
| G2 | 有 | 有 | 有 | 无 | global token |
| S-aux | 有 | 有 | 无 | 有 | 无 |
| S2 | 有 | 有 | 无 | 有 | spatial token |
| S-residual v2 | 有 | 有 | 无 | 有 | fusion后视觉残差 |

## 13. 推理数据路径

~~~mermaid
sequenceDiagram
    participant Env as RoboTwin
    participant Inf as Inference
    participant DINO as Frozen DINOv3
    participant Intent as Optional intent
    participant DiT as Action DiT
    participant Exec as Action execution

    Env->>Inf: 当前RGB qpos 任务名
    Inf->>DINO: 当前RGB
    DINO-->>Inf: 当前多层patch
    Inf->>Inf: 读取VTT任务条件
    Inf->>Intent: 当前patch qpos VTT H=32
    Intent-->>Inf: optional intent
    Inf->>DiT: random action + conditions
    loop Euler Flow Matching steps
        DiT-->>Inf: predicted velocity
        Inf->>Inf: update x_t
    end
    Inf->>Inf: denormalize and smooth
    Inf->>Exec: action chunk前16步
    Exec-->>Env: execute action
~~~

推理阶段不读取未来图像，不计算 future feature loss。

## 14. 当前未解决问题

1. v2微调版 clean提升明显，randomized提升较小；
2. v2 scratch50仍在训练，需要epoch40测试；
3. 固定2×4区域会混合目标物体、机械臂和背景；
4. 未来DINO差分包含部分不由动作决定的随机化变化；
5. 单一scalar gate不能按区域选择性抑制；
6. Global Intent scratch在randomized下降；
7. 当前正式结果主要是seed0、每任务50次，仍需seed1和seed2验证。

## 15. 下一步实验

1. 完成v2 scratch50，比较epoch20、30、40、50；
2. 以epoch40和C32做严格比较；
3. v2 scratch提升约2个百分点以上后，再做seed1和seed2；
4. 如果固定网格不稳定，先改为8个region gate；
5. 有clean/randomized严格配对样本后，再加空间一致性loss；
6. 如果固定网格仍无效，开发task-conditioned interaction query；
7. Global Intent暂时不和Spatial Residual同时启用。

## 16. 常用命令

快速微调：

~~~bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config 对应的ft4配置.yaml \
  --norm_stats_path ./artifacts/yf10/norm_stats.json \
  --init_from baseline_epoch40_checkpoint.pt \
  --save_dir 对应输出目录 \
  2>&1 | tee 对应日志路径
~~~

scratch训练：

~~~bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config 对应的scratch50配置.yaml \
  --norm_stats_path ./artifacts/yf10/norm_stats.json \
  --save_dir 对应输出目录 \
  2>&1 | tee 对应日志路径
~~~

统一测试：

~~~bash
export CKPT_SETTING="模型目录/时间戳目录"
export CHECKPOINT_EP=40
export CONFIG_PATH="对应配置文件.yaml"
export NORM_STATS_PATH="artifacts/yf10/norm_stats.json"
export INSTRUCTION_TYPE=unseen
export SEED=0
export TEST_NUM=50
export NUM_WORKERS=1

for CFG in demo_clean demo_randomized; do
  CUDA_VISIBLE_DEVICES=0 \
  CKPT_SETTING="$CKPT_SETTING" \
  CHECKPOINT_EP="$CHECKPOINT_EP" \
  TASK_CONFIG="$CFG" \
  CONFIG_PATH="$CONFIG_PATH" \
  NORM_STATS_PATH="$NORM_STATS_PATH" \
  INSTRUCTION_TYPE="$INSTRUCTION_TYPE" \
  SEED="$SEED" \
  TEST_NUM="$TEST_NUM" \
  NUM_WORKERS="$NUM_WORKERS" \
  RUN_ID="model_ep$CHECKPOINT_EP-$CFG-seed0-n50" \
  bash eval_yf10.sh
done
~~~


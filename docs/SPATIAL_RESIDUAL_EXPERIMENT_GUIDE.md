# 空间意图后续实验指南

## 当前实验结论

固定网格空间意图实验没有超过 C32 参照模型：

| 模型 | clean | randomized | 总体 |
|---|---:|---:|---:|
| C32 参照模型 | 75.2% | 63.0% | 69.1% |
| S-aux | 74.6% | 63.6% | 69.1% |
| S2 token 注入 | 74.0% | 63.4% | 68.7% |

空间目标并不是零，但预测范数明显小于目标范数。S2 的标量 gate 也基本保持在
初始化值附近。这说明“追加 8 个空间 token”可能不是有效的条件注入路径，
而不能说明空间信息本身没有价值。

## 下一项实验：S-residual

S-residual 保留原有的因果预测器和 2×4 空间目标，只修改空间信息的注入路径：

```text
当前 DINO patch
        + 预测得到的 2×4 未来残差，上采样到 20×15 个 patch
        -> 原有 DINO adapter -> Action DiT
```

只修改目标 `-2` 层的 DINO patch 特征，CLS token 和 register token 保持不变。
这样可以避免新增 token 被 DiT 忽略，同时保留原始视觉特征的空间布局。

配置文件：

```text
configs/yf10_dinov3b_h512_spatial_sres_c32_ft4.yaml
```

S-residual 从同一个 epoch40 C32 baseline 开始微调 4 个 epoch，不从 S-aux 或
S2 初始化，从而保持实验对比条件一致。

## 训练命令

```bash
cd /home/lab347-no10/Tianyi/LiLa-WAM

export BASE_CKPT="/home/lab347-no10/Tianyi/LiLa-WAM/checkpoints_vla/yf10_dinov3b_small/sft_2026-09-20_15-07-28/checkpoint_epoch_40.pt"
export NORM_STATS="/home/lab347-no10/Tianyi/LiLa-WAM/artifacts/yf10/norm_stats.json"

mkdir -p log/phase4_spatial_sres_c32
set -o pipefail

CUDA_VISIBLE_DEVICES=0 python train.py \
  --config ./configs/yf10_dinov3b_h512_spatial_sres_c32_ft4.yaml \
  --norm_stats_path "$NORM_STATS" \
  --init_from "$BASE_CKPT" \
  --save_dir ./checkpoints_vla/yf10_phase4_spatial_sres_c32 \
  2>&1 | tee ./log/phase4_spatial_sres_c32/train.log
```

训练时重点观察以下日志字段：

```text
Spatial
SpatialTargetNorm
SpatialPredNorm
SpatialGate
```

残差路径使用 gate 作为较小的初始残差系数（`gate_init=0.1`），同时保持
baseline 的视觉 token 布局和 DiT token 数量不变。

## 测试流程

训练完成后，将实际生成的时间戳填入：

```bash
export CKPT_SETTING="yf10_phase4_spatial_sres_c32/sft_YYYY-MM-DD_HH-MM-SS"
export CHECKPOINT_EP=4
export CONFIG_PATH="configs/yf10_dinov3b_h512_spatial_sres_c32_ft4.yaml"
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
  RUN_ID="spatial_sres_ep4_${CFG}_seed0_n50" \
  bash eval_yf10.sh
done
```

## 实验判断标准

1. 如果 S-residual 在 clean 和 randomized 上都提升约 2 个百分点以上，再运行
   seed1 和 seed2，并保留这条研究路线；
2. 如果它只在一类环境上提升，应先检查逐任务结果，不要立即增加模型规模或
   训练轮数；
3. 如果它仍然与 C32 持平，应停止固定网格空间意图路线，转向任务条件化的
   interaction query，而不是把相同模型继续训练 40 轮。

下一版 interaction query 应该从当前 DINO 特征、任务条件和 qpos 中预测少量
以物体或接触关系为中心的 token，而不是把所有未来变化分配到固定图像网格中。

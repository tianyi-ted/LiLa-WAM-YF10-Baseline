# LiLa-WAM YF10 模型版本与命令总览

## 1. 固定路径与共同结构

~~~text
项目根目录：/home/lab347-no10/Tianyi/LiLa-WAM
数据集：    /home/lab347-no10/Tianyi/LiLa-WAM/data/yf10
VTT 条件：  /home/lab347-no10/Tianyi/LiLa-WAM/embeddings/yf10_vtt_dinov3b
DINOv3-B：  /home/lab347-no10/Tianyi/LiLa-WAM/pretrained/dinov3-vitb16
统计文件：  /home/lab347-no10/Tianyi/LiLa-WAM/artifacts/yf10/norm_stats.json
~~~

共同视觉配置为 head_camera、320×240、feat_layers=[-7,-4,-2]、include_cls_register=true、fusion_mode=concat。DINOv3-B产生300个patch token、1个CLS token和4个register token。共同Action Expert为depth=12、hidden_size=512、num_heads=8、adapter queries=64、adapter depth=4。动作chunk为32，推理执行前16步。

## 2. 结果比较口径

C32不是最开始的baseline：

| 模型 | 训练方式 | Clean | Randomized | 总体 |
|---|---|---:|---:|---:|
| 原始baseline epoch40 | scratch | 70.2% | 64.6% | 67.4% |
| C32 | baseline+4轮微调 | 75.2% | 63.0% | 69.1% |
| G-aux | baseline+4轮微调 | 72.8% | 62.8% | 67.8% |
| G2 | baseline+4轮微调 | 73.2% | 63.8% | 68.5% |
| S-aux | baseline+4轮微调 | 74.6% | 63.6% | 69.1% |
| S2 | baseline+4轮微调 | 74.0% | 63.4% | 68.7% |
| S-residual | baseline+4轮微调 | 75.8% | 64.8% | 70.3% |
| G2 scratch epoch40 | scratch50 scheduler | 75.4% | 60.2% | 67.8% |
| S-residual scratch epoch40 | scratch50 scheduler | 72.4% | 61.0% | 66.7% |

微调4轮用于快速消融；正式结构比较使用相同的scratch50 scheduler并比较epoch40。

## 3. Baseline与C32

原始baseline配置：

~~~text
configs/yf10_small_dinov3b.yaml
checkpoints_vla/yf10_dinov3b_small/sft_2026-09-20_15-07-28/checkpoint_epoch_40.pt
~~~

从头训练：

~~~bash
cd /home/lab347-no10/Tianyi/LiLa-WAM
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config ./configs/yf10_small_dinov3b.yaml \
  --norm_stats_path ./artifacts/yf10/norm_stats.json \
  --save_dir ./checkpoints_vla/yf10_dinov3b_small \
  2>&1 | tee ./log/yf10_small_dinov3b/train.log
~~~

C32配置为 configs/yf10_dinov3b_h512_c32_ft4.yaml，从baseline epoch40微调4轮：

~~~bash
BASE_CKPT=/home/lab347-no10/Tianyi/LiLa-WAM/checkpoints_vla/yf10_dinov3b_small/sft_2026-09-20_15-07-28/checkpoint_epoch_40.pt
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config ./configs/yf10_dinov3b_h512_c32_ft4.yaml \
  --norm_stats_path ./artifacts/yf10/norm_stats.json \
  --init_from "$BASE_CKPT" \
  --save_dir ./checkpoints_vla/yf10_phase2_c32 \
  2>&1 | tee ./log/phase2_c32/train.log
~~~

F16跨度消融配置为 configs/yf10_dinov3b_h512_f16_ft4.yaml，C32使用future offset=32。

## 4. Global Intent系列

| 配置 | 结构 |
|---|---|
| yf10_dinov3b_h512_gaux_c32_ft4.yaml | 全局意图辅助监督，不注入DiT |
| yf10_dinov3b_h512_g2_c32_ft4.yaml | 全局意图预测并注入DiT |
| yf10_dinov3b_h512_g2_c32_gate01_ft4.yaml | 可训练gate，初值0.1 |
| yf10_dinov3b_h512_g2_c32_fixed01_ft4.yaml | 固定gate=0.1 |
| yf10_dinov3b_h512_g2_c32_ft40.yaml | scratch50 scheduler，比较epoch40 |

G2 scratch训练：

~~~bash
cd /home/lab347-no10/Tianyi/LiLa-WAM
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config ./configs/yf10_dinov3b_h512_g2_c32_ft40.yaml \
  --norm_stats_path ./artifacts/yf10/norm_stats.json \
  --save_dir ./checkpoints_vla/yf10_phase4_g2_c32_scratch \
  2>&1 | tee ./log/phase4_g2_c32_scratch/train.log
~~~

## 5. Spatial Intent系列

固定网格把300个patch划分为2×4共8个区域：

| 配置 | 结构 |
|---|---|
| yf10_dinov3b_h512_spatial_saux_c32_ft4.yaml | 只计算空间辅助loss |
| yf10_dinov3b_h512_spatial_s2_c32_ft4.yaml | 8个意图token追加到DiT |
| yf10_dinov3b_h512_spatial_sres_c32_ft4.yaml | fusion前加入空间残差 |
| yf10_dinov3b_h512_spatial_sres_v2_c32_ft4.yaml | fusion后加入空间残差 |

v2路径：

~~~text
DINO多层特征 → MultiLayerConcatFusion → 残差注入 → Visual adapter → Action DiT
~~~

关键配置：

~~~yaml
spatial_intent:
  enabled: true
  inject_into_dit: false
  inject_as_residual: true
  residual_injection: "after_fusion"
  region_grid: [2, 4]
  gate_init: 0.1
  gate_trainable: true
  residual_scale: 1.0
~~~

v2微调：

~~~bash
cd /home/lab347-no10/Tianyi/LiLa-WAM
BASE_CKPT=/home/lab347-no10/Tianyi/LiLa-WAM/checkpoints_vla/yf10_dinov3b_small/sft_2026-09-20_15-07-28/checkpoint_epoch_40.pt
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config ./configs/yf10_dinov3b_h512_spatial_sres_v2_c32_ft4.yaml \
  --norm_stats_path ./artifacts/yf10/norm_stats.json \
  --init_from "$BASE_CKPT" \
  --save_dir ./checkpoints_vla/yf10_phase4_spatial_sres_v2_ft4_fp32gate \
  2>&1 | tee ./log/phase4_spatial_sres_v2_ft4_fp32gate/train.log
~~~

v2 scratch50：

~~~bash
cd /home/lab347-no10/Tianyi/LiLa-WAM
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config ./configs/yf10_dinov3b_h512_spatial_sres_v2_c32_scratch50.yaml \
  --norm_stats_path ./artifacts/yf10/norm_stats.json \
  --save_dir ./checkpoints_vla/yf10_phase4_spatial_sres_v2_scratch50_fp32gate \
  2>&1 | tee ./log/phase4_spatial_sres_v2_scratch50_fp32gate/train.log
~~~

scratch训练不使用 --init_from 或 --resume，正式比较取epoch40。

## 6. 统一测试命令

十个任务固定为：

~~~text
adjust_bottle grab_roller hanging_mug move_stapler_pad open_microwave
press_stapler scan_object stack_blocks_two stamp_seal turn_switch
~~~

~~~bash
cd /home/lab347-no10/Tianyi/LiLa-WAM
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
  RUN_ID="model_ep${CHECKPOINT_EP}_${CFG}_seed0_n50" \
  bash eval_yf10.sh
done
~~~

结果位于 eval_result/yf10/运行ID/summary.json。测试时优先使用训练目录中的 config.yaml。

## 7. gate修复与匹配规则

此前bfloat16会量化gate在0.1附近的小更新，日志长期显示约0.0996。现在intent_gate和spatial_gate保持float32，在注入时再转为特征dtype。修复后v2微调日志中的gate从约0.099765下降到约0.063，说明gate已经参与优化。

baseline和C32没有gate，不需要因为修复重训。S-aux不把gate用于动作注入，也不需要重训。S2、旧版S-residual和G2如果要研究可学习gate，应使用修复后的代码重训。v2 S-residual应使用修复后的checkpoint。

~~~text
baseline checkpoint      → yf10_small_dinov3b.yaml
C32 checkpoint           → yf10_dinov3b_h512_c32_ft4.yaml
G2 checkpoint            → G2对应YAML
S2 checkpoint            → spatial_s2对应YAML
S-residual v2 checkpoint → spatial_sres_v2对应YAML
~~~

每个训练目录中的config.yaml是最可靠的历史配置记录。baseline配置没有global intent、spatial intent或residual injection时，不会创建或使用这些模块。

## 8. 推荐实验流程

~~~text
快速筛选：baseline epoch40 → init_from + 4轮 → seed0快速测试
正式比较：从随机初始化开始 → 相同50轮scheduler → 比较epoch40 → seed0/1/2
~~~

推荐顺序：

1. 完成gate修复后的S-residual v2微调测试；
2. 如果v2超过C32，再训练v2 scratch50；
3. scratch版本与原始baseline epoch40、C32、G2 scratch40对比；
4. 只有提升约2个百分点以上时，才增加seed1和seed2；
5. 如果固定网格仍不稳定，再开发task-conditioned interaction query；
6. Global intent暂时作为独立路线，不与spatial residual同时启用。

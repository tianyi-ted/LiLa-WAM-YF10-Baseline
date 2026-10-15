# LiLa-WAM / RoboTwin VLA 研究上下文摘要

更新时间：2026-10-12（Asia/Shanghai）

本文档用于新对话接续当前研究。内容明确区分已经完成的工作、当前正在进行的工作和计划中的工作。新对话开始后应优先阅读本文档，再检查项目中的实际文件和最新日志。

## 1. 研究目标

当前研究基于 LiLa-WAM，在 RoboTwin YF10 十任务数据集上研究 DINOv3 + Flow Matching Action DiT 的视觉语言动作策略。

主要目标：

1. 建立可复现的 DINOv3-B + h512 LiLa-WAM baseline；
2. 研究 future visual feature prediction 对动作策略的作用；
3. 研究 global intent 和 spatial intent 是否能提高动作成功率；
4. 解决 clean 场景和 randomized 场景之间的泛化差异；
5. 最终形成可以在 RoboTwin 仿真平台和 Edulite A3 真机上验证的 VLA 研究路线；
6. 论文阶段优先关注空间表示、交互区域选择、未来视觉表征和动作条件注入。

当前不再把 OPD 作为近期主线。OPD 计算和显存开销较大，应作为后续独立方向。

## 2. 项目和数据固定路径

项目根目录：

~~~text
/home/lab347-no10/Tianyi/LiLa-WAM
~~~

YF10 数据集：

~~~text
/home/lab347-no10/Tianyi/LiLa-WAM/data/yf10
~~~

任务条件 VTT：

~~~text
/home/lab347-no10/Tianyi/LiLa-WAM/embeddings/yf10_vtt_dinov3b
~~~

DINOv3-B checkpoint：

~~~text
/home/lab347-no10/Tianyi/LiLa-WAM/pretrained/dinov3-vitb16
~~~

动作和状态归一化统计：

~~~text
/home/lab347-no10/Tianyi/LiLa-WAM/artifacts/yf10/norm_stats.json
~~~

YF10 任务：

~~~text
adjust_bottle
grab_roller
hanging_mug
move_stapler_pad
open_microwave
press_stapler
scan_object
stack_blocks_two
stamp_seal
turn_switch
~~~

每个任务通常测试50个 episode，当前常用测试协议为：

~~~text
task_config: demo_clean / demo_randomized
instruction_type: unseen
seed: 0
test_num: 50
num_workers: 1
~~~

## 3. 共同基础网络

当前主要模型使用：

~~~yaml
vision_encoder:
  checkpoint_path: /home/lab347-no10/Tianyi/LiLa-WAM/pretrained/dinov3-vitb16
  feat_layers: [-7, -4, -2]
  include_cls_register: true
  fusion_mode: concat

action_expert:
  depth: 12
  hidden_size: 512
  num_heads: 8
  vlm_adapter_num_queries: 64
  adapter_depth: 4
~~~

图像配置：

~~~yaml
camera_names: ["head_camera"]
image_size: [320, 240]
~~~

DINOv3-B 在该图像尺寸下有20×15=300个 patch token，另外有1个 CLS token和4个 register token。完整视觉序列长度为305。

动作生成：

~~~yaml
action_chunk_size: 32
action_execution_horizon: 16
num_inference_steps: 10
~~~

模型使用 Flow Matching 预测32步动作 chunk，推理时通常执行前16步。

## 4. 已完成的 baseline 和参照版本

### 4.1 原始 DINOv3-B+h512 baseline

配置：

~~~text
configs/yf10_small_dinov3b.yaml
~~~

关键训练设置：

~~~yaml
batch_size: 96
learning_rate: 2e-4
lr_min: 5e-5
weight_decay: 1e-2
epochs: 50
lambda_future_feat: 0.5
~~~

原始 epoch40 checkpoint：

~~~text
checkpoints_vla/yf10_dinov3b_small/sft_2026-09-20_15-07-28/checkpoint_epoch_40.pt
~~~

测试结果：

~~~text
clean:       351/500 = 70.2%
randomized:  323/500 = 64.6%
overall:     674/1000 = 67.4%
~~~

### 4.2 C32 参照版本

配置：

~~~text
configs/yf10_dinov3b_h512_c32_ft4.yaml
~~~

训练方式：从原始 baseline epoch40 使用 init_from 继续训练4轮，future_frame_offset=32。

C32 是 baseline 微调版本，不是原始 baseline。

测试结果：

~~~text
clean:       376/500 = 75.2%
randomized:  315/500 = 63.0%
overall:     691/1000 = 69.1%
~~~

C32 是后续快速消融的主要参照。

### 4.3 F16

配置：

~~~text
configs/yf10_dinov3b_h512_f16_ft4.yaml
~~~

future_frame_offset=16。该版本没有表现出优于C32的稳定优势，因此当前正式路线保留 future offset=32。

## 5. Global Intent 实验

### 5.1 G-aux

配置：

~~~text
configs/yf10_dinov3b_h512_gaux_c32_ft4.yaml
~~~

结构：

~~~text
当前 DINO patch + qpos + VTT
          ↓
全局未来意图预测器
          ↓
全局意图辅助 loss
~~~

全局意图不注入 Action DiT。

结果：

~~~text
clean:       364/500 = 72.8%
randomized:  314/500 = 62.8%
overall:     678/1000 = 67.8%
~~~

### 5.2 G2

配置：

~~~text
configs/yf10_dinov3b_h512_g2_c32_ft4.yaml
~~~

G2 把全局意图编码为一个 token 注入 Action DiT。

结果：

~~~text
clean:       366/500 = 73.2%
randomized:  319/500 = 63.8%
overall:     685/1000 = 68.5%
~~~

其他 gate 消融：

~~~text
configs/yf10_dinov3b_h512_g2_c32_gate01_ft4.yaml
configs/yf10_dinov3b_h512_g2_c32_fixed01_ft4.yaml
~~~

### 5.3 G2 scratch50 scheduler

配置：

~~~text
configs/yf10_dinov3b_h512_g2_c32_ft40.yaml
~~~

实际训练使用50轮 scheduler，正式比较取 epoch40。训练目录：

~~~text
checkpoints_vla/yf10_phase4_g2_c32_scratch/sft_2026-10-06_13-13-48/
~~~

epoch40测试结果：

~~~text
clean:       377/500 = 75.4%
randomized:  301/500 = 60.2%
overall:     678/1000 = 67.8%
~~~

结论：从头训练的 G2 在 clean 上接近C32，但 randomized 明显下降。当前不建议把 global intent 与 spatial residual 直接组合。

## 6. Spatial Intent 实验

空间意图将300个patch固定划分为2×4共8个区域，预测未来空间特征变化。

### 6.1 S-aux

配置：

~~~text
configs/yf10_dinov3b_h512_spatial_saux_c32_ft4.yaml
~~~

只进行空间意图辅助监督，不将空间 token 注入DiT。

结果：

~~~text
clean:       373/500 = 74.6%
randomized:  318/500 = 63.6%
overall:     691/1000 = 69.1%
~~~

### 6.2 S2

配置：

~~~text
configs/yf10_dinov3b_h512_spatial_s2_c32_ft4.yaml
~~~

将8个空间意图区域编码为额外token，追加到 Action DiT。

结果：

~~~text
clean:       370/500 = 74.0%
randomized:  317/500 = 63.4%
overall:     687/1000 = 68.7%
~~~

结论：追加空间token没有带来稳定提升。

### 6.3 旧版 S-residual

配置：

~~~text
configs/yf10_dinov3b_h512_spatial_sres_c32_ft4.yaml
configs/yf10_dinov3b_h512_spatial_sres_c32_scratch50.yaml
~~~

旧版先在DINO多层fusion之前将预测空间残差加入patch：

~~~text
DINO patch
    ↓
空间残差注入
    ↓
MultiLayerConcatFusion + LayerNorm
    ↓
Visual adapter
~~~

旧版微调4轮结果：

~~~text
clean:       379/500 = 75.8%
randomized:  324/500 = 64.8%
overall:     703/1000 = 70.3%
~~~

旧版 scratch epoch40结果：

~~~text
clean:       362/500 = 72.4%
randomized:  305/500 = 61.0%
overall:     667/1000 = 66.7%
~~~

因此旧版微调提升不能直接说明旧版结构在scratch条件下有效。

## 7. S-residual v2

### 7.1 结构调整

新版配置：

~~~text
configs/yf10_dinov3b_h512_spatial_sres_v2_c32_ft4.yaml
configs/yf10_dinov3b_h512_spatial_sres_v2_c32_scratch50.yaml
~~~

新版把残差注入位置移动到多层DINO fusion之后：

~~~text
DINO多层特征
    ↓
MultiLayerConcatFusion
    ↓
空间残差注入
    ↓
Visual adapter
    ↓
Action DiT
~~~

配置关键项：

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

### 7.2 Gate float32 修复

此前 action model 整体转换为 bfloat16，spatial_gate 也被转为 bfloat16，导致0.1附近的小更新被量化，日志长期显示约0.0996。

当前代码已经修改：

- intent_gate 保持 float32；
- spatial_gate 保持 float32；
- 注入特征时再转换到特征 dtype；
- 日志 gate 显示精度提高到6位小数。

修改文件：

~~~text
models/vla_model_fm.py
models/model_runner.py
train.py
robotwin_infer.py
~~~

Gate 修复后 v2 微调日志：

~~~text
初始 SpatialGate ≈ 0.099765
训练后 SpatialGate ≈ 0.06318
~~~

说明 gate 已经真正参与优化。

### 7.3 v2 微调结果

训练目录：

~~~text
checkpoints_vla/yf10_phase4_spatial_sres_v2_ft4_fp32gate/sft_2026-10-11_13-44-31/
~~~

配置：

~~~text
configs/yf10_dinov3b_h512_spatial_sres_v2_c32_ft4.yaml
~~~

结果：

~~~text
clean:       386/500 = 77.2%
randomized:  319/500 = 63.8%
overall:     705/1000 = 70.5%
~~~

相对于C32：

~~~text
clean:       +2.0个百分点
randomized:  +0.8个百分点
overall:     +1.4个百分点
~~~

v2在clean提升明显，但randomized提升较小。相对于旧版 S-residual 微调的randomized 64.8%，v2为63.8%，下降1个百分点。

逐任务观察表明，v2主要改善 open_microwave、turn_switch 和部分空间交互任务，但 move_stapler_pad 在 randomized 下下降。

## 8. 当前正在进行的工作

### 8.1 v2 scratch50

配置：

~~~text
configs/yf10_dinov3b_h512_spatial_sres_v2_c32_scratch50.yaml
~~~

训练输出：

~~~text
checkpoints_vla/yf10_phase4_spatial_sres_v2_scratch50_fp32gate/
log/phase4_spatial_sres_v2_scratch50_fp32gate/train.log
~~~

截至当前摘要更新时间，训练大约进行到epoch11，尚未完成50轮。

epoch11附近日志：

~~~text
LR ≈ 1.85e-4
SpatialTargetNorm ≈ 9.6～10.6
SpatialPredNorm ≈ 4.8～5.7
SpatialGate ≈ 0.16
~~~

Gate从0.1上升到约0.16，说明 scratch 模型正在更积极使用空间残差。当前不能根据epoch11判断最终效果，应至少训练到epoch40。

## 9. 未解决问题

1. v2微调版clean提升明显，但randomized提升有限；
2. 固定2×4网格可能把背景、光照、机械臂和目标物体混在同一区域；
3. 空间未来特征包含部分不由动作决定的随机化变化；
4. 单一scalar gate不能对不同区域进行选择性抑制；
5. 当前 spatial target 仍然是DINO未来特征差分，不是明确的物体位置、接触点或动作相关位移；
6. v2 scratch50尚未完成，不能确认v2结构是否在scratch协议下有效；
7. Global intent scratch在randomized下表现下降，暂时不应与spatial residual直接组合；
8. S-aux、S2和旧版S-residual使用旧 gate 实现，正式研究 gate 时需要重新训练；
9. 当前正式结论仍主要基于seed0和每任务50次测试，统计稳定性不足。

## 10. 计划中的后续实验

### 阶段A：完成v2 scratch50

训练50轮，重点比较epoch20、30、40、50。正式主比较取epoch40：

~~~text
原始baseline epoch40
C32
G2 scratch epoch40
S-residual v2 scratch epoch40
~~~

只有 v2 scratch 相对于C32提升约2个百分点以上，才继续seed1和seed2。

### 阶段B：Region Gate

把当前一个scalar gate改为8个region gate：

~~~text
gate[0] ... gate[7]
~~~

让模型对目标物体区域保留残差，对背景和随机光照区域抑制残差。此实验应先使用init_from+4轮快速筛选，再决定是否scratch50。

### 阶段C：随机化一致性 loss

如果有clean/randomized的严格配对样本，可以约束两种视觉条件下预测的空间意图一致：

~~~text
L_total =
L_action
+ lambda_future * L_future
+ lambda_spatial * L_spatial
+ lambda_consistency * L_consistency
~~~

初始候选：

~~~yaml
lambda_spatial_intent: 0.1
lambda_spatial_consistency: 0.05
~~~

如果没有严格配对样本，不应直接使用该loss。

### 阶段D：Interaction Query

如果固定2×4网格仍然不稳定，改用任务条件化interaction query：

~~~text
当前DINO patch + VTT + qpos
          ↓
4个或少量 interaction queries
          ↓
目标物体、抓取点、接触区域、放置区域
          ↓
未来交互残差
          ↓
fusion后的视觉条件
~~~

Interaction Query 应在 v2 scratch和Region Gate验证后引入，避免同时修改多个因素。

### Global Intent 的后续位置

Global Intent暂时作为独立路线保留。只有当：

- v2 spatial residual 已经稳定；
- randomized成功率提高；
- 需要表示任务阶段或全局运动方向；

才考虑独立的 hierarchical global + spatial 结构，并使用两个独立gate和两个独立loss。

## 11. 常用训练命令

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

scratch训练不能使用 --init_from 或 --resume。

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

测试输出：

~~~text
eval_result/yf10/运行ID/summary.json
~~~

## 17. 2026-10-13 之后的最新结果

### 17.1 S-residual v2 微调版

配置：

~~~text
configs/yf10_dinov3b_h512_spatial_sres_v2_c32_ft4.yaml
~~~

gate float32 修复后的 checkpoint：

~~~text
checkpoints_vla/yf10_phase4_spatial_sres_v2_ft4_fp32gate/sft_2026-10-11_13-44-31/checkpoint_epoch_4.pt
~~~

测试结果：

~~~text
clean:       386/500 = 77.2%
randomized:  319/500 = 63.8%
overall:     705/1000 = 70.5%
~~~

相比 C32 的69.1%，总体提升1.4个百分点；clean提升明显，randomized提升较小。训练结束时 gate 从约0.099765下降到约0.063，说明 gate 已经可以正常更新。

### 17.2 S-residual v2 scratch epoch40

配置：

~~~text
configs/yf10_dinov3b_h512_spatial_sres_v2_c32_scratch50.yaml
~~~

测试结果：

~~~text
clean:       354/500 = 70.8%
randomized:  301/500 = 60.2%
overall:     655/1000 = 65.5%
~~~

该结果低于原始 baseline epoch40和C32。epoch40 checkpoint中的有效 gate约为0.236，说明从随机初始化训练时空间残差过早或过强地干扰了动作策略。不能把微调4轮的70.5%直接解释为S-residual结构在scratch条件下有效。

## 18. Gate实现的精确定义

当前代码中有三个不同量：

1. raw learnable gate：模型中被优化器更新的 float32 参数 spatial_gate；
2. tanh raw gate：tanh(spatial_gate)，范围约为[-1,1]；
3. effective gate：真正乘到空间残差上的系数。

v3配置下有效gate为：

~~~text
effective_gate = max_gate × tanh(raw_gate) × schedule(epoch)
~~~

实际注入为：

~~~text
fused_visual += effective_gate × residual_scale × spatial_residual
~~~

max_gate用于限制有效残差强度，不代表 raw gate 的上限。warmup_epochs控制前期完全关闭残差，ramp_epochs控制残差从0逐渐增加到完整强度所需要的epoch数。

当 max_gate=0.1、gate_init=0.0 时，raw gate从0开始学习；warmup结束后不会自动跳到0.1。gate_init是初始化目标，schedule另外控制实际注入时间。

当 gate_init=max_gate=0.1 时，gate_ratio接近1，raw gate会接近atanh(0.999)，处于tanh饱和区。因此四轮v3微调配置已改为 gate_init=0.05、max_gate=0.1；scratch50配置保留 gate_init=0.0、max_gate=0.1。

## 19. v3 Warm-up 配置

新增配置：

~~~text
configs/yf10_dinov3b_h512_spatial_sres_v3_warmup_ft4.yaml
configs/yf10_dinov3b_h512_spatial_sres_v3_warmup_scratch50.yaml
~~~

scratch50配置：

~~~yaml
gate_init: 0.0
max_gate: 0.1
warmup_epochs: 5
ramp_epochs: 10
~~~

训练过程：

~~~text
epoch 1～5：残差完全关闭，但空间预测器可通过spatial loss训练
epoch 6～15：残差注入从0逐渐增加
epoch 16以后：effective gate不超过0.1
~~~

ft4配置：

~~~yaml
gate_init: 0.05
max_gate: 0.1
warmup_epochs: 1
ramp_epochs: 2
~~~

四轮配置不能把 warmup 设置过长，否则整个实验结束时残差还没有真正参与动作生成。

## 20. 当前正确的实验诊断顺序

当前不能直接跳到 region gate 或 interaction query，应按下面顺序执行。

### 20.1 Residual-off 推理消融

使用同一个 v2/v3 epoch40 checkpoint，把测试配置中的 residual_scale 设为0，只重新推理，不重新训练。

判断：

- off明显恢复：残差注入正在干扰动作策略；
- on/off基本相同：策略没有真正使用残差，或scratch模型本身已经较差；
- off更差：残差有帮助，但scalar gate或区域分配不稳定。

### 20.2 梯度和实际残差比例诊断

需要记录 raw gate、tanh(raw gate)、schedule、effective gate、残差范数、融合视觉特征范数、residual ratio、spatial predictor梯度和gate梯度。仅观察SpatialGate不足以判断实际扰动强度。

### 20.3 Stop-gradient 残差

将注入Action DiT的空间残差detach，使 spatial predictor只由spatial loss训练，避免action loss和spatial loss之间的梯度冲突。先用 baseline epoch40微调4轮筛选，再决定是否scratch50。

### 20.4 S-aux-shared

当前S-aux的空间预测器是独立head，spatial loss主要只更新该head，不能证明空间监督改善了动作策略。S-aux-shared需要让spatial loss回传到Action DiT使用的共享视觉表示，测试未来空间监督是否能改善动作条件。

### 20.5 Region Gate

只有在 residual-on 比 residual-off有效、但不同任务或区域表现不一致时进行。第一版使用8个静态region gate，除gate数量外保持其他参数相同。

### 20.6 Task-conditioned Interaction Query

如果 residual-off、stop-gradient、S-aux-shared和region gate均无效，再停止固定2×4网格，转向由当前DINO、VTT和qpos动态选择交互区域的interaction query。

## 21. Stop-gradient 与 S-aux-shared 的区别

两者目的不同：

~~~text
Stop-gradient：保留空间残差注入，切断action loss到spatial predictor的梯度，诊断梯度冲突。

S-aux-shared：不要求直接注入预测残差，让spatial loss更新共享视觉表示，诊断空间监督是否能改善动作表征。
~~~

Stop-gradient关注优化路径；S-aux-shared关注表示学习路径。两者不能互相替代。

## 22. 当前未解决问题与下一步

已知问题：

1. v2微调版clean提升明显，但randomized提升有限；
2. v2 scratch版clean和randomized都低于C32；
3. 固定网格将物体、机械臂、背景和随机化因素混在一起；
4. global intent scratch在randomized下下降；
5. 当前S-aux独立head不改变Action DiT动作路径；
6. 现有正式结果主要是seed0、每任务50次，统计稳定性仍不足。

下一步建议：

1. 完成v3 warm-up ft4快速筛选；
2. 对已训练v2/v3 checkpoint做residual-off测试；
3. 做stop-gradient快速消融；
4. 实现S-aux-shared并进行4轮快速验证；
5. 只有空间残差确定有用但区域差异明显时才做region gate；
6. 固定网格路线失败后再开发interaction query；
7. 暂时不把Global Intent与Spatial Residual同时启用。


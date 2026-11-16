# MetaWorld：任务条件多 Query 迟汇聚预测器实验

日期：2026-11-11。状态：代码、独立配置和CPU管线验证已完成；正式50轮训练及闭环成功率未执行。

2026-11-12补充：原4query无KD/有KD实验均已完成，epoch40 cold分别为89.20%/89.08%。新增随机初始化4query action-only控制，配置与命令见第8节；该新控制尚未正式训练。以上首行状态为实现当日记录。

## 1. 本轮只改变什么

用可选 `pooling: task_attention_late` 替换学生的单query池化器。

```text
当前DINO patch [B,196,768]
             ↓ 投影
当前视觉memory [B,196,256] ← 2D位置只加入读取keys
             ↑ cross-attention
VTT＋当前state → 条件向量＋4个不同slot embedding [B,4,256]
             ↓ 残差读取
4个摘要 → 1层pre-norm self-attention＋FFN → 4个交互摘要
             ↓ 可学习summary query再次汇聚
一个256维摘要 → Linear → 一个768维摘要
             ↓ 原fixed LN和GlobalIntentPredictor（VTT、state、H仍单独输入）
一个768维latent → 原LN＋Linear → 一个512维token → 原DiT
```

4个query不是4个未来时刻，不预定义物体/机械臂语义，也不保证自动分工。它们只访问当前帧。`return_attention=True`返回slot平均的初次视觉读取权重，不能当作整个迟汇聚网络的最终归因图；本轮不启用attention-target。

DiT深度/宽度/视觉适配器、意图MLP、最终意图token数量、DINO层、VTT、数据、统计、C8/E4、H8、无gate和无直接VTT注入均不变。不新增稠密future loss、空间残差、OPD、历史帧或动作先验训练。

当前配置下动作模型总参数约80.96M→82.01M，增加1,055,232个参数（约1.30%），不包含冻结DINO。池化器参数约0.86M→1.91M；这不等于训练时间或峰值显存只增加1.30%，后两项仍需GPU实测。

## 2. 实验矩阵

| 预测器 | 无KD，仅动作loss | 有KD，动作loss＋0.1 latent KD |
|---|---|---|
| 旧单query | C-uniform，epoch40 cold 88.84% | D-uniform，epoch40 cold 89.16%，tail 89.52% |
| 新4query迟汇聚 | LatePool-Q4-C，epoch40 cold 89.20% | LatePool-Q4-D，epoch40 cold 89.08% |

两组都从同一原教师epoch40继承兼容动作网络，新的池化器和意图MLP重新初始化。D仍使用同一个冻结posterior产生标签。**不从C-uniform或D-uniform学生继续训练，不需要重训教师。** 第一轮先做cold对比；tail单独记录，不能混合归因。

两个新配置分别为：

- `metaworld_vla/configs/metaworld_mt50_easy_planwam_latepool_q4_action_only_uniformlr.yaml`
- `metaworld_vla/configs/metaworld_mt50_easy_planwam_latepool_q4_student_uniformlr.yaml`

相对对应的C/D-uniform，只改变三个结构字段：

```yaml
model:
  global_intent:
    pooling: task_attention_late
    pool_num_queries: 4
    pool_interaction_depth: 1
```

宽度256、heads4沿用原值。仍用batch96、seed0、50轮余弦LR 2e-4→5e-5，每5轮保存，epoch40作为主比较点。不能把总epochs直接改为40，否则改变学习率轨迹。

## 3. 旧模型和教师兼容性

- `task_attention`、`mean`旧路径不创建新模块，不改变参数键、初始化随机数消耗或forward。
- 新模块全部位于`intent_pool`内。教师初始化只跳过这个被显式替换的子模块；意图MLP仍不继承。兼容动作模块保持白名单、键集合、shape、有限值检查。
- 新学生checkpoint保留完整新架构合同，resume仍严格核对，不把新模型伪装成旧模型。
- 仅在比较旧教师合同的时候，将`task_attention_late`映射为原`task_attention`并去除两个新增结构字段。数据、相机、图像方向、state/action、VTT/stats哈希、H、DiT配置及posterior规格不被忽略；仍需精确匹配。
- 初始化报告新增`predictor_bridge: task_attention_late_v1`及未复制的教师池化器键列表。
- 2×2离线诊断会按原教师结构构造teacher action network，学生按新结构构造；两边各自严格加载，不能强行将教师state载入新学生池化器。
- 新checkpoint必须配自己的保存配置。不要用旧YAML加载新checkpoint，也不要用新YAML测试旧学生。

## 4. 正式训练命令

先训练无KD组：

```bash
cd /home/lab347-no10/Tianyi/LiLa-WAM
conda activate MetaWorldVLA
mkdir -p logs/planwam_comparison
set -o pipefail

TEACHER="checkpoints/metaworld_mt50_easy_planwam_teacher_seed0/checkpoint_epoch_40.pt"
RUN="checkpoints/student_action_init/metaworld_mt50_easy_planwam_latepool_q4_action_only_uniformlr_seed0"

CUDA_VISIBLE_DEVICES=0 python metaworld_vla/train_planwam.py \
  --config metaworld_vla/configs/metaworld_mt50_easy_planwam_latepool_q4_action_only_uniformlr.yaml \
  --stats stats/metaworld_mt50_corner2_480_50ep_4d.json \
  --init-teacher-action "$TEACHER" \
  --save-dir "$RUN" \
  2>&1 | tee logs/planwam_comparison/latepool_q4_action_only_uniformlr_seed0.log
```

再独立训练有KD组：

```bash
cd /home/lab347-no10/Tianyi/LiLa-WAM
conda activate MetaWorldVLA
mkdir -p logs/planwam_comparison
set -o pipefail

TEACHER="checkpoints/metaworld_mt50_easy_planwam_teacher_seed0/checkpoint_epoch_40.pt"
RUN="checkpoints/student_action_init/metaworld_mt50_easy_planwam_latepool_q4_student_uniformlr_seed0"

CUDA_VISIBLE_DEVICES=0 python metaworld_vla/train_planwam.py \
  --config metaworld_vla/configs/metaworld_mt50_easy_planwam_latepool_q4_student_uniformlr.yaml \
  --stats stats/metaworld_mt50_corner2_480_50ep_4d.json \
  --init-teacher-action "$TEACHER" \
  --teacher-checkpoint "$TEACHER" \
  --save-dir "$RUN" \
  2>&1 | tee logs/planwam_comparison/latepool_q4_student_uniformlr_seed0.log
```

不要添加`--resume`或从已有学生载入。真正恢复本轮中断时，保留同一新配置、原教师文件和原save-dir，再加`--resume "$RUN/checkpoint_epoch_XX.pt"`。

启动应看到`Causal late-pool ENABLED: queries=4, width=256, interaction_depth=1`。无KD组的KD/TargetNorm为0正常；有KD组应看到非零KD和TargetNorm。batch=1的开发smoke中LatentStd=0是统计定义导致，不能据此诊断坍缩。

## 5. epoch40 cold测试

无KD组：

```bash
cd /home/lab347-no10/Tianyi/LiLa-WAM
conda activate MetaWorldVLA
RUN="checkpoints/student_action_init/metaworld_mt50_easy_planwam_latepool_q4_action_only_uniformlr_seed0"

CUDA_VISIBLE_DEVICES=0 python metaworld_vla/eval_parallel.py \
  --config "$RUN/config.yaml" \
  --stats stats/metaworld_mt50_corner2_480_50ep_4d.json \
  --checkpoint "$RUN/checkpoint_epoch_40.pt" \
  --workers 8 --test-num 50 --horizon 500 \
  --seed 1000 --policy-seed 456 --num-inference-steps 10 \
  --output eval_result/planwam_comparison/latepool_q4_action_only_uniformlr_ep40_cold_env1000_policy456.json
```

有KD组：

```bash
cd /home/lab347-no10/Tianyi/LiLa-WAM
conda activate MetaWorldVLA
RUN="checkpoints/student_action_init/metaworld_mt50_easy_planwam_latepool_q4_student_uniformlr_seed0"

CUDA_VISIBLE_DEVICES=0 python metaworld_vla/eval_parallel.py \
  --config "$RUN/config.yaml" \
  --stats stats/metaworld_mt50_corner2_480_50ep_4d.json \
  --checkpoint "$RUN/checkpoint_epoch_40.pt" \
  --workers 8 --test-num 50 --horizon 500 \
  --seed 1000 --policy-seed 456 --num-inference-steps 10 \
  --output eval_result/planwam_comparison/latepool_q4_student_uniformlr_ep40_cold_env1000_policy456.json
```

若补tail，增加`--action-reuse --reuse-strength 1 --reuse-sigma 0.5`，并把输出名的`cold`改为`tail`。不修改YAML；原G2 tail为89.48%，旧D-uniform tail为89.52%，C-uniform tail尚未纳入本表。

## 6. 如何判断结果

1. 新无KD组 vs C-uniform：池化器本身是否改善动作学习？
2. 新有KD组 vs D-uniform：同监督下新池化器是否改善？
3. 新有KD−新无KD vs 旧D-uniform−旧C-uniform：是否有更清楚的KD收益？
4. latent拟合下降不等于闭环提升；离线教师latent替换也不是闭环oracle。
5. 多query增加了参数和计算；若有收益，后续应补参数量接近的加宽MLP控制、多seed和跨平台实验。单纯增加query不是已经成立的独立创新。

## 7. 验证记录及边界

- `python -m unittest discover -s utils -p 'test_planwam*.py' -v`：29项通过，覆盖旧路径、可选分支、输入合同保护、bf16/autocast梯度、冻结教师、因果性、单token接口、保存加载和优化器恢复。
- 修改前保存的旧单query模型，在修改后严格加载，latent与动作输出逐元素完全一致。
- 真实2500episode数据索引、冻结DINO-B、完整h512、原教师epoch40：新C/D各完成一个CPU优化步骤，保存于`/tmp/lila_latepool_q4_{C,D}_smoke_20261111`；均非正式checkpoint。
- 新D开发checkpoint通过现有Policy严格加载，不读取教师/未来图像即可生成动作；5个step产生2个chunk，第二个chunk实际复用4个位置，reset清空先验。
- 新D开发checkpoint通过CPU一条真实anchor的2×2离线生成诊断，确认原教师网络与新学生网络可分别构造和严格加载。输出仅在`/tmp`，不计入研究成功率。
- 尚未验证GPU整轮速度/显存和闭环成功率；不承诺性能提升。原模型与数据文件未改写。

## 8. 随机初始化4query action-only控制（2026-11-12）

新增配置：`metaworld_vla/configs/metaworld_mt50_easy_planwam_latepool_q4_action_only_scratch.yaml`。

与原4query action-only uniformlr配置相比，仅将`planwam.action_init.enabled`改为`false`、`planwam.action_init.checkpoint`改为`null`。没有改变模型、数据或优化日程；无KD、无教师载入，仍冻结同一预训练DINO并保留既有VTT。随机初始化针对可训练动作模型，不是把DINO随机化。

本控制补齐矩阵：单query随机N（87.68%）/兼容初始化C-uniform（88.84%）；4query随机（待训练）/兼容初始化LatePool-C（89.20%）。该矩阵隔离结构与初始化，不等同于“同预算无未来教师”控制。

训练：

```bash
cd /home/lab347-no10/Tianyi/LiLa-WAM
conda activate MetaWorldVLA
mkdir -p logs/planwam_comparison
set -o pipefail

RUN="checkpoints/student_action_init/metaworld_mt50_easy_planwam_latepool_q4_action_only_scratch_seed0"

CUDA_VISIBLE_DEVICES=0 python metaworld_vla/train_planwam.py \
  --config metaworld_vla/configs/metaworld_mt50_easy_planwam_latepool_q4_action_only_scratch.yaml \
  --stats stats/metaworld_mt50_corner2_480_50ep_4d.json \
  --save-dir "$RUN" \
  2>&1 | tee logs/planwam_comparison/latepool_q4_action_only_scratch_seed0.log
```

保留50轮cosine规划，主比较epoch40；不传`--init-teacher-action`、`--teacher-checkpoint`或`--resume`。启动日志应为`stage=action_only`、`teacher=None`、`initialization=fresh action/predictor weights (legacy)`，同时有late-pool启用日志；KD/TargetNorm为0正常。

epoch40 cold测试：

```bash
cd /home/lab347-no10/Tianyi/LiLa-WAM
conda activate MetaWorldVLA
RUN="checkpoints/student_action_init/metaworld_mt50_easy_planwam_latepool_q4_action_only_scratch_seed0"

CUDA_VISIBLE_DEVICES=0 python metaworld_vla/eval_parallel.py \
  --config "$RUN/config.yaml" \
  --stats stats/metaworld_mt50_corner2_480_50ep_4d.json \
  --checkpoint "$RUN/checkpoint_epoch_40.pt" \
  --workers 8 --test-num 50 --horizon 500 \
  --seed 1000 --policy-seed 456 --num-inference-steps 10 \
  --output eval_result/planwam_comparison/latepool_q4_action_only_scratch_ep40_cold_env1000_policy456.json
```

先与N比较结构本身，再与LatePool-C比较兼容初始化收益；不要将cold与tail混合归因。新配置是独立文件，未修改既有训练/测评代码或原配置。

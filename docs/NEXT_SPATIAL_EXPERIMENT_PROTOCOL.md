# LiLa-WAM 后续空间意图实验协议

更新时间：2026-10-14

本文档记录当前 S-residual v2/v3 之后的实验顺序。实验必须按照诊断优先、结构升级在后的原则执行，避免同时改变多个因素。

## 1. 当前已知结果

主要参照结果：

| 模型 | 训练方式 | Clean | Randomized | 总体 |
|---|---|---:|---:|---:|
| 原始 baseline epoch40 | scratch | 70.2% | 64.6% | 67.4% |
| C32 | baseline epoch40 + 微调4轮 | 75.2% | 63.0% | 69.1% |
| S-residual v2 微调4轮 | baseline epoch40 + 微调4轮 | 77.2% | 63.8% | 70.5% |
| S-residual v2 scratch epoch40 | scratch50 scheduler | 70.8% | 60.2% | 65.0% |

v2 scratch epoch40 的 gate 约为0.236，明显高于 v2 微调4轮结束时的约0.063。当前主要假设是：scratch训练初期空间预测器尚未稳定，残差过早进入视觉条件并干扰动作策略。

## 2. 实验固定协议

除非实验明确要求改变，否则统一使用：

~~~text
数据集：/home/lab347-no10/Tianyi/LiLa-WAM/data/yf10
统计：/home/lab347-no10/Tianyi/LiLa-WAM/artifacts/yf10/norm_stats.json
DINO：/home/lab347-no10/Tianyi/LiLa-WAM/pretrained/dinov3-vitb16
任务：YF10十任务
测试：demo_clean + demo_randomized
instruction_type：unseen
测试 seed：0
每任务测试次数：50
~~~

正式 scratch 对比：

~~~text
learning_rate: 2e-4
lr_min: 5e-5
batch_size: 96
epochs: 50
比较 checkpoint_epoch_40.pt
~~~

快速结构筛选：

~~~text
从 baseline epoch40 使用 init_from
训练4轮
seed0快速测试
~~~

## 3. 第一优先级：Residual-off 推理消融

### 目的

判断当前性能下降是否由空间残差实际注入造成。

### 方法

使用同一个 v2/v3 scratch epoch40 checkpoint，只在测试配置中关闭残差：

~~~yaml
spatial_intent:
  inject_as_residual: true
  residual_scale: 0.0
~~~

这一步不重新训练，只重新运行 clean 和 randomized 测试。

### 结果解释

| 结果 | 解释 | 下一步 |
|---|---|---|
| residual off 明显恢复 | 残差注入正在干扰动作策略 | 做 stop-gradient 和 gate 调度 |
| residual on/off 基本相同 | 策略没有真正使用残差，或模型本身已较差 | 做 S-aux-shared |
| residual off 更差 | 残差有用，但统一 gate 或区域分配不稳定 | 做 region gate |

关闭残差的模型仍然是已经联合训练过的模型，不能将它当作重新训练的 baseline。

## 4. 第二优先级：梯度和残差比例诊断

### 目的

区分 gate 小、预测残差小、视觉特征尺度不匹配和梯度冲突。

训练时记录：

~~~text
raw spatial_gate
tanh(raw spatial_gate)
warmup/ramp schedule
effective spatial gate
||spatial residual||
||fused visual feature||
residual ratio
spatial predictor gradient norm
spatial gate gradient
Action DiT visual-path gradient
~~~

残差比例定义：

~~~text
residual_ratio =
  ||effective_gate × residual||
  /
  ||fused_visual_feature||
~~~

不能只看 SpatialGate。即使 gate=0.1，如果预测残差范数很大，实际视觉扰动仍可能很强。

### 判断标准

- residual_ratio 很大：降低 max_gate 或增加 residual warm-up；
- SpatialPredNorm 很小：预测器没有学到未来变化；
- gate 梯度长期为零：检查注入路径和优化器参数；
- gate 变化明显但性能下降：模型可能学习了随机化 shortcut。

## 5. 第三优先级：Stop-gradient 残差实验

### 目的

判断空间预测器和动作策略的联合反向传播是否互相干扰。

原始路径：

~~~text
空间预测器 → spatial residual → Action DiT → action loss
~~~

Stop-gradient 路径：

~~~text
空间预测器 → detach spatial residual → Action DiT
空间预测器 ← spatial loss
~~~

实现含义：

- Action DiT 仍然使用空间残差；
- action loss 不再更新空间预测器；
- 空间预测器只由 spatial loss 训练。

### 训练协议

先做快速微调：

~~~text
baseline epoch40
init_from
4 epochs
seed0
clean/randomized
~~~

结果如果优于普通 S-residual，说明原来的主要问题是 action loss 和 spatial loss 对预测器的梯度冲突。

如果有效，再做 scratch50；如果无效，不继续扩大该分支。

## 6. 第四优先级：S-aux-shared

### 目的

验证未来空间监督能否改善动作模型使用的共享视觉表示。

当前 S-aux 的空间 predictor 是独立 head，空间 loss 主要只更新 predictor：

~~~text
current DINO → independent Spatial Predictor → spatial loss
~~~

S-aux-shared 应改为：

~~~text
current DINO → shared visual representation
                      ├── Action DiT
                      └── Spatial Predictor → spatial loss
~~~

这样 spatial loss 才会更新 Visual Adapter 或共享融合层，并间接影响 Action DiT。

### 推荐顺序

1. baseline epoch40 + init_from；
2. 训练4轮；
3. 与 baseline/C32 对比；
4. 如果快速版本有效，再训练 scratch50。

### 结果解释

- S-aux-shared 提升：空间监督能改善共享视觉表示；
- S-aux-shared 不变：空间目标可能不适合当前任务；
- S-aux-shared 提升但 residual 下降：空间表示有用，直接注入方式有问题。

## 7. 第五优先级：Region Gate

### 适用条件

只有在以下情况出现时才做：

- residual-on 比 residual-off 有效；
- scalar gate 已经影响策略；
- 不同任务或区域表现明显不一致；
- 怀疑少数错误区域迫使整体 gate 降低。

### 结构

当前 scalar gate：

~~~text
一个 gate → 控制全部8个区域
~~~

Region gate：

~~~text
gate[0] ... gate[7] → 分别控制8个空间区域
~~~

建议第一版使用静态可学习 region gate：

~~~yaml
region_gate: true
region_gate_init: 0.05
max_gate: 0.1
~~~

实现时先对 B×8×768 的区域残差乘 gate，再上采样到 B×300×768。

### 实验控制

只改变：

~~~text
scalar gate → region gate
~~~

其余保持：

- 相同 checkpoint；
- 相同 warm-up；
- 相同 max_gate；
- 相同 spatial loss；
- 相同测试 seed。

如果静态 region gate仍然不能提升 randomized，再研究动态 region gate。

## 8. 第六优先级：Task-conditioned Interaction Query

### 适用条件

当以下实验均没有稳定收益时：

- residual-off；
- stop-gradient；
- S-aux-shared；
- region gate。

### 结构

~~~text
当前 DINO patch + VTT + qpos
              ↓
少量 task-conditioned interaction queries
              ↓
目标物体、抓取位置、接触区域、放置区域
              ↓
动作条件或未来交互预测
~~~

第一版建议使用4个query，避免一开始引入过多自由度。先验证当前交互表示是否改善动作策略，再加入未来预测目标。

### 训练原则

Interaction Query 不应同时加入：

- Global Intent；
- OPD；
- region gate；
- cosine/norm 多个新 loss。

每次只引入一个结构变化。

## 9. Global Intent 的位置

Global Intent 暂时保持独立路线。

原因：

- G2 scratch 在 randomized 下明显下降；
- global intent 可能学习任务先验；
- 与空间残差同时加入会混淆问题来源。

只有在 spatial residual 或 interaction query 稳定后，才考虑：

~~~text
global intent：任务阶段和全局运动方向
spatial intent：局部物体和接触关系
~~~

组合时使用独立 predictor、独立 gate 和独立 loss。

## 10. 决策树

~~~text
Residual-off
    │
    ├─ off 明显恢复
    │      ↓
    │   Stop-gradient
    │      ↓
    │   若有效：保留残差路线
    │
    ├─ on 比 off 好但任务差异大
    │      ↓
    │   Region gate
    │
    └─ on/off 都没有提升
           ↓
        S-aux-shared
           ↓
        若仍无效：Interaction Query
~~~

## 11. 停止条件

满足以下任一条件时，不继续增加训练轮数：

1. scratch 和微调版本都低于 C32；
2. residual-off、stop-gradient 和 S-aux-shared 均无改善；
3. randomized 下降而 clean 偶尔提升；
4. gate 需要持续增大才能获得 clean 提升；
5. spatial predictor loss下降但动作成功率不变。

出现这些情况时，应停止固定网格空间残差路线，转向 task-conditioned interaction query 或更明确的对象/接触监督。

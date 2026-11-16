# AdaPolicy 风格的部署实验

本文三组原配置保持未对齐。若要单独比较时间对齐，使用[时间对齐实验配置](TIME_ALIGNMENT.md)；不要覆盖既有对照配置。

2026-11-13：在同一个 `real_vla.py` 内增加两个独立的可选模式，无需重训。
硬件归位、急停、100Hz控制滤波、关节/夹爪保护共用现有实现。

| 实验配置 | 队列更新 | B样条方法 | 输出目录 |
|---|---|---|---|
| `config.real_multitask.yaml` | drain_queue：执行完当前队列 | weighted_history：原有整体加权拟合 | inference_outputs_multitask |
| `config.real_multitask_latest.yaml` | latest_ready：新预测就绪即替换 | weighted_history | inference_outputs_multitask_latest |
| `config.real_multitask_adapolicy.yaml` | latest_ready | prefix_refit：局部前三控制点 | inference_outputs_multitask_adapolicy |

现有配置缺省时仍使用 drain_queue/weighted_history。
三份配置使用同一checkpoint、统计量、32步预测/16步执行、预取阈值8、30Hz动作和100Hz控制、相同安全参数。
原配置现在显式写出了默认模式，便于保存配置快照和回退。

## 推荐运行顺序

前一轮基准已测过，可先测“只改切换时机”，再测“切换＋局部重拟合”，每组至少相同次数和摆放条件。
此前记录是入盒任务，以下显式选择同一任务，并修正显示用的instruction：

```bash
conda activate edulite_deploy
cd /home/lab347-no10/Tianyi/LiLa-WAM

# 只改切换时机；先静态检查，再把 --check-only 换成 --execute。
python robot/edulite_a3_vla/real_vla.py \
  --config robot/edulite_a3_vla/config.real_multitask_latest.yaml \
  --task-name place_red_block_into_box \
  --instruction "pick up the red block and place it in the box" \
  --check-only

# 切换＋前三控制点重拟合；同样检查后把 --check-only 换成 --execute。
python robot/edulite_a3_vla/real_vla.py \
  --config robot/edulite_a3_vla/config.real_multitask_adapolicy.yaml \
  --task-name place_red_block_into_box \
  --instruction "pick up the red block and place it in the box" \
  --check-only
```

进入交互后仍是 a 归位、p 预览、r 开始、s 保持并保存。回到原方案时，使用原配置即可。
叠蓝块任务用 `--task-name place_red_block_on_blue_block` 并修改显示instruction。

## 队列切换

```yaml
runtime:
  chunk_update_mode: latest_ready
  prefetch_threshold_actions: 8
```

剩余8步时提交一次后台推理，且最多只有一个推理任务。
新结果在下一次动作调度中被发现后，先使用切换当时最新历史目标完成拟合和校验，
再丢弃旧队列剩余动作并装入新16步；每个调度周期仍只取一个动作，不会补发/突发动作。
新块装入后从16步重新计数，降到阈值才再次推理。
所以16是装入上限，不再表示每块一定执行满16步。不会额外增加GPU并发数。
停止后，旧generation的推理结果仍被丢弃，不能进入新一轮。

该模式不根据延迟跳过新预测前d步；不要将它视为ABPolicy式delay_steps补偿。
首块仍使用p预览的结果，包含原来的预览等待和启动倒计时；本次未改该行为。

## 局部重拟合的确切实现

```yaml
runtime:
  joint_bspline:
    enabled: true
    method: prefix_refit
    degree: 3
    control_points: 8
    history_actions: 2
    history_weight: 4.0
    refit_free_control_points: 3
    refit_regularization: 0.05
    fallback_if_worse: true
```

1. 在未来索引0..15上，对新预测L1～L6拟合三次、8控制点样条。
2. 最近H个已取出目标对应过去索引-H..-1。仅为建立历史拟合方程，将首段样条多项式向过去延伸，
   然后调整前3个控制点；后5个控制点固定。未来执行只在0..15内采样，不执行外推值。
3. 解软约束目标：

   `history_weight² * ||B_history * C_new - historical_targets||² + refit_regularization * ||ΔC_first3||²`。

   正则约束的是前3个控制点相对原预测的变化；不是ABPolicy的“最后一个自由控制点权重”。
4. 重建16个关节目标，进行原有边界裁剪和fallback判断，再由原控制器限速、滤波。
   L7原值不经过该拟合，继续使用独立控制和力保护。

这是对 [AdaPolicy附录B](https://arxiv.org/html/2610.34280v1#A2) 的可控借鉴实验，
不是官方代码复现：附录没有明确历史坐标和完整拟合目标，以上历史外推/正则是此处明确选择的实现。
它没有硬约束跨块位置、速度、加速度相等，不保证严格C1/C2连续或任务效果改善。
历史仍来自实际取出的模型目标，不是电机反馈或100Hz滤波后的发送轨迹；被丢弃的队列尾部不进入历史。

调参时先保留2/8/4、前三自由控制点和0.05正则。更大正则会更保留原预测，更小正则更强调历史贴合；
改变history_weight也会改变相对正则强度。两者不要同时调整。
保留fallback_if_worse=true。此处局部拟合不改变学习到的控制点输出维度，不涉及checkpoint结构变更。

## 新增诊断

原有CSV/JSON均继续保存，并增加：

- `chunk_executions.csv`：每块planned_actions、actions_selected、discarded_actions、end_reason，
  以及首末取出目标的时间。actions_selected是交给主循环的目标数，不是机器人实际达到的路点数。
- `inference_chunks.csv`：discarded_queue_actions、previous_chunk_actions_selected、previous_chunk_id、
  chunk_update_mode、conditioning_s。可区分主动替换与队列等待。
- 平滑指标新增bspline_method、fit_compute_s、candidate_boundary_rad、candidate_max_step_rad；
  candidate指标保留回退前的拟合结果，最终输出仍看smooth_boundary_rad/smooth_max_step_rad。
- prefix_refit另有历史RMSE前后值、控制点最大改变量，均为候选拟合诊断，不能直接当作反馈误差。
- 摘要新增chunk_replacements、replacement_discarded_actions、stopped_discarded_actions、
  background_chunk_actions_selected、chunk_conditioning_s、bspline_fit_s。

重点比较边界跳变、观测年龄、实际每块目标数、回退比例、跟踪误差和任务成功率。
latest_ready丢弃旧目标是预期行为，不是队列断供。

## 验证范围

离线回归使用假模型和假控制器，覆盖原队列行为、新结果提前替换、重复预取抑制、
接入时历史、异常结果原子校验、过期generation丢弃、局部拟合只改前三控制点、L7不变和边界处理。
静态检查核对三份配置、任务向量和checkpoint路径。
这些检查不连接CAN、不打开相机、不使能电机；现场效果需通过上述对照测量判断。

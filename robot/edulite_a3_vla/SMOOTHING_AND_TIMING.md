# 真机异步推理与 B 样条调参

当前实现保持预测32步、执行16步、动作更新30 Hz、控制滤波100 Hz。
异步预取一直启用，B 样条按配置选择。此次增加测量，没有增加动作延迟补偿、轨迹跳步或 ABPolicy 控制点输出。
训练 checkpoint 和训练数据无需改动。

## 第一轮：记录当前基准

`config.real_multitask.yaml` 已启用诊断，B 样条仍关闭：

```yaml
runtime:
  action_rate_hz: 30.0
  control_rate_hz: 100.0
  prefetch_threshold_actions: 8
  diagnostics:
    enabled: true
  joint_bspline:
    enabled: false
    degree: 3
    history_actions: 2
    control_points: 8
    history_weight: 4.0
    fallback_if_worse: true
```

```bash
conda activate edulite_deploy
cd /home/lab347-no10/Tianyi/LiLa-WAM
python robot/edulite_a3_vla/real_vla.py \
  --config robot/edulite_a3_vla/config.real_multitask.yaml \
  --task-name place_red_block_on_blue_block --check-only

# 现场确认后运行；a 归位、p 预览、r 执行、s 保持并保存、q 退出。
python robot/edulite_a3_vla/real_vla.py \
  --config robot/edulite_a3_vla/config.real_multitask.yaml \
  --task-name place_red_block_on_blue_block --execute
```

入盒任务改为 `--task-name place_red_block_into_box`，可添加
`--instruction "pick up the red block and place it in the box"` 更新显示文本。
两任务使用相同 checkpoint、统计量，分别做固定次数、相同摆放条件的对照，人工记录成功与否。

## 新增输出

文件与原有 `execution_trace.csv` 放在同一个 `inference_outputs_multitask/run_*` 目录。
若关闭 `visualization.enabled`，诊断仍保存，在同一 output_root 下创建 `diagnostics_*` 目录。
记录缓存在内存中，在按 s、达到时限、急停、退出或异常清理时保存；进程被强制 kill/断电时可能丢失。

| 文件 | 主要内容 |
|---|---|
| `runtime_summary.json` | P50/P95/最大值、断供、配置与 checkpoint 路径 |
| `inference_completed.csv` | 在活动轮次中被主线程接收到的后台预测，包括尚未执行的 pending 预测 |
| `inference_chunks.csv` | 真正接入执行的动作块；包含首块 preview、后续 background、时间戳和样条指标 |
| `runtime_ticks.csv` | 每次动作调度的间隔、工作耗时、迟到时间、队列状态、观测年龄、安全警告次数 |
| `control_timing.csv` | 100 Hz线程的实际周期、工作耗时、发送指令与反馈的最大关节误差、模型指令超时状态 |

CSV 用 `*_mono_s` 表示同一进程单调时钟的绝对秒数，可用于表间关联；不是日期时间。
每步 `chunk_id` 对应 `inference_chunks.csv` 的同名列。
关闭诊断用 `runtime.diagnostics.enabled: false`，不会关闭异步推理。

重点查看以下摘要项（时间单位均为秒）：

- `background_inference_s`：后台实际推理耗时，含状态缓冲/预处理、DINO、采样和结果传回 CPU。
  当前推理函数最终执行 `.cpu().numpy()`，因此测到 GPU 完成后的结果，不只是提交 CUDA 的耗时。
  不包含线程池等待；计时也不是纯 GPU kernel 时间。
- `worker_queue_wait_s`：提交至后台函数开始的时间；`result_poll_delay_s`：结果完成至主线程发现的时间。
- `queue_wait_events / queue_wait_total_s`：动作队列已空且下一块未就绪时的等待次数、总时长。
  这是动作调度轮询观察到的等待，精度受约33 ms周期影响；等待期间仍可能继续向最后目标运动，并非机械臂静止时长。
  未恢复便停止的最后一次等待也计入，结束时间取保存时刻。
- `action_update_interval_s`：新目标实际交给控制器的时间间隔，理想约0.0333；
  `control_interval_s`：100 Hz线程周期，理想约0.010。
- `action_tick_work_s / schedule_resets`：动作循环工作量、严重迟到后重置调度的次数。
  工作量包含本轮视频/可视化记录；用来判断图像编码是否拖慢调度。
- `following_error_rad`：100 Hz处的 `max(abs(filtered_command - feedback))`，不是模型目标误差。
- `background_chunk_first_observation_age_s`：后台块被接入时，距其状态读取时刻的时间。
  `preview_chunk_first_observation_age_s` 单独列出首块的预览等待、画图和启动倒计时影响，不能混入推理速度。
  状态读取时间是应用层读取时刻，不是电机原始反馈采样时间。
- `action_observation_age_s`：每步新目标对应的原始状态观测年龄，会随块内执行而增大。
  CSV另有 `action_camera_age_s`（预测所用图像年龄）、`camera_age_s`（当前读取图像年龄），两者含义不同。
- `model_command_timeout_cycles`：底层因模型目标长期未更新而切换保持的控制周期数。

后台预测若停止时仍未完成/未被轮询接收，不会等待它完成再计入该轮摘要。
因此很短的轮次样本量可能不足，应查看每个统计项的 `count`。

动作块的 `raw_boundary_rad / emitted_boundary_rad` 是相对上一个已取出动作目标的最大关节跳变，
不等于实际反馈跳变；还需结合原有 `execution_trace.csv` 的 filtered_command 和 feedback。
B 样条开启后额外包含 `smooth_boundary_rad`、`raw_max_step_rad`、`smooth_max_step_rad`、
`fit_mae_rad`、`clipped_values`、`fallback`。这些是平滑层输出指标，之后仍有控制器限幅和滤波。

## 第二轮：仅启用 B 样条

复制基准配置后，仅改 `runtime.joint_bspline.enabled: true`，保持2/8/4和所有控制限速不变。

| 参数 | 调节方向 |
|---|---|
| `degree` | 当前实现固定为3 |
| `control_points` | 先8；仍有抖动试6；轨迹被改动过多试10。控制点必须不超过history+horizon |
| `history_actions` | 先2；再单独比较4。至少2 |
| `history_weight` | 先4；可单独比较2或8。越大越强调历史目标拟合 |
| `fallback_if_worse` | 保持true，若首步衔接或最大步长变差则回退原动作 |

每轮摘要的 `bspline_fallback_chunks` 可判断是否经常回退。
现有加权拟合没有跨块位置/速度等式约束，不保证严格C1/C2连续；L7保持独立滤波和力保护。
不要同时打开训练配置 `inference.smooth_actions` 的全维样条，那会把夹爪也纳入通用平滑。

## 第三轮：根据测量调整预取

`prefetch_threshold_actions=8` 在30 Hz下提供约8/30=0.267秒准备窗口。
若存在队列断供，可根据后台推理、线程等待、结果轮询的测量值估计所需窗口：

```text
阈值 ≈ ceil(30 × (推理P95 + 线程等待P95 + 轮询延迟P95 + 余量))
```

各项P95相加只是保守调参参考，不等于总延迟的P95。余量可先试0.03～0.05秒。
例如推理0.20秒、线程等待可忽略、轮询0.03秒、余量0.03秒，对应ceil(7.8)=8。
更慢时可逐项比较阈值10、12；必须小于实际执行horizon（当前16）。
若需要16或更多，不能继续增加阈值；应排查推理耗时与调度开销。
阈值越大越早推理，预测到实际执行的观测延迟也可能越大；同时比较断供和任务成功率。

执行horizon来自 `model.config_path` 指向的配置，修改根目录训练YAML不会改变已加载运行。
首轮保留32/16，不要用降低 `action_rate_hz` 掩盖断供，因为这会改变轨迹执行时间尺度。
当前没有 `async.enabled` 开关，也没有 ABPolicy 的 `n_free/last_pt_weight/delay_steps` 参数。
可选的 `chunk_update_mode: latest_ready` 和 `joint_bspline.method: prefix_refit` 已另外实现，
实验配置、正则定义和命令见 [AdaPolicy风格实验](ADAPOLICY_DEPLOY_EXPERIMENT.md)；本文上方保持原基准的调参说明。

若不断供但仍有块间跳变，优先比较B样条；若指令平滑而反馈滞后，再检查控制滤波。
可最后单独比较 `safety.filter_omega: 8.0` 与当前10.0，降低会更柔和，也更滞后。
不要同时修改速度/加速度限制、推理步数和样条参数，否则难以定位收益来源。

## 离线回归

```bash
conda activate edulite_deploy
python -m unittest discover -s robot/edulite_a3_vla -p test_runtime_diagnostics.py -v
```

使用假模型、假控制器验证异步顺序、停止后的过期结果丢弃、统计与轮次隔离，以及夹爪不被关节样条修改；
不连接CAN、不打开相机、不使能电机。

# 可选的异步动作时间对齐

时间对齐默认关闭。原有 `config.real_multitask.yaml`、`config.real_multitask_latest.yaml`、
`config.real_multitask_adapolicy.yaml` 没有修改其配置值和执行方式，仍可作为对照。
不需要重新训练；没有增加速度桥接、改变滤波/限速或移动机械臂进行验证。

## 实验配置

| 配置 | 基于哪组 | 日志输出目录 |
|---|---|---|
| `config.real_multitask_latest_aligned.yaml` | 就绪即替换＋原整体加权拟合 | inference_outputs_multitask_latest_aligned |
| `config.real_multitask_adapolicy_aligned.yaml` | 就绪即替换＋局部前三控制点拟合 | inference_outputs_multitask_adapolicy_aligned |

与对应未对齐配置比较，控制配置只新增：

```yaml
runtime:
  time_alignment:
    enabled: true
```

另有独立输出目录。checkpoint、任务向量、动作频率30Hz、预测32步、装入16步、预取8步、样条参数和安全参数一致。
设置false或省略该节会恢复原方式，包括首块使用p的预览结果；不要通过修改checkpoint内训练配置打开对齐。

## 运行

```bash
conda activate edulite_deploy
cd /home/lab347-no10/Tianyi/LiLa-WAM

python robot/edulite_a3_vla/real_vla.py \
  --config robot/edulite_a3_vla/config.real_multitask_adapolicy_aligned.yaml \
  --task-name place_red_block_into_box \
  --instruction "pick up the red block and place it in the box" \
  --check-only
```

现场确认后，将最后的 `--check-only` 换成 `--execute`。仍按a归位、p预览、r开始、s保持保存。
对照“只改变时间对齐”时使用与前一轮相同的任务、布局、checkpoint和次数。
如果比较整体加权拟合，将配置改为 `config.real_multitask_latest_aligned.yaml`。
叠蓝块任务使用 `--task-name place_red_block_on_blue_block` 并相应修改显示instruction。

## 如何选择时间窗口

推理输出仍是原生绝对目标，不转为相对动作。保留完整32步，不在后台结果返回时提前截断成16步。

```text
t_obs = 构造状态观测时的应用层单调时钟
t_admit = 主线程拟接入新预测的时刻
d = floor((t_admit - t_obs) * action_rate_hz)
执行窗口 = prediction[d : d + 16]
```

例：接入时距观测约203ms，30Hz下d=6，取原预测索引6..21；约236ms时d=7，取7..22。
这里选择不晚于当前时刻的离散采样点，未引入分数步插值，名义残余时间误差小于一个动作周期。
时间原点使用状态观测读取时刻，不使用模型完成时刻或相机帧时刻；相机年龄继续独立记录。
状态读取时刻不是电机底层反馈采样时间，也无法消除图像/状态之间原有的采集不同步。

先选时间窗口，再对选出的16步做样条处理和校验，而不是先对0..15拟合后再切掉前缀。
所有7维使用同一时间索引，包括L7。过期前缀中的夹爪事件不会补发；直接使用所选窗口的当前夹爪目标，
之后仍走原有夹爪滤波、限速、锁存/力保护。日志记录被跳过前缀的夹爪范围，以便检查时序变化。

### 可执行范围和拒绝策略

- 必须留够完整16步，最大允许起始索引为16。若d>16，不把偏移强行截成16，也不退回执行第0步。
- 过期结果丢弃，旧队列不被清空，立即请求基于新观测的下一次推理。
  旧队列为空时不发新模型目标，继续使用原控制器的模型指令超时保持逻辑。
- 校验实际选中的窗口，覆盖现在可能使用的索引16..31；无效目标在替换旧队列前被拒绝。
- 若拟合/校验期间跨越采样索引，重新选窗拟合一次。两次都跨界则拒绝候选，避免无限重算阻塞动作循环。
- 接入之后仍以30Hz消费队列，不对已经接入的队列逐帧追赶或补发。调度迟到由相位误差记录反映。

### 首块刷新

p的预览通常已等待数秒，远超32步预测的时间范围，因此不能对这份旧预览强行计算跳步后执行。
对齐开启时，r倒计时结束后，在保持状态下读取新图像/状态并同步刷新一次预测，
待其完成后按同样公式取窗，再开始动作调度。
日志首块source为startup_refresh；后台其余块为background。
若首块刷新本身过期，启动中止并进入程序原有硬件清理，而不会执行过期预览。

p仍用于检查模型/场景、通过交互预览门槛，但实际首块来自刷新后的预测，未承诺与p的可视化逐点相同。
首块额外推理耗时在startup_refresh_inference_s中单独统计，不混入后台推理速度或运行中的队列断供。
运行记录时长包含首块刷新保持期间，不能直接当作有效动作执行时长或完成任务用时。

## 新增日志

`inference_chunks.csv`：

- aligned_start_index / aligned_end_index_exclusive：在原32步预测中的选窗范围。
- alignment_reference_mono_s：原预测的状态观测时刻。
- alignment_evaluated_age_s：选窗时的完整观测年龄。
- alignment_residual_age_s：完成拟合检查时，距所选采样点时间的残差。
- aligned_raw_start_to_command_max_rad：所选原始首目标与最近100Hz发送位置的最大关节差。
- aligned_start_to_command_max_rad：样条/裁剪后首目标与该发送位置的差；之后仍有控制器目标步长限制。
- aligned_start_gripper_norm、skipped_gripper_min_norm/max_norm：所选夹爪值和跳过前缀的范围。
- alignment_conditioning_attempts：本次选窗/拟合次数，最多2。

`runtime_ticks.csv`：

- prediction_action_index：当前目标在原预测中的时间索引（经样条处理后仍沿用该采样时间）。
- prediction_action_time_mono_s：原状态观测时间＋索引/30。
- alignment_phase_error_s：实际把目标交给控制器时刻－该名义动作时刻。
  此值包含拟合、主线程及调度误差，是应重点比较的时间对齐指标，不是电机实际到位误差。

`alignment_rejections.csv`仅有拒绝事件时生成，记录拒绝原因、计算的偏移、观测年龄和所需窗口长度。
摘要新增aligned_start_index、alignment_residual_age_s、alignment_phase_error_s、
aligned_raw_start_to_command_max_rad、aligned_start_to_command_max_rad、time_alignment_rejected_chunks，
以及首块刷新专用耗时和观测年龄。

观测年龄本身不会因跳步而变小：约200ms的原观测现在用于选择约第6步的预测。
应观察相位误差是否接近一个动作周期以内，并结合拒绝次数、断供、目标偏差、边界折返、夹爪时序和任务成功率判断。
实际交付时间还受调度影响，不保证每个样本严格落在33ms以内。

## 验证与限制

离线测试包括关闭功能的旧首块/截断行为、32步保留、接入时重新计算延迟、floor边界、
尾部安全检查、过期不清旧队列、拟合超时的有限重试、首块刷新、夹爪一致切片、先选窗后局部拟合、
drain_queue接入时间，以及原有异步/样条回归。

```bash
conda activate edulite_deploy
python -m unittest discover -s robot/edulite_a3_vla -p 'test_*.py' -v
```

没有增加位置/速度桥接，也未硬性保证时间对齐后的目标等于实时位置。
这是一项单独的时间索引实验，是否降低折返和顿挫需真机对照记录验证。

以下内容可直接复制到新对话中：

---

项目位于 `/home/lab347-no10/Tianyi/LiLa-WAM`。我正在使用 LiLa-WAM 进行 EDULITE A3 单臂 VLA 研究，任务为抓取红色小方块放入盒中，以及将红色小方块放在蓝色小方块上。机器人控制组件最初从 `/home/lab347-no10/Tianyi/robotwin_code/robot/` 迁移而来。

### 研究目标与已确定方案

- 使用主从双机械臂进行示教，但任务执行及训练采用原生单臂数据：`action_dim=7`、`state_dim=8`，不再补零伪装成双臂。
- 主臂接 `can1`，从臂接 `can0`。
- `action` 使用从臂的控制目标，保留当前训练标签来源，不改成 feedback。
- `feedback` 用于诊断实际运动，但已发现 L7 反馈缓存刷新问题，需要处理。
- 真机训练当前使用胸部相机 `chest_camera`。
- 仿真和真机使用各自配置入口，不进行联合训练，需保留原 RoboTwin 仿真训练兼容性。

### 已完成的代码与配置

采集目录：

```text
robot/edulite_a3_collect/
```

主要文件：

- `collect.py`：采集入口，支持 `--config`、`--task-profile`。
- `hardware.py`：主从控制、夹爪映射、滤波、力保护、数据快照。
- `home.py`：抽出的共享归位模块，采集和轨迹回放复用。
- `dataset_io.py`：写入 HDF5 和语言描述文件。
- `camera_controls.yaml`：独立相机控制参数。
- `config.master_slave.yaml`：基础采集配置。
- `config.master_slave.linear.yaml`：连续线性跟随配置，继承基础配置。
- `config.master_slave.binary.yaml`：二值开合配置。
- `task_profiles/place_red_block_into_box.yaml`。
- `task_profiles/place_red_block_on_blue_block.yaml`。

真机部署与查看目录：

```text
robot/edulite_a3_vla/
```

主要文件：

- `real_vla.py`：模型真机部署。
- `config.real_single.yaml`：单臂部署配置。
- `replay_dataset.py`：真机轨迹回放。
- `inspect_dataset_tk.py`：离线图像与轨迹查看。
- `README.md`：真机使用说明。

离线查看脚本已增加：

```bash
--trajectory-source action
--trajectory-source feedback
```

默认显示 `joint_action/vector`，feedback 显示 `teleop/slave_joint_feedback`。此前已通过 Python 语法检查。

### 数据记录定义

原生单臂 HDF5 的主要字段：

| 字段 | 内容 |
|---|---|
| `joint_action/vector` | `[T,7]`，从臂 L1～L6 目标绝对关节角（rad）＋归一化夹爪目标 |
| `teleop/slave_joint_feedback` | `[T,7]`，从臂反馈绝对关节角＋归一化夹爪反馈 |
| `teleop/master_joint_feedback` | `[T,7]`，主臂反馈；第 7 维为原始 L7 电机角度（rad），不是归一化值 |
| `endpose/left_endpose` | 从臂反馈计算的末端位置及四元数，共 7 维 |
| `endpose/left_gripper` | 从臂归一化夹爪反馈 |
| `observation/<camera>/rgb` | JPEG 编码图像 |
| `timing/monotonic` | 采样时间 |
| `timing/camera_<camera>` | 相机帧时间 |

`state_dim=8` 对应末端位姿 7 维＋反馈夹爪 1 维。关节 action 为绝对目标角度，不是相对增量。

当前采集配置：

```yaml
collection:
  output_root: ./collected_data_single
  rate_hz: 30
  jpeg_quality: 95
  single_arm_padding: none
```

两路相机当前采集申请值均为 `640×480、30 FPS`；训练/推理图像处理曾统一为 `320×240`，新对话应读取现有训练和部署配置确认最终值。

相机固定参数写在 `camera_controls.yaml`。最近胸部相机日志确认：

```text
/dev/video8
手动曝光
exposure_time_absolute=100
gain=4
white_balance_automatic=0
white_balance_temperature=4600
power_line_frequency=1（50 Hz）
```

头部相机为 `/dev/video0`，曾使用自动白平衡。胸部相机频闪调整后已消失。

图像颜色链路经过历史调整：采集 JPEG 使用 OpenCV 直接编码 RGB 数值数组，以适配训练 loader 用 OpenCV 解码后直接传 PIL 的方式。因此普通 JPG 查看器显示可能红蓝颠倒，不能仅凭 JPG 外观判断模型输入错误。此前增加过训练输入预览，并检查过训练/部署转换；新对话如需修改颜色，应重新对照当前代码，避免再次重复转换。

### 当前夹爪线性配置：已读取确认，尚未给用户最终检查结论

用户最新修改要求：

> 主臂输入 0%～50%，线性映射为从臂 0%～80%。

基础配置和 `.linear.yaml` 当前均有：

```yaml
master_gripper_control:
  enabled: true
  mapping_mode: linear
  activation_open_norm: 0.10
  input_full_open_norm: 0.50
  output_full_open_norm: 0.80
  command_rate_hz: 80
  feedback_refresh_rate_hz: 50
  feedback_refresh_timeout_s: 0.04
  teaching_kd: 0.05
  filter_alpha: 0.85
  deadband_norm: 0.005
  max_target_rate_norm_s: 2.50
  reopen_latch_margin_norm: 0.05
  feedback_timeout_s: 0.30
```

已确认 `hardware.py` 读取并校验 `output_full_open_norm`，实际映射为：

```python
mapped_norm = float(
    np.clip(master_norm / self._master_gripper_full_open_input, 0.0, 1.0)
    * self._master_gripper_full_open_output
)
```

位置约为 `hardware.py:1187`。因此配置不是只有注释修改，代码确实支持输出上限 0.80。

对应关系：

```text
主臂 0%  → 从臂 0%
主臂 25% → 从臂 40%
主臂 50% → 从臂 80%
主臂 >50% → 从臂保持映射目标80%
```

首次需要主臂开度超过 10% 才激活；激活后再闭合可跟随到 0%。滤波、变化速度限制和夹持保护会使最终目标与理想映射存在暂态区别。

当前端点配置：

```yaml
# 主臂
closed_rad: 1.8325957146  # 105°
open_rad: 0.2617993878    # 15°

# 从臂
closed_rad: 1.2217304764  # 70°
open_rad: -0.3490658504   # -20°
```

这里以当前 YAML 为准，历史口述端点曾有矛盾，不要直接依据早期对话翻转。

从臂开度 80% 对应约 `-2°`；主臂输入 50% 对应约 `60°`。

注意：输出 80% 是线性映射目标上限，不能未经检查就说键盘控制、启动保持、力保护等所有路径均被限制为 80%。

### 夹爪控制与保护：已实现

- 主臂 L7 使用 MIT 运控示教模式：`Kp=0`、`Kd=0.05`、前馈力矩 0，周期刷新反馈。
- 从臂 L7 使用 PP 位置模式。
- 当前从臂速度配置：

```yaml
sdk:
  gripper_velocity: 2.20
  gripper_acceleration: 2.80
```

- 电机限值：力矩 `0.10 Nm`、电流 `0.30 A`。
- 软件保护包括 HARD、SOFT、STALL。
- 当前软阈值 `0.12 Nm`，硬阈值 `0.18 Nm`。
- STALL：低速、存在闭合误差、EMA 力矩达到 `0.07 Nm` 并持续约 `0.30 s`。
- 触发后把目标设为检测到的当前开度并锁存。
- `backoff_norm=0`，不主动额外张开。
- 主臂明确打开超过保持目标一定幅度后解除锁存。

### 最新数据诊断结果：已完成只读分析，尚未修复

检查文件：

```text
robot/edulite_a3_collect/collected_data_single/
place_red_block_on_blue_block/demo_clean/data/episode3.hdf5
```

该文件为 1069 帧、30 Hz。用户截图在第 626 帧附近，发现 action “过冲”和 feedback 阶梯。

已确认两种现象来源不同：

1. **action 向上跳变符合力保护改写保持目标的逻辑。**

第 626～627 帧：

```text
主臂归一化开度：0.168 → 0.145
从臂 action：  0.214 → 0.506
从臂 feedback：0.525 → 0.500
```

主臂继续闭合，但从臂夹住物体后，保护把较小的闭合目标改为当前实际保持开度。不能仅凭 action 上跳认定机械夹爪实际弹开。

2. **从臂 L7 feedback 存在明显缓存过期现象。**

文件中完全相同的 L7 值连续保持：

```text
138～450帧：10.40秒，开度0.794
531～587帧：1.87秒，开度0.696
791～1068帧：9.23秒，开度0.641
```

第 51～52 帧，约 30 ms 内记录开度从 `0.215` 跳到 `0.708`，对应约 `44.4°`。结合配置运动速度，强烈表明“旧缓存保持，随后刷新”，而非真实瞬时运动。

代码依据：

- SDK 文件：

```text
robot/EDULITE_A3/el_a3_sdk/el_a3_sdk/interface.py
```

- `GetArmJointMsgs()` 约第 1310 行读取每个电机缓存，整臂时间戳取所有电机时间戳最大值。
- L1～L6 持续更新会掩盖 L7 已过期的问题。
- SDK 控制循环约第 513 行，只有 `_gripper_dirty` 为真才发送 PP 位置指令。
- 从臂没有像主臂一样独立周期刷新 L7 反馈，也没有独立检查 L7 的反馈年龄。

因此 `teleop/slave_joint_feedback` 第 7 维、`state` 的夹爪维度，以及力保护读取的 L7 位置/速度/力矩均需要进一步检查新鲜度。

绘图 `min/max` 数字是加了显示边距后的边界，不是真实数据极值。截图的 `max=1.067` 不代表记录开度超过 1；这条数据 action/feedback 实际最大值均约 0.998。

用户仍明确选择 action 作为训练目标，因为实测它与夹爪动作几乎没有延迟。不要擅自替换为 feedback。

### 训练及部署现状

真机训练配置：

```text
configs/edulite_single.yaml
configs/edulite_single_multitask.yaml
utils/stat-edulite-single-chest.json
utils/stat-edulite-single-chest-clean.json
```

曾训练完成 100 epochs，部署日志确认：

```text
action=7
state=8
camera=chest_camera
image_size=(320,240)
```

历史 checkpoint：

```text
checkpoints_vla/edulite_single_chest_clean/
sft_2026-09-27_19-00-44/checkpoint_epoch_100.pt
```

当时加载冻结 DINOv3：

```text
pretrained/dinov3-vitl16
hidden_size=1024
feat_layers=[-12,-8,-4]
fusion=concat
```

这些是历史运行结果，新的训练命令、checkpoint 和部署配置应以当前文件为准。

部署曾关闭 L1～L6 B 样条平滑；`history_actions=0` 在禁用时仍报校验错误的问题曾被提出。不要假设当前已经修复，需读代码确认。

### 归位与回放实验结果

- 初始 L1～L6：`[0,5,-5,0,0,0]°`。
- 采集：按 `a` 使能并归位；按 `r` 延迟约 3 秒开启示教和录制；保存/丢弃后可自动归位。
- 回放与采集曾共享 `home.py`，但在 RoboTwin 环境回放时 L3 始终约 `-1°`，达不到 `-5°`；在 `edulite_collect` 环境运行回放正常。
- 用户因此确认环境差异是重要因素。此前提出过撤销“额外补发归位位置”修改；是否已撤销以及具体环境依赖原因，应检查当前 `home.py` 和 SDK/依赖，不要宣称已明确定位。
- 部署按 `h` 的夹爪闭合归位功能因反复超时和弹开已要求完全移除，L7 归位阶段保持当前开度。

### 常用命令

线性夹爪示教采集堆叠任务：

```bash
conda activate edulite_collect
cd /home/lab347-no10/Tianyi/LiLa-WAM

python robot/edulite_a3_collect/collect.py \
  --config robot/edulite_a3_collect/config.master_slave.linear.yaml \
  --task-profile robot/edulite_a3_collect/task_profiles/place_red_block_on_blue_block.yaml
```

离线查看目标轨迹：

```bash
python robot/edulite_a3_vla/inspect_dataset_tk.py \
  robot/edulite_a3_collect/collected_data_single/place_red_block_on_blue_block/demo_clean/data/episode3.hdf5 \
  --camera chest_camera \
  --trajectory-source action
```

查看反馈轨迹：把最后一个参数改为 `--trajectory-source feedback`。

### 未解决问题与下一步计划

1. **先完成最新映射检查的反馈。**已读代码确认 0～50% → 0～80% 生效；需说明首次 10% 激活、80% 饱和、保护改写目标，以及这些百分比代表电机角度归一化。
2. **修复从臂 L7 反馈新鲜度。**计划在保持 PP 模式的情况下采用 SDK/协议支持的周期查询，按 L7 独立时间戳检测过期；目前没有实施。
3. **增加夹爪诊断记录与叠加图。**计划记录主臂输入、映射目标、发送目标、反馈及反馈年龄、力保护锁存，帮助区分目标改写和机械过冲；目前没有实施。
4. 旧数据缺失的 L7 动态过程无法通过简单插值可靠恢复。优先保留原数据；不要未经授权平滑改写训练标签。
5. 实验时不要自动连接 CAN、使能电机或执行机械臂动作，诊断先使用离线检查。

仓库当前有大量用户修改、未跟踪文件及部分任务条件文件删除。`robot/` 在当前 Git 状态下显示未跟踪。不要使用整体回滚或清理操作，不要碰与当前任务无关的训练和模型修改。

---
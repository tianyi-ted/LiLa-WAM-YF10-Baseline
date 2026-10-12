# LiLa-WAM 原生单臂真机推理

当前程序是 `real_vla.py`，只接受训练配置中的 `action_dim=7`、`state_dim=8`，执行从臂 can0 的 L1～L6 和 L7。使用 [config.real_single.yaml](config.real_single.yaml)；其中 checkpoint 路径仍是占位符，必须在完成新数据训练后填入实际目录。旧 RoboTwin 14D checkpoint、旧统计量和[归档配置](../docs/legacy/config.real_vla.14d.yaml)与当前程序不兼容。

## 采集数据真机回放

`replay_dataset.py` 只读取 HDF5 的 `joint_action/vector`，不加载 VLA 模型、不打开相机。默认是只读 dry preview；只有显式加入 `--execute` 才会连接 can0 并使能机械臂。程序会使用当前 collector 配置的 L7 力矩/电流限制、关节边界和夹爪力矩急停保护，默认以录制速度的 0.5 倍回放：

```bash
conda activate edulite_collect
cd /home/lab347-no10/Tianyi/LiLa-WAM

# 先只检查 episode，不连接 CAN、不运动
python robot/edulite_a3_vla/replay_dataset.py \
  --dataset-root robot/edulite_a3_collect/collected_data_single \
  --task place_red_block_into_box \
  --episode episode0

# 确认无误后才执行；首次建议半速或更低速
python robot/edulite_a3_vla/replay_dataset.py \
  --config robot/edulite_a3_collect/config.master_slave.yaml \
  --dataset-root robot/edulite_a3_collect/collected_data_single \
  --task place_red_block_into_box \
  --episode episode0 \
  --speed 0.5 \
  --execute
```

启动后程序只连接 can0，不会使能电机；输入 `a` 回车后才使能、建立当前位置保持并执行 collector 的 `episode_reset.target_deg` 回位。回位完成后按 Enter 才开始回放。运行中按 `e` 触发软件急停，按 `q` 中止回放，空格暂停/继续；物理急停必须始终可触达。`--max-seconds` 默认 90 秒，超过后会停止并断使能。`--no-home` 只应在机械臂已经确认处于该 episode 的起始姿态时使用。回放不会修改 HDF5。

回位使用与采集程序相同的 SDK `MoveJ + wait_for_motion + 反馈误差校验`，目标来自 `config.master_slave.yaml` 的 `episode_reset.target_deg`（当前为 `[0, 5, -5, 0, 0, 0]` 度）。回放只控制单臂任务臂 can0，不会让 can1 主臂运动。启动日志应出现 `REPLAY HOMING (same SDK MoveJ path as collector)`；如果仍显示旧的 `REPLAY HOMING ...`，说明运行的是旧路径下的脚本。

## 数据集 Tkinter 可视化

`inspect_dataset_tk.py` 是完全离线的 HDF5 查看器，不连接 CAN、不使能电机。它可以播放/暂停 episode，拖动帧号，同时显示胸部/头部相机图像、L1～L6 曲线、L7 归一化夹爪值和当前动作。

```bash
conda activate RoboTwin
cd /home/lab347-no10/Tianyi/LiLa-WAM

# 自动打开数据根目录下找到的第一个 episode
python robot/edulite_a3_vla/inspect_dataset_tk.py \
  --dataset-root robot/edulite_a3_collect/collected_data_single_static_clean \
  --camera chest_camera

# 指定 episode
python robot/edulite_a3_vla/inspect_dataset_tk.py \
  robot/edulite_a3_collect/collected_data_single/place_red_block_into_box/demo_clean/data/episode104.hdf5 \
  --camera chest_camera
```

界面按键/控件：`Play` 播放、`Pause` 暂停、`<<`/`>>` 前后跳 30 帧、底部滑块跳转、`Open episode` 选择其他 HDF5。重点检查 action 与 `teleop/slave_joint_feedback` 是否存在明显偏差、L3 是否跳变、夹爪方向是否反向，以及相机画面和动作时间是否同步。

## 当前真机相机输入

真机部署使用 `config.real_single.yaml` 中的 `chest_camera`。相机采集线程输出 RGB；`real_vla.py` 在送入模型前统一经过 `_model_camera_image()` 的通道转换，数据集原文件不改变。训练配置中的 `dataset.camera_names` 必须仍为 `["chest_camera"]`。

在 `RoboTwin` 环境下，先检查配置、checkpoint、统计量和任务向量；该检查不会打开相机或 CAN：

```bash
cd /home/lab347-no10/Tianyi/LiLa-WAM
python robot/edulite_a3_vla/real_vla.py \
  --config robot/edulite_a3_vla/config.real_single.yaml --check-only
```

检查通过且现场安全后，才使用 `--execute`。程序先加载模型和相机、连接 CAN，**不会自动使能电机**。交互按键为 `a` 使能并回位、`p` 静止预览、`r` 倒计时执行、`s` 保持、`h` 回 L1～L6 初始位（L7 不动）、`e` 软件急停、`q` 退出。每轮执行前先看 `p` 的预测可视化；物理急停始终保持可及。

多任务 checkpoint 可用 `--task-name` 选择对应的预计算任务向量，并用 `--instruction` 改日志描述。例如红块叠蓝块使用 `--task-name place_red_block_on_blue_block`。这两个参数不会改变 checkpoint 或归一化统计量。

完整的采集、统计、训练和部署步骤见[单臂指南](../edulite_a3_collect/SINGLE_ARM_GUIDE.md)。旧版 14D/16D 部署说明保存在[历史文档](../docs/legacy/REAL_VLA_README_14D.md)，不能直接用于 LiLa-WAM。

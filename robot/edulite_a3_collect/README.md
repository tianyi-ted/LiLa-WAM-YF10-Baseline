# EDULITE-A3 单臂主从采集

当前入口是 `collect.py` + 主从配置：can1 主臂示教，can0 从臂执行；当前配置暂时只采集 `chest_camera`（头部相机已禁用）。硬件主配置与任务 profile 分离，action 为 `[T,7]`，训练 state 为 `[T,8]`，不再补零右臂。不要使用 `config.example.yaml` 代替当前配置。

夹爪映射有两个可直接传给 `--config` 的入口。它们继承 `config.master_slave.yaml` 的 CAN、相机和安全参数，只覆盖主从夹爪映射：

```text
config.master_slave.linear.yaml  连续线性跟随
config.master_slave.binary.yaml  二档开合
```

连续线性模式（推荐用于记录连续夹爪动作）：

```bash
python robot/edulite_a3_collect/collect.py \
  --config robot/edulite_a3_collect/config.master_slave.linear.yaml
```

主臂输入 0%～50% 映射为从臂 0%～80%，超过 50% 后饱和为 80%；主臂打开超过 10% 后激活。

二档模式：

```bash
python robot/edulite_a3_collect/collect.py \
  --config robot/edulite_a3_collect/config.master_slave.binary.yaml
```

主臂打开超过 40% 时从臂打开到 80%，低于 40% 时从臂闭合。两种模式都保留夹爪滤波、速度限制、力矩/电流限制和停转保护。

第一个任务“红块放入盒中”使用独立 profile：

```bash
conda activate edulite_collect
cd /home/lab347-no10/Tianyi/LiLa-WAM
python robot/edulite_a3_collect/collect.py \
  --config robot/edulite_a3_collect/config.master_slave.yaml \
  --task-profile robot/edulite_a3_collect/task_profiles/place_red_block_into_box.yaml
```

第二个任务“红块放到蓝块上”同样只覆盖任务名和指令，不复制硬件参数：

```bash
python robot/edulite_a3_collect/collect.py \
  --config robot/edulite_a3_collect/config.master_slave.yaml \
  --task-profile robot/edulite_a3_collect/task_profiles/place_red_block_on_blue_block.yaml
```

`a` 使能并回到 episode 初始位；`r` 倒计时后开始示教和记录；`s` 保存；`d` 丢弃；`h` 回位；`t` 打印夹爪状态；`e` 软件急停；`q` 退出。主臂夹爪的具体映射由上述 linear/binary 配置决定；实际运动仍受限速与力保护约束。首次使用或修改参数后，请先在空载、物理急停可及的条件下测试。

```bash
python robot/edulite_a3_collect/validate_dataset.py \
  robot/edulite_a3_collect/collected_data_single --require-single
```

## 轨迹回放与离线检查

回放脚本位于 `../edulite_a3_vla/replay_dataset.py`，默认只检查 HDF5；加入 `--execute` 后先连接 can0，但不会立即使能。输入 `a` 回车后才使能、建立当前位置保持并触发回位，回位完成后再按 Enter 开始回放。建议先用 `--speed 0.5`：

```bash
python robot/edulite_a3_vla/replay_dataset.py \
  --config robot/edulite_a3_collect/config.master_slave.yaml \
  --dataset-root robot/edulite_a3_collect/collected_data_single \
  --task place_red_block_into_box \
  --episode episode104.hdf5 \
  --speed 0.5 \
  --execute
```

回放初始化采用和采集相同的 `MoveJ` 轨迹规划、等待和反馈误差检查；它只控制单臂任务臂 can0，不会驱动 can1 主臂。`e` 为软件急停，`q` 中止，空格暂停/继续，物理急停必须可触达。

离线查看轨迹和图像可使用：

```bash
python robot/edulite_a3_vla/inspect_dataset_tk.py \
  robot/edulite_a3_collect/collected_data_single/place_red_block_into_box/demo_clean/data/episode104.hdf5 \
  --camera chest_camera
```

该程序只读 HDF5，不连接机械臂。建议先检查轨迹曲线和夹爪开合，再进行 `--execute` 回放。

进一步阅读：[主从操作与安全](MASTER_SLAVE_GUIDE.md)、[完整单臂数据/训练/部署流程](SINGLE_ARM_GUIDE.md)、[相机固定参数](CAMERA_CONTROLS.md)。迁移前的补零数据说明与旧命令保存在[历史文档](../docs/legacy/README.md)，不适用于当前配置。

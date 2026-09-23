# EDULITE-A3 单臂主从采集

当前入口是 `collect.py` + `config.master_slave.yaml`：can1 主臂示教，can0 从臂执行；同时采集 `head_camera` 和 `chest_camera`。数据写入 `collected_data_single/place_red_block_into_box/demo_clean/`，action 为 `[T,7]`，训练 state 为 `[T,8]`，不再补零右臂。不要使用 `config.example.yaml` 代替当前配置。

```bash
conda activate edulite_collect
cd /home/lab347-no10/Tianyi/LiLa-WAM
python robot/edulite_a3_collect/collect.py \
  --config robot/edulite_a3_collect/config.master_slave.yaml
```

`a` 使能回位；`r` 倒计时后开始示教和记录；`s` 保存；`d` 丢弃；`h` 回位；`t` 打印夹爪状态；`e` 软件急停；`q` 退出。主臂夹爪开度大于 40% 对应从臂打开目标 80%，小于 40% 对应闭合；实际运动仍受限速与力保护约束。首次使用或修改参数后，请先在空载、物理急停可及的条件下测试。

```bash
python robot/edulite_a3_collect/validate_dataset.py \
  robot/edulite_a3_collect/collected_data_single --require-single
```

进一步阅读：[主从操作与安全](MASTER_SLAVE_GUIDE.md)、[完整单臂数据/训练/部署流程](SINGLE_ARM_GUIDE.md)、[相机固定参数](CAMERA_CONTROLS.md)。迁移前的补零数据说明与旧命令保存在[历史文档](../docs/legacy/README.md)，不适用于当前配置。

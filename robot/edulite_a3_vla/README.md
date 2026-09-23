# LiLa-WAM 原生单臂真机推理

当前程序是 `real_vla.py`，只接受训练配置中的 `action_dim=7`、`state_dim=8`，执行从臂 can0 的 L1～L6 和 L7。使用 [config.real_single.yaml](config.real_single.yaml)；其中 checkpoint 路径仍是占位符，必须在完成新数据训练后填入实际目录。旧 RoboTwin 14D checkpoint、旧统计量和[归档配置](../docs/legacy/config.real_vla.14d.yaml)与当前程序不兼容。

在 `RoboTwin` 环境下，先检查配置、checkpoint、统计量和任务向量；该检查不会打开相机或 CAN：

```bash
cd /home/lab347-no10/Tianyi/LiLa-WAM
python robot/edulite_a3_vla/real_vla.py \
  --config robot/edulite_a3_vla/config.real_single.yaml --check-only
```

检查通过且现场安全后，才使用 `--execute`。程序先加载模型和相机、连接 CAN，**不会自动使能电机**。交互按键为 `a` 使能并回位、`p` 静止预览、`r` 倒计时执行、`s` 保持、`h` 回 L1～L6 初始位（L7 不动）、`e` 软件急停、`q` 退出。每轮执行前先看 `p` 的预测可视化；物理急停始终保持可及。

完整的采集、统计、训练和部署步骤见[单臂指南](../edulite_a3_collect/SINGLE_ARM_GUIDE.md)。旧版 14D/16D 部署说明保存在[历史文档](../docs/legacy/REAL_VLA_README_14D.md)，不能直接用于 LiLa-WAM。

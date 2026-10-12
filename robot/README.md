# LiLa-WAM 真机目录

当前执行任务的是 EDULITE-A3 **can0 从臂**；can1 只用作主臂示教。正式数据与模型使用原生 7 维 action、8 维 state，不使用旧 RoboTwin 的右臂补零格式。

```text
robot/
├── edulite_a3_collect/   当前采集程序、相机配置和数据校验
├── edulite_a3_vla/       当前 7D/8D 真机推理与安全控制
├── EDULITE_A3/          上游机械臂 SDK/ROS 源码，采集器按此路径导入 SDK
├── ABPolicy-code/       参考实现，不是当前采集或推理入口
└── docs/
    ├── legacy/          迁移前的 14D/16D 文档和配置快照
    └── research/        参考模型的设计备忘
```

从[采集入口](edulite_a3_collect/README.md)和[真机部署入口](edulite_a3_vla/README.md)开始阅读；包含数据统计、任务向量、训练命令的完整流程见[单臂指南](edulite_a3_collect/SINGLE_ARM_GUIDE.md)。相机参数请看[相机说明](edulite_a3_collect/CAMERA_CONTROLS.md)。

`edulite_a3_collect/collected_data_single/` 是正式采集数据，目前不随代码整理移动；`edulite_a3_vla/inference_outputs/` 是旧推理记录，也原样保留。两套数据与输出都不是源码。`EDULITE_A3/` 和 `ABPolicy-code/` 各自带独立 `.git`，是迁移来的第三方/参考仓库；本次没有修改或重排，以免破坏 SDK 导入路径和既有环境。

`config.master_slave.yaml` 是当前采集配置，`config.real_single.yaml` 是当前部署模板。`config.example.yaml` 只是通用示例；旧版配置及旧模型说明已移到 `docs/legacy/`，**不要把旧 14D checkpoint 或统计量用于当前 7D 模型**。

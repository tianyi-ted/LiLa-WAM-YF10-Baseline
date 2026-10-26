# 独立真机部署环境 edulite_deploy

2026-10-22 已在本机创建 `/home/lab347-no10/anaconda3/envs/edulite_deploy`。
以 RoboTwin 克隆为基础，保留模型推理版本，安装 edulite_collect 的 Pinocchio 动力学依赖。
SDK 以 editable 方式注册到本仓库 `robot/EDULITE_A3/el_a3_sdk`；代码更新会立即生效。
两个源 conda 环境没有修改。

| 组件 | 部署环境版本 |
|---|---|
| Python | 3.10.20 |
| PyTorch / CUDA wheel | 2.4.1 / cu121 |
| torchvision | 0.19.1 |
| Transformers | 5.13.0 |
| NumPy | 2.2.6 |
| Pinocchio (`pin`) | 4.1.0 |
| OpenCV | 4.11.0 |
| SDK | 本仓库 el_a3_sdk 1.0.0 |

这不是两个环境的所有包直接合并：Pinocchio 的 cmeel-boost 1.90 要求 NumPy >=2，
而 RoboTwin 的 mplib 0.2.1 要求 NumPy <2。部署环境升级 NumPy 并移除了部署不使用的 mplib。
未添加采集环境中的 Qt/VTK 调试器依赖。训练和仿真实验继续使用各自原环境；此环境用于真机 VLA 部署。
Pinocchio 直接从 edulite_deploy 内导入，不需要设置 EDULITE_PINOCCHIO_PATH 或借用其他环境路径。

## 使用

```bash
conda activate edulite_deploy
cd /home/lab347-no10/Tianyi/LiLa-WAM

# 离线：路径、统计量、包来源、NumPy/Torch 转换、SDK 正运动学和重力计算。
python robot/edulite_a3_vla/check_deploy_env.py

# 离线：额外加载当前部署配置中的 checkpoint，对录制首帧进行 GPU 推理。
python robot/edulite_a3_vla/check_deploy_env.py \
  --episode robot/edulite_a3_collect/collected_data_single/place_red_block_on_blue_block/demo_clean/data/episode0.hdf5

python robot/edulite_a3_vla/real_vla.py \
  --config robot/edulite_a3_vla/config.real_on_blue_block.yaml --check-only

# 现场确认后运行交互部署：a 使能归位，p 预览，r 执行，s 保持，e 急停，q 退出。
python robot/edulite_a3_vla/real_vla.py \
  --config robot/edulite_a3_vla/config.real_on_blue_block.yaml --execute
```

## 重建参考（已有环境无需重复执行）

```bash
conda create -n edulite_deploy --clone RoboTwin --offline -y
conda activate edulite_deploy
cd /home/lab347-no10/Tianyi/LiLa-WAM
python -m pip uninstall -y mplib
python -m pip install -r robot/edulite_a3_vla/environment/requirements-hardware.txt
python -m pip install --no-deps --no-build-isolation -e ./robot/EDULITE_A3/el_a3_sdk
python -m pip check
```

完整已安装版本见 [pip 清单](environment/edulite_deploy-pip-freeze.txt)，
conda 管理的包见 [conda 清单](environment/edulite_deploy-conda-explicit.txt)。
conda 清单只覆盖 conda 管理的包，不能单独恢复 pip 包；克隆命令也依赖执行时 RoboTwin 的状态。
pip 清单中 editable/本地路径依赖需要对应源码，因此不是跨机器的一键安装文件。

## 已验证及边界

- `pip check` 无冲突。
- Pinocchio 来自新环境内部；NumPy/Torch 数组双向转换通过。
- SDK home 位姿 FK 和重力计算得到有限值。
- 当前叠蓝块配置的 epoch 40 checkpoint、DINOv3-L、统计量和 1024D task condition 加载通过，
  对 episode0 首帧完成 GPU 推理并输出有限的 7D action。

以上均为离线检查，没有连接 CAN、打开相机或使能电机。
建立独立环境减少了依赖混用，但不能据此认定之前的归位/跟踪误差已解决；现场仍需验证。

# RoboTwin VLA 与 EDULITE-A3 真实数据实验指南

> 历史归档：本文针对旧 `robotwin_code` 的 14D/16D 补零训练流程。
> 当前 LiLa-WAM 原生单臂流程见
> [SINGLE_ARM_GUIDE.md](../../edulite_a3_collect/SINGLE_ARM_GUIDE.md)。

> 本文档参照项目根目录 `EXPERIMENT_GUIDE.md` 编写，记录同一套 VLA
> 代码在两种独立数据入口下的完整实验流程。RoboTwin 仿真数据与
> EDULITE-A3 真实数据不联合训练，不共享归一化统计量，不混用 checkpoint。

---

## 目录

1. [实验原则与当前数据](#1-实验原则与当前数据)
2. [环境准备](#2-环境准备)
3. [数据目录与格式](#3-数据目录与格式)
4. [数据检查与统计量](#4-数据检查与统计量)
5. [训练入口参数](#5-训练入口参数)
6. [RoboTwin 仿真实验](#6-robotwin-仿真实验)
7. [EDULITE-A3 真实数据实验](#7-edulite-a3-真实数据实验)
8. [Checkpoint 与推理检查](#8-checkpoint-与推理检查)
9. [实验记录模板](#9-实验记录模板)
10. [常见问题排查](#10-常见问题排查)

---

## 1. 实验原则与当前数据

### 1.1 两个独立入口

| 项目 | RoboTwin 仿真入口 | EDULITE-A3 真实入口 |
|---|---|---|
| 数据根目录 | `/home/lab347-no10/Tianyi/RoboTwin/data` | `/home/lab347-no10/Tianyi/robotwin_code/robot/edulite_a3_collect/collected_data` |
| 数据分片 | `both` | `clean` |
| 统计文件 | `utils/stat-robotwin-current-400.json` | `utils/stat-edulite-smoke-20.json` |
| Checkpoint 根目录 | `checkpoints_vla/robotwin_current_400` | `checkpoints_vla/real_edulite_20` |
| 仿真评估 | `eval_vla.py` | 不适用 |

切换入口只通过训练命令参数完成，不需要反复编辑
`configs/robotwin.yaml`。原有 RoboTwin 默认路径仍保留在该配置中。

### 1.2 当前数据规模

```text
RoboTwin:
  4 个任务
  400 episodes
  81,389 frames
  demo_clean + demo_randomized

EDULITE-A3:
  1 个测试任务 master_slave_test
  20 episodes
  17,045 frames
  约 9.46 分钟
  仅 demo_clean
```

当前20条真实数据是流程测试数据：

- 使用通用测试指令，不代表正式操作任务；
- `episode0～15` 与 `episode16～19` 跨越了两版夹爪映射配置；
- 图像实际保存为 `640×360`，不是 YAML 请求的 `320×240`；
- 单臂数据通过 `zero_right` 补成 action 14维、state 16维；
- 只能用于验证训练流程、观察 loss 和检查 checkpoint，不能用于评价真实成功率。

---

## 2. 环境准备

### 2.1 激活训练环境

```bash
conda activate RoboTwin
cd /home/lab347-no10/Tianyi/robotwin_code
```

### 2.2 检查 PyTorch、CUDA 与 GPU

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print('BF16:', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)"

nvidia-smi
```

当前机器预期为 RTX 4070 Ti 12 GB。只有 `CUDA: True` 时才开始 GPU
训练。已有成功的 RoboTwin Stage 2 实验使用 `batch_size=128`；若更换模型、
图像配置或出现 OOM，再逐步降低。

### 2.3 检查模型与核心依赖

```bash
test -f Qwen3-VL-2B-Instruct/model.safetensors && echo "Qwen weights OK"

python -c "import torch, transformers, h5py, cv2, omegaconf; print('dependencies OK')"
```

### 2.4 检查磁盘

```bash
df -h /home/lab347-no10/Tianyi
du -sh checkpoints_vla
```

Stage 1 测试 checkpoint 约331 MB，Stage 2 测试 checkpoint 约292 MB。
避免每个 epoch 都保存一次。

---

## 3. 数据目录与格式

### 3.1 标准目录

两类数据都必须满足：

```text
{dataset_root}/
└── {task_name}/
    ├── demo_clean/
    │   ├── data/
    │   │   ├── episode0.hdf5
    │   │   └── ...
    │   └── instructions/
    │       ├── episode0.json
    │       └── ...
    └── demo_randomized/        # 可选
        ├── data/
        └── instructions/
```

文件名允许 `episode0.hdf5` 或 `episode_0.hdf5`，但同一条 HDF5 必须有
同名 instruction JSON。

### 3.2 模型使用字段

```text
joint_action/vector:             [T, 14]
endpose/left_endpose:            [T, 7]
endpose/left_gripper:            [T]
endpose/right_endpose:           [T, 7]
endpose/right_gripper:           [T]
observation/head_camera/rgb:     [T] JPEG bytes
```

训练时 state 拼接为：

```text
[left xyz(3), left quaternion wxyz(4), left gripper(1),
 right xyz(3), right quaternion wxyz(4), right gripper(1)]
= 16维
```

EDULITE 单臂数据中：

```text
action[0:7]   = 从臂六关节目标 + 夹爪目标
action[7:14]  = 0
state[0:8]    = 从臂末端位姿 + 实际夹爪
state[8:16]   = 0
```

`teleop/*` 与 `timing/*` 用于诊断，不参与当前模型训练。

### 3.3 真实数据采集

```bash
conda activate edulite_collect
cd /home/lab347-no10/Tianyi/robotwin_code

python robot/edulite_a3_collect/collect.py \
  --config robot/edulite_a3_collect/config.master_slave.yaml
```

当前主从流程：

```text
a -> 两臂使能并回初始位
r -> 3秒倒计时 -> 开始主从示教与记录
s -> 保存 episode -> 自动回初始位
d -> 丢弃 episode -> 自动回初始位
e -> 软件急停
q -> 退出
t -> 打印夹爪诊断
h -> 未录制时手动回初始位
```

完整机械臂操作与安全说明见同目录 `README.md` 和
`MASTER_SLAVE_GUIDE.md`。

---

## 4. 数据检查与统计量

### 4.1 验证真实 HDF5

```bash
conda activate edulite_collect
cd /home/lab347-no10/Tianyi/robotwin_code

python robot/edulite_a3_collect/validate_dataset.py \
  robot/edulite_a3_collect/collected_data/master_slave_test/demo_clean \
  --require-dual
```

当前预期：

```text
Checked 20 episode(s), failures=0
```

### 4.2 统计 episode 和帧数

```bash
conda activate RoboTwin
cd /home/lab347-no10/Tianyi/robotwin_code

python - <<'PY'
from pathlib import Path
import h5py

profiles = {
    "robotwin": Path("/home/lab347-no10/Tianyi/RoboTwin/data"),
    "edulite": Path("/home/lab347-no10/Tianyi/robotwin_code/robot/edulite_a3_collect/collected_data"),
}

for name, root in profiles.items():
    files = list(root.glob("*/demo_clean/data/*.hdf5"))
    files += list(root.glob("*/demo_randomized/data/*.hdf5"))
    frames = 0
    for path in files:
        with h5py.File(path, "r") as f:
            frames += int(f["joint_action/vector"].shape[0])
    print(name, "episodes=", len(files), "frames=", frames)
PY
```

### 4.3 计算 RoboTwin 统计量

```bash
python utils/calc_stat.py \
  --root_dir /home/lab347-no10/Tianyi/RoboTwin/data \
  --output_path ./utils/stat-robotwin-current-400.json \
  --outlier_path ./utils/outlier-robotwin-current-400.txt \
  --num_processes 8 \
  --data_mode both
```

当前预期：

```text
Processed 400/400 files
Action Dim: 14
State Dim: 16
```

### 4.4 计算 EDULITE 统计量

每次新增、删除或重新采集真实 episode 后都必须重新执行：

```bash
python utils/calc_stat.py \
  --root_dir /home/lab347-no10/Tianyi/robotwin_code/robot/edulite_a3_collect/collected_data \
  --output_path ./utils/stat-edulite-smoke-20.json \
  --outlier_path ./utils/outlier-edulite-smoke-20.txt \
  --num_processes 4 \
  --data_mode clean
```

正式任务数据建议把文件名改成不含 `smoke-20` 的名称，例如：

```text
utils/stat-edulite-real-pick-v1.json
```

并在所有真实 Stage 1、Stage 2 和离线推理命令中同步替换。

### 4.5 不要混用统计量

归一化将 action/state 从物理范围映射到 `[-1,1]`。使用错误统计文件会造成：

- 真实关节角被按仿真范围错误缩放；
- 模型 loss 异常或看似下降但输出无意义；
- 推理反归一化得到错误动作；
- 真机执行时产生越界风险。

当前仿真 outlier 报告有24个 episode 出现绝对关节值大于3 rad。流程测试可以
继续，正式实验前必须结合任务关节范围逐条确认，不要直接按固定 π 阈值删除
EDULITE 数据。

---

## 5. 训练入口参数

`train_vla.py` 现在保留配置文件默认值，同时允许命令行选择独立数据入口：

| 参数 | 含义 |
|---|---|
| `--config` | 模型与公共训练配置 |
| `--dataset_dir` | 本次训练唯一的数据根目录 |
| `--data_mode` | `clean`、`randomized` 或 `both` |
| `--norm_stats_path` | 与本次数据严格匹配的统计文件 |
| `--save_dir` | 本次实验的 checkpoint 根目录 |
| `--stage1` | 启用 Stage 1 |
| `--stage1_checkpoint` | Stage 2 使用的 Stage 1 权重 |
| `--freeze_stage1` | Stage 2 冻结 Stage 1 参数 |
| `--batch_size` | 覆盖当前阶段 batch size |
| `--num_workers` | 覆盖 DataLoader worker 数 |
| `--max_train_steps` | 可选的 optimizer 更新次数上限，仅用于冒烟短跑或明确的按 step 实验 |
| `--save_interval` | 每多少 epoch 保存一次 |

`--stage1 True` 与 `--stage1` 都能使用；推荐简写为 `--stage1`。
`--freeze_stage1 True` 与 `--freeze_stage1` 同理。

### 5.1 `max_train_steps` 语义

```text
一个 train step = 完成一次 optimizer.step()
```

它不是 batch 数，也不是原始帧数。正式按 epoch 训练时不要传这个参数；
否则会在达到步数上限后提前终止。启用梯度累积时：

```bash
--grad_accum_steps 4
```

每4个 batch 才计一个 train step。达到限制后程序会保存最终 checkpoint
并退出。

---

## 6. RoboTwin 仿真实验

### 6.1 Stage 1

```bash
conda activate RoboTwin
cd /home/lab347-no10/Tianyi/robotwin_code

python train_vla.py \
  --config ./configs/robotwin.yaml \
  --dataset_dir /home/lab347-no10/Tianyi/RoboTwin/data \
  --data_mode both \
  --stage1 \
  --norm_stats_path ./utils/stat-robotwin-current-400.json \
  --save_dir ./checkpoints_vla/robotwin_current_400 \
  --epochs 25 \
  --batch_size 128 \
  --num_workers 0 \
  --save_interval 25
```

预期输出：

```text
Successfully scanned 400 total episode files
Dataset Size: 81389
Action Model: 57.79M trainable parameters
VLAWrapperTwoStage initialized in Stage 1
```

输出路径：

```text
checkpoints_vla/robotwin_current_400/stage1/{timestamp}/checkpoint_epoch_25.pt
```

### 6.2 Stage 2

```bash
python train_vla.py \
  --config ./configs/robotwin.yaml \
  --dataset_dir /home/lab347-no10/Tianyi/RoboTwin/data \
  --data_mode both \
  --norm_stats_path ./utils/stat-robotwin-current-400.json \
  --stage1_checkpoint ./checkpoints_vla/robotwin_current_400/stage1/{stage1_timestamp}/checkpoint_epoch_25.pt \
  --freeze_stage1 \
  --save_dir ./checkpoints_vla/robotwin_current_400 \
  --epochs 50 \
  --batch_size 128 \
  --num_workers 12 \
  --save_interval 5
```

预期输出：

```text
Loading Frozen VLM
Loading Stage 1 weights
Freezing Stage 1 parameters
Total Parameters: 89.34M
Trainable Parameters: 31.55M
Epoch 50 Completed
```

### 6.3 RoboTwin 仿真评估

Stage 2 保存路径为：

```text
checkpoints_vla/robotwin_current_400/stage2/{timestamp}/checkpoint_epoch_50.pt
```

因此评估命令为：

```bash
python eval_vla.py \
  --config_path ./configs/robotwin.yaml \
  --norm_stats_path ./utils/stat-robotwin-current-400.json \
  --ckpt_setting robotwin_current_400/stage2/{timestamp} \
  --checkpoint_ep 50 \
  --test_num 50 \
  --instruction_type unseen
```

评估前将 `eval_vla.py` 的 `tasks_to_test` 调整为当前存在的任务：

```python
tasks_to_test = [
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "grab_roller",
]
```

---

## 7. EDULITE-A3 真实数据实验

### 7.1 10步/3步冒烟测试

流程已经实测通过：

```text
Stage 1: 10 steps, loss=1.3021
Stage 2: 3 steps, loss=4.5713
```

复现命令见同目录 `TRAIN_SMOKE_TEST.md`。

### 7.2 真实数据 Stage 1

```bash
conda activate RoboTwin
cd /home/lab347-no10/Tianyi/robotwin_code

python train_vla.py \
  --config ./configs/robotwin.yaml \
  --dataset_dir /home/lab347-no10/Tianyi/robotwin_code/robot/edulite_a3_collect/collected_data \
  --data_mode clean \
  --stage1 \
  --norm_stats_path ./utils/stat-edulite-smoke-20.json \
  --save_dir ./checkpoints_vla/real_edulite_20 \
  --epochs 10 \
  --batch_size 128 \
  --num_workers 0 \
  --save_interval 10
```

预期：

```text
Successfully scanned 20 total episode files
Dataset Size: 17045
Batches per Epoch: 133
```

输出：

```text
checkpoints_vla/real_edulite_20/stage1/{timestamp}/checkpoint_epoch_10.pt
```

### 7.3 真实数据 Stage 2

```bash
python train_vla.py \
  --config ./configs/robotwin.yaml \
  --dataset_dir /home/lab347-no10/Tianyi/robotwin_code/robot/edulite_a3_collect/collected_data \
  --data_mode clean \
  --norm_stats_path ./utils/stat-edulite-smoke-20.json \
  --stage1_checkpoint ./checkpoints_vla/real_edulite_20/stage1/{stage1_timestamp}/checkpoint_epoch_10.pt \
  --freeze_stage1 \
  --save_dir ./checkpoints_vla/real_edulite_20 \
  --epochs 10 \
  --batch_size 128 \
  --num_workers 12 \
  --save_interval 2
```

这里冻结 Stage 1，因为当前真实数据很少。Stage 2 输出：

```text
checkpoints_vla/real_edulite_20/stage2/{timestamp}/checkpoint_epoch_10.pt
```

### 7.4 正式真实任务前必须修改

在 `config.master_slave.yaml` 中至少修改：

```yaml
collection:
  task_name: real_pick_and_place_v1
  instructions:
    seen:
      - "pick up the object and place it in the target container"
    unseen:
      - "grasp the object and move it into the container"
```

新任务应使用新目录，不要继续把正式数据写进 `master_slave_test`。建议：

```text
collected_data/
└── real_pick_and_place_v1/
    └── demo_clean/
```

正式数据必须：

1. 全部使用同一版夹爪与滤波参数；
2. 每条 episode 从相同或明确随机化的初态开始；
3. 失败示范不混入 `demo_clean`；
4. 单独保留验证 episode；
5. 重新计算真实统计文件。

### 7.5 真机测试边界

`eval_vla.py` 只连接 RoboTwin/SAPIEN 仿真环境，不能用于控制 EDULITE。
当前仓库已经完成真实示教采集，但尚未提供经过完整安全验证的 VLA 真机执行器。

真实推理执行器至少需要：

```text
相机帧 + 16维真实state
        -> VLA推理
        -> 只取左侧action[0:7]
        -> 反归一化
        -> 关节/夹爪硬裁剪
        -> 速度和加速度限制
        -> 动作chunk执行与反馈看门狗
        -> 物理急停/软件急停
```

在该执行器完成前，只做 checkpoint 加载和离线前向，不能把模型输出直接发送
到电机。

---

## 8. Checkpoint 与推理检查

### 8.1 查找最新 checkpoint

RoboTwin：

```bash
find checkpoints_vla/robotwin_current_400 -type f \
  -name 'checkpoint_epoch_*.pt' -printf '%T@ %p\n' | sort -nr | head
```

真实数据：

```bash
find checkpoints_vla/real_edulite_20 -type f \
  -name 'checkpoint_epoch_*.pt' -printf '%T@ %p\n' | sort -nr | head
```

### 8.2 检查 checkpoint 元数据

```bash
python - <<'PY'
import torch

path = "checkpoints_vla/real_edulite_20/stage2/{timestamp}/checkpoint_epoch_10.pt"
ckpt = torch.load(path, map_location="cpu", weights_only=False)
print("stage:", ckpt["stage"])
print("epoch:", ckpt["epoch"])
print("loss:", ckpt["loss"])
print("model tensors:", len(ckpt["model_state_dict"]))
PY
```

Stage 1 的 `stage` 应为1，Stage 2 应为2。

### 8.3 离线加载 Stage 2

以下命令只加载模型，不连接机械臂：

```bash
python - <<'PY'
from robotwin_infer import RobotWinInference

model = RobotWinInference(
    config_path="./configs/robotwin.yaml",
    checkpoint_path="./checkpoints_vla/real_edulite_20/stage2/{timestamp}/checkpoint_epoch_10.pt",
    norm_stats_path="./utils/stat-edulite-smoke-20.json",
    two_stage_mode=True,
)
print("real-data Stage 2 checkpoint loaded successfully")
PY
```

加载时必须使用训练该 checkpoint 的同一统计文件。

---

## 9. 实验记录模板

建议每次训练新建记录：

```markdown
## 实验编号

### 基本信息
- 日期:
- 数据入口: RoboTwin / EDULITE
- 数据根目录:
- 任务:
- Episodes:
- Frames:
- 统计文件:

### Stage 1
- Epochs:
- Max train steps:
- Batch size:
- Learning rate:
- Checkpoint:
- 初始 loss:
- 最终 loss:

### Stage 2
- Stage 1 checkpoint:
- Freeze Stage 1:
- Epochs:
- Max train steps: 不使用（正式训练）/ 数值（仅短跑）
- Batch size:
- Learning rate:
- Checkpoint:
- 初始 loss:
- 最终 loss:

### 评估
- 环境: RoboTwin / 离线真实数据
- 指令类型:
- 成功率:
- 备注:
```

不要只记录 timestamp。至少同时记录数据目录、统计文件和完整命令。

---

## 10. 常见问题排查

### 10.1 扫描到错误的数据数量

检查：

```bash
grep -E "Successfully scanned|Dataset Size" training.log
```

预期：

```text
RoboTwin: 400 episodes, 81389 frames
EDULITE: 20 episodes, 17045 frames
```

如果数量不符，优先检查 `--dataset_dir` 和 `--data_mode`。

### 10.2 统计文件混用

```text
RoboTwin -> stat-robotwin-current-400.json
EDULITE -> stat-edulite-smoke-20.json
```

不要因为两边都是14/16维就认为统计量可以共用。

### 10.3 Stage 2 OOM

先把 batch size 从128逐步降低，例如：

```bash
--batch_size 64 --num_workers 8 --freeze_stage1
```

检查 GPU：

```bash
nvidia-smi
```

仍然 OOM 时继续尝试32、16、8、4、2、1，并结束其他占用显存的进程。
不要把 batch size 设置为0。

### 10.4 Dataloader 为0个 batch

训练使用 `drop_last=True`。如果 batch size 大于数据集长度，会得到0个 batch。
降低 `--batch_size`。

### 10.5 Stage 2 找不到 Stage 1 checkpoint

```bash
find checkpoints_vla -type f -name 'checkpoint_epoch_*.pt' | sort
```

确认路径属于同一个数据入口，且 Stage 1 使用了相同统计文件。

### 10.6 Loss 不下降

依次检查：

1. 数据根目录与统计文件是否对应；
2. action/state 是否包含 NaN/Inf；
3. instruction JSON 是否为有效 `seen` 列表；
4. 学习率是否过大；
5. 真实夹爪目标与实际反馈差异是否符合任务；
6. 是否在很小数据集上训练过多导致过拟合。

### 10.7 真机不能直接使用 `eval_vla.py`

这是正常现象。`eval_vla.py` 创建 SAPIEN 环境，不包含 EDULITE CAN 控制、
反馈看门狗和急停执行层。不要通过修改几个路径参数就把它用于真机。

---

## 快速参考

### RoboTwin

```bash
python train_vla.py \
  --config ./configs/robotwin.yaml \
  --dataset_dir /home/lab347-no10/Tianyi/RoboTwin/data \
  --data_mode both \
  --stage1 \
  --norm_stats_path ./utils/stat-robotwin-current-400.json \
  --save_dir ./checkpoints_vla/robotwin_current_400 \
  --epochs 25 \
  --batch_size 128 \
  --num_workers 0 \
  --save_interval 25
```

### EDULITE

```bash
python train_vla.py \
  --config ./configs/robotwin.yaml \
  --dataset_dir /home/lab347-no10/Tianyi/robotwin_code/robot/edulite_a3_collect/collected_data \
  --data_mode clean \
  --stage1 \
  --norm_stats_path ./utils/stat-edulite-smoke-20.json \
  --save_dir ./checkpoints_vla/real_edulite_20 \
  --epochs 10 \
  --batch_size 128 \
  --num_workers 0 \
  --save_interval 10
```

两套命令只共用模型代码和 `configs/robotwin.yaml`，其数据、统计量和输出目录
保持独立。

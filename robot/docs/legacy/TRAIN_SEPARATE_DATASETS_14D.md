# RoboTwin 与 EDULITE 真实数据：独立训练入口

> 历史归档：本文的训练脚本与路径属于旧 `robotwin_code`。
> 当前 LiLa-WAM 原生单臂流程见
> [SINGLE_ARM_GUIDE.md](../../edulite_a3_collect/SINGLE_ARM_GUIDE.md)。

两类数据不做联合训练。它们分别使用独立数据目录、统计文件和 checkpoint
目录，但共用同一个 `train_vla.py` 和模型结构。

## 公共准备

```bash
conda activate RoboTwin
cd /home/lab347-no10/Tianyi/robotwin_code

python -c "import torch; print('cuda=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

## 入口 A：原有 RoboTwin 仿真训练

数据入口：

```text
/home/lab347-no10/Tianyi/RoboTwin/data
```

统计文件：

```text
utils/stat-robotwin-current-400.json
```

### RoboTwin Stage 1

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

### RoboTwin Stage 2

把 `{timestamp}` 替换为 Stage 1 日志中的目录：

```bash
python train_vla.py \
  --config ./configs/robotwin.yaml \
  --dataset_dir /home/lab347-no10/Tianyi/RoboTwin/data \
  --data_mode both \
  --norm_stats_path ./utils/stat-robotwin-current-400.json \
  --stage1_checkpoint ./checkpoints_vla/robotwin_current_400/stage1/{timestamp}/checkpoint_epoch_25.pt \
  --freeze_stage1 \
  --save_dir ./checkpoints_vla/robotwin_current_400 \
  --epochs 50 \
  --batch_size 128 \
  --num_workers 12 \
  --save_interval 5
```

如果完全沿用旧配置，也可以不传 `--dataset_dir` 和 `--data_mode`；显式写出
它们可以避免切换过真实数据后误用目录。

## 入口 B：EDULITE 真实机械臂训练

数据入口：

```text
/home/lab347-no10/Tianyi/robotwin_code/robot/edulite_a3_collect/collected_data
```

统计文件：

```text
utils/stat-edulite-smoke-20.json
```

### 真实数据 Stage 1

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

### 真实数据 Stage 2

```bash
python train_vla.py \
  --config ./configs/robotwin.yaml \
  --dataset_dir /home/lab347-no10/Tianyi/robotwin_code/robot/edulite_a3_collect/collected_data \
  --data_mode clean \
  --norm_stats_path ./utils/stat-edulite-smoke-20.json \
  --stage1_checkpoint ./checkpoints_vla/real_edulite_20/stage1/{timestamp}/checkpoint_epoch_10.pt \
  --freeze_stage1 \
  --save_dir ./checkpoints_vla/real_edulite_20 \
  --epochs 10 \
  --batch_size 128 \
  --num_workers 12 \
  --save_interval 2
```

## 参数隔离规则

| 项目 | RoboTwin | EDULITE 真实数据 |
|---|---|---|
| `dataset_dir` | `/home/lab347-no10/Tianyi/RoboTwin/data` | `.../edulite_a3_collect/collected_data` |
| `data_mode` | `both` | `clean` |
| `norm_stats_path` | `stat-robotwin-current-400.json` | `stat-edulite-smoke-20.json` |
| `save_dir` | `robotwin_current_400` | `real_edulite_20` |

同一次 Stage 1/Stage 2 训练必须使用同一列中的参数，不要交叉使用统计文件或
checkpoint。真实机械臂执行模型输出时，还必须由真实控制程序强制忽略/清零
右臂7维输出，并经过关节限位、速度限制和急停层。

# EDULITE A3：20 条测试数据训练流程检查

> 历史归档：本文针对旧 `robotwin_code` 的 14D/16D 补零数据，
> 不适用于当前 LiLa-WAM 原生单臂训练。当前流程见
> [SINGLE_ARM_GUIDE.md](../../edulite_a3_collect/SINGLE_ARM_GUIDE.md)。

这套命令只用于验证数据扫描、归一化、前向、反向和 checkpoint 保存。
20 条数据使用测试指令，而且跨越了两版夹爪映射参数，不能用训练 loss
判断真实任务效果。

## 1. 检查 GPU

```bash
conda activate RoboTwin
cd /home/lab347-no10/Tianyi/robotwin_code

python -c "import torch; print('cuda=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print('bf16=', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)"
```

只有 `cuda=True` 时才继续。短跑结果放在已被 Git 忽略的
`checkpoints_vla/smoke_edulite/`，避免和正式 checkpoint 混淆。

## 2. Stage 1 短跑

```bash
python train_vla.py \
  --config ./configs/robotwin.yaml \
  --dataset_dir /home/lab347-no10/Tianyi/robotwin_code/robot/edulite_a3_collect/collected_data \
  --data_mode clean \
  --stage1 \
  --norm_stats_path ./utils/stat-edulite-smoke-20.json \
  --save_dir ./checkpoints_vla/smoke_edulite \
  --epochs 1 \
  --batch_size 32 \
  --num_workers 0 \
  --max_train_steps 10
```

正常情况下日志应包含：

```text
Successfully scanned 20 total episode files
Dataset Size: 17045
Reached --max_train_steps=10
Saved checkpoint to .../stage1/.../checkpoint_epoch_1.pt
Training Complete
```

如果显存不足，把 `--batch_size 32` 降为 `16`、`8` 或 `4`。

## 3. Stage 2 短跑

把下面的 `{timestamp}` 替换为 Stage 1 日志中生成的目录名：

```bash
python train_vla.py \
  --config ./configs/robotwin.yaml \
  --dataset_dir /home/lab347-no10/Tianyi/robotwin_code/robot/edulite_a3_collect/collected_data \
  --data_mode clean \
  --norm_stats_path ./utils/stat-edulite-smoke-20.json \
  --stage1_checkpoint ./checkpoints_vla/smoke_edulite/stage1/{timestamp}/checkpoint_epoch_1.pt \
  --freeze_stage1 \
  --save_dir ./checkpoints_vla/smoke_edulite \
  --epochs 1 \
  --batch_size 1 \
  --num_workers 0 \
  --max_train_steps 3
```

正常情况下日志应包含冻结 VLM 和 Stage 1、完成 3 个 optimizer step，
以及保存 `stage2/{timestamp}/checkpoint_epoch_1.pt`。如果 Stage 2 仍然
显存不足，不要降低到小于 batch 1；先检查是否存在其他占用 GPU 的进程。

## 4. 统计量和限制

- 统计文件：`utils/stat-edulite-smoke-20.json`
- 异常值报告：`utils/outlier-edulite-smoke-20.txt`
- action/state 维度：14/16
- 右臂 action/state 是常量零；归一化代码会保护零范围维度不除零。
- 此数据没有验证集，且所有指令都是测试文本，只能证明训练程序能运行。
- 正式训练前应使用统一最终采集参数、真实任务名和真实任务指令重新采集，
  重新计算统计量，并划分独立验证 episode。

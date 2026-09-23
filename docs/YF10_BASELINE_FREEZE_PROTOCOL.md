# YF10 Baseline 冻结与实验记录协议

> 目的：为后续 Action-Flow OPD、Self-OPD 和模型缩放实验建立不可随结果变化的参照系。  
> 当前状态：大模型 clean 评测完成；randomized 评测完成前，epoch 40 只能标记为“暂定主 checkpoint”。

## 1. 当前 clean 结果与选择建议

| Checkpoint | Macro / Micro | 成功数 | 结论 |
|---|---:|---:|---|
| epoch 30 | 72.8% | 364/500 | 明显低于 40/50，可作为低训练量或弱初始化对照 |
| epoch 40 | 78.2% | 391/500 | clean 暂定最佳，任务间相对均衡 |
| epoch 50 | 77.8% | 389/500 | 与 epoch 40 基本并列，但部分任务继续提升、部分任务退化 |

epoch 40 与 epoch 50 只差 2/500，不能据此断言 epoch 40 在真实分布上显著更优。正式选择顺序应固定为：

1. 比较相同 `seed=0, test_num=50` 的 clean 与 randomized macro success rate。
2. 使用预先声明的综合分数，例如 `(clean_macro + randomized_macro) / 2`。
3. 如果综合分数差距小于 2 个百分点，优先选择更早的 epoch 40，以降低训练成本；再用开发 seed 1 复核。
4. 选定后不得针对不同改进实验在 epoch 30/40/50 中重新挑选最有利结果。
5. 最终论文结果对固定 checkpoint 使用未参与选择的 seeds，并报告均值和置信区间。

推荐基线命名：

```text
B0-L: LiLa-WAM YF10 large, depth=12, hidden=768, checkpoint=epoch40（暂定）
B0-S: LiLa-WAM YF10 small, depth=8, hidden=512, checkpoint=待重新选择
```

`B0-L` 和 `B0-S` 是两个不同 baseline，不能混写成同一个模型。

## 2. 为什么保留 epoch 30

后续主结果统一使用选定的 `B0-L`，目前建议 epoch 40。epoch 30 只适合以下预先声明的辅助问题：

- OPD 对尚未完全收敛的模型是否更有效。
- 不同初始化强度下的学习曲线。
- 低训练预算对照。

不能在某个任务上 epoch 30 提升更大，就报告 epoch 30；另一个任务上 epoch 40 更强，又切换到 epoch 40。主表必须统一起点。

## 3. 小模型协议

小模型配置为：[configs/yf10_small_d8_h512.yaml](../configs/yf10_small_d8_h512.yaml)。相对大模型只改变：

```yaml
model:
  action_expert:
    depth: 8
    hidden_size: 512
```

已按当前代码实际实例化统计：

| 模型 | Action model 参数量 | 相对大模型 |
|---|---:|---:|
| depth 12 / hidden 768 | 211.95M | 100% |
| depth 8 / hidden 512 | 77.70M | 36.7% |

Transformer 主体粗略计算比例为：

```text
(8 × 512²) / (12 × 768²) ≈ 0.296
```

实际 wall-clock 不会严格缩短到 29.6%，因为 DINO 前向、数据加载、未来特征 decoder 和评测环境是固定开销。必须实际记录 throughput 和峰值显存。

### 3.1 不能加载大模型 checkpoint

`hidden_size`、DiT block 数量、adapter 和输出层形状均发生变化，因此不能使用：

```bash
--init_from checkpoints_vla/yf10/.../checkpoint_epoch_40.pt
```

小模型应从随机初始化训练自己的 CFM baseline。未来如果要利用大模型，只能实现显式 teacher distillation/Flow-OPD，而不是直接加载 state dict。

### 3.2 首次训练保持其他参数不变

第一轮仍使用 batch 64，从而把性能和速度变化归因于架构缩小。命令：

```bash
cd /home/lab347-no10/Tianyi/LiLa-WAM

mkdir -p log/yf10_small_d8_h512

CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/yf10_small_d8_h512.yaml \
  --norm_stats_path artifacts/yf10/norm_stats.json \
  --save_dir checkpoints_vla/yf10_small_d8_h512 \
  2>&1 | tee log/yf10_small_d8_h512/train.log
```

第一轮完成后再单独做 batch throughput 实验：batch 64、96、128 各运行相同数量 optimizer steps。若改变 batch 或梯度累积，它应当是新的训练 run，并记录有效 batch，不能覆盖第一轮架构对照。

## 4. 必须冻结的六类信息

### 4.1 数据协议

必须记录：

- 数据集根目录的绝对路径。
- 十个任务名称和顺序。
- 每任务 clean/randomized episode 数。
- 有效 episode 与总搜索帧数。
- 数据 subset manifest。
- 删除的 outlier 文件列表及其哈希。
- 静止片段处理版本。
- 摄像头名称、数量、分辨率、RGB/BGR 约定。
- state/action 维度及 indices。
- action chunk 长度和 episode 尾部 padding 语义。

数据集目录不应只靠绝对路径识别；至少保存文件清单的哈希，防止同一路径内容以后被替换。

### 4.2 归一化与任务条件

必须记录：

- `norm_stats.json` 路径、SHA256、生成脚本和生成数据范围。
- action/state mean、std 的维度。
- VTT 根目录和 manifest。
- 每任务 `task_cond.npy` 的 SHA256。
- DINO checkpoint 标识、配置与权重 SHA256。
- DINO feature layers、是否包含 CLS/register、fusion mode。

### 4.3 模型协议

必须记录：

- 完整 YAML 快照，而不只是文件路径。
- action expert depth、hidden size、heads、adapter depth、query 数。
- future feature head 是否开启及 loss 权重。
- 总参数量、训练参数量。
- 模型代码文件 SHA256。
- padding/mask 的实际语义。

### 4.4 训练协议

必须记录：

- 完整训练命令和启动时间。
- 随机种子；当前训练若未显式设置，应如实记录“未固定”，不能事后补写。
- batch size、gradient accumulation、有效 batch。
- optimizer、betas、weight decay、gradient clipping。
- 初始/最小 LR、scheduler 和总 optimizer steps。
- epoch 数、每 epoch batch 数、drop-last 行为。
- BF16/FP32、GPU 型号、峰值显存。
- 每 epoch 时间、samples/s、总 GPU 小时。
- 是否 resume/init_from，以及父 checkpoint SHA256。
- 原始训练日志和 loss CSV。

### 4.5 推理协议

必须记录：

- checkpoint 文件和 SHA256。
- 自动读取的 checkpoint `config.yaml`。
- `num_inference_steps=10`。
- `action_execution_horizon=16`。
- 是否 B-Spline、degree、control points。
- 每次生成 32 步、执行前 16 步的缓存语义。
- action type（当前为 qpos）。
- state/history 更新方式。
- 推理代码 SHA256。

### 4.6 评测协议

必须记录：

- RoboTwin 仓库 commit 与 dirty diff。
- `demo_clean` / `demo_randomized`。
- 十任务列表和任务配置文件 SHA256。
- `_eval_step_limit.yml`、embodiment 和 camera 配置 SHA256。
- `instruction_type=unseen`；LiLa-WAM 当前不直接使用文本，但仍固定官方调用参数。
- `SEED` 与起始环境 seed 的映射。
- `TEST_NUM` 是每任务有效 episode 数。
- unstable seed 的跳过规则。
- 是否保存视频。
- 每任务成功数、macro/micro average、失败日志。
- checkpoint 选择用的开发 seeds 与最终报告 seeds 必须分开。

## 5. 推荐归档结构

```text
artifacts/yf10/baselines/
├── B0-L_d12-h768_ep40/
│   ├── manifest.yaml
│   ├── train_config.yaml
│   ├── train_command.txt
│   ├── train.log
│   ├── code.patch
│   ├── sha256sums.txt
│   └── eval/
│       ├── clean_seed0_n50_summary.json
│       └── randomized_seed0_n50_summary.json
└── B0-S_d8-h512_selected-epXX/
    └── ...相同结构...
```

不必复制 1.2GB checkpoint；可以记录其不可变路径和 SHA256。但如果磁盘允许，论文定稿时应把最终 checkpoint 单独备份到只读存储。

## 6. Manifest 模板

```yaml
baseline_id: B0-L_d12-h768_ep40
status: provisional  # randomized 完成且选择规则执行后改为 frozen
created_at: 2026-09-15T00:00:00+08:00

data:
  root: /home/lab347-no10/Tianyi/LiLa-WAM/data/yf10
  subset_manifest: <path>
  subset_manifest_sha256: <sha256>
  mode: both
  tasks: [adjust_bottle, grab_roller, hanging_mug, move_stapler_pad,
          open_microwave, press_stapler, scan_object, stack_blocks_two,
          stamp_seal, turn_switch]
  cameras: [head_camera]
  image_size: [320, 240]

model:
  config_snapshot: <path>
  config_sha256: <sha256>
  action_expert: {depth: 12, hidden_size: 768, num_heads: 8}
  action_params: 211948046
  dino_checkpoint: /home/lab347-no10/Tianyi/LiLa-WAM/pretrained/dinov3-vitl16

training:
  checkpoint: <absolute path>
  checkpoint_sha256: <sha256>
  epoch: 40
  batch_size: 64
  grad_accum_steps: 1
  learning_rate: 2.0e-4
  lr_min: 5.0e-5
  random_seed: not_explicitly_fixed
  command_file: <path>
  log_file: <path>

inference:
  num_inference_steps: 10
  action_chunk_size: 32
  action_execution_horizon: 16
  smooth_actions: true
  bspline_degree: 3
  bspline_control_points: 8

evaluation:
  selection_seed: 0
  test_num_per_task: 50
  instruction_type: unseen
  clean_summary: <path>
  randomized_summary: <path>
  final_report_seeds: [1, 2, 3]

code:
  lila_git_commit: <commit>
  lila_dirty_patch: <path>
  robotwin_git_commit: <commit>
  robotwin_dirty_patch: <path>
```

## 7. 生成哈希和环境记录

最终 checkpoint 选择以后执行，输出应保存进对应归档目录：

```bash
sha256sum \
  checkpoints_vla/yf10/sft_2026-09-11_12-37-35/checkpoint_epoch_40.pt \
  checkpoints_vla/yf10/sft_2026-09-11_12-37-35/config.yaml \
  artifacts/yf10/norm_stats.json \
  train.py models/model_runner.py models/vla_model_fm.py robotwin_infer.py \
  > artifacts/yf10/baselines/B0-L_d12-h768_ep40/sha256sums.txt

git rev-parse HEAD \
  > artifacts/yf10/baselines/B0-L_d12-h768_ep40/lila_git_commit.txt

git diff \
  > artifacts/yf10/baselines/B0-L_d12-h768_ep40/code.patch

conda list --explicit \
  > artifacts/yf10/baselines/B0-L_d12-h768_ep40/conda_explicit.txt

nvidia-smi -q \
  > artifacts/yf10/baselines/B0-L_d12-h768_ep40/nvidia_smi.txt
```

RoboTwin 是另一个 Git 仓库，应在其目录单独记录 commit 和 diff。

## 8. 后续 OPD 的公平比较规则

每个 OPD 方法必须明确选择一种基础架构：

```text
B0-L -> B1-L continued SFT / Flow-OPD / Self-OPD
B0-S -> B1-S continued SFT / Flow-OPD / Self-OPD
```

不能把 `B0-L` 的成功率作为 baseline，却让 OPD 从 `B0-S` 初始化；也不能只比较训练速度、不报告参数量与成功率损失。

推荐论文组织：

- 大模型 `B0-L`：证明方法在原 LiLa-WAM 容量上有效。
- 小模型 `B0-S`：证明方法在更低计算预算下有效。
- 可选：用冻结的 `B0-L` 作为 `B0-S` 的 Flow-OPD teacher，研究能力蒸馏。
- 每个架构都必须有 compute-matched continued SFT 对照。

## 9. 何时可以标记为 frozen

只有满足以下条件才把 manifest 的 `status` 从 `provisional` 改为 `frozen`：

- [ ] clean 和 randomized 的 epoch 30/40/50 均测试完成。
- [ ] 按预先声明规则选出一个主 checkpoint。
- [ ] checkpoint、配置、norm stats 和相关代码哈希已保存。
- [ ] 数据/VTT/DINO manifest 已保存。
- [ ] 训练命令、日志、环境版本已保存。
- [ ] 开发评测结果完整且非空。
- [ ] 最终评测 seeds 已预先声明且未用于调参。
- [ ] 后续所有实验统一引用 baseline ID，而不是模糊地写“epoch 30 或 40”。


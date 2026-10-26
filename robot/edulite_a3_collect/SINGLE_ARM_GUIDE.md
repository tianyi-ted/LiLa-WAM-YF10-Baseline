# LiLa-WAM 原生单臂采集、训练与部署

本方案只把 **can0 从臂**视为执行任务的机械臂；can1 主臂仅用于示教和诊断，不作为第二个任务臂。新 episode 的 `joint_action/vector` 为 `[T,7]`：L1～L6 目标关节角（rad）和 L7 归一化开度（0 闭、1 开）。训练 state 为 `[T,8]`：从臂末端 `xyz+quaternion(wxyz)` 共 7 维，加从臂夹爪实际开度 1 维。HDF5 不再保存假右臂 action/endpose 字段；`teleop/master_joint_feedback` 仍保留，用于诊断，不参与训练。

新数据写到 `robot/edulite_a3_collect/collected_data_single/place_red_block_into_box/demo_clean/`。旧 14D 补零数据不会自动转换；当前 LiLa-WAM 的 `robot/edulite_a3_collect/` 下没有旧 `collected_data/` 目录。`configs/edulite_single.yaml` 只读新目录；仿真的 YF10 配置仍是 14D/16D，不受影响。不要把不同维度的数据目录一起传给同一次统计或训练。

## 1. 采集与检查

使用已经配置好的 `edulite_collect` 环境，启动前确认 can1=主臂、can0=从臂、物理急停可及，相机位置与固定参数无误：

```bash
conda activate edulite_collect
cd /home/lab347-no10/Tianyi/LiLa-WAM
python robot/edulite_a3_collect/collect.py \
  --config robot/edulite_a3_collect/config.master_slave.yaml \
  --task-profile robot/edulite_a3_collect/task_profiles/place_red_block_into_box.yaml
```

操作顺序沿用当前采集器：`a` 使能并让两臂 L1～L6 回初始位；`r` 倒计时 3 秒后开始主从示教和记录；`s` 保存并回位；`d` 丢弃并回位；`h` 手动回位；`t` 打印夹爪诊断；`[`/`]` 小步闭/开、`c`/`o` 全闭/开为键盘后备；`e` 软件急停，`q` 退出。主臂 L7 首次打开大于 40% 才激活控制；之后大于 40% 时从臂目标为 80% 打开，小于 40% 时目标为闭合。新档位连续出现 3 个控制样本才切换，以避免阈值附近的单帧噪声抖动；恰好等于 40% 时保持上一档。闭合途中接触物体仍由力矩/停转保护锁存，**不是强制关到机械 0%**。开合目标仍经 EMA、1.50 norm/s 限速，当前配置中的 L7 PP 速度/加速度均为 1.50 rad/s、1.50 rad/s²；电机力矩和电流限制仍为 0.10 Nm、0.30 A。回位只处理 L1～L6，L7 保持当前开度。采集器会按现有配置同时记录头、胸两路 JPEG 帧（640×480、目标采样率 30 Hz）；当前真机训练与部署配置使用 `chest_camera`，`head_camera` 继续保留在原始 HDF5 中备用。

第一次验证新的两档夹爪时，先在空载且远离物体的空间运行，用 `t` 观察 `master_gripper_norm`、`target_norm` 与力保护状态；确认开合方向无误后再抓取软物体。打开档只到 80%，若此前力保护锁存在高于 80% 的开度，主臂打开档不能进一步张开并解锁，可用键盘 `o` 后备打开。不要为了提高速度而提高力矩或电流限值。

HDF5 的第 7 维 action 记录实际下发的**限速后目标**，而非未经处理的两档开关，因此开合过渡帧会出现 0～0.8 之间的数值；state 的第 8 维仍记录从臂夹爪实测开度，夹到物体并触发保护时可能停在中间位置。

每次采完先校验：

```bash
python robot/edulite_a3_collect/validate_dataset.py \
  robot/edulite_a3_collect/collected_data_single --require-single
```

检查 HDF5 中 `joint_action/vector` 是否为 `[T,7]`、`endpose/left_endpose` 是否为 `[T,7]`、`endpose/left_gripper` 是否为 `[T]`，并确认没有 `endpose/right_*`。建议检查任务成功率、相机帧内容、动作/反馈差和时间戳；格式正确不代表示教质量足够。

## 2. 训练准备与训练

下面使用现有 `RoboTwin` 训练环境；不要在只安装了机械臂 SDK 的 `edulite_collect` 环境运行 DINOv3 训练。统计命令不带 `--delete_outliers`，只报告并排除越界 episode，不删除原始文件：

```bash
conda activate RoboTwin
cd /home/lab347-no10/Tianyi/LiLa-WAM
python utils/calc_stat_remove_outlier.py \
  --root_dir robot/edulite_a3_collect/collected_data_single \
  --data_mode clean --num_processes 4 \
  --output_path utils/stat-edulite-single.json \
  --outlier_path utils/outliers-edulite-single.txt
```

打开统计 JSON，确认 `robotwin2.action.dim=7`、`robotwin2.state.dim=8`，`valid_files` 与预期一致，越界报告无异常。接着计算当前 LiLa-WAM 所需的任务条件向量（冻结 DINOv3 的每轮首末帧特征差）；增加或筛选正式示教数据后要重新计算：

```bash
python utils/precompute_task_cond.py \
  --config configs/edulite_single.yaml \
  --output_dir embeddings/edulite_single_chest

python train.py \
  --config configs/edulite_single.yaml \
  --norm_stats_path utils/stat-edulite-single.json \
  --save_dir checkpoints_vla/edulite_single
```

checkpoint 位于 `checkpoints_vla/edulite_single/sft_时间戳/checkpoint_epoch_N.pt`，同目录有训练时的 `config.yaml`。配置为 `action_dim=7`、`state_dim=8`、胸部相机输入 `320×240`、动作块 32、执行 horizon 16、按 epoch 保存。8×8 的 batch/梯度累积只是初始配置；显存不足时优先降低 `training.batch_size` 并相应调整 `grad_accum_steps`。**旧 14D checkpoint 的输入/输出层形状不同，不能直接用 `--init_from` 或 `--resume` 载入 7D 模型。** 训练、推理必须使用同一统计 JSON、同一配置和同一任务条件向量。

当前 LiLa-WAM 使用预计算的视觉任务向量作为任务条件，`instructions/episode*.json` 随数据保留，但训练代码不会把其中的文字实时编码输入模型；部署配置中的 `instruction` 主要用于界面显示。新增任务时必须为新任务名重新计算任务向量。

## 3. 真机推理（训练完成后）

真机推理使用独立的 `edulite_deploy` 环境，版本与离线验证方式见[部署环境说明](../edulite_a3_vla/DEPLOY_ENV.md)。训练继续使用 RoboTwin。

编辑 `robot/edulite_a3_vla/config.real_single.yaml`，把 `model.checkpoint_path` 中的 `sft_REPLACE_WITH_RUN_TIMESTAMP` 改为实际训练目录；最好同时把 `model.config_path` 指向该目录的 `config.yaml`，确保是训练时的配置副本。先执行不打开相机和 CAN 的静态检查，再检查现场后执行：

```bash
conda activate edulite_deploy
cd /home/lab347-no10/Tianyi/LiLa-WAM
python robot/edulite_a3_vla/real_vla.py \
  --config robot/edulite_a3_vla/config.real_single.yaml --check-only

python robot/edulite_a3_vla/real_vla.py \
  --config robot/edulite_a3_vla/config.real_single.yaml --execute
```

推理键位：`a` 使能并回初始位，`p` 只生成/可视化一块 7D 预测且机械臂不动，`r` 倒计时后执行，`s` 保持当前位置，`h` 回 L1～L6 初始位（L7 不动），`e` 软件急停，`q` 退出。先目视审核 `p` 输出，确认预测轨迹和夹爪开度合理再按 `r`。这里仍沿用从臂 L7 电机 0.10 Nm/0.30 A 限幅、软件力矩/停转锁存和 L1～L6 安全滤波；真实安全性只能通过低速、空载、小范围现场验证，静态检查不能代替硬件测试。

部署代码只接收原生 `[N,7]` 动作和 `[1,8]` state，绝不把右臂补零。模型通用推理的全维 B 样条对该配置关闭；真机运行时只对 L1～L6 做块间 B 样条，L7 继续独立 EMA、限速与力保护。预览、执行记录在 `robot/edulite_a3_vla/inference_outputs_single/`。

## 4. 两任务联合训练：方块入盒 + 红块叠蓝块

不要修改当前硬件主配置中的原任务名来回覆盖，而是使用任务 profile。第二任务采集命令是：

```bash
conda activate edulite_collect
cd /home/lab347-no10/Tianyi/LiLa-WAM
python robot/edulite_a3_collect/collect.py \
  --config robot/edulite_a3_collect/config.master_slave.yaml \
  --task-profile robot/edulite_a3_collect/task_profiles/place_red_block_on_blue_block.yaml
```

它复用完全相同的 CAN、相机、夹爪和安全参数，但写入独立目录 `collected_data_single/place_red_block_on_blue_block/demo_clean/`。开始每轮前将红、蓝方块恢复到训练初始布局；episode 只包含一次完整的“抓红块并稳定放到蓝块上”，失败、倒塌或需要人工补救的轨迹不要保存为 clean。两任务在 dataloader 中按帧采样，建议第二任务也采集约 100 条，并使两类 episode 的时长分布接近，避免较长或较多的任务主导训练。

采完先验证两类任务：

```bash
python robot/edulite_a3_collect/validate_dataset.py \
  robot/edulite_a3_collect/collected_data_single --require-single
```

随后在训练环境生成**两任务共享的全局归一化统计量**。明确指定两个任务，防止以后加入的测试目录被误统计；不要使用 `--delete_outliers`：

```bash
conda activate RoboTwin
cd /home/lab347-no10/Tianyi/LiLa-WAM
python utils/calc_stat_remove_outlier.py \
  --root_dir robot/edulite_a3_collect/collected_data_single \
  --task_names place_red_block_into_box place_red_block_on_blue_block \
  --data_mode clean --num_processes 4 \
  --output_path utils/stat-edulite-single-multitask.json \
  --outlier_path utils/outliers-edulite-single-multitask.txt
```

检查统计文件必须是 action 7 维、state 8 维，`valid_files` 等于两任务有效 episode 总数。然后为每个任务分别计算首末帧 DINO 特征差；输出目录会包含两个子目录：

```bash
python utils/precompute_task_cond.py \
  --config configs/edulite_single_multitask.yaml \
  --output_dir embeddings/edulite_single_multitask_chest
```

联合模型从头训练：

```bash
python train.py \
  --config configs/edulite_single_multitask.yaml \
  --norm_stats_path utils/stat-edulite-single-multitask.json \
  --save_dir checkpoints_vla/edulite_single_multitask
```

这里不要直接使用当前打开的 `yf10_dinov3b_h512_g2_c32_ft4.yaml`：它是 YF10 仿真的 14D action/16D state 四轮微调配置，并假定有匹配的 14D 初始化 checkpoint。新配置是 7D/8D、50 epoch 的独立真机训练入口。旧单任务 checkpoint 虽然维度相同，但继续微调容易偏向新增任务；第一版两任务基线建议从头训练，之后再单独比较微调方案。

训练完成后，将 `config.real_single.yaml` 中的三条路径改成同一训练实验的 `config.yaml`、checkpoint 和 `stat-edulite-single-multitask.json`。同一个 checkpoint 通过任务名选择不同的 `task_cond.npy`。先检查再执行红块叠蓝块任务：

```bash
python robot/edulite_a3_vla/real_vla.py \
  --config robot/edulite_a3_vla/config.real_single.yaml \
  --task-name place_red_block_on_blue_block \
  --instruction "pick up the red block and place it on top of the blue block" \
  --check-only

python robot/edulite_a3_vla/real_vla.py \
  --config robot/edulite_a3_vla/config.real_single.yaml \
  --task-name place_red_block_on_blue_block \
  --instruction "pick up the red block and place it on top of the blue block" \
  --execute
```

测试原来的入盒任务时省略覆盖参数，或显式使用 `--task-name place_red_block_into_box`。当前模型不会在线编码 `--instruction` 文本；真正区分任务的是 `--task-name` 对应的预计算向量，instruction 只是日志中的可读描述，因此任务名必须与数据目录完全一致。

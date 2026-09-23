# LiLa-WAM 十任务 baseline：训练与 RoboTwin 评测流程

> 后续论文创新路线见：[执行后果驱动双层 On-Policy 研究路线](EG_DUAL_OPD_RESEARCH_ROADMAP.md)。
> 动作生成 Flow 专项实验见：[Flow-OPD / Self-OPD 实验指南](ACTION_FLOW_OPD_EXPERIMENT_GUIDE.md)。
> Baseline 选择、归档和小模型对照见：[YF10 Baseline 冻结协议](YF10_BASELINE_FREEZE_PROTOCOL.md)。

本文固定一套可重复的十任务 baseline，供后续 Flow-OPD、视觉表征、空间建模或动作解码改动统一对比。默认使用完整数据集的 clean 最多 50 条 + randomized 200 条/任务；全部数据用于训练，测试只在 RoboTwin 仿真环境中进行。

> 这是“自有十任务数据上的 LiLa-WAM baseline”，不是论文 50 任务成绩的复现。论文使用了另行处理的 50 任务数据，两者不能直接横向比较。

## 1. 固定实验定义

任务：

~~~text
adjust_bottle
grab_roller
hanging_mug
move_stapler_pad
open_microwave
press_stapler
scan_object
stack_blocks_two
stamp_seal
turn_switch
~~~

| 项目 | baseline 设置 |
|---|---|
| 数据源 | /media/lab347-no10/data/tianyi/RoboTwin_datasets |
| 子集 | 每任务 clean min(50, 实有数量) + randomized 200 |
| 筛选前 episode 数 | clean 492 + randomized 2000 = 2492 |
| 官方预处理后 episode 数 | clean 485 + randomized 1956 = **2441** |
| 官方预处理后可索引帧数 | **392736** |
| 摄像头 | 仅 head_camera，与 LiLa-WAM 默认一致 |
| 输入分辨率 | 320 × 240 |
| 动作块 | 32 步 |
| 任务条件 | VTT：每任务 DINO CLS 的“末帧-首帧”episode 均值 |
| 视觉编码器 | 冻结 DINOv3 ViT-L/16，层 [-12,-8,-4] |
| 辅助目标 | 开启未来 DINO 特征预测，权重 0.5 |
| 推理 | 10 个 flow steps，执行 horizon 16，开启 B-Spline |
| 训练划分 | 所有 2441 条预处理后 episode 用于训练，不设离线 validation/test |

各任务最终数量以 [processed_manifest.json](../data/yf10/processed_manifest.json) 为准。clean 少于 50 是源数据不足或异常轨迹被过滤，不重复 episode 补齐。

## 2. 已准备的实验文件

- utils/prepare_yf10_subset.py：建立十任务符号链接视图，不复制、不修改完整数据。
- configs/yf10_base.yaml：高 LR 阶段，12 epochs，LR 2e-4。
- configs/yf10_finetune.yaml：低 LR 阶段，4 epochs，LR 4e-5。
- eval_yf10.sh：固定十任务评测并汇总 JSON。
- train.py：梯度累积时的余弦调度周期已改为按 optimizer step 计算；不改变模型结构。
- docs/YF10_CONFIG_REFERENCE.md：逐项解释 YAML 参数的实际代码语义、资源影响和兼容性风险。

配置默认物理 batch 8、累积 16，等效 batch 128。

## 3. 环境

训练建议使用项目指定的独立环境：

~~~bash
conda create -n LiLaWAM python=3.10 -y
conda activate LiLaWAM
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==5.0.0rc0 omegaconf accelerate h5py opencv-python tqdm
~~~

检查：

~~~bash
cd /home/lab347-no10/Tianyi/LiLa-WAM
python - <<'PY'
import torch, transformers
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("bf16:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY
~~~

本机 RoboTwin 环境虽然是 Torch 2.4.1，与上游 README 指定的 2.7.1 不同，但已经通过 LiLa-WAM 导入和 DINOv3 配置识别检查，可用于本项目 baseline。后续所有对比实验应固定同一环境，不要在其中途升级 PyTorch/Transformers；论文中记录实际版本即可。

## 4. 数据子集与核验

原始筛选视图（符号链接、未预处理，已在整理时删除；需要时可重建）：

~~~text
/home/lab347-no10/Tianyi/LiLa-WAM/data/yf10_raw_links
~~~

供正式 baseline 使用的物理预处理数据已经放在：

~~~text
/home/lab347-no10/Tianyi/LiLa-WAM/data/yf10
~~~

需要在新目录重建时执行：

~~~bash
python utils/prepare_yf10_subset.py \
  --source /media/lab347-no10/data/tianyi/RoboTwin_datasets \
  --output /home/lab347-no10/Tianyi/LiLa-WAM/data/yf10_raw_links \
  --clean-limit 50 \
  --randomized-limit 200
~~~

核验：

~~~bash
find data/yf10_raw_links -mindepth 1 -maxdepth 1 -type d | wc -l
find data/yf10_raw_links -type l -name '*.hdf5' | wc -l
find -L data/yf10_raw_links -type l -print
~~~

原始筛选视图期望为 10、2492，最后一项无输出。它只作为可重建的 staging view，不作为当前正式 baseline 输入。

### 数据预处理口径

当前正式 baseline 已按 LiLa-WAM 仓库提供的顺序完成：

1. 将 staging view 解引用复制为物理文件；
2. 使用 threshold=1e-6、margin=1 删除静止段；
3. 删除非夹爪关节超出 ±π 的 episode；
4. 根据最终数据重新计算归一化统计；
5. VTT 也必须在最终数据上重新计算。

处理结果：静止清理将 580370 帧降至 397939 帧；action outlier 过滤删除 51 个 episode，最终为 2441 个 episode、392736 帧。dry-run、正式清理和 outlier 清单位于 artifacts/yf10/preprocess_reports。

最终统计文件已生成。需要复算时执行：

~~~bash
mkdir -p artifacts/yf10/preprocess_reports
python utils/calc_stat_remove_outlier.py \
  --root_dir ./data/yf10 \
  --output_path ./artifacts/yf10/norm_stats.json \
  --outlier_path ./artifacts/yf10/preprocess_reports/final_outlier_check.txt \
  --num_processes 8 \
  --data_mode both
~~~

不要对 /media 下的完整数据或符号链接 staging view 运行原地静止清理。该脚本会以写模式覆盖 HDF5；只能对独立物理副本操作。

检查统计：

~~~bash
python - <<'PY'
import json
s = json.load(open("artifacts/yf10/norm_stats.json"))["robotwin2"]
print("tasks:", s["meta"]["num_tasks"])
print("action dim:", s["action"]["dim"])
print("state dim:", s["state"]["dim"])
PY
~~~

应为 10、14、16。不能复用旧的 stat-200-10.json 或 50 任务统计。

## 5. 计算 VTT 任务条件

VTT 与 episode 集相关，不直接复用旧 data-200-10-taskcond：

~~~bash
CUDA_VISIBLE_DEVICES=0 python utils/precompute_task_cond.py \
  --config ./configs/yf10_base.yaml \
  --output_dir /home/lab347-no10/Tianyi/LiLa-WAM/embeddings/yf10_vtt \
  --device cuda \
  2>&1 | tee ./artifacts/yf10/preprocess_reports/vtt.log
~~~

核验：

~~~bash
find embeddings/yf10_vtt -name task_cond.npy | wc -l
python - <<'PY'
import json
m = json.load(open("embeddings/yf10_vtt/meta.json"))
print("tasks:", len(m["tasks"]))
for name, info in sorted(m["tasks"].items()):
    print(name, info["num_episodes"], info["dim"], info["norm"])
PY
~~~

应有 10 个向量。VTT 是任务标签式条件，当前模型不读取自然语言，因此 seen/unseen instruction 不能解释为语言泛化实验。

## 6. 显存、DataLoader 与 smoke test

4070 Ti 从下列组合开始，保持等效 batch 128：

| 情况 | batch_size | grad_accum_steps | 等效 batch |
|---|---:|---:|---:|
| 默认起点 | 8 | 16 | 128 |
| OOM | 4 | 32 | 128 |
| 显存有余量 | 16 | 8 | 128 |

num_workers=4 是稳妥起点；GPU 等数据时再试 8，内存/HDF5 不稳定时降为 2。当前 train.py 没有 step-limit，先启动并观察若干 step；Ctrl+C 只会丢失当前未完成 epoch，完整 epoch 末尾才保存。

训练前记录：

~~~bash
git status --short
git rev-parse HEAD
sha256sum configs/yf10_base.yaml \
  configs/yf10_finetune.yaml \
  artifacts/yf10/norm_stats.json \
  data/yf10/processed_manifest.json
~~~

## 7. 两阶段训练

正式启动前可用一条命令检查所有路径、数量和维度：

~~~bash
python utils/verify_yf10_setup.py
~~~

### 7.1 Base：12 epochs，LR 2e-4

~~~bash
cd /home/lab347-no10/Tianyi/LiLa-WAM
conda activate LiLaWAM
mkdir -p logs checkpoints_vla/yf10

CUDA_VISIBLE_DEVICES=0 python train.py \
  --config ./configs/yf10_base.yaml \
  --norm_stats_path ./artifacts/yf10/norm_stats.json \
  --save_dir ./checkpoints_vla/yf10 \
  2>&1 | tee ./logs/yf10_base.log
~~~

启动日志应有 Task Condition: ENABLED、Future-Feat Pred: ENABLED、effective_batch: 128，以及 LR 2e-4 -> 5e-5。

中断恢复使用 --resume：

~~~bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config ./configs/yf10_base.yaml \
  --norm_stats_path ./artifacts/yf10/norm_stats.json \
  --save_dir ./checkpoints_vla/yf10 \
  --resume ./checkpoints_vla/yf10/sft_<BASE_RUN>/checkpoint_epoch_<N>.pt \
  2>&1 | tee -a ./logs/yf10_base_resume.log
~~~

它恢复模型、optimizer、scheduler 和 epoch。程序会建立新输出 run，但学习率进度接续旧 checkpoint。

### 7.2 Fine-tune：4 epochs，LR 4e-5

切换阶段必须使用 --init_from，仅加载权重并重建低 LR optimizer/scheduler：

~~~bash
BASE_CKPT=/home/lab347-no10/Tianyi/LiLa-WAM/checkpoints_vla/yf10/sft_<BASE_RUN>/checkpoint_epoch_12.pt

CUDA_VISIBLE_DEVICES=0 python train.py \
  --config ./configs/yf10_finetune.yaml \
  --norm_stats_path ./artifacts/yf10/norm_stats.json \
  --save_dir ./checkpoints_vla/yf10 \
  --init_from "$BASE_CKPT" \
  2>&1 | tee ./logs/yf10_finetune.log
~~~

日志必须出现 fresh optimizer + lr schedule、4e-5 -> 1e-5、missing=0 和 unexpected=0。最终 baseline 为新 run 的 checkpoint_epoch_4.pt；4 表示低 LR 阶段第 4 轮，不是总第 16 轮。

## 8. RoboTwin 评测准备

桥脚本必须位于 RoboTwin 根目录。用符号链接避免两份代码漂移：

~~~bash
test -e /home/lab347-no10/Tianyi/RoboTwin/eval_vla_bridge.py || \
ln -s /home/lab347-no10/Tianyi/LiLa-WAM/eval_vla_bridge.py \
      /home/lab347-no10/Tianyi/RoboTwin/eval_vla_bridge.py

conda activate RoboTwin
cd /home/lab347-no10/Tianyi/LiLa-WAM
~~~

### Step limit 口径

当前 RoboTwin/task_config/_eval_step_limit.yml 是原始上限的约 2/3，适合快速筛查，但不是原始 RoboTwin 上限。备份 _eval_step_limit.before_two_thirds_20260910.yml 保留原值。

- 快速协议：保留 2/3 上限，结果标记 custom-2/3-limit。
- 正式 baseline（推荐）：恢复原始上限，之后所有方法保持相同。

恢复：

~~~bash
cp /home/lab347-no10/Tianyi/RoboTwin/task_config/_eval_step_limit.before_two_thirds_20260910.yml \
   /home/lab347-no10/Tianyi/RoboTwin/task_config/_eval_step_limit.yml
sha256sum /home/lab347-no10/Tianyi/RoboTwin/task_config/_eval_step_limit.yml
~~~

两种上限的成功率不能放在同一主表比较。

## 9. 测试命令

若 checkpoint 是 LiLa-WAM/checkpoints_vla/yf10/sft_2026-09-10_20-00-00/checkpoint_epoch_4.pt：

~~~bash
export CKPT_SETTING=yf10/sft_2026-09-10_20-00-00
export CHECKPOINT_EP=4
~~~

快速 clean，每任务 5 次：

~~~bash
CUDA_VISIBLE_DEVICES=0 \
CKPT_SETTING="$CKPT_SETTING" CHECKPOINT_EP="$CHECKPOINT_EP" \
TASK_CONFIG=demo_clean TEST_NUM=5 SEED=0 NUM_WORKERS=1 \
bash ./eval_yf10.sh
~~~

每个 worker 都加载完整模型；单张 4070 Ti 默认 NUM_WORKERS=1。

正式 clean，3 seeds × 每任务 50 次：

~~~bash
for S in 0 1 2; do
  CUDA_VISIBLE_DEVICES=0 \
  CKPT_SETTING="$CKPT_SETTING" CHECKPOINT_EP="$CHECKPOINT_EP" \
  TASK_CONFIG=demo_clean TEST_NUM=50 SEED="$S" NUM_WORKERS=1 \
  RUN_ID="formal_clean_seed${S}" \
  bash ./eval_yf10.sh
done
~~~

正式 randomized：

~~~bash
for S in 0 1 2; do
  CUDA_VISIBLE_DEVICES=0 \
  CKPT_SETTING="$CKPT_SETTING" CHECKPOINT_EP="$CHECKPOINT_EP" \
  TASK_CONFIG=demo_randomized TEST_NUM=50 SEED="$S" NUM_WORKERS=1 \
  RUN_ID="formal_randomized_seed${S}" \
  bash ./eval_yf10.sh
done
~~~

要尽快得到第一版，可先只跑 seed=0、TEST_NUM=50。论文主表建议至少 3 seeds 的均值和标准差。由于训练集同时含 clean/randomized，这里应称为 in-distribution clean/randomized evaluation，不能直接声称官方 C2C/C2R，除非数据划分和官方协议完全一致。

## 10. 输出和主指标

每次运行输出：

~~~text
eval_result/yf10/<RUN_ID>/
├── logs/worker_0.log
├── bridge_results/worker_0.json
└── summary.json
~~~

summary.json 包含逐任务成功率、macro average 和 micro average。各任务次数相同时二者相同；主指标建议使用十任务等权 macro SR，同时单列每个任务。

| Method | Train episodes | Camera | Step limit | Clean SR | Randomized SR | Seeds |
|---|---:|---|---|---:|---:|---|
| LiLa-WAM baseline | 2441 | head | official/original | 待填 | 待填 | 0,1,2 |

## 11. 公平对比必须固定

后续创新方法除目标消融外，固定：

1. subset_manifest、归一化统计和训练 episode；
2. epochs、有效 batch、LR schedule 和训练 seed；
3. DINO checkpoint、320×240、单 head camera；
4. VTT；改变任务条件时须列为明确消融；
5. flow steps、action horizon、B-Spline；
6. RoboTwin commit、任务配置、step-limit、测试 seeds 和次数；
7. checkpoint 选择规则。

第一版固定使用 12+4 的最终 checkpoint，避免按测试成功率挑 epoch。如果以后用仿真结果选 checkpoint，应保留不参与选择的最终 seeds，或明确 selection protocol。

## 12. 常见故障

- 缺 task condition：确认十个 task_cond.npy 都存在。
- Missing/Unexpected 非零：训练和评测 config 不一致；bridge 优先读取 checkpoint 同目录 config.yaml。
- OOM：降物理 batch并同比增加累积；评测只开一个 worker。
- GPU 利用率低：逐步把 num_workers 4 调到 8，再尝试增大物理 batch。
- 成功率全零：核对统计、VTT、DINO 路径和 checkpoint；日志应显示权重完整加载。
- 结果不可比：核对 step-limit、task config、B-Spline、TEST_NUM 和 SEED。
- unseen 指令仍很高：模型通过 set_task(task_name) 注入 VTT，不消费自然语言。

最终 checkpoint、六份正式 summary.json、两份 config、统计、manifest、代码 commit 与 step-limit 哈希共同组成 baseline 冻结包。

# YF10 DINOv3-Base＋h512 基线冻结记录

冻结日期：2026-09-22。这里的“冻结”表示后续实验使用明确的只读基线标识与校验值，不改变checkpoint文件权限，也不复制544 MiB权重。C32/F16的共同起点固定为：

```text
checkpoints_vla/yf10_dinov3b_small/sft_2026-09-20_15-07-28/checkpoint_epoch_40.pt
SHA256 98bd63e1c760d2c3bbb1daaa7675a409d1efc1c53d3b079cee2fbd1edf0eb421
```

## 模型和数据协议

| 项目 | 冻结值 |
|---|---|
| Vision encoder | frozen DINOv3 ViT-B/16，768维，layers `[-7,-4,-2]` |
| Action Expert | depth=12，hidden=512，heads=8 |
| Future target layer | `-2` |
| Input | head camera RGB，320×240；state 16D；task VTT 768D |
| Action | 32×14 chunk；执行前16步；Flow NFE=10 |
| Inference smoothing | cubic B-Spline，8 control points |
| Training data | `data/yf10`，clean＋randomized |
| Dataset inventory | 10 tasks，2441 episodes，392736 processed frames |
| Baseline training batch | 96 |
| Normalization | `artifacts/yf10/norm_stats.json` |
| VTT | `embeddings/yf10_vtt_dinov3b`，10×768 |

数据集由处理后的本地HDF5组成，未对数百GB文件逐文件计算内容哈希。本记录冻结数据根目录、任务/episode/frame数量，以及统计量和VTT内容校验值。若HDF5可能被改写，应在实验前重新核对清单，或另行生成全量数据哈希。

## 已完成基线评测

评测使用环境seed 0、每个任务每种场景50个有效episode：

| 配置 | 成功数 | 成功率 |
|---|---:|---:|
| demo_clean | 351/500 | 70.2% |
| demo_randomized | 323/500 | 64.6% |
| 合计 | 674/1000 | 67.4% |

这是一个训练run和一个开发环境seed的结果。后续方法以同一Base基线、相同评测seed和有效场景集合进行配对比较；最终候选再用预先保留的环境seed复核。

## 校验方法

从LiLa-WAM根目录执行：

```bash
sha256sum -c artifacts/yf10/baseline_freeze/dinov3b_h512_ep40.sha256
```

校验清单包含基线checkpoint、保存配置、归一化统计、DINO权重、全部十个VTT向量、评测汇总及关键训练/推理源文件。后续修改源代码不会改变已保存checkpoint；实验记录应同时保存本次代码版本或新的源文件哈希。

基线训练完成后、加入本轮显式seed功能前，`train.py`的SHA256为`cbcaf74ab50917580701f664a69cfec405a68670cb2d55b782fcd709800ab533`。当前校验清单中的`train.py`是用于C32/F16的新版本；历史基线训练脚本未另存副本，因此基线可复现身份以checkpoint、保存配置和相关数据资产为准，不能声称能够从训练seed逐位重放原40轮训练。

## Phase 2 共同起点

```text
C32: configs/yf10_dinov3b_h512_c32_ft4.yaml
F16: configs/yf10_dinov3b_h512_f16_ft4.yaml
```

两份配置的解析内容只允许在`dataset.future_frame_offset`上不同。共同参数为：训练seed=0、batch=96、4 epochs、LR `4e-5 → 1e-5`、future loss权重0.5。两组均使用`--init_from`加载上述epoch40权重，建立全新的optimizer和scheduler；不能用`--resume`。

当前数据加载器会将越过episode结尾的future索引截到末帧。按392736个anchor统计：F16约有39056个（9.94%）future target被截到末帧，C32约有78112个（19.89%）。这是跨度消融的一部分，也是解释结果时必须报告的混杂因素；后续可增加terminal mask或共同有效anchor消融，首轮不同时更改采样规则。

## 训练命令

两组在同一张GPU上顺序运行。以下命令从LiLa-WAM根目录执行；`pipefail`保证Python失败时命令也返回失败状态。

```bash
cd /home/lab347-no10/Tianyi/LiLa-WAM
set -o pipefail

BASE_CKPT="/home/lab347-no10/Tianyi/LiLa-WAM/checkpoints_vla/yf10_dinov3b_small/sft_2026-09-20_15-07-28/checkpoint_epoch_40.pt"
NORM_STATS="/home/lab347-no10/Tianyi/LiLa-WAM/artifacts/yf10/norm_stats.json"

mkdir -p log/phase2_c32 log/phase2_f16

CUDA_VISIBLE_DEVICES=0 python train.py \
  --config ./configs/yf10_dinov3b_h512_c32_ft4.yaml \
  --norm_stats_path "$NORM_STATS" \
  --init_from "$BASE_CKPT" \
  --save_dir ./checkpoints_vla/yf10_phase2_c32 \
  2>&1 | tee ./log/phase2_c32/train.log

CUDA_VISIBLE_DEVICES=0 python train.py \
  --config ./configs/yf10_dinov3b_h512_f16_ft4.yaml \
  --norm_stats_path "$NORM_STATS" \
  --init_from "$BASE_CKPT" \
  --save_dir ./checkpoints_vla/yf10_phase2_f16 \
  2>&1 | tee ./log/phase2_f16/train.log
```

启动日志必须同时出现：

```text
Training seed: 0
Epochs: 4 | Batch: 96
Model loaded (init_from epoch=40). missing=0, unexpected=0
Starting new stage: lr=4.00e-05 -> 1.00e-05, epochs=4
```

C32还应显示future offset 32，F16显示16。输出checkpoint命名为新阶段的`checkpoint_epoch_1.pt`到`checkpoint_epoch_4.pt`；这里的epoch 4表示继续微调4轮，不是原始训练的总epoch 44。比较两组各自的epoch 4。

# LiLa-WAM / RoboTwin / Meta-World 研究上下文摘要

更新时间：2026-10-19（Asia/Tokyo）

本文用于新对话接续。项目根目录：`/home/lab347-no10/Tianyi/LiLa-WAM`。

## 一、研究目标

当前研究以 LiLa-WAM 为主要 baseline，使用 DINOv3、VTT 任务条件和 Flow Matching Action DiT，在 RoboTwin YF10 十任务上研究视觉动作生成。重点方向是 future visual feature prediction、global intent、spatial intent、空间交互表示和动作条件注入。OPD 暂不作为近期主线，因为计算量和显存开销较大。

## 二、RoboTwin / LiLa-WAM 已完成内容

YF10 数据集：

```text
/home/lab347-no10/Tianyi/LiLa-WAM/data/yf10
```

YF10 DINOv3-B VTT：

```text
/home/lab347-no10/Tianyi/LiLa-WAM/embeddings/yf10_vtt_dinov3b
```

DINOv3-B checkpoint：

```text
/home/lab347-no10/Tianyi/LiLa-WAM/pretrained/dinov3-vitb16
```

YF10 任务为 `adjust_bottle`、`grab_roller`、`hanging_mug`、`move_stapler_pad`、`open_microwave`、`press_stapler`、`scan_object`、`stack_blocks_two`、`stamp_seal`、`turn_switch`。

主要 baseline 结构：DINOv3-B，视觉层 `[-7,-4,-2]`，concat 融合，Action Expert depth 12、hidden size 512、8 heads，动作 chunk 长度 32，Flow Matching 推理步数 10。常用评测协议是 `demo_clean`、`demo_randomized`、`instruction_type=unseen`、`seed=0`、每任务 50 episodes、`num_workers=1`。

已记录的结果：

```text
原始 DINOv3-B+h512 baseline：clean 351/500=70.2%，randomized 323/500=64.6%，overall 67.4%
C32 微调版本：clean 376/500=75.2%，randomized 315/500=63.0%，overall 69.1%
G-aux：overall 67.8%
G2：overall 68.5%
G2 scratch50 epoch40：overall 67.8%
```

当前建议将原始 baseline 或 C32 版本作为后续实验参照，并保持相同测试协议。

## 三、Meta-World 平行代码：已完成

为避免影响 RoboTwin，新增独立目录：

```text
/home/lab347-no10/Tianyi/LiLa-WAM/metaworld_vla
```

新增文件：`README.md`、`check_env.py`、`collect_expert.py`、`dataset.py`、`calc_stats.py`、`precompute_vtt.py`、`train.py`、`eval.py` 和 `configs/metaworld_mt10_b4_h512.yaml`。

Meta-World 代码只复用 `models/model_runner.py` 和 `models/vla_model_fm.py`，没有修改 RoboTwin 的 `train_vla.py`、`hdf5_dataloader/`、`robotwin_infer.py` 或原有配置。

### 1. 环境

已创建环境：

```bash
conda create -n MetaWorldVLA --clone RoboTwin
conda activate MetaWorldVLA
python -m pip install --upgrade "metaworld==3.1.1"
```

已验证 Meta-World 3.1.1、Gymnasium 1.4.0、MuJoCo 3.3.0 可以工作。服务器使用 `MUJOCO_GL=egl` 离屏渲染，相关脚本会自动设置，也可以手动执行：

```bash
export MUJOCO_GL=egl
```

环境检查已经成功输出：

```text
observation shape: (39,) float64
action shape: (4,) float32
rgb shape: (240, 320, 3) uint8
```

Gymnasium 的 Box observation 边界警告属于 Meta-World 环境定义提示，不影响环境创建、渲染或动作执行。

### 2. 当前数据集

数据目录：

```text
/home/lab347-no10/Tianyi/LiLa-WAM/data/metaworld_mt10
```

当前已经采集十个 MT10 任务，每个任务 10 个 episode，总计 100 个 episode。每个 episode 500 帧，专家成功轨迹统计为 100/100，数据目录大小约 665 MB。

任务为：`reach-v3`、`push-v3`、`pick-place-v3`、`door-open-v3`、`drawer-open-v3`、`drawer-close-v3`、`button-press-topdown-v3`、`peg-insert-side-v3`、`window-open-v3`、`box-close-v3`。

每个 HDF5 文件包含：

```text
observations/state       [T, 4]
observations/rgb_jpeg    [T]
actions                  [T, 4]
rewards                  [T]
success                  [T]
```

状态是 `obs[:4]`，即末端 xyz 和夹爪状态；动作是 Meta-World 原生 `[dx, dy, dz, gripper]`。当前已经验证十个任务的专家策略都可以启动并写入 HDF5。

## 四、Meta-World 尚未完成的步骤

以下内容还没有正式完成：

1. 计算 `stats/metaworld_mt10_4d.json`；
2. 使用 DINOv3-Base 计算 `embeddings/metaworld_mt10_vtt_dinov3b/`；
3. 运行真实 DataLoader smoke test；
4. 运行一步 Flow Matching 训练 smoke test；
5. 完成 MT10 训练和正式评测。

当前 10 episodes/task 的数据适合调试管线，不适合作为最终论文训练规模。

## 五、Meta-World 训练配置和命令

配置文件：

```text
metaworld_vla/configs/metaworld_mt10_b4_h512.yaml
```

关键设置是 state/action 维度 4/4，chunk 长度 32，每次执行 4 步，DINOv3-Base，Action Expert hidden size 512，Flow Matching 推理步数 10，训练 50 epochs。

计算统计量：

```bash
python metaworld_vla/calc_stats.py \
  --dataset-dir data/metaworld_mt10 \
  --output stats/metaworld_mt10_4d.json
```

计算 VTT：

```bash
python metaworld_vla/precompute_vtt.py \
  --dataset-dir data/metaworld_mt10 \
  --output-dir embeddings/metaworld_mt10_vtt_dinov3b \
  --dino-path pretrained/dinov3-vitb16 \
  --tasks reach-v3 push-v3 pick-place-v3 door-open-v3 drawer-open-v3 \
  drawer-close-v3 button-press-topdown-v3 peg-insert-side-v3 \
  window-open-v3 box-close-v3
```

训练：

```bash
CUDA_VISIBLE_DEVICES=0 python metaworld_vla/train.py \
  --config metaworld_vla/configs/metaworld_mt10_b4_h512.yaml \
  --stats stats/metaworld_mt10_4d.json \
  --save-dir checkpoints/metaworld_mt10_b4_h512
```

断点续训：

```bash
python metaworld_vla/train.py \
  --config metaworld_vla/configs/metaworld_mt10_b4_h512.yaml \
  --stats stats/metaworld_mt10_4d.json \
  --save-dir checkpoints/metaworld_mt10_b4_h512 \
  --resume checkpoints/metaworld_mt10_b4_h512/checkpoint_epoch_10.pt
```

训练 smoke test：

```bash
python metaworld_vla/train.py \
  --config metaworld_vla/configs/metaworld_mt10_b4_h512.yaml \
  --stats stats/metaworld_mt10_4d.json \
  --save-dir /tmp/metaworld_smoke \
  --max-steps 1
```

测试：

```bash
CUDA_VISIBLE_DEVICES=0 python metaworld_vla/eval.py \
  --config metaworld_vla/configs/metaworld_mt10_b4_h512.yaml \
  --stats stats/metaworld_mt10_4d.json \
  --checkpoint checkpoints/metaworld_mt10_b4_h512/checkpoint_epoch_50.pt \
  --test-num 50 --seed 0 \
  --output eval_result/metaworld_mt10_ep50.json
```

推理保持 RoboTwin 的 receding-horizon 语义：生成 32 步动作，执行前 4 步，再重新读取图像和状态。

## 六、未解决问题和风险

- 需要在真实 DINOv3-Base checkpoint 上运行 VTT，确认条件向量维度与模型一致；
- 需要完成真实 batch 的 DataLoader、前向、反向和 checkpoint 保存验证；
- 需要确认 Meta-World reset 时的 goal 采样与正式评测协议一致；
- Meta-World 当前使用 4 维 state，不能与 RoboTwin 16 维 state 的成功率做绝对数值比较，应比较各自平台内的相对提升；
- 当前数据规模适合管线调试，最终实验建议增加每任务 episode 数量。

## 七、下一步顺序

```text
1. 计算 Meta-World 4D normalization stats
2. 计算 DINOv3-B VTT
3. DataLoader smoke test
4. --max-steps 1 训练 smoke test
5. 2~5 epoch 小规模训练
6. 单任务测试
7. MT10 50 epoch 训练
8. 每任务 50 episodes、固定 seed 的正式评测
9. 进行 state-rich、DINOv3-L 或空间模块消融
```

RoboTwin 主线继续使用原来的配置和 checkpoint；Meta-World 作为平行平台，用于验证 4D state/action 接口和模型结构的跨环境适应性。

## 八、2026-10-20：Residual-off 强制配置消融结果

已修改 `eval_vla_bridge.py` 的配置优先级：显式传入 `--config_path` 时优先使用该配置，不再被 checkpoint 目录中的 `config.yaml` 覆盖；只有未传入配置时才回退到 checkpoint 配置，最后使用默认配置。评测日志已确认出现：

```text
[Config] Using explicitly supplied config:
```

本次消融使用同一个空间残差 v2 gate0 scratch checkpoint：

```text
checkpoints_vla/yf10_phase4_spatial_sres_v2_scratch50_fp32gate0/sft_2026-10-14_16-20-07/checkpoint_epoch_40.pt
```

Residual-off 配置为：

```text
configs/yf10_dinov3b_h512_spatial_sres_v2_c32_scratch50_residual_off_eval.yaml
```

其中仅将 `spatial_intent.residual_scale` 设为 `0.0`，其余模型结构和测试协议保持一致。

结果如下，均为 seed0、每任务50次、YF10十任务：

| 设置 | Clean | Randomized | Overall |
|---|---:|---:|---:|
| Residual-on | 365/500 = 73.0% | 306/500 = 61.2% | 671/1000 = 67.1% |
| Residual-off | 368/500 = 73.6% | 302/500 = 60.4% | 670/1000 = 67.0% |

Residual-off 相比 Residual-on 在 clean 上增加3个成功，在 randomized 上减少4个成功，总体只差1个样本。这个差异远小于当前每个任务50次评测的统计波动，不能说明残差注入带来可靠收益，也不能据此断言残差只在训练阶段起作用。更准确的结论是：当前 scalar residual 在推理时的直接贡献很小，或者其影响已被主干/action expert吸收；同时残差注入没有改善 randomized 鲁棒性。

当前不建议立即更换研究路线。后续优先级为：

1. 对 Residual-on/off 各追加 seed1、seed2，确认1个百分点差异不是偶然波动；
2. 在推理日志中记录 residual 范数、融合特征范数、`residual_norm / base_norm` 和 effective gate，确认残差是否真正进入 action expert；
3. 若多 seed 仍接近，停止把 scalar residual 作为主改进点，转向 `S-aux-shared`：保留空间预测器和辅助监督，但不注入DiT，用于判断空间表征本身是否有用；
4. 若 Residual-off 稳定优于 Residual-on，测试 region/confidence gate，让不同空间区域按状态自适应注入；
5. 若固定网格和 region gate 均无效，再转向 task-conditioned interaction query 或 cross-attention，将任务条件用于选择与动作相关的局部视觉区域；
6. Global Intent G2 保持独立路线，用户后续可自行完成复测，不与本次空间残差结论混合。

当前阶段的论文结论应写成“固定网格 scalar residual 未显示稳定的推理增益”，而不是“空间信息无效”。要证明空间信息无效，还需要完成 S-aux-shared 和多 seed 对照。

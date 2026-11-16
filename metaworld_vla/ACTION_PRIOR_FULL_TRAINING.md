# easy 基线的 C/N/R 完整训练实验

2026-11-02。推理固定10次调用，C8/E4，完整MT50，从头训练50轮，以epoch40为预先固定的主比较，epoch50为附加结果。不是epoch40微调，也不是G2实验。

## 1. 三组配置与训练数据通路

| 组 | 配置（位于metaworld_vla/configs） | 起点 |
|---|---|---|
| C | `metaworld_mt50_easy_prior_c_scratch50.yaml` | 100%标准高斯 |
| N | `metaworld_mt50_easy_prior_n_scratch50.yaml` | 50%标准高斯，50%合法重叠位置缩噪至sigma=0.5 |
| R | `metaworld_mt50_easy_prior_r_scratch50.yaml` | 50%标准高斯，50%合法重叠位置用旧预测+sigma=0.5噪声 |

保持easy基线：DINO-B、DiT depth12/hidden512、batch96、seed0、LR2e-4→5e-5、future-feature关闭、VTT注入关闭。保留原模型参数形状，不修改原始YAML。训练样本shuffle、epoch覆盖、原有padding目标保持不变。

每个样本目标A为当前专家8步动作。起点参数为mean与scale：`Z=mean+scale*epsilon`，训练插值`X_s=(1-s)Z+sA`，速度监督`A-Z`。N/R按样本使用独立RNG进行混合，未选中样本严格使用标准高斯；原始FM噪声与时间采样继续使用原RNG路径。

R的旧计划来自冻结easy epoch40在同episode前4帧观测下的预测：例如观测96帧预测96～103步，将预测100～103步存入当前100帧样本缓存。不会读取未来图像生成先验，不复制专家目标作为先验，不把上一个batch当成上一时刻。前4帧无先验；末尾超出记录范围的位置不使用先验。专家动作原有终止padding不变。

教师只生成缓存。三个新模型均随机初始化，不使用教师权重初始化，不使用--init-from。固定教师缓存与部署新策略自生成旧计划仍存在偏差，本实验不保证性能改善。

## 2. 先一次性生成R组缓存

```bash
cd /home/lab347-no10/Tianyi/LiLa-WAM
conda activate MetaWorldVLA
set -o pipefail
mkdir -p logs/action_prior_full

CUDA_VISIBLE_DEVICES=0 python metaworld_vla/precompute_action_prior.py \
  --config metaworld_vla/configs/metaworld_mt50_easy_prior_r_scratch50.yaml \
  --stats stats/metaworld_mt50_corner2_480_50ep_4d.json \
  --checkpoint checkpoints/metaworld_mt50_easy_noffvtt/checkpoint_epoch_40.pt \
  --output-dir artifacts/action_prior/easy_ep40_c8_e4 \
  --batch-size 64 --workers 4 --seed 321 \
  2>&1 | tee logs/action_prior_full/cache.log
```

缓存全量预测所有合法帧，不是对每4帧取一帧；这样训练仍能覆盖全部frame anchor。缓存保存归一化后4步预测，不保存图像或梯度。批量生成避免逐帧完整推理的高开销，但仍有一次离线GPU成本，应单独报告。

中断恢复：相同命令加`--resume`；batch-size、seed、配置和教师不可改变，已完成episode会校验后跳过。没有--resume时拒绝覆盖已有缓存。显存不足可在首次运行前改batch-size；中途要改变batch-size应换新缓存目录并同步R配置。

manifest记录教师hash、stats hash、模型/观测配置、数据文件身份、每episode缓存hash与complete状态。R训练拒绝不完整、错配置、改过数据或损坏的缓存。C/N不需要读取此缓存，可独立训练。

## 3. 三组从头完整训练

```bash
for GROUP in c n r; do
  CUDA_VISIBLE_DEVICES=0 python metaworld_vla/train.py \
    --config "metaworld_vla/configs/metaworld_mt50_easy_prior_${GROUP}_scratch50.yaml" \
    --stats stats/metaworld_mt50_corner2_480_50ep_4d.json \
    --save-dir "checkpoints/metaworld_mt50_easy_prior_${GROUP}_scratch50" \
    2>&1 | tee "logs/action_prior_full/train_${GROUP}.log" || break
done
```

依次运行，不能在同一GPU同时启动三组。沿用原50轮scheduler，每5轮保存，主比较epoch40。不要将epochs直接改40，否则epoch40学习率轨迹不同。输出已有checkpoint时拒绝覆盖，必须明确resume或使用新目录。

日志应出现：`Flow source protocol`与`FlowSource mode=cold/scale/tail`。N/R的modified_sample_fraction约接近0.5但受无合法先验的开头帧影响，每个batch会波动。三组loss目标分布不同，不能只按loss大小排名；以实际推理指标为准。

### 中断恢复

```bash
CUDA_VISIBLE_DEVICES=0 python metaworld_vla/train.py \
  --config checkpoints/metaworld_mt50_easy_prior_r_scratch50/config.yaml \
  --stats stats/metaworld_mt50_corner2_480_50ep_4d.json \
  --save-dir checkpoints/metaworld_mt50_easy_prior_r_scratch50 \
  --resume checkpoints/metaworld_mt50_easy_prior_r_scratch50/checkpoint_epoch_20.pt
```

C/N同理。新实验checkpoint另外保存source混合RNG、全局RNG和协议；恢复时检查source协议、stats和R缓存manifest。按已完成epoch恢复，smoke-test的部分epoch不能作为正式训练中断点。未宣称跨硬件/软件逐bit复现，原训练入口的旧配置行为不变。

## 4. 主比较：C-cold、N-scale、R-tail

训练完成后：

```bash
export MUJOCO_GL=egl
EP=40
for GROUP in c n r; do
  EXTRA=()
  if [[ "$GROUP" == n ]]; then
    EXTRA=(--action-reuse --reuse-strength 0 --reuse-sigma 0.5)
  fi
  if [[ "$GROUP" == r ]]; then
    EXTRA=(--action-reuse --reuse-strength 1 --reuse-sigma 0.5)
  fi
  RUN="checkpoints/metaworld_mt50_easy_prior_${GROUP}_scratch50"
  CUDA_VISIBLE_DEVICES=0 python metaworld_vla/eval_parallel.py \
    --config "$RUN/config.yaml" \
    --stats stats/metaworld_mt50_corner2_480_50ep_4d.json \
    --checkpoint "$RUN/checkpoint_epoch_${EP}.pt" \
    --workers 1 --test-num 50 --horizon 500 \
    --seed 1000 --policy-seed 456 --num-inference-steps 10 \
    --profile-generation "${EXTRA[@]}" \
    --output "eval_result/action_prior_full/${GROUP}_ep${EP}_matched_env1000_policy456.json" || break
done
```

默认全部50任务；快速检查可加此前5任务--tasks，但正式全任务不能据此省略其他任务。单worker用于可比延迟；只筛选成功率可提高workers但不把竞争耗时当部署延迟。EP=50作为附加结果，勿按各组最高分任意选择轮次。新的场景seed区间1000～1049不与此前0～19筛选完全重合。

## 5. 额外交叉测试（建议保留）

在相同测试命令基础上变更组与EXTRA，使用独立输出名：

| 模型 | 推理选项 | 目的 |
|---|---|---|
| C | cold：不传--action-reuse | 主参照 |
| C | --action-reuse --reuse-strength 1 --reuse-sigma 0.5 | 未匹配训练的复用 |
| N | cold | 是否损害首次启动/回退 |
| N | --action-reuse --reuse-strength 0 --reuse-sigma 0.5 | 缩噪匹配训练 |
| R | cold | 是否保留标准起点能力 |
| R | --action-reuse --reuse-strength 1 --reuse-sigma 0.5 | 尾部匹配训练 |

新训练配置不会自动切换评测source，评测必须显式给出上述开关。N的--action-reuse权重0只用作重叠位置mask，旧预测对mean贡献为0。R推理用当前新模型上次预测，不读取离线教师缓存。

## 6. 兼容性与验证范围

共享FM只新增默认None的source_mean/source_noise_scale；旧batch不含fm_source字段时仍抽取原标准高斯，插值与监督路径不变。Meta-World只有显式training.action_prior.enabled=true才包装数据和构造新source。没有新增模型参数，旧checkpoint仍可strict加载；没有改采集、真机与RoboTwin调用。

CPU测试：`python metaworld_vla/test_prior_training.py`；原尾部复用测试：`python metaworld_vla/test_action_reuse.py`。覆盖标准source等价性、RNG与梯度、A-Z目标、mask、缓存索引、N/R控制及不完整缓存拒绝。

已做真实GPU单episode缓存、C/N/R各一步前向/反向及保存检查；使用/tmp小配置，不是完整50任务训练。完整缓存和50轮训练需用户执行以上命令。缓存训练引入的闭环偏差、性能收益和全任务稳定性仍待实验验证。

先比较成功率、生成延迟与边界控制变化。成功前/后、内部变化、真实运动轨迹等更细指标尚未新增，不把现有边界指标当成完整物理平滑性结论。

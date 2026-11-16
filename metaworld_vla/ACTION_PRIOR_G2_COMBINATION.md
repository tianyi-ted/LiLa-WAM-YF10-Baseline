# C/R交叉推理与G2组合实验

状态：新增独立配置；不修改既有C/N/R或G2配置，不自动启动正式训练。以下使用epoch40、10次生成、原测试场景seed1000与policy seed456。

## 1. 先完成现有C/R交叉推理

已测C-cold与R-tail，补C-tail、R-cold即可构成训练×推理的2×2对照，不需重训。

```bash
cd /home/lab347-no10/Tianyi/LiLa-WAM
conda activate MetaWorldVLA
export MUJOCO_GL=egl
STATS=stats/metaworld_mt50_corner2_480_50ep_4d.json

# C模型开启回传
RUN=checkpoints/metaworld_mt50_easy_prior_c_scratch50
CUDA_VISIBLE_DEVICES=0 python metaworld_vla/eval_parallel.py \
  --config "$RUN/config.yaml" --stats "$STATS" \
  --checkpoint "$RUN/checkpoint_epoch_40.pt" \
  --workers 8 --test-num 50 --horizon 500 \
  --seed 1000 --policy-seed 456 --num-inference-steps 10 \
  --action-reuse --reuse-strength 1 --reuse-sigma 0.5 --profile-generation \
  --output eval_result/action_prior_cross/c_ep40_tail_env1000_policy456.json

# R模型关闭回传：不传--action-reuse即可
RUN=checkpoints/metaworld_mt50_easy_prior_r_scratch50
CUDA_VISIBLE_DEVICES=0 python metaworld_vla/eval_parallel.py \
  --config "$RUN/config.yaml" --stats "$STATS" \
  --checkpoint "$RUN/checkpoint_epoch_40.pt" \
  --workers 8 --test-num 50 --horizon 500 \
  --seed 1000 --policy-seed 456 --num-inference-steps 10 --profile-generation \
  --output eval_result/action_prior_cross/r_ep40_cold_env1000_policy456.json
```

workers=8与已有成功率评测一致；显存不足可降低。并行竞争下时间不能作为单模型部署速度结论；比较延迟需将所有候选统一单worker复测。

## 2. G2组合固定哪个版本？

沿用当前 `metaworld_mt50_corner2_easy_g2.yaml` 中G2：task_attention输入汇聚、mean未来目标、无gate，lambda_intent=0.1。显式固定target_pooling=mean，不是另外的attntarget版本。

未来特征decoder仍关闭；但G2训练需要读取t+8真实未来图像，由冻结DINO提取意图监督目标。推理只用当前图像/状态/VTT预测意图，不读取未来图像。

虽然直接VTT token注入仍关闭，task_attention的G2会间接使用VTT作为任务条件，因此不要把组合版称为“完全不使用VTT”。

两条数据通路独立：

```text
当前图像/状态/VTT → G2预测全局意图 → 一个条件token → DiT
上一轮未执行动作 → 加噪起点Z ─────────────────────→ DiT
```

训练总loss为动作FM loss + 0.1×原G2意图loss。G2只增加自己的模块，不改变旧动作先验定义。

## 3. 两组配对配置

| 配置 | 训练source | G2 |
|---|---|---|
| `configs/metaworld_mt50_easy_prior_c_g2_scratch50.yaml` | 纯噪声 | 开启 |
| `configs/metaworld_mt50_easy_prior_r_g2_scratch50.yaml` | 50%cold、50%旧尾部+sigma0.5噪声 | 开启 |

两组均从头50轮、主比较epoch40，保持batch96、LR2e-4→5e-5、seed0、C8/E4。不能用旧无G2 checkpoint的resume/init-from直接创建新模块。

R+G2通过显式 `cache_reference_config: checkpoints/metaworld_mt50_easy_noffvtt/config.yaml` 复用既有easy教师缓存。兼容性检查只允许global_intent模块差异，其余数据、相机、H/E、归一化与基础模型仍须严格匹配。未配置reference的旧R依然执行原严格检查。无需重新生成缓存，也不改原manifest。

保持同一教师先验有助于比较G2的增量作用；若同时换成G2教师重新生成缓存，会混入教师变化这一因素，应作为后续独立实验。

## 4. 正式训练命令

```bash
set -o pipefail
mkdir -p logs/action_prior_g2

for GROUP in c r; do
  CUDA_VISIBLE_DEVICES=0 python metaworld_vla/train.py \
    --config "metaworld_vla/configs/metaworld_mt50_easy_prior_${GROUP}_g2_scratch50.yaml" \
    --stats "$STATS" \
    --save-dir "checkpoints/metaworld_mt50_easy_prior_${GROUP}_g2_scratch50" \
    2>&1 | tee "logs/action_prior_g2/train_${GROUP}.log" || break
done
```

先运行以上环境变量设置。日志中应有Global intent=True、Intent损失，以及FlowSource cold/tail。无gate版本日志Gate为常数1正常，不是学习失败。若显存不足，先停止并使用相同batch/累积协议修改两份新配置，不改单边参数后声称等预算。

## 5. epoch40测试命令

```bash
for GROUP in c r; do
  EXTRA=()
  if [[ "$GROUP" == r ]]; then EXTRA=(--action-reuse --reuse-strength 1 --reuse-sigma 0.5); fi
  RUN="checkpoints/metaworld_mt50_easy_prior_${GROUP}_g2_scratch50"
  CUDA_VISIBLE_DEVICES=0 python metaworld_vla/eval_parallel.py \
    --config "$RUN/config.yaml" --stats "$STATS" \
    --checkpoint "$RUN/checkpoint_epoch_40.pt" \
    --workers 8 --test-num 50 --horizon 500 \
    --seed 1000 --policy-seed 456 --num-inference-steps 10 \
    --profile-generation "${EXTRA[@]}" \
    --output "eval_result/action_prior_g2/${GROUP}_g2_ep40_matched_env1000_policy456.json" || break
done
```

四个主结果形成C、R、C+G2、R+G2的配对组合。只有R+G2同时优于R和C+G2，才有理由进一步讨论两者互补；仍需额外场景/训练seed确认。训练50轮的epoch50可作为预先声明的附加对照，不按每组最好epoch挑选。

新增G2配置不影响当前交叉推理命令。推荐先完成交叉推理，再决定是否投入两组训练；也可以将组合实验作为独立分支运行，不能取代交叉消融。

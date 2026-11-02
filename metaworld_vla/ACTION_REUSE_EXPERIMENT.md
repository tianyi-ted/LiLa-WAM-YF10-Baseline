# short epoch40：动作尾部复用推理实验

2026-10-29。已实现推理侧尾部复用；没有修改模型结构、训练代码、原始YAML或RoboTwin入口。关闭复用时保留原始噪声采样路径。这里是未匹配训练的起点分布实验，不保证提升。

## 原理与参数

short保存配置为C8/E4：每次生成8步、执行4步。第一次从纯噪声开始；之后把上一完整归一化预测中未执行的后4步移到新chunk前4步：

```text
新起点前4步 = reuse_strength * 旧后4步 + reuse_sigma * 新噪声
新起点后4步 = 标准高斯噪声
```

结合最新观测从生成时间0积分到1；没有新增条件token、没有锁定旧动作、没有训练反传。episode reset和任务切换会清空动作先验。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--action-reuse` | 不启用 | 显式传入才复用 |
| `--reuse-sigma` | 1.0 | 仅对有效旧尾部加噪的标准差 |
| `--reuse-strength` | 1.0 | 旧尾部均值权重 |
| `--num-inference-steps` | 使用保存配置 | 可覆盖为5或10，不改网络形状 |
| `--policy-seed` | 未指定 | 指定后使用独立动作采样RNG；每任务重置，每episode不重置 |
| `--profile-generation` | 不启用 | 同步CUDA记录生成时间，有额外计时开销 |

旧命令不增加这些参数时，不会启用复用、独立RNG或强制计时同步。JSON增加协议/指标字段，原success字段保持不变。若指定独立policy seed，请用相同开关重跑cold对照，不与旧全局RNG结果当作严格配对实验。

## 快速运行检查（不代表成功率）

```bash
cd /home/lab347-no10/Tianyi/LiLa-WAM
conda activate MetaWorldVLA
export MUJOCO_GL=egl

CUDA_VISIBLE_DEVICES=0 python metaworld_vla/eval.py \
  --config checkpoints/metaworld_mt50_short/config.yaml \
  --stats stats/metaworld_mt50_corner2_480_50ep_4d.json \
  --checkpoint checkpoints/metaworld_mt50_short/checkpoint_epoch_40.pt \
  --tasks push-v3 --test-num 2 --horizon 8 --seed 0 \
  --action-reuse --reuse-sigma 0.5 --policy-seed 123 \
  --output eval_result/action_reuse/short_ep40_smoke.json
```

每episode8步应产生2次chunk生成，其中第2次复用4个位置；2个episode合计4次生成、2次复用、8个复用位置。不要用这个短horizon评价任务成功率。

## 第一轮：B10、W10-S1、W10-S05

```bash
CONFIG=checkpoints/metaworld_mt50_short/config.yaml
CKPT=checkpoints/metaworld_mt50_short/checkpoint_epoch_40.pt
STATS=stats/metaworld_mt50_corner2_480_50ep_4d.json

for MODE in B10 W10-S1 W10-S05; do
  EXTRA=()
  if [[ "$MODE" == W10-S1 ]]; then EXTRA=(--action-reuse --reuse-sigma 1.0); fi
  if [[ "$MODE" == W10-S05 ]]; then EXTRA=(--action-reuse --reuse-sigma 0.5); fi
  CUDA_VISIBLE_DEVICES=0 python metaworld_vla/eval_parallel.py \
    --config "$CONFIG" --stats "$STATS" --checkpoint "$CKPT" \
    --tasks push-v3 pick-place-v3 door-open-v3 peg-insert-side-v3 button-press-topdown-v3 \
    --workers 1 --test-num 20 --horizon 500 --seed 0 \
    --policy-seed 123 --num-inference-steps 10 --profile-generation \
    "${EXTRA[@]}" \
    --output "eval_result/action_reuse/short_ep40_${MODE}_screen_env0_policy123.json"
done
```

并行入口会拒绝覆盖已有结果。任务并行会竞争GPU；计时对比推荐workers=1。需要并行筛选可改为2，但不要把竞争下延迟当作单策略部署延迟。

## 后续

选出sigma后，把上面生成步数改为5，分别跑cold与reuse，使用新的输出名B5/W5。正式50任务时删除 `--tasks ...`，把test-num改为50。环境seed可用0、1000、2000以避免当前seed+episode协议的大量重叠。

输出中 `action_generation.action_reuse` 确认开关，任务的 `generation_metrics.reused_chunks/reused_positions` 确认真复用。还记录边界xyz动作L2差分与夹爪绝对差分（实际发送、裁剪后动作）、每episode成功计数，以及可选生成延迟均值/p95。边界指标受各策略轨迹不同影响，不单独作为因果结论。

计时覆盖当前观测预处理、视觉特征、DiT积分和动作转CPU，不含环境渲染/step；不是整任务运行效率，也不是异步控制频率。此版未实现内部动作变化、失败类型分类、训练先验缓存或匹配训练。

验证命令：`python metaworld_vla/test_action_reuse.py`。测试不加载DINO或MuJoCo，覆盖源构造、时间对齐、reset、禁用时旧Euler结果与RNG一致性、独立采样RNG。

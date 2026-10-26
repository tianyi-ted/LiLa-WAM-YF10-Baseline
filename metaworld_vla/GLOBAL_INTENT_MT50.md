# MT50 C8 baseline 与 G2 全局意图实验

2026-10-24。当前 base 保存的评测结果为 86.56%（`eval_result/metaworld_mt50_base/metaworld_mt50_parallel2.json`）。原实验及用户现有 base/short/tiny YAML 不修改；以保存的训练配置为准。

新增配对配置：

| YAML | chunk / 执行 | future offset | 全局意图 |
|---|---|---|---|
| configs/metaworld_mt50_corner2_c8_base.yaml | 8 / 4 | 32 | 关闭 |
| configs/metaworld_mt50_corner2_c8_global_intent.yaml | 8 / 4 | 32 | G2，mean pooling，gate 0.1 |

先比较 C32 base 与 C8 baseline，再比较 C8 baseline 与 C8 G2。保持 H=32 是为了避免同时改变未来特征辅助监督的预测时距；H 无需等于 action chunk。现有 short 是 C8/E4/H12，tiny 是 C8/E4/H16 且结构更小，都不能视为这个严格配对 baseline。新配置仍为 DINO-B、[-7,-4,-2]、h512/depth12/heads8、batch96、50 epochs、LR2e-4，数据、stats、VTT 完全复用。

G2 目标为冻结 DINO -2 层的 `LN(mean(F[t+32])) - LN(mean(F[t]))`，stop-gradient。预测器只用当前 patch、当前 state、VTT 和 H，损失权重 lambda_intent=0.1，余弦损失权重=0。预测结果经过可学习 tanh gate 作为一个 token 注入 Action DiT，保留原 future-feature loss（权重0.5）。推理每个 chunk 预测一次，并在10步 Flow Matching 积分中复用，不读取未来帧。

此前 Meta-World train.py 写死 use_global_intent=False、eval.py 未提供意图条件；这次已补齐实际训练/推理路径。global_intent.enabled 不是仅构造一个未训练的 head。日志会显示 action/future/intent loss 与有效 gate，同时修复 epoch 平均 loss 的分母（原 smoke test 的均值会被完整 epoch batch 数稀释）。

## 命令

从头训练两个独立实验，不使用旧 C32 checkpoint 的 --resume / --init-from。改变动作长度和新增模块后 strict 参数形状不同。无需重新采集、计算 stats 或 VTT。

```bash
cd /home/lab347-no10/Tianyi/LiLa-WAM
conda activate MetaWorldVLA

# 先训练并评测 C8 baseline。
CUDA_VISIBLE_DEVICES=0 python metaworld_vla/train.py \
  --config metaworld_vla/configs/metaworld_mt50_corner2_c8_base.yaml \
  --stats stats/metaworld_mt50_corner2_480_50ep_4d.json \
  --save-dir checkpoints/metaworld_mt50_c8_base

# G2 一步检查；确认日志 intent loss 非零、gate 存在。
CUDA_VISIBLE_DEVICES=0 python metaworld_vla/train.py \
  --config metaworld_vla/configs/metaworld_mt50_corner2_c8_global_intent.yaml \
  --stats stats/metaworld_mt50_corner2_480_50ep_4d.json \
  --save-dir checkpoints/metaworld_mt50_c8_g2_smoke --max-steps 1

# G2 正式训练。
CUDA_VISIBLE_DEVICES=0 python metaworld_vla/train.py \
  --config metaworld_vla/configs/metaworld_mt50_corner2_c8_global_intent.yaml \
  --stats stats/metaworld_mt50_corner2_480_50ep_4d.json \
  --save-dir checkpoints/metaworld_mt50_c8_g2

# 两组使用相同评测协议；保持50个任务、每任务50次、seed0。
CUDA_VISIBLE_DEVICES=0 python metaworld_vla/eval_parallel.py \
  --config checkpoints/metaworld_mt50_c8_base/config.yaml \
  --stats stats/metaworld_mt50_corner2_480_50ep_4d.json \
  --checkpoint checkpoints/metaworld_mt50_c8_base/checkpoint_epoch_50.pt \
  --workers 2 --test-num 50 --seed 0 \
  --output eval_result/metaworld_mt50_c8_base_ep50.json

CUDA_VISIBLE_DEVICES=0 python metaworld_vla/eval_parallel.py \
  --config checkpoints/metaworld_mt50_c8_g2/config.yaml \
  --stats stats/metaworld_mt50_corner2_480_50ep_4d.json \
  --checkpoint checkpoints/metaworld_mt50_c8_g2/checkpoint_epoch_50.pt \
  --workers 2 --test-num 50 --seed 0 \
  --output eval_result/metaworld_mt50_c8_g2_ep50.json
```

若需要串行，使用 eval.py，删除 --workers 即可。只做推理检查可加 --tasks reach-v3 --test-num 2。不要将 smoke checkpoint 的 epoch1 当成完整一轮训练。

## 验证与下一步

已使用真实自采 batch、实际 DINO-B、CPU FP32 验证 baseline/G2 一步训练、反向和 checkpoint；确认 intent predictor 梯度非零、gate 梯度有限；加载 smoke checkpoint 后输出 [8,4]，每 chunk 只预测1次意图，10次去噪使用同一意图张量。未进行完整 GPU 训练或正式成功率评测。

后续先保持 mean 全局意图作为对照，再单独测试 task_attention 输入汇聚；分块空间意图留待下一项独立实验。目前 Meta-World 入口遇到 spatial_intent.enabled=true 会明确报未接入，不会静默忽略。RoboTwin 共享模型与主训练入口未修改。

# LiLa-WAM 隔离式 RoboTwin RL 实验方案

> 2026-10-22 更新：低资源 RL 的优先验证平台调整为 Meta-World，详细推进步骤、现有本地资源、可执行预检命令及原型问题见 [Meta-World RL 实验路线](RL_METAWORLD_EXPERIMENT.md)。本文保留为 RoboTwin 迁移参考；不是已完成的在线训练系统。现有 `rl/` 控制器只能解析 RoboTwin endpose，不能直接用于 Meta-World。没有改动旧模型代码或启动训练。

## 1. 目的与兼容性边界

本方案在已有 SFT/Flow-Matching VLA 之外增加一个轻量 RL residual policy。它不修改 `train.py`、`robotwin_infer.py`、`eval_yf10.sh` 或任何已有 YAML，因此原有训练、checkpoint 和测试命令保持不变。

新增代码位于 `rl/`：

- `residual_policy.py`：零初始化 residual actor 与 value head；
- `reward.py`：RoboTwin 成功奖励/可选阶段奖励；
- `rollout_env.py`：将 RoboTwin 的 `get_obs()` + `take_action()` 封装成宏步接口；
- `configs/rl_residual_yf10_single.yaml`：独立实验配置。

RL 不导入时，现有流程不会调用本模块。residual head 的最后一层初始化为 0，在确定性推理、相同 SFT 随机数及动作缓存下，新控制器第一次输出满足：

```text
a_RL = a_SFT + residual_scale * 0 = a_SFT
```

随机探索时，即使均值为0仍会采样非零残差，不能称为完全等价。目前原型将高斯样本clamp后求log_prob，必须修复其概率处理，补全超时bootstrap与优化循环后才能用于PPO训练；详见新文档第5节。

## 2. 当前 RoboTwin 的 RL 数据通路

```text
TASK_ENV.get_obs()
       │
       ├─ RGB / endpose / qpos ──> 原有 RobotWinInference
       │                              │
       │                              └─ a_SFT
       │
       └─ endpose ──> ResidualActionPolicy ──> Δa
                                      │
                         a = a_SFT + α·Δa
                                      │
                         TASK_ENV.take_action(a)
                                      │
                         eval_success / step_lim / stage_reward
                                      │
                            (s, a, r, done, info)
```

一次 RL 宏步与现有评测中的一次 `model.step()` + 一次 `TASK_ENV.take_action()` 完全对应。`take_action()` 内部可能推进多个物理子步，但 RL 缓冲区只记录一个宏步，避免改变已有 VLA 的动作执行语义。

## 3. 为什么第一版只训练 residual

若直接对整个 Flow Matching 策略做 PPO，需要额外定义可用的策略概率或增广随机过程；普通确定性 ODE 输出没有现成的 PPO log-prob。独立 residual PPO 只计算 residual 策略概率，不要求计算冻结 Flow Matching 的似然。第一版计划只训练：

- 14 维动作 residual head；
- 一个 value head；
- 可选共享的小型状态编码器。

DINO、VTT、Action Expert 和原 SFT checkpoint 冻结。这样 RL 主要改善的是“当前 SFT 动作附近的局部修正”，而不是重新学习视觉理解或动作生成器。

## 4. 奖励设置

默认只使用 terminal binary reward：成功为 `1`，失败为 `0`，并可选极小 step penalty。部分 RoboTwin 任务提供 `stage_reward()`，但第一轮不启用，避免不同任务的阶段奖励尺度不一致。

后续若需要 dense reward，应先统一尺度并记录：成功奖励、阶段进度、动作平滑惩罚、超时惩罚；仿真器中的 privileged 状态只能用于 reward，不能输入 policy。

## 5. 资源预算与安全配置

4070 Ti 首轮建议固定：

```yaml
num_envs: 1
rollout_length: 64
train_batch_size: 4
freeze_sft: true
freeze_dino: true
freeze_action_expert: true
```

关闭视频、depth、point cloud 和多相机，只保留当前 baseline 所需的 head RGB 与 endpose。rollout buffer 放 CPU；如果 OOM，先降低 `train_batch_size`，不要修改已有 baseline 的 `action_chunk_size=32`。

## 6. 推荐实验顺序

1. **接口冒烟测试**：加载 epoch40 SFT，`enabled=false`，确认输出和旧评测一致。
2. **零 residual 测试**：`enabled=true` 但保持 residual head 零初始化，成功率应与 SFT 基本一致。
3. **单任务 residual RL**：先选 `place_can_basket` 或 `adjust_bottle`，单环境、64 宏步 rollout。
4. **十任务训练**：确认单任务有收益后再扩展到 YF10。
5. **轻微解冻**：只有 residual 明确提升后，才解冻 Action Expert 最后 1–2 层，并保留 BC/KL 约束。

每个实验必须单独保存 RL checkpoint、奖励曲线、成功率和随机种子；不要覆盖 `checkpoints_vla/` 下的 baseline 文件。

## 7. 当前限制

本目录提供隔离的策略/环境基础设施，不自动启动 PPO 训练，也不改变现有评测。`rl/ppo.py` 已有最小 GAE/PPO loss 函数，但没有完整训练循环。真正训练前仍需补充 rollout buffer、正确采样概率、终止/截断处理、优化循环、正则及独立 RL 评测入口。这样可以先验证接口与资源，再决定是否投入完整 RL 实现。此前显存范围为估计，尚未经过真实在线 RL 测量，不能作为12GB一定可运行的保证。

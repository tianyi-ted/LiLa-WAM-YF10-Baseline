# 新采样协议与历史测评的对比

新增 `eval_adapolicy.py`；现有 `eval.py` 和 `eval_parallel.py` 保持本次修改前的内容。

## 场景与随机数

- 历史协议：每任务构造一次带 seed 的 MT1，通过随机任务包装器有放回抽取50实例场景池。
- 新协议：参考 AdaPolicy，直接使用 `MT1.train_classes` 创建环境，设置任务后关闭 `_freeze_rand_vec`，每次 reset 重新采样物体/目标位置。
- 可复现补充：AdaPolicy 公开脚本没有完整设置实际场景 RNG。新脚本每回合同时设置全局 NumPy、Python 和环境自身 RNG，实际场景 seed 为 `--seed + episode_index`，不依赖被当前版本忽略的 `reset(seed=...)`。
- 模型噪声采用独立 Torch Generator，每回合 seed 为 `--policy-seed + episode_index`；`--policy-seed` 默认0。因此前一回合长度不改变下一回合的噪声流。不同模型若生成次数不同，同一回合后续对应噪声的使用仍可能不同；没有强制 CUDA 逐位确定性。
- JSON 保存每回合的 `initial_rand_vec`、有效场景 seed、模型噪声 seed、步数与成功标记，并汇总 `unique_initializations`。相同环境版本、任务和场景 seed 应复现相同初始化；不同 seed 本身不构成与训练集严格不重叠的证明。

保留你当前模型的相机、180度方向处理、480→224预处理、动作 chunk 和执行长度，不引入 AdaPolicy 的动作平滑或模型。默认500步便于与历史结果比较；AdaPolicy 默认200步，可传 `--horizon 200`。新协议首次成功即结束，该回合按曾经成功计分。

## 命令

以下选用本地已存在的 tiny 第40轮权重。使用 checkpoint 目录保存的配置，避免当前 YAML 改动导致不匹配。测试其他模型时同时替换 config、checkpoint 和 output。

```bash
cd /home/lab347-no10/Tianyi/LiLa-WAM
conda activate MetaWorldVLA

# 小规模验证：一个任务、2次测试。
CUDA_VISIBLE_DEVICES=0 python metaworld_vla/eval_adapolicy.py \
  --config checkpoints/metaworld_mt50_tiny/config.yaml \
  --stats stats/metaworld_mt50_corner2_480_50ep_4d.json \
  --checkpoint checkpoints/metaworld_mt50_tiny/checkpoint_epoch_40.pt \
  --tasks reach-v3 --test-num 2 --horizon 500 \
  --seed 10000 --policy-seed 0 \
  --output eval_result/metaworld_mt50_tiny_ep40_unfrozen_smoke.json

# 新协议：配置中的全部任务，每任务50次。
CUDA_VISIBLE_DEVICES=0 python metaworld_vla/eval_adapolicy.py \
  --config checkpoints/metaworld_mt50_tiny/config.yaml \
  --stats stats/metaworld_mt50_corner2_480_50ep_4d.json \
  --checkpoint checkpoints/metaworld_mt50_tiny/checkpoint_epoch_40.pt \
  --test-num 50 --horizon 500 --seed 10000 --policy-seed 0 \
  --output eval_result/metaworld_mt50_tiny_ep40_unfrozen_env10000_policy0.json

# 历史协议控制组，使用同一权重、500步预算，不覆盖旧结果。
CUDA_VISIBLE_DEVICES=0 python metaworld_vla/eval.py \
  --config checkpoints/metaworld_mt50_tiny/config.yaml \
  --stats stats/metaworld_mt50_corner2_480_50ep_4d.json \
  --checkpoint checkpoints/metaworld_mt50_tiny/checkpoint_epoch_40.pt \
  --test-num 50 --horizon 500 --seed 0 \
  --output eval_result/metaworld_mt50_tiny_ep40_legacy_seed0_control.json
```

这比较的是两个整体测评协议，并非仅改变场景池的单变量实验：新协议逐回合重置独立动作噪声；历史入口现也已改为成功后停止。输出文件已存在时会拒绝覆盖。

## 日志与并行（2026-10-31，按旧版打印方式更新）

`eval_adapolicy.py` 与旧版串行入口一样，在终端打印模型加载、逐回合 `episode N success=...`、任务成功率及最终结果，不再使用四分位进度输出。可选 `--log-file PATH` 额外保存 Python logging 明细；第三方进度条等标准输出不保证被该选项收集。要将全部明细保存在文件并保持前台简洁，使用以下并行入口。

新增 `eval_adapolicy_parallel.py`，把不同任务分配给多个子进程；每个进程拥有自己的模型及环境，同一张 GPU 上共享可见设备。通过 `--workers` 控制进程数，默认2；通过 `--threads-per-worker` 控制每进程 CPU 线程数，默认1。每任务的有效场景种子、独立策略噪声、相机预处理和成功判定与新版串行入口一致，不因任务分配而改变。

```bash
CUDA_VISIBLE_DEVICES=0 python metaworld_vla/eval_adapolicy_parallel.py \
  --config checkpoints/metaworld_mt50_tiny/config.yaml \
  --stats stats/metaworld_mt50_corner2_480_50ep_4d.json \
  --checkpoint checkpoints/metaworld_mt50_tiny/checkpoint_epoch_40.pt \
  --workers 4 --threads-per-worker 1 \
  --test-num 50 --horizon 500 --seed 10000 --policy-seed 0 \
  --output eval_result/metaworld_mt50_tiny_ep40_unfrozen_parallel4.json
```

先可在命令末尾加 `--tasks reach-v3 push-v3 --test-num 2 --workers 2` 做小规模测评（后面的同名参数会覆盖前面的值），或用 `--dry-run` 只打印进程命令。增加进程会增加模型副本和显存占用，实际加速取决于 GPU、CPU 与渲染负载。

终端会显示日志目录：`<output_stem>_parts_<run_id>/`。其中 `worker_XX.log` 完整记录模型加载、逐回合日志和异常，`worker_XX.json` 保存子任务结果，`manifest.json` 保存实际执行命令。不再拆分 `worker_XX_detail.log`。前台与旧版并行一致，只显示 Worker 启动、Worker 完成和最终汇总，不转发逐任务进度。验证子进程结果完整、协议一致后才写最终 JSON。

新版并行入口独立调度子进程，复用旧版结果校验函数。本次打印调整未改动 `eval.py`、`eval_parallel.py`。设备、模型或环境版本变化可能引入数值差异，不保证跨设备逐位一致。

## “50次只有30种场景”的范围

历史 seed0 的 reset-only 审计仅检查 reach-v3、push-v3、pick-place-v3，各自50次 reset 得到30种唯一初始化。这是三项任务的检查结果，并非已经核验全部MT50。审计未运行策略 rollout，也未将原始 HDF5 的完整场景状态用于比较；详见 `SEED_AUDIT_2026-10-30.md`。

## 参考源码

- https://github.com/teee000/AdaPolicy/blob/main/metaworld/scripts/eval_metaworld_parallel.py
- https://github.com/Farama-Foundation/Metaworld/blob/v3.1.1/metaworld/sawyer_xyz_env.py

## 本次验证

- 新协议 seed10000～10049：reach-v3、push-v3、pick-place-v3 各50次 reset，分别获得50种唯一初始化；同一环境再次用seed10000重置，初始化向量精确一致。
- 三个任务的 RGB 渲染形状均为480×480×3、uint8。
- tiny 第40轮已有权重在 CPU 完成1任务×1回合×1步真实推理，模型加载、动作执行、JSON保存成功。该检查不是完整成功率测评，尚未运行2500回合新协议实验。
- 本次编辑前后 `eval.py`、`eval_parallel.py` 的 SHA256 一致；原有工作区改动保持原样。

2026-10-31 日志/并行更新验证：CPU 上真实运行两个任务（reach、push）×各1回合×1步，串行和双进程的逐回合记录、初始化向量及策略统计一致；详细日志确实包含模型加载和逐回合明细。未运行完整成功率实验，也未测量 GPU 并行加速比。

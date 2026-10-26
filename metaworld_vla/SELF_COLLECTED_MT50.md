# 自采 MT50：corner2，480×480，50 条成功轨迹/任务

当前主线使用自采数据，外部 MetaWorld_YF 保留用于后续研究，详见 `METAWORLD_YF_MT50.md`。本方案采用相同的 MT50 任务列表和每任务 50 条规模，但不改写自采 HDF5 为 YF 格式。

配置：`configs/metaworld_mt50_corner2_480_50ep_b4_h512.yaml`。

短 chunk 备选配置（原 32/4 配置保留）：

| 配置文件 | 生成步数 | 执行步数 | future_frame_offset |
|---|---:|---:|---:|
| metaworld_mt50_corner2_480_50ep_c16_e8_b4_h512.yaml | 16 | 8 | 16 |
| metaworld_mt50_corner2_480_50ep_c12_e6_b4_h512.yaml | 12 | 6 | 12 |

三者共用相同数据、stats 和 VTT，不需要重采或重算。未来帧偏移随 chunk 同步，因此这些备选也改变了 future-feature 辅助任务的预测时距；若做只改变动作长度的严格消融，可单独将 future_frame_offset 固定为 32。feat_layers 仍为 [-7,-4,-2]，没有同时更改视觉层。

训练时替换 --config，并为两个实验分别使用带 c16_e8 / c12_e6 的独立 save-dir；评测也必须使用对应 YAML/checkpoint。旧 32 步 checkpoint 含不同长度的位置编码，不能直接 strict resume 到短 chunk。默认从头训练。执行 8 或 6 步后重新读取观测，不会执行完整生成 chunk。

采集完成后的实际目录为 `data/metaworld_mt50_corner2`，配置已同步。核验得到 50 个任务、2,500 条成功轨迹、190,451 帧；每任务恰好 50 条，首帧均为 480×480，原生 corner2，方向标记 180。下方 stats/VTT/checkpoint 的 `480_50ep` 后缀只是实验命名，保持不变。

- 原生 corner2，不指定自定义 camera_profile；渲染 480×480，原有旋转 180°后保存并记录方向标记，读取时不再次旋转。
- DataLoader、VTT、在线模型输入统一 resize 到 224×224；在线评测也先渲染 480×480，再执行同样方向/缩放处理。
- state/action 为 4D，chunk 32，执行前 4 步，尾部 xyz 补 0、夹爪保持最后控制值，padding 参与 loss。
- 每任务目标 50 条成功轨迹；最大尝试次数 500。失败不保存，日志若显示未达目标，需要单独补采；不能把最大尝试数当成保证成功数量。
- 最大长度设为 500，成功即停。它是时间上限，不是固定轨迹长度。

## 完整命令

在项目根目录执行，使用 MetaWorldVLA 环境。命令只供用户启动，未自动执行完整采集。

```bash
conda activate MetaWorldVLA
cd /home/lab347-no10/Tianyi/LiLa-WAM
export MUJOCO_GL=egl

python metaworld_vla/collect_expert.py \
  --config metaworld_vla/configs/metaworld_mt50_corner2_480_50ep_b4_h512.yaml \
  --output-dir data/metaworld_mt50_corner2 \
  --target-successes 50 --max-attempts 500 \
  --horizon 500 --success-only --stop-on-success --seed 0

python metaworld_vla/inspect_dataset_tk.py \
  --dataset-root data/metaworld_mt50_corner2 --camera corner2

python metaworld_vla/calc_stats.py \
  --dataset-dir data/metaworld_mt50_corner2 \
  --output stats/metaworld_mt50_corner2_480_50ep_4d.json

CUDA_VISIBLE_DEVICES=0 python metaworld_vla/precompute_vtt.py \
  --config metaworld_vla/configs/metaworld_mt50_corner2_480_50ep_b4_h512.yaml

CUDA_VISIBLE_DEVICES=0 python metaworld_vla/train.py \
  --config metaworld_vla/configs/metaworld_mt50_corner2_480_50ep_b4_h512.yaml \
  --stats stats/metaworld_mt50_corner2_480_50ep_4d.json \
  --save-dir checkpoints/metaworld_mt50_corner2_480_50ep_smoke --max-steps 1

CUDA_VISIBLE_DEVICES=0 python metaworld_vla/train.py \
  --config metaworld_vla/configs/metaworld_mt50_corner2_480_50ep_b4_h512.yaml \
  --stats stats/metaworld_mt50_corner2_480_50ep_4d.json \
  --save-dir checkpoints/metaworld_mt50_corner2_480_50ep_b4_h512

CUDA_VISIBLE_DEVICES=0 python metaworld_vla/eval.py \
  --config metaworld_vla/configs/metaworld_mt50_corner2_480_50ep_b4_h512.yaml \
  --stats stats/metaworld_mt50_corner2_480_50ep_4d.json \
  --checkpoint checkpoints/metaworld_mt50_corner2_480_50ep_b4_h512/checkpoint_epoch_50.pt \
  --test-num 50 --seed 0 --output eval_result/metaworld_mt50_corner2_50ep.json
```

`collect_expert.py --config` 读取 task_names、camera_names、camera_profile 和 render_size；这些相机设置以配置为准。任务子集可以用 `--tasks reach-v3` 覆盖。成功数量、horizon、seed 仍由 CLI 指定。`--seed` 是基础 seed，每次尝试使用 seed+episode_id，同时传给环境构造与 reset。

请使用全新输出目录。重新执行完整命令会覆盖同名 episode，不能当成断点续采。失败尝试会产生编号空缺，因此保存 50 条不代表最后文件名恰好是 episode_000049。

## 两种 DataLoader

| 配置 dataset.format | 读取类 | 结构 |
|---|---|---|
| native | dataset_native.NativeMetaWorldDataset | task/*.hdf5，observations/state、actions、observations/rgb_jpeg |
| metaworld_yf | dataset_yf.YFMetaWorldDataset | task/demo_clean/data/*.hdf5，proprio/state、joint_action/vector、observation/corner或behind/rgb |

两类独立限定目录和字段，防止误混用；训练入口根据 format 选择。未指定 format 的旧配置继续使用 auto 兼容读取器。模型看到的 batch 接口相同。stats 和 VTT 仍通过共享只读适配支持两种格式，不需要转换数据。

YF PNG 已正立，读取时保留原方向；native JPEG 依据方向标记处理。两个实验需要各自的数据路径、stats、VTT 和 checkpoint，不能混用条件向量或复用未对齐的在线相机。

## 单卡多进程评测

`eval_parallel.py` 按任务将工作分成多个独立进程，每个进程有自己的模型和 MuJoCo 环境，继承同一个 CUDA_VISIBLE_DEVICES。它不会将同一 episode 的动作分开执行，也不拆分同任务的 episode，以保留原任务 reset 顺序。

```bash
CUDA_VISIBLE_DEVICES=0 python metaworld_vla/eval_parallel.py \
  --config checkpoints/metaworld_mt50_corner2_480_50ep_b4_h512/config.yaml \
  --stats stats/metaworld_mt50_corner2_480_50ep_4d.json \
  --checkpoint checkpoints/metaworld_mt50_corner2_480_50ep_b4_h512/checkpoint_epoch_50.pt \
  --workers 2 --threads-per-worker 1 \
  --test-num 50 --seed 0 \
  --output eval_result/metaworld_mt50_parallel2.json
```

优先使用训练目录内保存的 config.yaml，保证视觉层和 chunk 设置与 checkpoint 一致。--workers 是总进程数，不是每任务进程数。可先用 `--tasks reach-v3 push-v3 --test-num 2` 做小规模测速，再比较 workers=1/2/4；显存空闲不代表计算、渲染和 CPU 不存在瓶颈，不保证线性加速。`--dry-run` 只打印命令。并行模式不启用 viewer。

子进程输出保存在自动创建的 *_parts_* 目录，分别含 worker_XX.log 和 worker_XX.json。只有所有子进程成功、任务完整且计数一致时才写最终汇总；失败则保留日志并停止其他子进程。输出文件已存在时拒绝覆盖。

为使分组不改变 Flow Matching 的噪声流，eval.py 现在每个任务开始时固定 Python/NumPy/PyTorch RNG 为 --seed。新旧串行入口使用相同规则，但此前未固定 Torch RNG 的历史结果不保证逐 episode 复现。CUDA 算子的数值非确定性仍可能造成差异，不能宣称跨硬件绝对一致。

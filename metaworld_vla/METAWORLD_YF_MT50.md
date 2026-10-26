# MetaWorld_YF MT50 使用说明

当前状态（2026-10-22）：已归档保留，暂不作为主线训练/在线评测数据。原因是提供方机械臂材质、相机姿态及采集协议尚未与本地环境对齐。后续主线改用自采 MT50，见 `SELF_COLLECTED_MT50.md`；本数据及已有统计量不删除、不覆盖。

原始压缩包来自 `/media/lab347-no10/data/tianyi/MetaWorld_YF`，解压到项目的 `data/metaworld_yf_mt50/`，不重编码、不修改源 HDF5。解压记录为数据目录内的 `extraction_manifest.json`。

2026-10-22 核验：50 个任务，每任务 50 条，共 2,500 条、225,304 帧；episode 长度 28～364，2,500 个 episode 的 success 属性均为 true。所有 state/action 数值有限，两路 PNG 的首尾帧均能解码为 256×256。50 个任务均通过 DataLoader 取样；真实单 episode 的 VTT 和一步 CPU 训练（含反向/checkpoint）通过。完整 VTT 和正式训练未启动。

发现三个任务存在完全相同的 state/action 轨迹：plate-slide-back-side-v3 为 50 条中 20 条唯一，plate-slide-back-v3 为 23 条唯一，plate-slide-side-v3 为 10 条唯一。此检查不代表所有图像也重复，未自动去重。完整报告为 `data/metaworld_yf_mt50/validation_report.json`。全数据归一化统计量已生成到 `stats/metaworld_yf_mt50_4d.json`。

结构与本项目原采集器不同：

```text
<task>/demo_clean/data/<task>_ep_XXXX.hdf5
  proprio/state           [T,4]
  joint_action/vector     [T,4]
  observation/corner/rgb  [T]  PNG，256×256
  observation/behind/rgb  [T]  PNG，256×256
  reward                  [T]
```

HDF5 的 `success` 是 episode 属性，不是逐帧标签。`episode_seed`、`reset_rng_mt` 等 reset 信息原样保留。数据集提供方的 README 没有完整采集协议；不能仅凭相机键名认定其是原生 corner/corner2。抽样 PNG 已正立，YF 读取分支保持原图方向，不执行旧采集器的 180°旋转。

新配置：`configs/metaworld_yf_mt50_corner_b4_h512.yaml`。使用 50 个任务、单路 corner、4D state/action、224×224 输入、32 步 chunk、执行前 4 步、DINOv3-B/h512、future-feature 辅助任务。两路相机都在原始文件中保留；当前模型只用一路，不是双相机融合。padding 为 xyz=0、夹爪重复末值，future image 越界取末帧。

在项目根目录、MetaWorldVLA 环境执行：

YF 专用 Tk 入口支持 `--task` 筛选和窗口内相机下拉切换，默认打开已下载目录的第一条数据：

```bash
python metaworld_vla/inspect_yf_dataset_tk.py --task pick-place-v3 --camera corner
python metaworld_vla/inspect_yf_dataset_tk.py --task pick-place-v3 --camera behind
```

同样支持直接传入一个 YF HDF5 路径作为位置参数。图像保持原 PNG 方向；状态区的 episode_success 是整条轨迹标签。需要桌面显示环境，本入口不启动 MuJoCo。

```bash
python metaworld_vla/inspect_dataset_tk.py \
  --dataset-root data/metaworld_yf_mt50 --camera corner

# 换另一路只改变查看参数，不修改数据。
python metaworld_vla/inspect_dataset_tk.py \
  --dataset-root data/metaworld_yf_mt50 --camera behind

python metaworld_vla/calc_stats.py \
  --dataset-dir data/metaworld_yf_mt50 \
  --output stats/metaworld_yf_mt50_4d.json

CUDA_VISIBLE_DEVICES=0 python metaworld_vla/precompute_vtt.py \
  --config metaworld_vla/configs/metaworld_yf_mt50_corner_b4_h512.yaml

CUDA_VISIBLE_DEVICES=0 python metaworld_vla/train.py \
  --config metaworld_vla/configs/metaworld_yf_mt50_corner_b4_h512.yaml \
  --stats stats/metaworld_yf_mt50_4d.json \
  --save-dir checkpoints/metaworld_yf_mt50_corner_smoke --max-steps 1

CUDA_VISIBLE_DEVICES=0 python metaworld_vla/train.py \
  --config metaworld_vla/configs/metaworld_yf_mt50_corner_b4_h512.yaml \
  --stats stats/metaworld_yf_mt50_4d.json \
  --save-dir checkpoints/metaworld_yf_mt50_corner_b4_h512
```

新 VTT 必须基于本数据集生成，不能复用旧 MT10 条件；`--config` 自动读取任务列表、选定相机及输入尺寸。切换 behind 时应复制 YAML、修改 camera_names 和独立 VTT/checkpoint 路径。

当前配置使用全部 episode，不包含离线 train/validation 划分。在线评测暂设 `dataset.online_eval_verified: false`：样本机械臂为蓝色，本地原始资产为红色，而且相机姿态、视觉朝向和环境生成参数未由提供方说明。应先取得采集代码或标定相机/材质及 reset 协议，再开放在线评测；不能只将此开关改为 true 就认为已完成对齐。

# 平行 Meta-World VLA 实验流程

本目录提供独立的 Meta-World MT10 实验路径。它只复用 `models/` 中的模型实现，不读取或修改现有 RoboTwin 的数据集、训练入口和推理代码。

**当前 MT50 主线**：自采、每任务 50 条成功轨迹、原生 corner2、480×480 保存及 224×224 输入，完整命令和分别适配 native/YF 的 DataLoader 说明见 [SELF_COLLECTED_MT50.md](SELF_COLLECTED_MT50.md)。下文 MT10 配置继续保留用于旧实验。

C8（8步生成/4步执行）与 G2 全局意图的配对配置、训练/评测命令见 [GLOBAL_INTENT_MT50.md](GLOBAL_INTENT_MT50.md)。两组保持 future offset32，先不混入视觉层数或空间模块改动。

训练终端日志默认每轮在约 25%/50%/75%/100% 输出4条进度，加1条耗时和平均 loss 汇总。格式与 RoboTwin 接近：`Epoch [i/N] Step [j/M] Loss ... Total ... LR ...`，按启用分支显示 FutureFeat、Intent、TargetNorm、PredNorm、Gate。Loss 为当前 batch 动作损失，Total 为当前 batch 加权总损失，轮末 Avg Loss 为本轮总损失均值。可通过 `training.log_updates_per_epoch: 4` 调整进度条数；短轮/smoke test 会合并重复位置，初始化日志不计入每轮5条。

外部 MetaWorld_YF MT50（双路 256×256 PNG，当前选单路 corner 训练）的配置与完整命令见 [METAWORLD_YF_MT50.md](METAWORLD_YF_MT50.md)。该数据格式、方向和原采集器不同，已通过只读适配支持，无需转换原文件。

## 当前默认协议：corner2，480×480 采集 → 224×224 模型输入

基础配置 `configs/metaworld_mt10_b4_h512.yaml` 现在使用原生 `corner2`，不启用 `camera_profile`。`dataset.render_size: [480, 480]` 表示原始采集/评测渲染尺寸；`dataset.image_size: [224, 224]` 表示 DataLoader、VTT 和评测模型输入尺寸。采用 OpenCV `INTER_LINEAR` 直接缩放，不裁剪；DINOv3-B 的 patch grid 为 14×14，future-feature queries 自动为 196。

以下是该协议完整命令，均在项目根目录和 MetaWorldVLA 环境执行。已有的 workspace 相机配置及独立 YAML 保留；后文旧目录/旧尺寸示例仅用于对应历史实验，不要与本协议混用。

```bash
# 采集器默认宽高现为 480；显式写出便于复现。
python metaworld_vla/collect_expert.py \
  --output-dir data/metaworld_mt10_corner2_480 \
  --camera corner2 --width 480 --height 480 \
  --target-successes 100 --max-attempts 300 \
  --horizon 250 --success-only --stop-on-success --seed 0

python metaworld_vla/inspect_dataset_tk.py \
  --dataset-root data/metaworld_mt10_corner2_480

python metaworld_vla/calc_stats.py \
  --dataset-dir data/metaworld_mt10_corner2_480 \
  --output stats/metaworld_mt10_corner2_480_4d.json

CUDA_VISIBLE_DEVICES=0 python metaworld_vla/precompute_vtt.py \
  --dataset-dir data/metaworld_mt10_corner2_480 \
  --output-dir embeddings/metaworld_mt10_corner2_480_input224_vtt_dinov3b \
  --dino-path pretrained/dinov3-vitb16 --image-size 224 224 \
  --tasks reach-v3 push-v3 pick-place-v3 door-open-v3 drawer-open-v3 \
  drawer-close-v3 button-press-topdown-v3 peg-insert-side-v3 window-open-v3 box-close-v3

CUDA_VISIBLE_DEVICES=0 python metaworld_vla/train.py \
  --config metaworld_vla/configs/metaworld_mt10_b4_h512.yaml \
  --stats stats/metaworld_mt10_corner2_480_4d.json \
  --save-dir checkpoints/metaworld_corner2_480_input224_smoke --max-steps 1

CUDA_VISIBLE_DEVICES=0 python metaworld_vla/train.py \
  --config metaworld_vla/configs/metaworld_mt10_b4_h512.yaml \
  --stats stats/metaworld_mt10_corner2_480_4d.json \
  --save-dir checkpoints/metaworld_corner2_480_input224

CUDA_VISIBLE_DEVICES=0 python metaworld_vla/eval.py \
  --config metaworld_vla/configs/metaworld_mt10_b4_h512.yaml \
  --stats stats/metaworld_mt10_corner2_480_4d.json \
  --checkpoint checkpoints/metaworld_corner2_480_input224/checkpoint_epoch_50.pt \
  --test-num 50 --seed 0 --output eval_result/metaworld_corner2_480_input224_ep50.json
```

图像方向不因分辨率变化而改变：原生相机仍执行已有的 180°旋转，新 JPEG 写入方向标记，训练/VTT/viewer 不重复旋转。此处仅对齐相机和分辨率，不宣称与某篇论文的完整协议一致。重新采集后重新计算 stats/VTT，并开始独立训练；不要续训旧分辨率 checkpoint。旧 320×240 图像虽可被 resize 到 224×224，但不能视为原始 480×480 数据。

`precompute_vtt.py --image-size` 的默认值现在是 224 224。复现旧 workspace 实验时，必须显式传其 YAML 中的尺寸，例如 `--image-size 320 240`；VTT meta 尺寸与模型不一致会报错。复现旧采集尺寸时也应显式传 `--width 320 --height 240`。

## 1. 配置环境

Meta-World 需要较新的 Gymnasium 和 MuJoCo，建议从 RoboTwin 环境复制一个独立环境：

```bash
conda create -n MetaWorldVLA --clone RoboTwin
conda activate MetaWorldVLA
python -m pip install --upgrade "metaworld==3.1.1"
cd /home/lab347-no10/Tianyi/LiLa-WAM
python metaworld_vla/check_env.py
```

无显示器服务器请设置：

```bash
export MUJOCO_GL=egl
```

检查结果应包含：

```text
observation shape: (39,)
action shape: (4,)
rgb shape: (480, 480, 3)
```

Meta-World Sawyer 的 4 维动作是 `[dx, dy, dz, gripper]`，每一维通常位于 `[-1, 1]`。本实验将 `obs[:4]` 作为 4 维状态，也就是末端位置 xyz 和夹爪状态。

采集器默认使用 `corner2` 相机。当前链路统一把原始 RGB **旋转 180°（不是镜像翻转）**，公共实现是 `image_orientation.py`。新采集数据保存旋转后的 JPEG 和 `image_rotation_degrees=180` 属性；没有方向属性的旧 HDF5 按原始方向处理，读取时旋转。因此旧数据不需要重写，新数据不会重复旋转。DataLoader 的当前/未来帧、VTT、离线 Tk viewer、环境检查和评测模型输入保持相同方向，state/action 坐标不做变换。

图像方向改变后必须重新计算 VTT 并重新训练。当前配置使用独立的 `embeddings/metaworld_mt10_vtt_dinov3b_rot180`，不要覆盖旧条件或旧 checkpoint。训练/评测会拒绝没有新方向标记的条件向量和 checkpoint。仅改变图像方向时 state/action 统计量不变；重新采集或裁剪数据后应重新计算统计量。

官网提醒示例图片可能为展示而旋转：[Rendering 文档](https://metaworld.farama.org/rendering/rendering/)。使用相机名称比依赖不同版本的相机 ID 更明确。

## 2. 采集专家数据

MT10 使用以下任务：

```text
reach-v3, push-v3, pick-place-v3, door-open-v3, drawer-open-v3,
drawer-close-v3, button-press-topdown-v3, peg-insert-side-v3,
window-open-v3, box-close-v3
```

每个 episode 保存为 `data/metaworld_mt10/<task>/episode_XXXXXX.hdf5`，包含：

```text
observations/state       [T, 4]
observations/rgb_jpeg    [T]
actions                  [T, 4]
rewards                  [T]
success                  [T]
```

RGB 使用 JPEG 压缩保存，DataLoader 读取时会解码成 RGB。训练使用固定长度 32 步动作块；episode 尾部不足的动作步按 `[0, 0, 0, g_last]` 补齐，其中 `g_last` 是最后一个真实 action 的夹爪控制量。xyz 是位移增量，夹爪不是位移增量；重复夹爪控制量用于保持末尾开合指令，并不保证物理开度严格不变。padding 继续参与 loss，不使用 mask 排除。future image 则截取到最后一帧。

先采集小规模数据进行检查：

```bash
python metaworld_vla/collect_expert.py \
  --output-dir data/metaworld_mt10 \
  --episodes-per-task 10 \
  --tasks reach-v3 push-v3 pick-place-v3 door-open-v3 drawer-open-v3 \
  drawer-close-v3 button-press-topdown-v3 peg-insert-side-v3 \
  window-open-v3 box-close-v3
```

正式实验可以使用每个任务 100 个 episode。第一次采集建议不加 `--success-only --stop-on-success`，先检查专家成功率。归一化统计量：

```bash
python metaworld_vla/calc_stats.py \
  --dataset-dir data/metaworld_mt10 \
  --output stats/metaworld_mt10_4d.json
```

如果只保留成功轨迹，并在环境第一次报告成功后立即结束 episode，可以使用：

```bash
python metaworld_vla/collect_expert.py \
  --output-dir data/metaworld_mt10_success \
  --episodes-per-task 100 \
  --horizon 250 \
  --success-only \
  --stop-on-success
```

这里 `--horizon` 是最大帧数，不是固定帧数；`--stop-on-success` 会在成功帧结束，所以每条轨迹长度可以不同。若需要每个任务最终确实保存 100 条成功轨迹，使用带重试的模式：

具体停止条件是执行一次 action 后检查 `info["success"]`，首次为 true 时保存该动作及其执行前的 state/RGB，然后结束；不要求连续多帧成功，也不额外记录执行后的终止图像。`success[t]` 对应执行 `actions[t]` 后的结果。episode 也会在环境 terminated/truncated 或达到 horizon 时结束。

采集器现在在创建环境和 reset 时都使用 `seed + episode_id`。旧版本只改变 reset seed，而 Meta-World wrapper 在应用 reset seed 前就采样任务，导致重复新建的环境选中相同场景。同任务等长应检查是否轨迹重复；已有旧数据中已发现前两条 episode 的 state/action 和首帧 JPEG 完全重复。此修复不会改变旧文件，应在新目录重采集并重新生成统计量/VTT。

```bash
python metaworld_vla/collect_expert.py \
  --output-dir data/metaworld_mt10_success \
  --target-successes 100 \
  --max-attempts 200 \
  --horizon 250 \
  --success-only \
  --stop-on-success
```

`--target-successes` 按任务计数成功保存的 episode，失败尝试不会写入 HDF5；`--max-attempts` 防止成功率不足时无限重试。建议把裁剪后的数据写到新目录，不要覆盖原始数据。当前已有的 100 条 episode 的 `expert_success` 属性均为 true；viewer 中早期帧显示 `success=false` 只表示该帧尚未完成任务，不代表整条 episode 失败。

计算每个任务的冻结 DINO VTT 条件向量：

```bash
python metaworld_vla/precompute_vtt.py \
  --dataset-dir data/metaworld_mt10 \
  --output-dir embeddings/metaworld_mt10_vtt_dinov3b_rot180 \
  --dino-path pretrained/dinov3-vitb16 \
  --tasks reach-v3 push-v3 pick-place-v3 door-open-v3 drawer-open-v3 \
  drawer-close-v3 button-press-topdown-v3 peg-insert-side-v3 \
  window-open-v3 box-close-v3
```

## 3. 训练

配置文件是 `metaworld_vla/configs/metaworld_mt10_b4_h512.yaml`，采用 4 维状态、4 维动作、32 步动作块、每次执行 4 步、DINOv3-Base、hidden size 512 和 10 步 Flow Matching 推理。

开始训练：

```bash
CUDA_VISIBLE_DEVICES=0 python metaworld_vla/train.py \
  --config metaworld_vla/configs/metaworld_mt10_b4_h512.yaml \
  --stats stats/metaworld_mt10_4d.json \
  --save-dir checkpoints/metaworld_mt10_b4_h512_rot180
```

断点续训：

```bash
python metaworld_vla/train.py \
  --config metaworld_vla/configs/metaworld_mt10_b4_h512.yaml \
  --stats stats/metaworld_mt10_4d.json \
  --save-dir checkpoints/metaworld_mt10_b4_h512_rot180 \
  --resume checkpoints/metaworld_mt10_b4_h512_rot180/checkpoint_epoch_10.pt
```

一步 smoke test：

```bash
python metaworld_vla/train.py \
  --config metaworld_vla/configs/metaworld_mt10_b4_h512.yaml \
  --stats stats/metaworld_mt10_4d.json \
  --save-dir /tmp/metaworld_smoke \
  --max-steps 1
```

`--max-steps 1` 会验证一次真实 batch 的前向、反向和 checkpoint 保存；它不是完整训练。正式训练前可先将 `training.epochs` 临时设为 2～5，确认 loss 和显存占用，再运行 50 epoch。

## 4. 训练后测试

测试程序为每个任务创建独立 MT1 环境，并采用与 RoboTwin 相同的 receding-horizon 推理：生成 32 步动作，执行前 4 步，再重新读取图像和状态。

```bash
CUDA_VISIBLE_DEVICES=0 python metaworld_vla/eval.py \
  --config metaworld_vla/configs/metaworld_mt10_b4_h512.yaml \
  --stats stats/metaworld_mt10_4d.json \
  --checkpoint checkpoints/metaworld_mt10_b4_h512_rot180/checkpoint_epoch_50.pt \
  --test-num 50 \
  --seed 0 \
  --output eval_result/metaworld_mt10_ep50.json
```

只测试一个任务：

```bash
CUDA_VISIBLE_DEVICES=0 python metaworld_vla/eval.py \
  --config metaworld_vla/configs/metaworld_mt10_b4_h512.yaml \
  --stats stats/metaworld_mt10_4d.json \
  --checkpoint checkpoints/metaworld_mt10_b4_h512_rot180/checkpoint_epoch_50.pt \
  --tasks pick-place-v3 \
  --test-num 50 \
  --seed 0 \
  --output eval_result/pick_place_ep50.json
```

JSON 会记录每个任务的成功次数、测试数量和成功率。消融实验应固定任务列表、摄像头、图像尺寸、随机种子、episode 数量、测试步数、checkpoint 和 DINO 路径。

## 5. 离线查看 HDF5 数据

Meta-World 数据可以像真机数据一样离线查看。viewer 只读取 HDF5，不启动 MuJoCo、不连接机械臂，也不会执行动作。它显示 RGB 帧、帧滑块、播放控制、4 维 state、4 维 delta action、reward、success，以及整条 action 曲线。

查看指定 episode：

```bash
cd /home/lab347-no10/Tianyi/LiLa-WAM
conda activate MetaWorldVLA
python metaworld_vla/inspect_dataset_tk.py \
  data/metaworld_mt10/pick-place-v3/episode_000000.hdf5
```

不指定文件时打开数据目录中排序后的第一条 episode：

```bash
python metaworld_vla/inspect_dataset_tk.py \
  --dataset-root data/metaworld_mt10
```

如果服务器没有图形桌面，Tk viewer 不能直接显示窗口；可以把 HDF5 复制到有桌面的机器上查看，或继续使用离线抽帧脚本生成 contact sheet。Meta-World HDF5 中的图像统一保存在 `observations/rgb_jpeg`，不需要传 `--camera` 参数。

## 6. 数据维度和代码边界

Meta-World 原始 observation 常见为 39 维或 49 维。本实验只读取 `obs[:4]`，用于研究视觉信息与紧凑 proprioception 的组合。后续如果研究丰富状态输入，可以单独把 Meta-World DataLoader 和配置改成 39 维状态，不需要修改 RoboTwin 代码。

RoboTwin 使用 `train.py`、`dataloader/`、`robotwin_infer.py` 和 `configs/robotwin*.yaml`。Meta-World 使用本目录下的 `dataset.py`、`train.py`、`eval.py`、`collect_expert.py` 和 `configs/`。两条路径共享的只有 `models/model_runner.py` 和 `models/vla_model_fm.py`。


## 7. MuJoCo 自由视角窗口

这是真实的三维场景交互窗口，可旋转、平移、缩放观察，不受 `corner2` 固定相机限制。它采用 MuJoCo 官方 `viewer.launch_passive`，不需要从源码安装或修改 Meta-World：[MuJoCo viewer 文档](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer)。需要图形桌面或可用的显示转发；`egl` 离屏渲染本身不能提供桌面窗口。

只打开 reset 场景，不执行动作：

```bash
conda activate MetaWorldVLA
python metaworld_vla/view_scene.py --task reach-v3
```

在自由视角里观察脚本专家执行（不保存数据）：

```bash
python metaworld_vla/view_scene.py --task pick-place-v3 --expert
```

鼠标左键拖动旋转、右键拖动平移、滚轮缩放；窗口可以自由调整观察角度。

在原有评测命令末尾加 `--viewer` 即可同时观察模型执行：

```bash
CUDA_VISIBLE_DEVICES=0 python metaworld_vla/eval.py \
  --config metaworld_vla/configs/metaworld_mt10_b4_h512.yaml \
  --stats stats/metaworld_mt10_4d.json \
  --checkpoint checkpoints/metaworld_mt10_b4_h512_rot180/checkpoint_epoch_50.pt \
  --tasks reach-v3 --test-num 1 --seed 0 \
  --output eval_result/metaworld_reach_rot180.json --viewer
```

自由视角只控制观察窗口；策略仍读取配置相机的 RGB 并旋转 180°，拖动观察视角不会改变模型输入。带交互窗口的评测只用于观察，窗口交互可能影响仿真；正式成功率评测请使用不带 `--viewer` 的命令。关闭窗口会终止评测，不写入不完整成功率。

## 8. 项目专属操作面相机：交互调节与保存

`camera_config.py` 在运行时覆盖 world-attached 的 `corner2` 相机槽位，不修改 Meta-World 安装包。示例配置为 `configs/workspace_camera.json`，是桌前偏侧的斜俯视起点，并非所有任务的最终标定。基础训练 YAML 中只有注释示例，未自动启用新相机，以免误用于旧数据。

进入项目根目录和 MetaWorldVLA 环境，在有桌面的终端运行：

```bash
python metaworld_vla/tune_camera.py \
  --task pick-place-v3 \
  --profile metaworld_vla/configs/workspace_camera.json \
  --output metaworld_vla/configs/workspace_camera.json
```

- 左键拖动旋转，右键拖动平移观察中心，滚轮调整距离。
- 按 **S** 保存当前自由视角到 JSON，并导出同名 PNG，PNG 使用采集/评测的固定相机接口。
- 按 **P** 重新导出当前已配置固定相机的 PNG。没有保存前，PNG 并不代表鼠标刚调好的视角。
- 加 `--expert` 查看专家运动，空格暂停/继续。静态 reset、接触、终止姿态都需要检查遮挡。
- 加 `--fovy 60` 调整垂直视场角。大视场看得更广，但相同分辨率下物体像素更少。默认 45°。
- 加 `--fixed` 直接查看已加载的固定相机。这个模式不能按 S 导出自由视角；调节请重新运行不带 `--fixed` 的命令。

建议先旋转避开基座，再平移让工作区位于中心，最后缩放；不要只把镜头推近夹爪。至少用 reach、pick-place、door-open、peg-insert-side 和 box-close 检查不同工作区与高度，并使用不同 seed。预览窗口尺寸可能与采集宽高比不同，以导出的固定 RGB PNG 为准。

确定参数后采集到新目录：

```bash
python metaworld_vla/collect_expert.py \
  --output-dir data/metaworld_mt10_workspace \
  --camera-profile metaworld_vla/configs/workspace_camera.json \
  --target-successes 100 --max-attempts 300 \
  --horizon 250 --success-only --stop-on-success
```

新相机姿态本身正立，不再施加旧相机的额外 180°旋转；保存的是最终正立像素，继续使用方向标记 `image_rotation_degrees=180` 表示已转换为项目统一方向，读取/VTT/viewer 不重复旋转。HDF5 额外保存完整 `camera_profile_json`，方便追溯。

随后重新计算新目录的统计量/VTT，使用独立输出目录，并在新实验 YAML 中配置：

```yaml
dataset:
  dataset_dir: "/home/lab347-no10/Tianyi/LiLa-WAM/data/metaworld_mt10_workspace"
  camera_names: ["corner2"]  # 复用的底层槽位
  camera_profile: "/home/lab347-no10/Tianyi/LiLa-WAM/metaworld_vla/configs/workspace_camera.json"
  task_cond_dir: "/home/lab347-no10/Tianyi/LiLa-WAM/embeddings/metaworld_mt10_workspace_vtt"
```

DataLoader/VTT/viewer 直接读取保存的图像；评测根据 `dataset.camera_profile` 还原同一固定相机。用此配置训练新的 checkpoint。相机确定后不要修改该实验使用的 JSON，应为下一组视角另存文件。`check_env.py` 也支持 `--camera-profile`。

# RoboTwin YF10：easy、G2、R、G2仅回传与C/D-uniform实验

更新：2026-11-11。训练命令放在第2～4节，测试命令单独放在第5节。原计划R+G2改为“普通G2权重，仅测试开启回传”，不再要求训练R+G2；旧配置保留，不删除、不覆盖。新增RoboTwin C-uniform与D-uniform配对组，不能继承MetaWorld的checkpoint或缓存。

## 1. 固定协议

使用已有 `data/yf10` 十任务处理数据，统计 `artifacts/yf10/norm_stats.json`、VTT `embeddings/yf10_vtt_dinov3b`。本次扫描得到2441个episode，不能按清洗前标称数声称2500个。

动作14维、状态16维、单路head_camera、输入320×240、DINO-B、depth12/h512、chunk32/执行16步、10次FM调用，batch96，50轮scheduler，主比较epoch40。

| YAML（位于configs） | G2 | 训练source | 推理回传 |
|---|---|---|---|
| robotwin_yf10_easy_noffvtt.yaml | 关闭 | 标准高斯 | 关闭 |
| robotwin_yf10_easy_g2.yaml | task_attention、mean target、无gate | 标准高斯 | 关闭 |
| robotwin_yf10_easy_r.yaml | 关闭 | 50%cold、50%旧尾部+sigma0.5噪声 | 开启 |
| robotwin_yf10_easy_g2_reuse_eval.yaml | 同G2，同一份权重 | 不新增训练，使用普通G2 checkpoint | 仅测试开启 |
| robotwin_yf10_planwam_c_uniform.yaml | 因果task_attention结构，无mean-target辅助loss | 教师兼容动作初始化后，仅标准高斯FM | 关闭 |
| robotwin_yf10_planwam_d_uniform.yaml | 与C-uniform同结构，增加0.1 latent KD | 与C相同初始化和标准高斯FM | 关闭 |

上述版本统一关闭dense future-feature decoder、直接VTT token注入和B样条后处理；普通G2仍用VTT条件和t+32真实未来帧作mean-target辅助监督，C/D-uniform不使用这一固定辅助loss。D-uniform以同一冻结教师posterior输出作latent KD目标。与旧原始LiLa-WAM（future-head/VTT开启）不是同一baseline，论文需另外保留原版对照及直接VTT条件控制，避免把恢复任务信息的收益全部归给预测器。

R+G2原指“旧尾部source训练＋G2”；本轮G2仅回传指“普通高斯起点训练＋推理时回传”。后者不需要缓存、不需要第二次G2训练。C-uniform的uniform指所有学生参数统一学习率，不是uniform时间采样；time_sampler仍为logit_normal。

easy/R保留相同任务条件参数构造，但没有活跃的VTT条件通路；G2和C/D-uniform在意图通路使用VTT。不改旧RGB解码、状态拼接和padding监督。新配置用strict_loading=true防止损坏样本被随机替换后错配动作缓存。推理独立policy_seed456，原YAML没有该字段时不改变随机采样路径。

数据训练模式为both，即clean+randomized混合训练；在clean和randomized分别评测。这不是官方严格C2C/C2R训练协议，不能混称。若以后需要官方协议，另建仅clean训练组及对应统计和教师缓存。

### 平台脚本隔离（2026-11-11）

| 平台 | PlanWAM训练入口 | 初始化/优化辅助代码 |
| --- | --- | --- |
| RoboTwin | `train_planwam_robotwin.py` | `robotwin_planwam_transfer.py` |
| MetaWorld | `metaworld_vla/train_planwam.py` | `metaworld_vla/planwam_transfer.py` |

两个平台不再调用同一个训练脚本，也不互相导入训练入口或初始化辅助代码。RoboTwin未来教师、C-uniform、D-uniform都使用根目录新入口；MetaWorld原命令不变。各自校验dataset.type，误用另一平台YAML会在训练前报错。

本次只隔离训练调度、数据入口和迁移/优化辅助代码；公共模型定义、posterior与FM数学loss仍使用models目录现有实现，不能理解为整个项目完全无共享依赖。普通easy/G2/R继续使用原train.py，测评命令不变。

保留原RoboTwin的VERSION、输入合同、参数键、优化器参数顺序、scheduler和RNG保存字段，支持同协议的拆分前checkpoint。已有教师不需要重训；已有C/D完整epoch checkpoint可用新脚本--resume继续，但仍需同一教师文件和配置。正在运行的旧进程无需中途切换，停在已保存的完整epoch后再换入口；只换入口不应同时改学习率、batch或epoch规划。没有绕过原协议校验。

## 2. 训练命令一：easy_noffvtt对照

本节先准备环境，后续训练和测试都从项目根目录执行。训练只启动训练，不会自动进入评测。

```bash
cd /home/lab347-no10/Tianyi/LiLa-WAM
conda activate RoboTwin
set -euo pipefail
mkdir -p logs/robotwin_prior

CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/robotwin_yf10_easy_noffvtt.yaml \
  --norm_stats_path artifacts/yf10/norm_stats.json \
  --save_dir checkpoints_vla/robotwin_yf10_easy_noffvtt \
  2>&1 | tee logs/robotwin_prior/easy_noffvtt.log
```

从头训练，不使用旧Meta-World/旧future-head checkpoint初始化。训练输出带sft时间戳。

## 3. R训练准备：用easy epoch40生成缓存

```bash
C_RUN=$(find checkpoints_vla/robotwin_yf10_easy_noffvtt \
  -mindepth 1 -maxdepth 1 -type d -name 'sft_*' | sort | tail -1)
test -f "$C_RUN/checkpoint_epoch_40.pt" || exit 1

CUDA_VISIBLE_DEVICES=0 python metaworld_vla/precompute_action_prior.py \
  --config "$C_RUN/config.yaml" \
  --stats artifacts/yf10/norm_stats.json \
  --checkpoint "$C_RUN/checkpoint_epoch_40.pt" \
  --output-dir artifacts/action_prior/robotwin_yf10_easy_ep40_c32_e16 \
  --batch-size 32 --workers 4 --seed 321 \
  2>&1 | tee logs/robotwin_prior/cache.log
```

同一组存在多个run时务必核对C_RUN，以训练日志记录为准。缓存生成脚本根据dataset.type自动选RoboTwin原reader与14/16维模型，不做Meta-World的图像旋转，也不读取/复用Meta-World缓存。

每个训练锚点t>=16，使用同episode t-16时刻的图像/状态，生成32步动作并缓存后16步。无需读取未来图像生成先验，教师不使用G2。第一次中断后相同命令加--resume；不能中途更换batch、seed、数据或配置。

旧RoboTwin future-head开启的checkpoint不能严格加载到easy教师；应先完成上面的新easy训练。本轮该缓存仅用于R训练，不用于G2仅回传或C-uniform。缓存是额外离线成本，需报告。

## 4. 其他独立训练命令

### 4.1 普通G2训练

G2不依赖easy缓存，可提前训练。只训练这一次，随后分别测试关闭和开启回传。

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/robotwin_yf10_easy_g2.yaml \
  --norm_stats_path artifacts/yf10/norm_stats.json \
  --save_dir checkpoints_vla/robotwin_yf10_easy_g2 \
  2>&1 | tee logs/robotwin_prior/easy_g2.log
```

### 4.2 R训练

等待第3节缓存完整生成；不要修改旧配置或缓存manifest来绕过校验。本轮不再启动旧R+G2训练。

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/robotwin_yf10_easy_r.yaml \
  --norm_stats_path artifacts/yf10/norm_stats.json \
  --save_dir checkpoints_vla/robotwin_yf10_easy_r \
  2>&1 | tee logs/robotwin_prior/easy_r.log
```

新先验checkpoint保存source协议及混合采样RNG；--resume需要相同配置/缓存。中断续训沿用train.py原接口，恢复后会创建新的sft输出目录。Smoke用--max_train_steps 1，不能把部分epoch的checkpoint作为正式轮次结果。

### 4.3 C-uniform准备：新训练RoboTwin未来特权教师

```bash
CUDA_VISIBLE_DEVICES=0 python train_planwam_robotwin.py \
  --config configs/robotwin_yf10_planwam_teacher.yaml \
  --stats artifacts/yf10/norm_stats.json \
  --save-dir checkpoints_vla/robotwin_yf10_planwam_teacher_seed0 \
  2>&1 | tee logs/robotwin_prior/planwam_teacher_seed0.log
```

独立RoboTwin入口使用原HDF5 reader，不旋转图像，不改RGB/BGR、状态拼接或末尾重复padding。教师使用当前帧与t+32未来帧，15×20个DINO patch汇聚为一个未来特权latent，经意图投影送入DiT，以动作FM训练。不是R的.npy尾部缓存，也不能在在线测试中偷看未来。必须使用RoboTwin教师，不能复用MetaWorld教师或easy_noffvtt模型；已有拆分前训练的RoboTwin教师仍可使用。

### 4.4 C-uniform学生训练

```bash
TEACHER="checkpoints_vla/robotwin_yf10_planwam_teacher_seed0/checkpoint_epoch_40.pt"
test -f "$TEACHER"

CUDA_VISIBLE_DEVICES=0 python train_planwam_robotwin.py \
  --config configs/robotwin_yf10_planwam_c_uniform.yaml \
  --stats artifacts/yf10/norm_stats.json \
  --init-teacher-action "$TEACHER" \
  --save-dir checkpoints_vla/robotwin_yf10_planwam_c_uniform_seed0 \
  2>&1 | tee logs/robotwin_prior/planwam_c_uniform_seed0.log
```

复制教师兼容的视觉融合、意图读出、投影、DiT及动作头；因果intent_pool和intent_predictor重新初始化。不恢复教师优化器、scheduler或epoch。所有学生参数统一2e-4→5e-5 cosine，batch96、seed0，新的50轮训练，主结果仍取epoch40。不是从教师第40轮续至第50轮。

学生只看当前图像、状态和VTT；FM目标来自专家动作。无latent KD、无mean-intent loss、无R缓存、无推理回传，也不读取未来图像。部署只需学生，不需教师。不能用train.py或--resume代替本命令；train.py会明确拒绝PlanWAM配置，避免静默跑错目标。

教师和C-uniform输出固定目录，不自动加sft时间戳。重复训练请换新save-dir；中断恢复使用同协议--resume。保存action_init_report.json、planwam_protocol.json及源权重SHA256；输入、stats、VTT和教师规格不匹配会拒绝初始化。教师50轮＋学生50轮的总成本须单列。

C-uniform与普通G2同时改变了初始化及辅助loss，是探索组而非“仅改变初始化”的严格消融。后续若要归因，应补同结构随机初始化action-only控制。当前不要求额外训练该控制。

### 4.5 D-uniform学生训练：与C共用同一个RoboTwin教师

```bash
TEACHER="checkpoints_vla/robotwin_yf10_planwam_teacher_seed0/checkpoint_epoch_40.pt"
test -f "$TEACHER"

CUDA_VISIBLE_DEVICES=0 python train_planwam_robotwin.py \
  --config configs/robotwin_yf10_planwam_d_uniform.yaml \
  --stats artifacts/yf10/norm_stats.json \
  --init-teacher-action "$TEACHER" \
  --teacher-checkpoint "$TEACHER" \
  --save-dir checkpoints_vla/robotwin_yf10_planwam_d_uniform_seed0 \
  2>&1 | tee logs/robotwin_prior/planwam_d_uniform_seed0.log
```

不需要为D再训练一个教师，也不必等C训练结束；二者可以在同一教师epoch40就绪后分别运行。D不是从C的训练结果继续，也不要传--resume或C的checkpoint初始化。

两个teacher参数职责不同：--init-teacher-action只复制兼容动作权重，--teacher-checkpoint加载冻结posterior产生KD监督；必须指向同一份教师权重，代码会检查SHA256。

| 对比项 | C-uniform | D-uniform |
| --- | --- | --- |
| 学生结构/部署输入 | 因果G2，当前图像＋状态＋VTT | 完全相同 |
| 初始化 | 同一教师动作权重＋新因果预测器 | 完全相同 |
| 优化 | 统一LR 2e-4→5e-5，50轮scheduler、epoch40比较 | 完全相同 |
| loss | L_FM | L_FM＋0.1×MSE(LN(z_student), LN(z_teacher)) |
| 训练未来帧 | 不读取 | 仅供冻结教师产生标签，学生不看未来 |
| 测试未来帧/教师/回传 | 全部不需要/关闭 | 完全相同 |

LN是无可训练仿射参数的固定LayerNorm；teacher标签stop-gradient。D增加未来帧DINO前向和轻量posterior前向，但不再复制整套教师DiT做每batch动作前向，也不做教师10步动作采样。具体显存和速度需实测。日志KD为未乘0.1的原始蒸馏项，Total=Action+0.1×KD；TargetNorm可在LN后接近sqrt(768)，不是未来特征本身的norm。

C/D配置仅planwam.stage和planwam.lambda_kd不同，初始化、模型、数据及training字段完全相同。D对C的比较用于检验latent KD的增量作用；单训练seed差异仍不能直接当作稳定收益。D续训需继续提供--teacher-checkpoint并保留同一初始化教师文件。

## 5. 独立测试命令（十任务clean+randomized，统一epoch40）

新终端先执行第2节环境准备部分。下面每组均独立启动测试，不包含训练；同组有多个sft run时自动取最新目录，必须打印核对，复现时建议直接固定具体RUN。已有RUN_ID重测请加后缀，避免覆盖结果。

### 5.1 easy_noffvtt：关闭回传

```bash
RUN=$(find checkpoints_vla/robotwin_yf10_easy_noffvtt \
  -mindepth 1 -maxdepth 1 -type d -name 'sft_*' | sort | tail -1)
echo "$RUN"
test -f "$RUN/checkpoint_epoch_40.pt"
for CFG in demo_clean demo_randomized; do
  CUDA_VISIBLE_DEVICES=0 \
  CONFIG_PATH="$RUN/config.yaml" \
  CKPT_SETTING="${RUN#checkpoints_vla/}" CHECKPOINT_EP=40 \
  NORM_STATS_PATH=artifacts/yf10/norm_stats.json \
  TASK_CONFIG="$CFG" INSTRUCTION_TYPE=unseen \
  SEED=0 TEST_NUM=50 NUM_WORKERS=1 \
  RUN_ID="robotwin_easy_noffvtt_ep40_${CFG}_seed0_n50" \
  bash eval_yf10.sh
done
```

### 5.2 普通G2：关闭回传

```bash
RUN=$(find checkpoints_vla/robotwin_yf10_easy_g2 \
  -mindepth 1 -maxdepth 1 -type d -name 'sft_*' | sort | tail -1)
echo "$RUN"
test -f "$RUN/checkpoint_epoch_40.pt"
for CFG in demo_clean demo_randomized; do
  CUDA_VISIBLE_DEVICES=0 \
  CONFIG_PATH="$RUN/config.yaml" \
  CKPT_SETTING="${RUN#checkpoints_vla/}" CHECKPOINT_EP=40 \
  NORM_STATS_PATH=artifacts/yf10/norm_stats.json \
  TASK_CONFIG="$CFG" INSTRUCTION_TYPE=unseen \
  SEED=0 TEST_NUM=50 NUM_WORKERS=1 \
  RUN_ID="robotwin_easy_g2_cold_ep40_${CFG}_seed0_n50" \
  bash eval_yf10.sh
done
```

### 5.3 R：开启回传

```bash
RUN=$(find checkpoints_vla/robotwin_yf10_easy_r \
  -mindepth 1 -maxdepth 1 -type d -name 'sft_*' | sort | tail -1)
echo "$RUN"
test -f "$RUN/checkpoint_epoch_40.pt"
for CFG in demo_clean demo_randomized; do
  CUDA_VISIBLE_DEVICES=0 \
  CONFIG_PATH="$RUN/config.yaml" \
  CKPT_SETTING="${RUN#checkpoints_vla/}" CHECKPOINT_EP=40 \
  NORM_STATS_PATH=artifacts/yf10/norm_stats.json \
  TASK_CONFIG="$CFG" INSTRUCTION_TYPE=unseen \
  SEED=0 TEST_NUM=50 NUM_WORKERS=1 \
  RUN_ID="robotwin_easy_r_tail_ep40_${CFG}_seed0_n50" \
  bash eval_yf10.sh
done
```

### 5.4 G2仅回传：复用5.2完全相同的G2 checkpoint

```bash
RUN=$(find checkpoints_vla/robotwin_yf10_easy_g2 \
  -mindepth 1 -maxdepth 1 -type d -name 'sft_*' | sort | tail -1)
echo "$RUN"
test -f "$RUN/checkpoint_epoch_40.pt"
for CFG in demo_clean demo_randomized; do
  CUDA_VISIBLE_DEVICES=0 \
  CONFIG_PATH=configs/robotwin_yf10_easy_g2_reuse_eval.yaml \
  CKPT_SETTING="${RUN#checkpoints_vla/}" CHECKPOINT_EP=40 \
  NORM_STATS_PATH=artifacts/yf10/norm_stats.json \
  TASK_CONFIG="$CFG" INSTRUCTION_TYPE=unseen \
  SEED=0 TEST_NUM=50 NUM_WORKERS=1 \
  RUN_ID="robotwin_easy_g2_reuse_ep40_${CFG}_seed0_n50" \
  bash eval_yf10.sh
done
```

显式CONFIG_PATH优先于checkpoint旁的config.yaml。新增eval YAML与当前G2仅action_reuse.enabled一处不同，模型/数据/训练字段完全一致。若以后改了G2训练结构，必须同步核对eval副本，不可盲用。

生成32步、执行前16步后，下一次起点的前16步用旧的未执行尾部×1.0＋sigma0.5高斯，其余16步为纯高斯。首次生成或reset无旧计划时全部纯高斯；仍从t=0积分10步。不是直接混合最终执行动作，也不是R的训练source。

### 5.5 C-uniform：只测学生，关闭回传

```bash
RUN="checkpoints_vla/robotwin_yf10_planwam_c_uniform_seed0"
test -f "$RUN/checkpoint_epoch_40.pt"
for CFG in demo_clean demo_randomized; do
  CUDA_VISIBLE_DEVICES=0 \
  CONFIG_PATH="$RUN/config.yaml" \
  CKPT_SETTING="${RUN#checkpoints_vla/}" CHECKPOINT_EP=40 \
  NORM_STATS_PATH=artifacts/yf10/norm_stats.json \
  TASK_CONFIG="$CFG" INSTRUCTION_TYPE=unseen \
  SEED=0 TEST_NUM=50 NUM_WORKERS=1 \
  RUN_ID="robotwin_planwam_c_uniform_cold_ep40_${CFG}_seed0_n50" \
  bash eval_yf10.sh
done
```

不要把RUN换成planwam_teacher目录。因果学生checkpoint提供RoboTwin严格加载的model_state_dict，teacher不作为在线策略使用，推理入口会明确拒绝未来特权教师。

### 5.6 D-uniform：只测学生，关闭回传

```bash
RUN="checkpoints_vla/robotwin_yf10_planwam_d_uniform_seed0"
test -f "$RUN/checkpoint_epoch_40.pt"
for CFG in demo_clean demo_randomized; do
  CUDA_VISIBLE_DEVICES=0 \
  CONFIG_PATH="$RUN/config.yaml" \
  CKPT_SETTING="${RUN#checkpoints_vla/}" CHECKPOINT_EP=40 \
  NORM_STATS_PATH=artifacts/yf10/norm_stats.json \
  TASK_CONFIG="$CFG" INSTRUCTION_TYPE=unseen \
  SEED=0 TEST_NUM=50 NUM_WORKERS=1 \
  RUN_ID="robotwin_planwam_d_uniform_cold_ep40_${CFG}_seed0_n50" \
  bash eval_yf10.sh
done
```

测试只加载学生权重，不需要传--teacher-checkpoint，也不需未来图像；与C的测试预算、环境和动作积分步数一致。训练完成不代表已验证有性能提升，正式比较以C/D同协议结果为准。

### 5.7 D-uniform＋回传：复用5.6同一个epoch40学生checkpoint

2026-11-13补充，仅增加推理消融，不新增训练。先在项目根目录激活RoboTwin环境；若原训练保存目录不同，替换RUN为实际相对路径。

```bash
RUN="checkpoints_vla/robotwin_yf10_planwam_d_uniform_seed0"
test -f "$RUN/checkpoint_epoch_40.pt"
for CFG in demo_clean demo_randomized; do
  CUDA_VISIBLE_DEVICES=0 \
  CONFIG_PATH=configs/robotwin_yf10_planwam_d_uniform_reuse_eval.yaml \
  CKPT_SETTING="${RUN#checkpoints_vla/}" CHECKPOINT_EP=40 \
  NORM_STATS_PATH=artifacts/yf10/norm_stats.json \
  TASK_CONFIG="$CFG" INSTRUCTION_TYPE=unseen \
  SEED=0 TEST_NUM=50 NUM_WORKERS=1 \
  RUN_ID="robotwin_planwam_d_uniform_tail_ep40_${CFG}_seed0_n50" \
  bash eval_yf10.sh || break
done
```

新增eval YAML与当前D训练配置仅inference.action_reuse.enabled不同，模型、数据、training和planwam字段不变。显式CONFIG_PATH覆盖checkpoint旁config.yaml。未来若改过D结构或输入规格，应先与该run保存配置核对，不能盲用旧eval副本。

复用上次生成32步chunk中未执行的后16步，构成下次起点的前16步：旧尾部×1.0＋sigma0.5噪声；剩余16步为纯高斯，仍从t=0做10步积分。首次chunk/reset无旧计划时使用纯高斯。每次环境step仍只执行一条动作，不读取真实未来，也不需要R教师缓存。这是D-uniform＋推理回传，不是R＋D训练，更不是减少积分步数的加速实验。两次测试需使用相同checkpoint、环境seed及policy_seed456，结果目录使用不同标签；重跑同标签前请换新RUN_ID。

RoboTwin的回传由显式/保存YAML的inference.action_reuse.enabled控制，不使用MetaWorld命令行--action-reuse。每次环境step仍只返回一条动作并更新状态buffer。原始图像、执行动作不被加噪，加噪发生在归一化生成起点。

RoboTwin环境资源相对更重，先单worker。评测目录不要重复使用；可更改RUN_ID。输出在eval_result/yf10对应目录。现有桥接结果主要是成功率，不会自动生成Meta-World那套generation_metrics；本轮未声称已实现RoboTwin完整延迟/平滑度统计。

当前本地 `_eval_step_limit.yml` 仍使用此前修改的部分步数限制（例如adjust_bottle=267），并保留 `_eval_step_limit.before_two_thirds_20260910.yml`。本轮不改动环境文件。所有内部组必须使用同一限制；投稿比较官方数值时，应核对官方协议，并明确区分本地改动设置，不能直接将内部结果写作官方榜单复现。

所有新组统一不做B样条平滑，避免掩盖R的动作连续性作用；旧配置中的平滑设置仍然有效。后续比较完成步数、真实关节轨迹与单worker延迟时应独立记录。

## 6. 验证与未完成项

已用实际YF10样本核对图像[3,240,320]、状态[1,16]、动作[32,14]和任务条件[768]。CPU单元测试覆盖平坦索引、不跨episode、真实推理方法的32/16回传及reset；旧Meta-World source数学测试继续通过。

历史验证（2026-11-04）：未占用GPU，已在CPU用真实DINO-B与单个128帧episode完成原四组各一步训练/保存、完整单episode教师缓存、R+G2 checkpoint重载、真实观测格式的17次step（跨越32/16回传边界）与reset检查。原R+G2仅作历史记录，不属于本轮必跑矩阵。

2026-11-10新增RoboTwin PlanWAM CPU测试：真实HDF5 reader与原reader逐张量一致、末尾重复padding、14维动作教师反向传播、兼容白名单复制、新因果预测器梯度、统一LR、严格部署权重重载、完整小数据teacher→C-uniform训练入口及resume协议检查。测试使用临时小模型/替代视觉编码器，不把它的loss当作正式实验结果。原MetaWorld PlanWAM及初始化测试也继续通过；G2回传配置经检查仅修改一个推理开关。

另用实际YF10的一个anchor、真实DINO-B和完整depth12/h512模型，在CPU完成教师及无未来输入因果路径的前向/反向传播，loss均有限；这不是初始化训练后的闭环效果验证。

2026-11-11补充D-uniform验证：C/D配置仅stage和lambda_kd不同；D训练reader返回未来标签帧。小模型完整入口完成同教师动作初始化＋冻结posterior KD、统一LR优化、保存、恢复，并通过原RobotWinInference严格加载（Missing=0、Unexpected=0）且仅用当前观测生成有限14维动作。原教师文件保持不变；文档13段Bash命令通过语法检查，未执行正式训练/评测。

同日的平台拆分验证：RoboTwin隔离测试共6项、原MetaWorld PlanWAM 6项、初始化迁移7项、离线诊断8项通过。另将拆分前脚本保存在/tmp做对照：两平台各使用8-anchor临时数据、微型视觉替代器和动作网络训练teacher/C/D两个epoch；拆分前后checkpoint全部字段（模型、posterior、optimizer、scheduler、输入合同及RNG等）逐项完全一致。用旧C/D的epoch1 checkpoint在新入口恢复到epoch2，也与原连续训练完全一致。这是CPU小数据的确定性回归，不代表完整GPU长训练的逐位保证或成功率验证。

所有开发临时数据位于/tmp；没有启动全量缓存、完整训练或SAPIEN仿真。正式GPU性能、成功率及完整50轮仍待运行上述命令。

## 7. 投稿定位：不能用分区代替创新判断

当前R与 [WarmPrior](https://arxiv.org/abs/2605.13959) 的时间先验source高度相关，[A2A](https://arxiv.org/abs/2602.07322) 也已研究历史动作驱动起点。不能将旧尾部+高斯噪声直接宣称首次提出。

G2与未来latent预测/控制条件化的已有工作相关：[LiLa-WAM](https://arxiv.org/abs/2608.03701) 已联合未来表征与动作学习；[StageWAM](https://arxiv.org/abs/2608.10780) 研究预测未来阶段latent以辅助控制。它们与当前G2具体结构不完全相同，但表明“预测未来特征帮助动作”本身不是足够的新贡献。

这些是本次核查的近邻，不等于全面证明有/无新颖性。就当前简单模块和单训练seed结果，方法创新仍偏增量，直接包装为两个独立新机制存在较大重合风险；R+G2目前也未出现叠加成功率收益，不应只靠组合命名增加贡献。

工程应用类论文可能以明确场景、可靠系统方案和充分实验形成贡献，但不能承诺中科院三区/四区录用。分区取决于年份和学科，也不意味着可忽略相关工作。先完成RoboTwin与Meta-World的配对验证、原始LiLa与直接VTT控制、WarmPrior等近邻对照，多训练seed、完成步数/真实轨迹平滑度、单模型延迟和包含教师缓存的成本报告。若真机可用，再用相同实际应用任务验证收益和失败恢复。

若只得到小幅成功率改善，优先聚焦已验证的连续性与控制质量问题；若设计新的可靠性/交互机制，需再次进行针对性的文献排重，不因加一个gate就声称解决创新性。

本模型当前使用视觉、状态与任务VTT，不直接处理自然语言。评测参数instruction_type=unseen沿用平台接口，不足以证明模型具备未见语言指令泛化，投稿时应准确说明任务条件来源。

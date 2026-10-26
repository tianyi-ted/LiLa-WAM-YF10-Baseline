# G2 单 query 条件化汇聚实验

日期：2026-10-22。此实验只改变全局意图预测器的输入汇聚，不改变未来监督目标。

## 已实现与兼容性

- `models/vla_model_fm.py`：增加 `TaskConditionedIntentPool`；`predict_global_intent` 按 `intent_pooling` 选择汇聚路径。
- `models/model_runner.py`：读取 `model.global_intent.pooling`，缺省为 `mean`。
- 原始 G2 配置 `configs/yf10_dinov3b_h512_g2_c32_ft40.yaml` 保持不变；旧模式不创建 attention 池化参数，旧 checkpoint 的参数名保持兼容。
- 新配置 `configs/yf10_dinov3b_h512_g2_taskattn_c32_scratch50.yaml` 启用 `pooling: task_attention`、256维汇聚、4 heads。
- 无全局意图的 baseline 不创建新模块。

## 数据通路

```text
VTT + 最新机器人状态 → LayerNorm/Linear/GELU → 单个 query
当前 DINO patch → LayerNorm/Linear → values
values + 固定二维正弦位置编码 → keys
query/keys/values → 一层 cross-attention → 投影回 DINO 维度 → fixed LayerNorm
    → 原有预测器（仍结合状态、VTT、跨度）→ 未来变化向量
    → 原有 gate 和意图 token 注入路径 → 动作 DiT
```

未来目标仍是 `LN(mean(F_future)) - LN(mean(F_current))`，并停止梯度。
未来图像不进入池化器/query；训练与推理共用 `predict_global_intent`。
没有新增未来编码器、EMA、空间残差、region gate 或可学习目标分支。

这是输入汇聚消融，不是任务条件化 target 实验。如果结果无提升，不能直接断言局部目标预测无效。
`G2_pro.md` 中提出的 CLS、mean/std、变化统计是另一组目标表示消融，本次暂不混入。

## 训练与比较协议

沿用旧 G2：DINO-B、h512、depth12、batch96、seed0、LR2e-4、50 epoch cosine scheduler、future offset32、gate_init0.1；主比较取 epoch40。
不要将 epochs 改为40，否则学习率轨迹与旧 G2 不同。
从头训练，不使用 `--init_from` 或 `--resume`。

```bash
cd /home/lab347-no10/Tianyi/LiLa-WAM
mkdir -p log/g2_taskattn_scratch50
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/yf10_dinov3b_h512_g2_taskattn_c32_scratch50.yaml \
  --norm_stats_path artifacts/yf10/norm_stats.json \
  --save_dir checkpoints_vla/yf10_g2_taskattn_c32_scratch50 \
  2>&1 | tee log/g2_taskattn_scratch50/train.log
```

训练后将 `CKPT_SETTING` 显式设置为本次实际 `yf10_g2_taskattn_c32_scratch50/sft_...`，使用新 YAML、epoch40，在 clean/randomized 上按相同协议测试；使用独立 RUN_ID。
不要用新 attention YAML 加载旧 mean checkpoint，两者新增参数不同。

## 验证范围与风险

`PYTHONPATH=. python utils/test_g2_task_pool.py` 验证缺省/显式 mean 的 strict 参数加载和数值一致、目标不变且 detached、新分支梯度及优化器更新。
目前已通过 CPU 模块测试和 Python 语法检查，未启动完整 DINO GPU 训练，未完成端到端训练/仿真评测；速度和显存需实际测量。
attention 不保证自动聚焦目标，应结合后续注意力可视化、多 seed 和成功率判断，不能只依据 Intent loss。

## 后备改进路径（计划，尚未实现）

以下是当前实验之后的候选方向，不代表已经支持的 YAML 选项。除 `mean` 和 `task_attention` 外，目前代码不能直接通过配置切换这些方案；选择后需单独实现、验证和创建配置。文中“VTT query”指当前 **VTT＋机器人状态生成 query** 的方案，并非直接用原始 VTT 点积。

### 先区分两种改动

- **输入汇聚改动**：改变预测器读取当前图像的方式，未来目标仍固定为平均池化变化。用于回答“预测器能否从当前观测提取更有效的信息”。
- **目标表示改动**：改变要预测的未来变化定义。用于回答“全局平均变化是不是适合动作学习的监督信号”。

不能在同一次实验中同时改变汇聚、目标、gate 和注入方式，否则难以归因。某种目标的 Intent loss 更低，也不意味着动作成功率更高；不同目标之间的 loss 数值通常不能直接比较。

### A. CLS / register 表示：低开销目标对照

**做法**：用冻结 DINO 的 CLS 表示替代平均 patch 表示，目标为 `LN(CLS_future) - LN(CLS_current)`。register 可作为独立对照，例如预先规定对多个 register 求均值，不与 CLS 混用。

第一组严格实验可以保持当前输入汇聚不变，仅替换目标；“输入和目标均改为 CLS”应作为另一组实验记录。维度相同时保留原预测器输出宽度和1个意图 token。

**价值**：检验全局语义表示是否比平均 patch 更适合作为未来监督。DINO 已返回相关 token 时无需额外编码器前向，但仍需检查当前特征切片接口，不能误把 patch 当作 CLS/register。

**风险**：CLS 不一定对接触和精确位置敏感；register 不保证对应物体或交互区域。目标变化范数大、与成功相关，都不直接证明其作为训练目标更有效。

### B. Mean＋Std：增加特征分布统计

对每个通道，沿空间 patch 维度计算：

```text
u(F) = concat(mean_patches(F), std_patches(F, correction=0))
target = LN(u(F_future)) - LN(u(F_current))
```

**做法**：作为目标表示消融，先保留输入路径，预测输出从 D 维改为2D维，再映射为1个意图 token。若随后输入也使用 mean/std，另记为第二项改动。需要记录参数量和 loss 尺度变化，不能称为严格等参数实验。

**价值**：均值之外补充通道特征的离散程度，统计操作开销较小，但预测器和 token 投影参数会增加。

**风险**：空间标准差不是“变化集中度”的直接测量，也不能定位目标。对 patch 做任意排列，mean/std 都不变；因此不能据此声称恢复了空间布局或识别了接触事件。

### C. Patch 变化统计：增加运动幅度辅助监督

仅在目标侧计算：

```text
delta_i = F_future[i] - F_current[i]
r_i = norm(delta_i)
signature = [mean(r), fixed_topk_mean(r), fixed_quantiles(r)]
```

**做法**：从当前观测预测低维 signature。优先作为独立辅助 head 评估，不直接替换原意图 token；加入该信号作为条件是后续独立实验。top-k比例、分位数、归一化和 loss 权重要事先固定。

**价值**：提供变化幅度分布信息，区分少数区域的大变化与大量区域的小变化；目标侧不增加 DINO 前向。

**风险**：相同 patch 下标是图像坐标对应，不是物体跟踪；相机运动、光照和遮挡也会造成变化。未来差值绝不能作为推理输入。该路径偏辅助监督，与“改进汇聚”应分开比较。

### D. 固定 VTT 点积池化：移除可学习选择器的对照

**做法**：仅当 VTT 与 patch 在维度、特征层和表示空间上兼容时，使用归一化向量计算权重 `alpha = softmax(sim(VTT, patch) / temperature)`；固定 temperature，不训练 query/K/V 投影。第一步只替换当前输入池化，仍保持平均变化目标。

**价值**：排查学习式池化器的优化问题，参数和计算增量小。

**风险**：维度相同不代表语义相似度有效。当前 VTT 是任务相关视觉表示，不可直接当成文本语义 query；固定点积也不保证抑制背景或优于平均池化。随机冻结 query 同样不是天然有效的选择器。

### E. 当前帧注意力加权的未来目标：对齐监督区域

在输入注意力路径验证后，再单独测试：由当前观测得到 `alpha_t`，用同一组空间权重汇聚当前和未来特征：

```text
target = LN(sum_i alpha_t[i] * F_future[i])
       - LN(sum_i alpha_t[i] * F_current[i])
```

**实现约束**：当前模块使用投影后的 values 和 output projection，不能直接将其输出等同于上述加权原始特征；需要明确输出注意力权重、多个 head 的汇聚规则，并单独实现目标构造。目标整体停止梯度，未来特征不参与当前 query 的计算。

**价值**：使监督更接近当前关注区域，而不是整幅图的平均变化。

**风险**：固定图像区域不等于跟踪运动物体；注意力可能选择容易预测的静态背景。stop-gradient 只阻断一次反向路径，不保证跨训练步目标稳定或不会坍缩。EMA可作为稳定化对照，但也不是防坍缩的充分条件；不要同时新增可学习目标投影和修改注入结构。

### F. 任务无关注意力 / 少量交互 query

**先做单 query 控制**：将任务/状态 query 改为一个学习的固定 query，其余保持一致，用于区分“注意力汇聚本身”与“任务/状态条件化”的作用。若要进一步区分任务和状态，可分别做仅任务、仅状态 query 对照。

**再考虑2～4个条件化 query**：提取多个交互表示，尝试覆盖目标、夹爪与放置区域；这些角色是研究假设，不应把无监督学出的 query 自动命名为对应物体。

**风险**：query 可能同质化，且多个 token 增加了条件容量。若复制同一个全局目标，不能证明获得不同空间语义；应先明确目标分配/监督，再决定是否添加多样性约束，并与等 token 数量控制组比较。与 S2 的区别在于自适应汇聚，而不是固定2×4区域。

## 推荐决策顺序

1. **完成当前对照**：先比较修复 FP32 gate 的 mean-G2 与本次单 query-G2；保持数据、50轮scheduler、epoch40评测和gate设置一致。保留不含全局意图的原始baseline作参照；C32属于额外4轮微调参照，不能与scratch40混称同预算baseline。
2. **先诊断，不急着新增50轮训练**：核对注意力是否近乎均匀、是否过度集中背景、任务/状态变化是否会改变汇聚结果；检查目标/预测范数、gate、相同噪声下开启/关闭意图条件的动作差异。先确认训练和推理实际使用了新分支。相关可视化和诊断工具尚需实现。
3. **如果attention没有学到有意义的选择**：优先做固定query或任务/状态query消融；固定VTT点积仅在表示兼容时作为低成本备选，不先堆叠更多query。
4. **如果注意力有选择性，但动作没有收益**：优先做 CLS 与 mean/std 的固定目标表示对照，或检查辅助梯度与动作梯度是否冲突。当前输入attention仍预测全局mean目标，可能存在输入选择与监督不匹配，不能直接认定attention无效。
5. **如果目标表示或输入选择已有正向信号**：再评估当前帧权重共享的未来目标；最后扩展到少量交互query，不直接混入空间残差和region gate。
6. **停止条件**：如果多次可比实验均无稳定收益，暂停全局未来意图作为主线，不继续盲调gate。转向任务条件化局部交互表示，或先研究空间辅助监督的共享梯度路径；OPD不自动恢复为近期路线。

短微调可以用于排查数值、梯度和开销，不应仅凭4轮结果淘汰需要从头共同学习的结构。先使用固定训练seed进行开发比较，候选方案出现收益后再追加评测seed与训练seed，分别记录两类随机性；最终不能只报告最好seed。

## 每项实验应记录

- 代码版本、配置副本、训练seed、checkpoint路径、参数量、峰值显存及每epoch耗时。
- 输入汇聚、目标表示、目标维度/归一化、目标梯度策略、意图token数量和gate定义。
- 原动作loss、未来特征loss、意图loss及权重、目标/预测范数；注意力方案增加空间权重可视化。
- Clean与Randomized每任务成功数和总次数、评测seed、步数限制、推理去噪步数、动作执行跨度和配置实际加载路径。
- 明确区分已完成实现、模块smoke test、完整训练和仿真评测；预测loss下降不能替代任务成功率证据。

## Meta-World MT50 迁移配置（2026-10-24）

保留均值G2：`metaworld_vla/configs/metaworld_mt50_corner2_c8_global_intent.yaml`。
新增单query：`metaworld_vla/configs/metaworld_mt50_corner2_c8_g2_taskattn.yaml`。

二者都为4D state/action、预测8步、执行4步、未来跨度8、corner2原生相机、渲染480×480、输入224×224、180度图像旋转、DINO-B、h512、50轮规划。原G2注释H=32有误，已改为H=8，数值未修改。已有 `metaworld_mt50_corner2_base.yaml` 是C32/H32，不能当成只差意图模块的严格C8对照；当前 `metaworld_mt50_corner2_tiny.yaml` 的实际值为C8/E4/H8，其头部注释也已过时，使用时须以数值和运行配置快照为准。

已验证配置只差池化选项、Meta-World训练器构造相同loss配置、50任务数据与768维VTT存在、224输入对应196个patch、单query模块真实维度CPU前后向通过。训练/推理均通过ModelFactory和同一个predict_global_intent创建/执行分支，且都调用keep_control_gates_fp32。未启动完整训练或仿真评测。

```bash
cd /home/lab347-no10/Tianyi/LiLa-WAM
conda activate MetaWorldVLA
mkdir -p log/metaworld_g2_taskattn_c8
CUDA_VISIBLE_DEVICES=0 python metaworld_vla/train.py \
  --config metaworld_vla/configs/metaworld_mt50_corner2_c8_g2_taskattn.yaml \
  --stats stats/metaworld_mt50_corner2_480_50ep_4d.json \
  --save-dir checkpoints/metaworld_mt50_corner2_c8_g2_taskattn_seed0 \
  2>&1 | tee log/metaworld_g2_taskattn_c8/train.log
```

从头训练，不加init-from/resume，目录应为本次专用空目录。Meta-World训练器直接向save-dir保存，不自动追加sft时间戳。训练50轮规划，预先固定epoch40为主比较点；原mean-G2也应采用相同checkpoint选择规则。

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python metaworld_vla/eval.py \
  --config metaworld_vla/configs/metaworld_mt50_corner2_c8_g2_taskattn.yaml \
  --stats stats/metaworld_mt50_corner2_480_50ep_4d.json \
  --checkpoint checkpoints/metaworld_mt50_corner2_c8_g2_taskattn_seed0/checkpoint_epoch_40.pt \
  --test-num 50 --horizon 500 --seed 0 \
  --output eval_result/metaworld_mt50_c8_g2_taskattn_ep40_seed0_n50.json
```

该命令评测配置中的全部50任务，不使用RoboTwin的demo_clean/demo_randomized；评测环境及goal采样协议应与已有Meta-World基线一致。新attention参数需要独立训练，不能用旧mean checkpoint直接评测新结构。

# Meta-World 低资源 SFT → Residual RL 实验路线

更新日期：2026-10-22。本文是实验设计与接口审计，不表示在线 RL 已实现或验证通过。与 RoboTwin 路线并行保存，不修改原有模型代码、训练配置或 checkpoint。

## 1. 结论与范围

**Meta-World 更适合当前阶段验证低成本 RL 接口和局部动作修正，但不会自动降低视觉 VLA 的推理成本，也不保证 RL 一定提升成功率。**

相比本地 RoboTwin 的自定义任务接口，它提供 Gymnasium reset/step、逐步奖励和 success 信号，4D 连续动作也更易做 residual RL。状态策略可以不渲染 RGB，因此适合先验证奖励、采样、GAE 和优化器。但视觉策略仍须渲染图像、运行 DINO 与 Flow Matching；环境简单不代表这部分免费。

Meta-World 的结论不能直接代替 RoboTwin 双臂、随机背景或真机结论。推荐定位为“方法验证平台”，有收益后再回到 RoboTwin 验证迁移性，而不是同时扩展两套 RL 系统。

## 2. 已核对的本地资源

以下是本次读取现有文件得到的事实；没有启动采集、训练或完整在线评测。

| 项目 | 当前状态 / 路径 |
|---|---|
| 独立环境 | `MetaWorldVLA`，已查询安装版本 |
| 软件版本 | metaworld 3.1.1、gymnasium 1.4.0、mujoco 3.3.0、torch 2.4.1 |
| 自采方案 | `metaworld_vla/SELF_COLLECTED_MT50.md` |
| 主线配置 | `metaworld_vla/configs/metaworld_mt50_corner2_480_50ep_b4_h512.yaml` |
| 数据路径 | `data/metaworld_mt50_corner2` |
| 数据规模 | 现有自采说明记录为 50 任务、2500 条成功轨迹、190451 帧；本轮未重新扫描 |
| 统计量 | `stats/metaworld_mt50_corner2_480_50ep_4d.json` |
| VTT 配置路径 | `embeddings/metaworld_mt50_corner2_480_50ep_input224_vtt_dinov3b`；完整性应在正式训练前核验 |
| SFT 训练 / 评测入口 | `metaworld_vla/train.py` / `metaworld_vla/eval.py` |
| 网络 | DINOv3-B，Action Expert depth12/h512，future-feature 辅助监督 |
| 动作协议 | state=4、action=4，生成32、执行前4步、未来偏移32 |
| 视觉协议 | 原生 corner2、480×480 渲染、共享180°方向处理、224×224模型输入 |

外部 `data/metaworld_yf_mt50` 已在现有文档中归档为“在线视觉未对齐”，不应作为首轮 RL 的 SFT 数据。优先使用同一套本地仿真器采集的数据，避免将相机/材质差异误判为 RL 方法失败。

**不能直接加载 RoboTwin epoch40 checkpoint 到 Meta-World。** RoboTwin 是14D动作/16D状态，Meta-World 是4D末端增量及夹爪控制；两者语义和形状不同。可以复用架构和冻结 DINO 权重，但必须建立 Meta-World 自己的 SFT baseline、stats 和 VTT。

## 3. 环境接口和评价协议

官方当前接口：

```python
obs, info = env.reset(seed=seed)
next_obs, reward, terminated, truncated, info = env.step(action)
```

动作空间为 `Box(-1, 1, (4,))`，对应末端 dx/dy/dz 控制及夹爪。这里的数值是环境控制量，不能把 residual=0.05 直接解读为移动5厘米；实际位移由环境缩放和动力学决定。

区分三个概念：

- `terminated`：环境终止条件；
- `truncated`：时间上限等外部截断，通常仍应 bootstrap 最后状态价值；
- `info['success']`：任务是否完成，不必天然等于 terminated。

GAE 应分别处理“能否bootstrap”和“是否跨episode递推”：超时可bootstrap，但不能把下一条新episode的优势串进上一条。单个 done mask 不足以覆盖所有情况。

最终评价看成功率，不是累计 reward。官方 MT 协议覆盖每任务训练目标集合中的50个目标，每个目标一条episode，期间任意一步 success=1即计成功；平均所有任务/episode。ML1/ML10/ML45 是另外的元学习设置，不是普通 SFT+RL 的同义词。

本地 `eval.py` 已使用 episode 内 success 的 OR 聚合，但它循环 `reset(seed=seed+ep)`，没有显式遍历官方50个目标，也未固定 PyTorch flow噪声种子。因此它目前是本地筛选评测，不应未经核验称为严格官方 MT50 分数。正式论文需增加独立、冻结的目标列表与policy RNG协议，不直接改动旧评测结果。

## 4. 推荐网络与数据通路

```text
当前 RGB + obs[:4] + task VTT
              │
       冻结的 Meta-World SFT
       DINO → Adapter → Flow DiT
              │
       chunk[32,4] → 原队列执行前4步
              │
       当前 base_action[4] ─────────────┐
                                     │
RGB特征 + proprio + task + base_action │
          + chunk phase/剩余动作       │
              │                      │
       轻量 residual actor           │
              │                      │
            delta[4] ──────加法/限幅──┘
                       │
              env.step(final_action)
                       │
        next_obs/reward/terminated/truncated/success
                       │
             CPU rollout buffer → PPO/GAE
```

SFT本身不更新。PPO只优化显式的 residual 随机分布，因此**不需要计算原始 Flow Matching 的动作似然**。这是与完整 flow-policy RL 的关键区别。

由于SFT有动作缓存，residual策略只看obs[:4]通常信息不足：它看不到目标物体，也不知道当前base action和chunk阶段。计划至少输入proprio、base action、task条件、phase；视觉版本再输入冻结DINO的紧凑特征。旧原型的“proprio-only”只适合接口测试，不能作为空间定位改进的充分方案。

低资源首版保持SFT现有4步队列语义，每个env.step更新residual。DINO特征可在SFT重规划时计算并缓存；若这样做，必须显式记录“视觉每4步刷新”，而不能声称residual使用每一步新图像。每步刷新视觉应作为额外开销实验，不能暗中修改基线刷新频率。

## 5. 当前 rl/ 原型的限制：训练前必须处理

本次只记录问题，不修改旧原型：

1. `ResidualVLAController` 解析RoboTwin endpose字典，不能直接包装Meta-World数组观测。
2. 原型 `act()` 将Normal样本clamp后再求Normal log_prob，边界处不是正确的截断分布概率。应选定可追踪的潜变量：采样u、保存u，执行delta=limit*tanh(u)，更新时计算同一u的概率；若报告变换后delta密度，则加入tanh Jacobian。不要用clamp后的值冒充原Normal样本。
3. actor均值零初始化只保证**确定性**零残差；随机采样仍有探索噪声。零残差兼容测试必须关闭采样噪声。
4. 最终动作可能还需clip到[-1,1]。保存原始潜变量、delta、未clip和实际动作、clip比例，避免PPO更新时错误重算概率。记录base action及队列状态，保证同一训练样本的条件不变。
5. 旧控制器实际在反归一化后的环境动作上加残差，不是在SFT统计归一化空间；文档/变量必须区分。Meta-World首版直接用[-1,1]环境动作单位更清楚。
6. 现有PPO/GAE函数是最小原型；还需补全terminated/truncated处理、value bootstrap、任务条件batch、优化循环、日志及checkpoint。不能把配置中出现 `ppo_epochs` 当成训练已实现。
7. 自定义reward实现和环境清理也需要单测；首轮Meta-World直接使用官方环境返回的reward，不复用RoboTwin reward adapter。

## 6. 分阶段实验与通过标准

### M0：接口和资源测量，无RL训练

固定metaworld版本、相机、reward_function_version、max_episode_steps、成功终止策略。先使用1个环境，完成reset/step/render/close测试。

分别测状态模式、RGB模式、SFT模式的steps/s、CPU RAM、nvidia-smi进程显存和PyTorch peak allocated/reserved。MuJoCo/EGL显存不一定被PyTorch统计覆盖。不要把此前估计的5～9GB视为实测保证。

### M1：状态RL控制组，可选但推荐

先用小型MLP SAC（推荐样本复用）或PPO在reach-v3验证环境与奖励，之后push-v3。状态RL不需DINO或渲染，不是VLA论文主结果。若使用完整环境state（含物体/目标信息），必须标为state oracle，不能与RGB+proprio策略混为公平比较。

建议预算先100k环境步检查是否学到有意义行为，最多300k步再诊断；这是预设预算，不是保证收敛步数。M1与后续residual PPO使用不同算法仅作工具验证，不据此归因网络优劣。

### M2：建立 Meta-World 视觉SFT baseline

沿用自采数据协议完成VTT、SFT训练和单任务在线筛选。先用reach-v3作接口测试，再选push-v3、drawer-open-v3、pick-place-v3等有改进空间的任务。若某任务baseline已经接近100%，不适合仅用它判断RL是否有效。

先检查SFT在同仿真器、同相机下能正常工作；若普遍接近0，优先排查state/action时序、图像方向、动作增量、VTT，不急于用RL“修复”严重接口错误。固定一个SFT checkpoint供所有分支启动。

### M3：零残差与rollout一致性

同一reset目标、相同SFT随机数/flow噪声下，原SFT与residual关闭/确定性零残差逐步比较动作。不要让两个模型依次消费不同的随机数后比较。

通过标准：动作一致、队列清理一致、无跨episode状态泄漏；reward与原生env一致，超时能正确bootstrap；SFT参数hash前后不变；新RL输出写到独立目录。

### M4：单任务 residual PPO，主线最小实验

计划起点：1个环境，rollout=1024，minibatch=64（OOM降16），PPO epochs=4，lr=1e-4，gamma=.99，lambda=.95，clip=.2，grad norm=.5，MLP hidden128。rollout长度与SFT batch96无关，缓存应detach并存CPU，不能保留1024步DINO计算图。

先给10k环境步做冒烟/成本检查，再做100k步筛选；训练中每10k步在固定开发目标集评估。首版使用环境dense reward，固定reward版本和归一化规则；不同时叠加人为几何奖励。若dense reward升高而成功率下降，应检查reward hacking而非宣布提升。

先只修正xyz，夹爪沿用SFT；把夹爪残差作为后续消融，避免先破坏开闭时机。residual幅度从0.05（环境动作单位上界）起步，可预注册0.02/0.05/0.10小扫。它是局部改动先验，不保证数值越大越好。

用mean-delta平方正则限制修正幅度；这不是专家BC，也不等价于对flow policy的KL。若加入专家BC或KL，必须明确监督对象和概率定义。

### M5：公平对照和扩展

最少比较：冻结SFT、确定性零residual、同等噪声但不学习、residual PPO。可补充“继续SFT同等墙钟时间”控制额外训练收益。统一相机、目标集合、horizon、chunk/执行长度、成功判定与评测flow随机数。

筛选用固定seed和10 episodes仅看趋势；最终至少3个RL训练seed，在相同50目标集评估，报告各任务成功数/50、任务平均、跨训练seed均值±标准差、环境交互总步数、训练时长和峰值显存。50条随机reset不自动等于官方50目标遍历。开发目标用于选checkpoint；最终报告目标只评估，不用其分数反复调参。此处是在线评估隔离，不要求重新拆分已有SFT示范数据。

单任务稳定后扩到3～5任务，再到固定自定义10任务或官方MT10。任意10任务需写“custom-10”，不能冒充官方MT10。暂不同时做MT50、解冻DiT、改chunk及RL，避免无法定位收益来源。

## 7. 当前能执行的命令

以下入口已存在。不会自动安装包或启动RL；不要反复采集覆盖现有数据。

```bash
conda activate MetaWorldVLA
cd /home/lab347-no10/Tianyi/LiLa-WAM
export MUJOCO_GL=egl

python metaworld_vla/check_env.py --task reach-v3 --seed 0

CFG=metaworld_vla/configs/metaworld_mt50_corner2_480_50ep_b4_h512.yaml
STATS=stats/metaworld_mt50_corner2_480_50ep_4d.json

# VTT若已完整存在则无需重新计算；会写入配置中的VTT目录。
# CUDA_VISIBLE_DEVICES=0 python metaworld_vla/precompute_vtt.py --config "$CFG"

# 独立目录做SFT一步测试，不覆盖已有正式run。
CUDA_VISIBLE_DEVICES=0 python metaworld_vla/train.py \
  --config "$CFG" --stats "$STATS" \
  --save-dir checkpoints/metaworld_rl_preflight_sft --max-steps 1

# 填写已完成的Meta-World SFT checkpoint，不能用RoboTwin checkpoint。
MW_CKPT=/absolute/path/to/metaworld/checkpoint_epoch_40.pt
test -f "$MW_CKPT" || { echo "请先设置真实MW_CKPT"; exit 1; }
CUDA_VISIBLE_DEVICES=0 python metaworld_vla/eval.py \
  --config "$CFG" --stats "$STATS" --checkpoint "$MW_CKPT" \
  --tasks reach-v3 push-v3 drawer-open-v3 pick-place-v3 \
  --test-num 10 --horizon 500 --seed 0 \
  --output eval_result/metaworld_rl_preflight/sft_seed0_n10.json
```

正式SFT仍按 `metaworld_vla/SELF_COLLECTED_MT50.md` 的命令执行。本轮没有新增Meta-World RL训练入口，所以**目前没有可执行的Meta-World PPO/SAC训练命令**；不要将 `configs/rl_residual_yf10_single.yaml` 传给Meta-World训练器，它是RoboTwin原型，不兼容4D控制。

## 8. 下一次代码实现的清单（计划，尚未完成）

建议只增加 `metaworld_rl/` 和独立配置，不修改共享 `models/`：

1. `env_adapter.py`：原生step、终止/截断、目标列表和奖励协议；
2. `frozen_sft.py`：连接现有Meta-World SFT，缓存base action/phase/视觉特征；
3. `residual_actor.py`：正确的潜变量采样与概率计算，4D控制和零残差评测；
4. `buffer.py`、`train.py`：CPU缓存、GAE/PPO、恢复训练和独立输出；
5. `evaluate.py`：开发/最终目标隔离、三seed、与baseline配对；
6. 单测：概率ratio在无更新时为1、动作界限、超时bootstrap、SFT参数不变；
7. 一次真实单环境短rollout与一次优化后，才提供并标为可用的训练命令。

单卡低成本取决于冻结骨干、单环境、缓存特征和CPU buffer，而不是平台名。先完成M0～M3，再决定是否投入M4；Meta-World残差RL取得收益后再考虑RoboTwin与真机迁移。

## 9. 官方参考

- [Basic Usage：MT/ML、环境构造及参数](https://metaworld.farama.org/introduction/basic_usage/)
- [Action Space：4D Cartesian displacement与gripper](https://metaworld.farama.org/benchmark/action_space/)
- [Evaluation：success定义及MT/ML评测](https://metaworld.farama.org/evaluation/evaluation/)
- [Rendering：相机与RGB渲染](https://metaworld.farama.org/rendering/rendering/)
- [官方代码](https://github.com/Farama-Foundation/Metaworld)

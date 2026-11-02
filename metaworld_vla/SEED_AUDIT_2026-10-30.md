# 采集与评测 seed 核验（2026-10-30）

## 从文件确认的事实

自采目录为 `data/metaworld_mt50_corner2`。50个任务各50条成功数据，HDF5 的 `seed` 为环境构造/初次 reset 使用的 `基础seed+尝试编号`。多数任务为0～49；存在失败重试的任务编号更大，例如 peg-insert-side-v3 为0～57，soccer-v3为1～59。不能将最大编号视为实际成功数量。

已保存的 base/short/tiny/G2 等汇总评测 JSON 的 seed 均为0、test_num为50、horizon为500，rng_protocol为 reset_python_numpy_torch_per_task_v1。

## 重要修正：reset seed 不是逐 episode 的有效场景 seed

当前安装的 Meta-World `SawyerXYZEnv.reset` 明确忽略 seed 参数；`RandomTaskSelectWrapper.reset` 在调用它之前从自身 RNG 选任务实例。`gym.make('Meta-World/MT1', env_name=task, seed=S)` 构造由 S 生成的50实例任务池；随后 repeated reset 默认有放回地抽取该池中的实例。

采集每次尝试都用 seed+episode_id 新建环境；评测每个任务只用 seed=0 新建一次，再反复 reset。因此：

- 训练不同 episode 来自不同构造 seed 的任务池。
- seed0评测不是分别构造 seed0～49 的50个场景，也不是每次 reset 同一个场景。
- 评测 JSON 中 episode_results.environment_seed（若存在）记录的是请求传入 reset 的数值，不能用它证明实际场景独立或不重叠。
- 同 seed 的训练与评测存在重叠风险，但不能据此断言所有测试都是训练场景。

## 重建检查方法与限制

`audit_seed_overlap.py` 读取训练 HDF5 的 seed，在当前安装环境重建 collection 的初始化 rand_vec；与 legacy评测 reset-only 序列比较（精度10位小数）。它不加载模型、不运行策略、不修改已有数据或评测协议。

结果在 `eval_result/seed_overlap_audit/report.json`。这是当前环境下的重建证据，不是从旧 HDF5 直接读取的完整物体/目标状态；旧文件仅保存4D proprioception。没有策略 rollout，不能据此计算重叠导致的成功率变化，也不能将少数任务的结论推广至全部MT50。

已完成 reach-v3、push-v3、pick-place-v3 三个任务：各有50个唯一训练初始化；seed0的50次评测reset分别只有30个唯一初始化，均有3次匹配训练初始化（episode0、25、30，匹配训练seed0），单个场景最多重复4次。三者合计9/150次reset重叠，不代表其余47个任务的重叠率。核对全部原始文件的 `seed - episode_id` 均为0，确认采集基础seed为0。

## 下一步建议

保留已有seed0结果用于历史内部消融，不覆盖它。对同样的checkpoint新增独立环境构造seed，如10000、20000、30000，并使用相同任务、相机、horizon、test-num和模型配置进行对比。不同seed仍应检查实际初始化重叠，不能仅凭seed数字不同声称严格持出。

可用 `--policy-seed 0` 单独固定动作噪声，并对所有环境seed使用同样该选项；为严格比较，应给环境seed0也重新运行一个带该选项的控制组。示例：

```bash
CUDA_VISIBLE_DEVICES=0 python metaworld_vla/eval_parallel.py \
  --config checkpoints/metaworld_mt50_base/config.yaml \
  --stats stats/metaworld_mt50_corner2_480_50ep_4d.json \
  --checkpoint checkpoints/metaworld_mt50_base/checkpoint_epoch_50.pt \
  --workers 2 --test-num 50 --horizon 500 \
  --seed 10000 --policy-seed 0 \
  --output eval_result/metaworld_mt50_base_env10000_policy0.json
```

重复采样会降低有效独立场景数量，但不必然抬高成功率；场景难度分布、接触容错和判成功方式也会影响结果。当前评测按500步内任意一次 info.success 为真计成功，不要求最终帧或连续多步成功。当前86.56%不能仅凭seed重叠认定为无效，也不能直接称为无重叠泛化性能。

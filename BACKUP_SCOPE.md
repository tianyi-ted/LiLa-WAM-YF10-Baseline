# LiLa-WAM YF10 baseline snapshot

Snapshot date: 2026-09-23. This repository is a clean code snapshot of the locally validated LiLa-WAM YF10 baseline and the first future-offset ablation setup. It was prepared from the working tree derived from [teee000/LiLa-WAM](https://github.com/teee000/LiLa-WAM), whose upstream commit at the time of the snapshot was `65a320397faae07f311680fcb32f6c2f27dca1a0`.

## Included

- DINOv3 LiLa-WAM training, dataset, model and RoboTwin evaluation code.
- DINOv3-Base＋h512 baseline configuration.
- C32/F16 four-epoch future-offset ablation configurations.
- Explicit training-seed and resumable RNG-state support.
- Ten-task Base and Large VTT vectors; YF10 normalization statistics.
- Baseline manifests, research protocols and evaluation summaries.
- EduLite A3 collection/VLA integration source, configs and local SDK patch.

## Deliberately omitted

- HDF5 datasets and real-robot recordings.
- DINOv3 pretrained weights and all model checkpoints.
- Evaluation videos, full worker logs and training logs.
- Generated inference images/videos.
- Nested third-party repositories `robot/ABPolicy-code` and `robot/EDULITE_A3`.
- Local credentials and editor files.

The omitted baseline checkpoint is larger than GitHub's normal per-file limit. Its identity remains recorded in `docs/YF10_DINOV3B_H512_BASELINE_FREEZE.md` and `artifacts/yf10/baseline_freeze/dinov3b_h512_ep40.sha256`. Store the actual checkpoint and DINO weights in local or institutional artifact storage.

## Frozen baseline

```text
architecture: DINOv3 ViT-B/16 + Action Expert depth=12, hidden=512
checkpoint: checkpoints_vla/yf10_dinov3b_small/sft_2026-09-20_15-07-28/checkpoint_epoch_40.pt
checkpoint SHA256: 98bd63e1c760d2c3bbb1daaa7675a409d1efc1c53d3b079cee2fbd1edf0eb421
clean seed0 n50: 351/500 = 70.2%
randomized seed0 n50: 323/500 = 64.6%
combined: 674/1000 = 67.4%
```

The compact evaluation summaries are under `baseline_records/eval/`. The complete protocol and commands are in `docs/YF10_DINOV3B_H512_BASELINE_FREEZE.md`.

## Restoring local assets

The checked-in YAML files currently preserve the absolute paths used on the experiment workstation. For a clone at another location, update these three path classes consistently:

```text
dataset.dataset_dir
dataset.task_cond_dir
model.vision_encoder.checkpoint_path
```

Restore the corresponding local assets:

```text
data/yf10/
pretrained/dinov3-vitb16/
checkpoints_vla/yf10_dinov3b_small/sft_2026-09-20_15-07-28/
```

Then validate the setup:

```bash
python utils/verify_yf10_setup.py \
  --config configs/yf10_small_dinov3b.yaml \
  --stats artifacts/yf10/norm_stats.json
```

The repository is intended to preserve code and small reproducibility assets. It is not a self-contained redistribution of RoboTwin, DINOv3 weights or the training dataset.

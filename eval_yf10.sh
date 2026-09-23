#!/usr/bin/env bash
# Ten-task LiLa-WAM evaluation. Configuration is supplied with environment
# variables so the script itself remains an auditable, unchanged protocol.
set -euo pipefail

VLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/home/lab347-no10/Tianyi/RoboTwin}"
BRIDGE_SCRIPT="$ROBOTWIN_ROOT/eval_vla_bridge.py"
PYTHON="${PYTHON:-python}"

# The legacy RoboTwin evaluation code imports modules from the repository root
# and opens ./assets/... paths during import.  Export both the explicit root and
# Python search path so evaluation is independent of the caller's cwd and also
# works when eval_vla_bridge.py is a symlink into this repository.
export ROBOTWIN_ROOT
export PYTHONPATH="$ROBOTWIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"

: "${CKPT_SETTING:?Set CKPT_SETTING, e.g. yf10/sft_2026-09-10_12-00-00}"
: "${CHECKPOINT_EP:?Set CHECKPOINT_EP, e.g. 12}"

TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-unseen}"
NORM_STATS_PATH="${NORM_STATS_PATH:-artifacts/yf10/norm_stats.json}"
CONFIG_PATH="${CONFIG_PATH:-configs/yf10_base.yaml}"
SEED="${SEED:-0}"
TEST_NUM="${TEST_NUM:-50}"
NUM_WORKERS="${NUM_WORKERS:-1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

TASKS=(
  adjust_bottle
  grab_roller
  hanging_mug
  move_stapler_pad
  open_microwave
  press_stapler
  scan_object
  stack_blocks_two
  stamp_seal
  turn_switch
)

if [[ ! -f "$BRIDGE_SCRIPT" ]]; then
  echo "Missing $BRIDGE_SCRIPT"
  echo "Create it once with:"
  echo "  ln -s $VLA_ROOT/eval_vla_bridge.py $BRIDGE_SCRIPT"
  exit 2
fi

for required_path in \
  "$ROBOTWIN_ROOT/envs/__init__.py" \
  "$ROBOTWIN_ROOT/assets" \
  "$ROBOTWIN_ROOT/task_config"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Invalid ROBOTWIN_ROOT=$ROBOTWIN_ROOT; missing $required_path" >&2
    exit 2
  fi
done

SAVE_ROOT="$VLA_ROOT/eval_result/yf10/$RUN_ID"
LOG_DIR="$SAVE_ROOT/logs"
RESULT_DIR="$SAVE_ROOT/bridge_results"
mkdir -p "$LOG_DIR" "$RESULT_DIR"

pids=()
worker_ids=()
for ((w=0; w<NUM_WORKERS; w++)); do
  chunk=()
  for ((i=w; i<${#TASKS[@]}; i+=NUM_WORKERS)); do
    chunk+=("${TASKS[i]}")
  done
  [[ ${#chunk[@]} -eq 0 ]] && continue
  echo "[worker $w] ${chunk[*]}"
  (
    cd "$ROBOTWIN_ROOT"
    "$PYTHON" "$BRIDGE_SCRIPT" \
      --task_names "${chunk[@]}" \
      --task_config "$TASK_CONFIG" \
      --instruction_type "$INSTRUCTION_TYPE" \
      --ckpt_setting "$CKPT_SETTING" \
      --checkpoint_ep "$CHECKPOINT_EP" \
      --model_base_path "$VLA_ROOT" \
      --norm_stats_path "$NORM_STATS_PATH" \
      --config_path "$CONFIG_PATH" \
      --vla_root "$VLA_ROOT" \
      --save_root "$SAVE_ROOT" \
      --result_json "$RESULT_DIR/worker_$w.json" \
      --seed "$SEED" \
      --test_num "$TEST_NUM"
  ) >"$LOG_DIR/worker_$w.log" 2>&1 &
  pids+=("$!")
  worker_ids+=("$w")
done

failed=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[i]}"; then
    echo "worker ${worker_ids[i]} failed; see $LOG_DIR/worker_${worker_ids[i]}.log"
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  echo "Evaluation failed; no summary was generated. Inspect logs under: $LOG_DIR" >&2
  exit "$failed"
fi

"$PYTHON" - "$RESULT_DIR" "$SAVE_ROOT/summary.json" <<'PY'
import glob, json, os, sys

result_dir, output = sys.argv[1:]
rows = []
for path in sorted(glob.glob(os.path.join(result_dir, "worker_*.json"))):
    with open(path, encoding="utf-8") as f:
        rows.extend(json.load(f)["results"])
total_success = sum(r["success"] for r in rows)
total_trials = sum(r["test_num"] for r in rows)
summary = {
    "results": rows,
    "macro_success_rate": (sum(r["success_rate"] for r in rows) / len(rows)) if rows else None,
    "micro_success_rate": (total_success / total_trials) if total_trials else None,
    "total_success": total_success,
    "total_trials": total_trials,
}
with open(output, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
for r in rows:
    print(f"{r['task_name']:24s} {r['success']:3d}/{r['test_num']:3d}  {100*r['success_rate']:6.2f}%")
if rows:
    print(f"Macro average: {100*summary['macro_success_rate']:.2f}%")
    print(f"Micro average: {total_success}/{total_trials} = {100*summary['micro_success_rate']:.2f}%")
print(f"Summary: {output}")
PY

exit "$failed"

"""Offline comparison of the six 2026-11-13 prefix-refit deployment runs.

Run from the repository root with a Python environment containing NumPy.
Only reads saved telemetry; never imports the hardware runtime.
"""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "robot/edulite_a3_vla"
GROUPS = {
    "unaligned": ("inference_outputs_multitask_adapolicy", ["172742", "172827", "172909"]),
    "aligned": ("inference_outputs_multitask_adapolicy_aligned", ["183156", "183234", "183306"]),
}


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return dict(count=len(values), min=float(values.min()),
                p50=float(np.percentile(values, 50)), p95=float(np.percentile(values, 95)),
                max=float(values.max()))


def values(rows, key):
    return [float(row[key]) for row in rows if row.get(key, "") != ""]


def matrix(rows, prefix):
    return np.array([[float(row[f"{prefix}_{j}"]) for j in range(6)] for row in rows])


def collect(folder, suffixes):
    pool = {}
    per_run = []
    counts = Counter()
    offsets = Counter()
    speed_ratios = []
    first_18s_cos, first_18s_min_ratio = [], []
    first_18s_speed_ratios = []
    settings = []
    warnings_detail = []

    def add(key, samples):
        pool.setdefault(key, []).extend(samples)

    for suffix in suffixes:
        run = ROOT / folder / f"run_20261113_{suffix}"
        summary = json.loads((run / "runtime_summary.json").read_text())
        settings.append(summary["settings"])
        chunks = read_csv(run / "inference_chunks.csv")
        bg = [row for row in chunks if row["source"] == "background"]
        ticks = read_csv(run / "runtime_ticks.csv")
        trace = read_csv(run / "execution_trace.csv")
        control = read_csv(run / "control_timing.csv")
        completed = read_csv(run / "inference_completed.csv")
        executions = read_csv(run / "chunk_executions.csv") if (run / "chunk_executions.csv").exists() else []
        assert len(ticks) == len(trace)
        times = np.array(values(ticks, "command_mono_s"))
        assert len(times) == len(ticks) and np.all(np.diff(times) > 0)
        ci = np.array(values(control, "control_mono_s"))
        speed = np.array(values(control, "max_command_velocity_rad_s"))
        safe, cmd, raw = (matrix(trace, key) for key in ("safe_target", "filtered_command", "raw_action"))
        gap = np.max(np.abs(safe - cmd), axis=1)
        v = np.diff(safe, axis=0) / np.diff(times)[:, None]
        add("target_to_filtered_command_rad", gap)
        chunk_map = {row["chunk_id"]: row for row in chunks}
        counts["background_chunks"] += len(bg)
        counts["fallback_background_chunks"] += sum(int(row.get("fallback") or 0) for row in bg)
        counts["warning_actions"] += summary["safety_warning_actions"]
        counts["model_updates"] += summary["model_updates"]
        for key in ("queue_wait_events", "schedule_resets", "time_alignment_rejected_chunks"):
            counts[key] += summary.get(key, 0)
        counts["duration_ms"] += round(summary["duration_s"] * 1000)
        for key in ("emitted_boundary_rad", "raw_boundary_rad", "smooth_max_step_rad",
                    "fit_mae_rad", "fit_compute_s", "observation_age_s", "conditioning_s"):
            add(key, values(bg, key))
        add("background_inference_s", values([row for row in completed if row["source"] == "background"], "infer_s"))
        add("following_error_rad", values(control, "following_error_rad"))
        add("action_interval_s", values(ticks, "action_interval_s"))
        add("control_interval_s", values(control, "interval_s"))
        add("background_actions_selected", values([row for row in executions
            if row["source"] == "background" and row["end_reason"] == "replaced"], "actions_selected"))
        for key in ("aligned_start_index", "alignment_residual_age_s", "aligned_raw_start_to_command_max_rad",
                    "aligned_start_to_command_max_rad", "alignment_conditioning_attempts"):
            add(key, values(chunks, key))
        offsets.update(int(row["aligned_start_index"]) for row in chunks if row.get("aligned_start_index", "") != "")
        add("startup_refresh_observation_age_s", values([row for row in chunks if row["source"] == "startup_refresh"], "observation_age_s"))
        add("startup_refresh_infer_s", values([row for row in chunks if row["source"] == "startup_refresh"], "infer_s"))
        phase = []
        selected_counts = Counter()
        for row in ticks:
            meta = chunk_map[row["chunk_id"]]
            selected_counts[row["chunk_id"]] += 1
            selected = int(row.get("chunk_actions_selected") or selected_counts[row["chunk_id"]])
            assert selected == selected_counts[row["chunk_id"]]
            start = int(meta.get("aligned_start_index") or 0)
            predicted_time = float(meta["observation_mono_s"]) + (start + selected - 1) / 30
            error = float(row["command_mono_s"]) - predicted_time
            if row.get("alignment_phase_error_s", "") != "":
                assert abs(error - float(row["alignment_phase_error_s"])) < 1e-8
            if meta["source"] == "background":
                phase.append(error)
        add("background_phase_error_s", phase)
        add("logged_alignment_phase_error_s", values(ticks, "alignment_phase_error_s"))
        counts["phase_background_over_one_tick"] += sum(abs(x) > 1 / 30 for x in phase)
        timeouts = [float(row["control_mono_s"]) for row in control if int(row["model_command_timed_out"])]
        timeout_breakdown = {
            "before_first_action": int(sum(t < times[0] for t in timeouts)),
            "during_action_span": int(sum(times[0] <= t <= times[-1] for t in timeouts)),
            "after_last_action": int(sum(t > times[-1] for t in timeouts)),
        }
        for key, n in timeout_breakdown.items():
            counts["timeout_cycles_" + key] += n

        boundaries = [i for i in range(3, len(ticks) - 6) if ticks[i]["chunk_id"] != ticks[i-1]["chunk_id"]]
        run_cos, run_min_ratio = [], []
        for i in boundaries:
            a, b = v[i-3:i-1].mean(0), v[i:i+2].mean(0)
            add("boundary_target_slope_change_rad_s", [np.max(np.abs(b-a))])
            if np.max(np.abs(a)) > .05 and np.max(np.abs(b)) > .05:
                run_cos.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
                if times[i] - times[0] <= 18:
                    first_18s_cos.append(run_cos[-1])
            stamp = times[i]
            pre = speed[(ci >= stamp-.10) & (ci < stamp-.02)]
            if len(pre) and pre.mean() > .05:
                ratios = []
                for j in range(6):
                    w = speed[(ci >= stamp+j*.05) & (ci < stamp+(j+1)*.05)]
                    ratios.append(float(w.mean()/pre.mean()) if len(w) else float("nan"))
                speed_ratios.append(ratios)
                w = speed[(ci >= stamp) & (ci < stamp+.15)]
                run_min_ratio.append(float(w.min()/pre.mean()))
                if times[i] - times[0] <= 18:
                    first_18s_min_ratio.append(run_min_ratio[-1])
                    first_18s_speed_ratios.append(ratios)
        add("boundary_target_direction_cos", run_cos)
        add("boundary_control_speed_min_ratio_150ms", run_min_ratio)
        interior = []
        for i in range(3, len(ticks)-6):
            if len({row["chunk_id"] for row in ticks[i-3:i+3]}) != 1:
                continue
            a, b = v[i-3:i-1].mean(0), v[i:i+2].mean(0)
            if np.max(np.abs(a)) > .05 and np.max(np.abs(b)) > .05:
                interior.append(float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b))))
        add("interior_target_direction_cos", interior)
        warning_indices = [i for i, row in enumerate(ticks) if int(row["safety_warning_count"])]
        warning_info = {"run": run.name, "count": len(warning_indices)}
        if warning_indices:
            delta = np.abs(raw-safe)
            warning_info.update(
                first_time_s=float(trace[warning_indices[0]]["time_s"]),
                last_time_s=float(trace[warning_indices[-1]]["time_s"]),
                affected_joints={f"L{j+1}": int(np.sum(delta[warning_indices, j] > 1e-6)) for j in range(6)},
                max_selected_to_safe_difference_rad=float(delta[warning_indices].max()),
                max_selected_step_rad=float(np.abs(np.diff(raw, axis=0)).max()),
                max_safe_step_rad=float(np.abs(np.diff(safe, axis=0)).max()),
            )
        warnings_detail.append(warning_info)
        per_run.append(dict(
            run=str(run.relative_to(REPO)), stop_reason=summary["stop_reason"],
            duration_s=summary["duration_s"], background_chunks=len(bg),
            boundary_rad=stats(values(bg, "emitted_boundary_rad")),
            inference_s=stats(values(completed, "infer_s")),
            following_error_rad=stats(values(control, "following_error_rad")),
            phase_s=stats(phase),
            direction_reversal_count=sum(x < 0 for x in run_cos),
            direction_eligible_count=len(run_cos),
            slowdown_count=sum(x < .5 for x in run_min_ratio),
            slowdown_eligible_count=len(run_min_ratio),
            warnings=len(warning_indices),
            fallbacks=sum(int(row.get("fallback") or 0) for row in bg),
            timeout_cycles=timeout_breakdown,
        ))
    result = dict(metrics={key: stats(samples) for key, samples in pool.items()},
                  counts=dict(counts), per_run=per_run, warnings=warnings_detail,
                  offset_counts=dict(sorted(offsets.items())),
                  speed_ratios_median_50ms_bins=np.nanmedian(speed_ratios, axis=0).tolist())
    result["first_18s_boundary_comparison"] = {
        "direction_eligible_count": len(first_18s_cos),
        "direction_reversal_fraction": float(np.mean(np.array(first_18s_cos) < 0)),
        "slowdown_eligible_count": len(first_18s_min_ratio),
        "speed_drop_below_half_fraction": float(np.mean(np.array(first_18s_min_ratio) < .5)),
        "speed_ratios_median_50ms_bins": np.nanmedian(first_18s_speed_ratios, axis=0).tolist(),
    }
    for key, samples in (("boundary_direction_reversal_fraction", pool["boundary_target_direction_cos"]),
                         ("interior_direction_reversal_fraction", pool["interior_target_direction_cos"])):
        result[key] = float(np.mean(np.array(samples) < 0))
    result["boundary_speed_drop_below_half_fraction"] = float(np.mean(np.array(pool["boundary_control_speed_min_ratio_150ms"]) < .5))
    result["warning_fraction"] = counts["warning_actions"] / counts["model_updates"]
    assert all(setting == settings[0] for setting in settings)
    result["settings_identical_within_group"] = True
    return result, settings[0]


def diff(a, b, prefix=""):
    changes = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(a.keys() | b.keys()):
            changes += diff(a.get(key), b.get(key), f"{prefix}.{key}".lstrip("."))
    elif a != b:
        changes.append(dict(field=prefix, unaligned=a, aligned=b))
    return changes


def plot_comparison(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#667c96", "#149989"]
    labels = ["Unaligned", "Time aligned"]
    groups = [result[key] for key in ("unaligned", "aligned")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axes[0, 0]
    for group, label, color in zip(groups, labels, colors):
        ax.plot(np.arange(25, 300, 50), group["speed_ratios_median_50ms_bins"],
                "o-", label=label, color=color, linewidth=2)
    ax.axhline(1, color="gray", linestyle="--", linewidth=1)
    ax.set(xlabel="Time after chunk switch (ms)", ylabel="Normalized command-speed envelope",
           title="A. Less speed dip at chunk switches", ylim=(0, 1.15))
    ax.legend()
    ax = axes[0, 1]
    fractions = [100 * group["boundary_speed_drop_below_half_fraction"] for group in groups]
    ax.bar(labels, fractions, color=colors, alpha=.8)
    for i, group in enumerate(groups):
        trials = [100 * run["slowdown_count"] / run["slowdown_eligible_count"] for run in group["per_run"]]
        ax.scatter(np.array([i-.08, i, i+.08]), trials, color="black", s=20, zorder=3)
        ax.text(i, fractions[i]+2, f"{fractions[i]:.1f}%", ha="center")
    ax.set(ylabel="Eligible switches (%)", title="B. Speed below 50% within 150 ms", ylim=(0, 60))
    ax = axes[1, 0]
    phases = [1000 * group["metrics"]["background_phase_error_s"]["p95"] for group in groups]
    ax.bar(labels, phases, color=colors)
    for i, value in enumerate(phases):
        ax.text(i, value+5, f"{value:.2f} ms", ha="center")
    ax.set(ylabel="P95 phase error (ms)", title="C. Background action time alignment", ylim=(0, 270))
    ax = axes[1, 1]
    for i, (group, label, color) in enumerate(zip(groups, labels, colors)):
        heights = [group["metrics"][key]["p95"] for key in
                   ("emitted_boundary_rad", "target_to_filtered_command_rad")]
        positions = np.arange(2) + (i-.5)*.32
        ax.bar(positions, heights, width=.32, label=label, color=color)
        for x, height in zip(positions, heights):
            ax.text(x, height+.006, f"{height:.3f}", ha="center", fontsize=9)
    ax.set(xticks=np.arange(2), xticklabels=["Goal jump at switch", "Goal to filtered command"],
           ylabel="P95 max joint difference (rad)", title="D. Goal lead remains despite smoother motion", ylim=(0, .30))
    ax.legend()
    fig.suptitle("EDULITE prefix refit: 3 unaligned vs 3 time-aligned runs (2026-11-13)", fontsize=14)
    fig.savefig(REPO / "docs/figures/edulite_time_alignment_comparison_2026-11-13.png", dpi=180)
    plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true", help="Also generate a comparison figure (requires Matplotlib).")
    args = parser.parse_args()
    result, settings = {}, {}
    for key, (folder, suffixes) in GROUPS.items():
        result[key], settings[key] = collect(folder, suffixes)
    result["settings_differences"] = diff(settings["unaligned"], settings["aligned"])
    destination = Path(__file__).with_name("edulite_time_alignment_comparison_2026-11-13.json")
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if args.plot:
        plot_comparison(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

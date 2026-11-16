"""Recompute the five deployment variants from 15 explicitly selected saved runs.

No hardware imports. Run with --plot to also generate the overview figure.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_edulite_time_alignment import collect, diff

REPO = Path(__file__).resolve().parents[2]
GROUPS = {
    "A_off": ("inference_outputs_multitask", ["153631", "153713", "153915"]),
    "B_weighted": ("inference_outputs_multitask", ["161246", "161333", "161414"]),
    "C_latest": ("inference_outputs_multitask_latest", ["172501", "172546", "172628"]),
    "D_prefix": ("inference_outputs_multitask_adapolicy", ["172742", "172827", "172909"]),
    "E_aligned": ("inference_outputs_multitask_adapolicy_aligned", ["183156", "183234", "183306"]),
}


def plot(groups):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["A\nNo spline", "B\nWeighted", "C\nLatest\nWeighted", "D\nLatest\nPrefix", "E\nAligned\nPrefix"]
    colors = ["#a1aab4", "#667c96", "#d0a145", "#a17db8", "#149989"]
    data = list(groups.values())
    x = np.arange(len(data))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    def bars(ax, numbers, title, ylabel, fmt, ceiling):
        ax.bar(x, numbers, color=colors)
        for i, number in enumerate(numbers):
            ax.text(i, number + ceiling * .02, format(number, fmt), ha="center", fontsize=10)
        ax.set(xticks=x, xticklabels=labels, title=title, ylabel=ylabel, ylim=(0, ceiling))

    bars(axes[0, 0], [d["metrics"]["emitted_boundary_rad"]["p95"] for d in data],
         "1. Smaller target jumps at chunk switches", "P95 max joint jump (rad)", ".3f", .33)
    bars(axes[0, 1], [100*d["boundary_speed_drop_below_half_fraction"] for d in data],
         "2. Fewer speed drops after time alignment", "Speed below 50% within 150 ms (%)", ".1f", 75)
    for annotation in list(axes[0, 1].texts):
        annotation.remove()
    for i, d in enumerate(data):
        trial = [100*r["slowdown_count"]/r["slowdown_eligible_count"] for r in d["per_run"]]
        axes[0, 1].scatter(i+np.array([-.08, 0, .08]), trial, color="black", s=14, zorder=3)
        fraction = 100*d["boundary_speed_drop_below_half_fraction"]
        axes[0, 1].text(i, max(trial+[fraction])+3, f"{fraction:.1f}", ha="center", fontsize=10)
    for i, d in enumerate(data):
        axes[1, 0].plot(np.arange(25, 300, 50), d["speed_ratios_median_50ms_bins"],
                        "o-", color=colors[i], label=labels[i].replace("\n", ": "), linewidth=2)
    axes[1, 0].axhline(1, color="gray", linestyle="--", linewidth=1)
    axes[1, 0].set(title="3. Command-speed envelope near switches", xlabel="Time after switch (ms)",
                   ylabel="Median normalized command speed", ylim=(0, 1.2))
    axes[1, 0].legend(fontsize=8)
    bars(axes[1, 1], [d["metrics"]["target_to_filtered_command_rad"]["p95"] for d in data],
         "4. Target-to-command gap still needs attention", "P95 max joint difference (rad)", ".3f", .31)
    fig.suptitle("EDULITE deployment comparison: 5 variants, 15 runs (2026-11-13)", fontsize=15)
    fig.savefig(REPO / "docs/figures/edulite_deployment_comparison_2026-11-13.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    groups, settings = {}, {}
    for key, (folder, suffixes) in GROUPS.items():
        groups[key], settings[key] = collect(folder, suffixes)
    reference = settings["A_off"]
    for setting in settings.values():
        assert setting["safety"] == reference["safety"]
        assert {k: v for k, v in setting["model"].items() if k != "instruction"} == {
            k: v for k, v in reference["model"].items() if k != "instruction"}
        for field in ("action_chunk_size", "action_execution_horizon", "num_inference_steps"):
            assert setting["model_interface"][field] == reference["model_interface"][field]
        for field in ("action_rate_hz", "control_rate_hz", "prefetch_threshold_actions"):
            assert setting["runtime"][field] == reference["runtime"][field]
    keys = list(groups)
    result = {
        "groups": groups,
        "settings": settings,
        "adjacent_settings_differences": {
            f"{a}_to_{b}": diff(settings[a], settings[b]) for a, b in zip(keys, keys[1:])},
        "run_count": sum(len(d["per_run"]) for d in groups.values()),
        "background_chunks": sum(d["counts"]["background_chunks"] for d in groups.values()),
        "model_updates": sum(d["counts"]["model_updates"] for d in groups.values()),
    }
    assert result["run_count"] == 15 and result["background_chunks"] == 915
    output = Path(__file__).with_name("edulite_deployment_comparison_2026-11-13.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if args.plot:
        plot(groups)
    print(f"Wrote {output}; {result['run_count']} runs, {result['background_chunks']} background chunks.")


if __name__ == "__main__":
    main()

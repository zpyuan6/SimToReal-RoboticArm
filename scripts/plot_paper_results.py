from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "results" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 1.0,
    }
)


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def plot_pseudo_real_main() -> None:
    shift_paths = {
        "Appearance": ROOT / "results" / "pseudo_real_appearance" / "backbone_suite_shift_main" / "feedforward" / "backbone_suite_shiftgrid_a8g05_fullff" / "suite_overall_metrics.csv",
        "Embodiment": ROOT / "results" / "pseudo_real_embodiment" / "backbone_suite_shift_main" / "feedforward" / "backbone_suite_shiftgrid_a8g05_fullff" / "suite_overall_metrics.csv",
        "Joint": ROOT / "results" / "pseudo_real_joint" / "backbone_suite_shift_main" / "feedforward" / "backbone_suite_shiftgrid_a8g05_fullff" / "suite_overall_metrics.csv",
    }
    order = ["no_adaptation", "input_normalization", "static_adapter", "ours"]
    labels = {
        "no_adaptation": "No adaptation",
        "input_normalization": "Input norm.",
        "static_adapter": "Static adapter",
        "ours": "PLICA",
    }
    colors = {
        "no_adaptation": "#9AA0A6",
        "input_normalization": "#6C8EBF",
        "static_adapter": "#B07AA1",
        "ours": "#D55E00",
    }

    values = {}
    for shift, path in shift_paths.items():
        df = _load_csv(path)
        values[shift] = {row["baseline"]: row["mean_success_rate"] * 100.0 for _, row in df.iterrows()}

    fig, ax = plt.subplots(figsize=(8.8, 3.8), constrained_layout=True)
    shifts = list(shift_paths.keys())
    x = range(len(shifts))
    width = 0.18
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

    for idx, baseline in enumerate(order):
        ys = [values[shift][baseline] for shift in shifts]
        bars = ax.bar([i + offsets[idx] for i in x], ys, width=width, label=labels[baseline], color=colors[baseline])
        for bar, y in zip(bars, ys):
            ax.text(bar.get_x() + bar.get_width() / 2, y + 0.7, f"{y:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(shifts)
    ax.set_ylabel("Mean success rate (%)")
    ax.set_ylim(60, 100.5)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12))

    fig.savefig(PLOTS / "paper_pseudo_real_main.png", dpi=300, bbox_inches="tight")
    fig.savefig(PLOTS / "paper_pseudo_real_main.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_pseudo_real_efficiency() -> None:
    shift_paths = {
        "Appearance": ROOT / "results" / "pseudo_real_appearance" / "backbone_suite_shift_main" / "feedforward" / "backbone_suite_shiftgrid_a8g05_fullff" / "suite_overall_metrics.csv",
        "Embodiment": ROOT / "results" / "pseudo_real_embodiment" / "backbone_suite_shift_main" / "feedforward" / "backbone_suite_shiftgrid_a8g05_fullff" / "suite_overall_metrics.csv",
        "Joint": ROOT / "results" / "pseudo_real_joint" / "backbone_suite_shift_main" / "feedforward" / "backbone_suite_shiftgrid_a8g05_fullff" / "suite_overall_metrics.csv",
    }
    order = ["no_adaptation", "input_normalization", "static_adapter", "ours"]
    labels = {
        "no_adaptation": "No adaptation",
        "input_normalization": "Input norm.",
        "static_adapter": "Static adapter",
        "ours": "PLICA",
    }
    colors = {
        "no_adaptation": "#9AA0A6",
        "input_normalization": "#6C8EBF",
        "static_adapter": "#B07AA1",
        "ours": "#D55E00",
    }

    values = {}
    for shift, path in shift_paths.items():
        df = _load_csv(path)
        values[shift] = {row["baseline"]: row["mean_steps"] for _, row in df.iterrows()}

    fig, ax = plt.subplots(figsize=(8.8, 3.8), constrained_layout=True)
    shifts = list(shift_paths.keys())
    x = range(len(shifts))
    width = 0.18
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

    for idx, baseline in enumerate(order):
        ys = [values[shift][baseline] for shift in shifts]
        bars = ax.bar([i + offsets[idx] for i in x], ys, width=width, label=labels[baseline], color=colors[baseline])
        for bar, y in zip(bars, ys):
            ax.text(bar.get_x() + bar.get_width() / 2, y + 0.18, f"{y:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(shifts)
    ax.set_ylabel("Mean episode length")
    ax.set_ylim(5.5, 16.8)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12))

    fig.savefig(PLOTS / "paper_pseudo_real_efficiency.png", dpi=300, bbox_inches="tight")
    fig.savefig(PLOTS / "paper_pseudo_real_efficiency.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_real_transition_results() -> None:
    baseline_path = ROOT / "results" / "fixed_protocol" / "real_eval_shiftgrid_a8g05_ff" / "backbone_suite_real" / "suite_overall_metrics.csv"
    ablation_path = ROOT / "results" / "fixed_protocol" / "real_eval_shiftgrid_a8g05_ff" / "ablation_shiftgrid_a8g05" / "feedforward" / "ablation_overall_metrics.csv"

    baseline_df = _load_csv(baseline_path)
    ablation_df = _load_csv(ablation_path)

    baseline_order = [
        "no_adaptation",
        "static_adapter",
        "tent_style",
        "few_shot_finetuning",
        "input_normalization",
        "probe_feature_alignment",
        "ours",
    ]
    baseline_labels = {
        "no_adaptation": "No adaptation",
        "static_adapter": "Static adapter",
        "tent_style": "Tent-style",
        "few_shot_finetuning": "Few-shot FT",
        "input_normalization": "Input norm.",
        "probe_feature_alignment": "Probe align.",
        "ours": "PLICA",
    }
    baseline_colors = {
        "ours": "#D55E00",
        "static_adapter": "#B07AA1",
        "no_adaptation": "#9AA0A6",
        "tent_style": "#6C8EBF",
        "few_shot_finetuning": "#4C78A8",
        "input_normalization": "#59A14F",
        "probe_feature_alignment": "#E15759",
    }

    ablation_order = [
        "no_adaptation",
        "static_adapter",
        "ours_w_o_transition",
        "ours_w_o_reg",
        "ours_full",
    ]
    ablation_labels = {
        "no_adaptation": "No adaptation",
        "static_adapter": "Static adapter",
        "ours_w_o_transition": "PLICA w/o trans.",
        "ours_w_o_reg": "PLICA w/o reg.",
        "ours_full": "PLICA",
    }
    ablation_colors = {
        "no_adaptation": "#9AA0A6",
        "static_adapter": "#B07AA1",
        "ours_w_o_transition": "#6C8EBF",
        "ours_w_o_reg": "#F28E2B",
        "ours_full": "#D55E00",
    }

    baseline_vals = [baseline_df.loc[baseline_df["baseline"] == b, "mean_transition_mse"].iloc[0] for b in baseline_order]
    ablation_vals = [ablation_df.loc[ablation_df["variant"] == b, "mean_transition_mse"].iloc[0] for b in ablation_order]

    fig, axes = plt.subplots(1, 2, figsize=(9.1, 3.9), constrained_layout=True)

    ax = axes[0]
    practical_order = ["no_adaptation", "static_adapter", "tent_style", "few_shot_finetuning", "ours"]
    practical_vals = [baseline_df.loc[baseline_df["baseline"] == b, "mean_transition_mse"].iloc[0] for b in practical_order]
    bars = ax.bar(range(len(practical_order)), practical_vals, color=[baseline_colors[b] for b in practical_order])
    ax.set_ylabel("Mean transition MSE")
    ax.set_ylim(0.0, 0.50)
    ax.set_xticks(range(len(practical_order)))
    ax.set_xticklabels([baseline_labels[b] for b in practical_order], rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    for bar, val in zip(bars, practical_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.text(-0.10, 1.04, "(a)", transform=ax.transAxes, fontsize=11, fontweight="bold")

    inset = inset_axes(ax, width="42%", height="45%", loc="upper right", borderpad=1.0)
    catastrophic_order = ["input_normalization", "probe_feature_alignment"]
    catastrophic_vals = [baseline_df.loc[baseline_df["baseline"] == b, "mean_transition_mse"].iloc[0] for b in catastrophic_order]
    inset.bar(range(len(catastrophic_order)), catastrophic_vals, color=[baseline_colors[b] for b in catastrophic_order])
    inset.set_yscale("log")
    inset.set_xticks(range(len(catastrophic_order)))
    inset.set_xticklabels(["Input\nnorm.", "Probe\nalign."], fontsize=7)
    inset.tick_params(axis="y", labelsize=7)
    inset.set_title("Failed baselines", fontsize=8, pad=2)
    for i, val in enumerate(catastrophic_vals):
        txt = f"{val:.1f}" if val < 1e6 else f"{val:.1e}"
        inset.text(i, val * 1.18, txt, ha="center", va="bottom", fontsize=6, rotation=90)

    ax2 = axes[1]
    bars2 = ax2.bar(range(len(ablation_order)), ablation_vals, color=[ablation_colors[b] for b in ablation_order])
    ax2.set_ylabel("Mean transition MSE")
    ax2.set_xticks(range(len(ablation_order)))
    ax2.set_xticklabels([ablation_labels[b] for b in ablation_order], rotation=25, ha="right")
    ax2.set_ylim(0, 0.26)
    ax2.grid(axis="y", alpha=0.25, linewidth=0.6)
    for bar, val in zip(bars2, ablation_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.006, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax2.text(-0.10, 1.04, "(b)", transform=ax2.transAxes, fontsize=11, fontweight="bold")

    fig.savefig(PLOTS / "paper_real_transition_results.png", dpi=300, bbox_inches="tight")
    fig.savefig(PLOTS / "paper_real_transition_results.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_real_taskwise_breakdown() -> None:
    task_path = ROOT / "results" / "fixed_protocol" / "real_eval_shiftgrid_a8g05_ff" / "backbone_suite_real" / "suite_task_metrics.csv"
    df = _load_csv(task_path)
    order = ["no_adaptation", "static_adapter", "ours"]
    labels = {
        "no_adaptation": "No adaptation",
        "static_adapter": "Static adapter",
        "ours": "PLICA",
    }
    colors = {
        "no_adaptation": "#9AA0A6",
        "static_adapter": "#B07AA1",
        "ours": "#D55E00",
    }
    task_order = ["level1_verify", "level2_approach", "level3_pick_place"]
    task_labels = {
        "level1_verify": "L1 Verify",
        "level2_approach": "L2 Approach",
        "level3_pick_place": "L3 Pick&Place",
    }

    fig, axes = plt.subplots(1, 2, figsize=(9.1, 3.9), constrained_layout=True)
    width = 0.22
    offsets = [-width, 0.0, width]
    x = range(len(task_order))

    for idx, baseline in enumerate(order):
        sub = df[df["baseline"] == baseline].set_index("task")
        ys = [float(sub.loc[task, "transition_mse"]) for task in task_order]
        bars = axes[0].bar([i + offsets[idx] for i in x], ys, width=width, color=colors[baseline], label=labels[baseline])
        for bar, y in zip(bars, ys):
            axes[0].text(bar.get_x() + bar.get_width() / 2, y + 0.008, f"{y:.3f}", ha="center", va="bottom", fontsize=8)

    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels([task_labels[t] for t in task_order])
    axes[0].set_ylabel("Transition MSE")
    axes[0].set_ylim(0.0, 0.42)
    axes[0].grid(axis="y", alpha=0.25, linewidth=0.6)
    axes[0].text(-0.10, 1.04, "(a)", transform=axes[0].transAxes, fontsize=11, fontweight="bold")

    for idx, baseline in enumerate(order):
        sub = df[df["baseline"] == baseline].set_index("task")
        ys = [float(sub.loc[task, "primitive_match"]) * 100.0 for task in task_order]
        bars = axes[1].bar([i + offsets[idx] for i in x], ys, width=width, color=colors[baseline], label=labels[baseline])
        for bar, y in zip(bars, ys):
            axes[1].text(bar.get_x() + bar.get_width() / 2, y + 1.3, f"{y:.1f}", ha="center", va="bottom", fontsize=8)

    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels([task_labels[t] for t in task_order])
    axes[1].set_ylabel("Primitive match (%)")
    axes[1].set_ylim(0.0, 45.0)
    axes[1].grid(axis="y", alpha=0.25, linewidth=0.6)
    axes[1].legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    axes[1].text(-0.10, 1.04, "(b)", transform=axes[1].transAxes, fontsize=11, fontweight="bold")

    fig.savefig(PLOTS / "paper_real_taskwise_breakdown.png", dpi=300, bbox_inches="tight")
    fig.savefig(PLOTS / "paper_real_taskwise_breakdown.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_calibration_budget() -> None:
    budget_path = ROOT / "results" / "fixed_protocol" / "real_budget_shiftgrid_a8g05_ff" / "feedforward" / "budget_sweep_overall_metrics.csv"
    df = _load_csv(budget_path)
    method_order = ["static_adapter", "ours"]
    labels = {"static_adapter": "Static adapter", "ours": "PLICA"}
    colors = {"static_adapter": "#B07AA1", "ours": "#D55E00"}
    no_adapt = float(df[(df["method"] == "no_adaptation") & (df["budget"] == 0)]["mean_transition_mse"].iloc[0])
    no_adapt_match = float(df[(df["method"] == "no_adaptation") & (df["budget"] == 0)]["mean_primitive_match"].iloc[0]) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(9.1, 3.9), constrained_layout=True)

    budget_ticks = sorted(df[df["method"].isin(method_order)]["num_samples"].unique().tolist())

    for method in method_order:
        sub = df[df["method"] == method].sort_values("budget")
        axes[0].plot(sub["num_samples"], sub["mean_transition_mse"], marker="o", linewidth=2.0, color=colors[method], label=labels[method])
        axes[1].plot(sub["num_samples"], sub["mean_primitive_match"] * 100.0, marker="o", linewidth=2.0, color=colors[method], label=labels[method])

    axes[0].axhline(no_adapt, color="#666666", linestyle="--", linewidth=1.2, label="No adaptation")
    axes[1].axhline(no_adapt_match, color="#666666", linestyle="--", linewidth=1.2, label="No adaptation")
    axes[0].set_xscale("log", base=2)
    axes[1].set_xscale("log", base=2)
    axes[0].set_xticks(budget_ticks, [str(v) for v in budget_ticks])
    axes[1].set_xticks(budget_ticks, [str(v) for v in budget_ticks])
    axes[0].set_xlabel("Real calibration transitions")
    axes[1].set_xlabel("Real calibration transitions")
    axes[0].set_ylabel("Transition MSE")
    axes[1].set_ylabel("Primitive match (%)")
    axes[0].grid(alpha=0.25, linewidth=0.6)
    axes[1].grid(alpha=0.25, linewidth=0.6)
    axes[0].set_ylim(0.0, 0.28)
    axes[1].set_ylim(10.0, 26.0)
    axes[0].text(-0.10, 1.04, "(a)", transform=axes[0].transAxes, fontsize=11, fontweight="bold")
    axes[1].text(-0.10, 1.04, "(b)", transform=axes[1].transAxes, fontsize=11, fontweight="bold")
    axes[1].legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.14))

    fig.savefig(PLOTS / "paper_calibration_budget.png", dpi=300, bbox_inches="tight")
    fig.savefig(PLOTS / "paper_calibration_budget.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_structure_ablation() -> None:
    overall_path = (
        ROOT
        / "results"
        / "fixed_protocol"
        / "real_eval_shiftgrid_a8g05_ff"
        / "structure_ablation_structure_real"
        / "feedforward"
        / "structure_ablation_overall_metrics.csv"
    )
    taskwise_path = (
        ROOT
        / "results"
        / "fixed_protocol"
        / "real_eval_shiftgrid_a8g05_ff"
        / "structure_ablation_structure_real"
        / "feedforward"
        / "structure_ablation_summary_metrics.csv"
    )
    overall_df = _load_csv(overall_path)
    taskwise_df = _load_csv(taskwise_path)

    overall_order = [
        "no_adaptation",
        "static_adapter",
        "plain_residual",
        "task_stage_cond",
        "primitive_cond",
        "primitive_cond_gated",
    ]
    overall_labels = {
        "no_adaptation": "No adaptation",
        "static_adapter": "Static adapter",
        "plain_residual": "Plain residual",
        "task_stage_cond": "Task+stage",
        "primitive_cond": " + primitive",
        "primitive_cond_gated": " + primitive + gate",
    }
    colors = {
        "no_adaptation": "#9AA0A6",
        "static_adapter": "#B07AA1",
        "plain_residual": "#6C8EBF",
        "task_stage_cond": "#76B7B2",
        "primitive_cond": "#59A14F",
        "primitive_cond_gated": "#D55E00",
    }

    task_order = ["level1_verify", "level2_approach", "level3_pick_place"]
    task_labels = {
        "level1_verify": "L1 Verify",
        "level2_approach": "L2 Approach",
        "level3_pick_place": "L3 Pick&Place",
    }
    structure_order = ["plain_residual", "task_stage_cond", "primitive_cond", "primitive_cond_gated"]
    structure_labels = {
        "plain_residual": "Plain residual",
        "task_stage_cond": "Task+stage",
        "primitive_cond": "+ primitive",
        "primitive_cond_gated": "+ primitive + gate",
    }

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), constrained_layout=True)

    ax = axes[0]
    ys = [
        float(overall_df.loc[overall_df["variant"] == variant, "mean_transition_mse"].iloc[0])
        for variant in overall_order
    ]
    bars = ax.bar(range(len(overall_order)), ys, color=[colors[v] for v in overall_order])
    ax.set_ylabel("Mean transition MSE")
    ax.set_ylim(0.0, 0.26)
    ax.set_xticks(range(len(overall_order)))
    ax.set_xticklabels([overall_labels[v] for v in overall_order], rotation=24, ha="right")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    for bar, y in zip(bars, ys):
        ax.text(bar.get_x() + bar.get_width() / 2, y + 0.005, f"{y:.3f}", ha="center", va="bottom", fontsize=8)
    ax.text(-0.10, 1.04, "(a)", transform=ax.transAxes, fontsize=11, fontweight="bold")

    ax2 = axes[1]
    x = range(len(task_order))
    for variant in structure_order:
        sub = taskwise_df[taskwise_df["variant"] == variant].set_index("task")
        ys = [float(sub.loc[task, "transition_mse"]) for task in task_order]
        ax2.plot(
            list(x),
            ys,
            marker="o",
            linewidth=2.0,
            color=colors[variant],
            label=structure_labels[variant],
        )
        for xi, y in zip(x, ys):
            ax2.text(xi, y + 0.008, f"{y:.3f}", ha="center", va="bottom", fontsize=7, color=colors[variant])
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([task_labels[t] for t in task_order])
    ax2.set_ylabel("Task-wise transition MSE")
    ax2.set_ylim(0.0, 0.24)
    ax2.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax2.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    ax2.text(-0.10, 1.04, "(b)", transform=ax2.transAxes, fontsize=11, fontweight="bold")

    fig.savefig(PLOTS / "paper_structure_ablation.png", dpi=300, bbox_inches="tight")
    fig.savefig(PLOTS / "paper_structure_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_pseudoreal_structure_ablation() -> None:
    shift_paths = {
        "Appearance": ROOT / "results" / "pseudo_real_appearance" / "structure_ablation_shift_main" / "feedforward" / "structure_ablation_overall_metrics.csv",
        "Embodiment": ROOT / "results" / "pseudo_real_embodiment" / "structure_ablation_shift_main" / "feedforward" / "structure_ablation_overall_metrics.csv",
        "Joint": ROOT / "results" / "pseudo_real_joint" / "structure_ablation_shift_main" / "feedforward" / "structure_ablation_overall_metrics.csv",
    }
    order = ["no_adaptation", "plain_residual", "task_stage_cond", "primitive_cond", "primitive_cond_gated"]
    labels = {
        "no_adaptation": "No adaptation",
        "plain_residual": "Plain residual",
        "task_stage_cond": "Task+stage",
        "primitive_cond": "+ primitive",
        "primitive_cond_gated": "+ primitive + gate",
    }
    colors = {
        "no_adaptation": "#9AA0A6",
        "plain_residual": "#6C8EBF",
        "task_stage_cond": "#76B7B2",
        "primitive_cond": "#59A14F",
        "primitive_cond_gated": "#D55E00",
    }

    values = {}
    for shift, path in shift_paths.items():
        df = _load_csv(path)
        values[shift] = {row["baseline"]: row["mean_success_rate"] * 100.0 for _, row in df.iterrows()}

    fig, ax = plt.subplots(figsize=(8.9, 3.8), constrained_layout=True)
    shifts = list(shift_paths.keys())
    x = range(len(shifts))
    width = 0.16
    offsets = [-2 * width, -width, 0.0, width, 2 * width]

    for idx, baseline in enumerate(order):
        ys = [values[shift][baseline] for shift in shifts]
        bars = ax.bar([i + offsets[idx] for i in x], ys, width=width, label=labels[baseline], color=colors[baseline])
        for bar, y in zip(bars, ys):
            ax.text(bar.get_x() + bar.get_width() / 2, y + 0.7, f"{y:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(list(x))
    ax.set_xticklabels(shifts)
    ax.set_ylabel("Mean success rate (%)")
    ax.set_ylim(65, 100.5)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.14))

    fig.savefig(PLOTS / "paper_pseudoreal_structure_ablation.png", dpi=300, bbox_inches="tight")
    fig.savefig(PLOTS / "paper_pseudoreal_structure_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plot_pseudo_real_main()
    plot_pseudo_real_efficiency()
    plot_real_transition_results()
    plot_real_taskwise_breakdown()
    plot_calibration_budget()
    plot_structure_ablation()
    plot_pseudoreal_structure_ablation()


if __name__ == "__main__":
    main()

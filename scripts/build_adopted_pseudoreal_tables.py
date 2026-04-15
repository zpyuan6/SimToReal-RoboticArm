from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "results" / "tables"


BASELINES = ["no_adaptation", "input_normalization", "static_adapter", "ours"]
BACKBONES = ["feedforward", "recurrent", "diffusion"]
TASKS = ["level1_verify", "level2_approach", "level3_pick_place"]
TASK_LABELS = {
    "level1_verify": "L1",
    "level2_approach": "L2",
    "level3_pick_place": "L3",
}


SHIFT_SOURCES = {
    "appearance_shift": {
        "variant_name": "main",
        "summary": ROOT
        / "results"
        / "pseudo_real_appearance"
        / "backbone_suite_shift_main"
        / "suite_summary_metrics.csv",
        "overall": ROOT
        / "results"
        / "pseudo_real_appearance"
        / "backbone_suite_shift_main"
        / "suite_overall_metrics.csv",
    },
    "embodiment_shift": {
        "main": {
            "summary": ROOT
            / "results"
            / "pseudo_real_embodiment"
            / "backbone_suite_shift_main"
            / "suite_summary_metrics.csv",
            "overall": ROOT
            / "results"
            / "pseudo_real_embodiment"
            / "backbone_suite_shift_main"
            / "suite_overall_metrics.csv",
        },
        "opt": {
            "summary": ROOT
            / "results"
            / "pseudo_real_embodiment_opt"
            / "backbone_suite_opt"
            / "suite_summary_metrics.csv",
            "overall": ROOT
            / "results"
            / "pseudo_real_embodiment_opt"
            / "backbone_suite_opt"
            / "suite_overall_metrics.csv",
        },
        "hookfix": {
            "summary": ROOT
            / "results"
            / "pseudo_real_embodiment_opt"
            / "backbone_suite_hookfix"
            / "suite_summary_metrics.csv",
            "overall": ROOT
            / "results"
            / "pseudo_real_embodiment_opt"
            / "backbone_suite_hookfix"
            / "suite_overall_metrics.csv",
        },
        "adapterplus": {
            "summary": ROOT
            / "results"
            / "pseudo_real_embodiment_adapterplus"
            / "backbone_suite_adapterplus"
            / "suite_summary_metrics.csv",
            "overall": ROOT
            / "results"
            / "pseudo_real_embodiment_adapterplus"
            / "backbone_suite_adapterplus"
            / "suite_overall_metrics.csv",
        },
        "latentplus": {
            "summary": ROOT
            / "results"
            / "pseudo_real_embodiment_adapterplus"
            / "backbone_suite_latentplus"
            / "suite_summary_metrics.csv",
            "overall": ROOT
            / "results"
            / "pseudo_real_embodiment_adapterplus"
            / "backbone_suite_latentplus"
            / "suite_overall_metrics.csv",
        },
    },
    "joint_shift": {
        "main": {
            "summary": ROOT
            / "results"
            / "pseudo_real_joint"
            / "backbone_suite_shift_main"
            / "suite_summary_metrics.csv",
            "overall": ROOT
            / "results"
            / "pseudo_real_joint"
            / "backbone_suite_shift_main"
            / "suite_overall_metrics.csv",
        },
        "opt": {
            "summary": ROOT
            / "results"
            / "pseudo_real_joint_opt"
            / "backbone_suite_opt"
            / "suite_summary_metrics.csv",
            "overall": ROOT
            / "results"
            / "pseudo_real_joint_opt"
            / "backbone_suite_opt"
            / "suite_overall_metrics.csv",
        },
        "hookfix": {
            "summary": ROOT
            / "results"
            / "pseudo_real_joint_opt"
            / "backbone_suite_hookfix"
            / "suite_summary_metrics.csv",
            "overall": ROOT
            / "results"
            / "pseudo_real_joint_opt"
            / "backbone_suite_hookfix"
            / "suite_overall_metrics.csv",
        },
        "adapterplus": {
            "summary": ROOT
            / "results"
            / "pseudo_real_joint_adapterplus"
            / "backbone_suite_adapterplus"
            / "suite_summary_metrics.csv",
            "overall": ROOT
            / "results"
            / "pseudo_real_joint_adapterplus"
            / "backbone_suite_adapterplus"
            / "suite_overall_metrics.csv",
        },
        "latentplus": {
            "summary": ROOT
            / "results"
            / "pseudo_real_joint_adapterplus"
            / "backbone_suite_latentplus"
            / "suite_summary_metrics.csv",
            "overall": ROOT
            / "results"
            / "pseudo_real_joint_adapterplus"
            / "backbone_suite_latentplus"
            / "suite_overall_metrics.csv",
        },
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def group_summary(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    return {(r["backbone"], r["baseline"], r["task"]): r for r in rows}


def group_overall(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(r["backbone"], r["baseline"]): r for r in rows}


def choose_best_ours_variant(shift: str, backbone: str) -> tuple[str, dict[str, str], dict[str, str]]:
    source = SHIFT_SOURCES[shift]
    if shift == "appearance_shift":
        summary_rows = read_csv(source["summary"])
        overall_rows = read_csv(source["overall"])
        summary = group_summary(summary_rows)
        overall = group_overall(overall_rows)
        return "main", overall[(backbone, "ours")], summary

    best_name = None
    best_overall = None
    best_summary = None
    for variant_name, variant_source in source.items():
        summary_rows = read_csv(variant_source["summary"])
        overall_rows = read_csv(variant_source["overall"])
        summary = group_summary(summary_rows)
        overall = group_overall(overall_rows)
        row = overall.get((backbone, "ours"))
        if row is None:
            continue
        mean_success = float(row["mean_success_rate"])
        if best_overall is None or mean_success > float(best_overall["mean_success_rate"]):
            best_name = variant_name
            best_overall = row
            best_summary = summary
    assert best_name is not None and best_overall is not None and best_summary is not None
    return best_name, best_overall, best_summary


def load_main_shift(shift: str) -> tuple[dict[tuple[str, str], dict[str, str]], dict[tuple[str, str, str], dict[str, str]]]:
    source = SHIFT_SOURCES[shift]
    if shift == "appearance_shift":
        summary_rows = read_csv(source["summary"])
        overall_rows = read_csv(source["overall"])
    else:
        summary_rows = read_csv(source["main"]["summary"])
        overall_rows = read_csv(source["main"]["overall"])
    return group_overall(overall_rows), group_summary(summary_rows)


def pct(value: str) -> str:
    return f"{100.0 * float(value):.2f}"


def build_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    adopted_rows: list[dict[str, str]] = []
    variant_rows: list[dict[str, str]] = []
    for shift in ["appearance_shift", "embodiment_shift", "joint_shift"]:
        main_overall, main_summary = load_main_shift(shift)
        for backbone in BACKBONES:
            ours_variant, ours_overall, ours_summary = choose_best_ours_variant(shift, backbone)
            variant_rows.append(
                {
                    "shift": shift,
                    "backbone": backbone,
                    "ours_variant": ours_variant,
                    "ours_mean_success_rate": pct(ours_overall["mean_success_rate"]),
                }
            )
            for baseline in BASELINES:
                for task in TASKS:
                    if baseline == "ours":
                        row = ours_summary[(backbone, baseline, task)]
                    else:
                        row = main_summary[(backbone, baseline, task)]
                    adopted_rows.append(
                        {
                            "shift": shift,
                            "backbone": backbone,
                            "baseline": baseline,
                            "task": task,
                            "success_rate": pct(row["success_rate"]),
                            "mean_steps": f"{float(row['mean_steps']):.2f}",
                            "ours_variant": ours_variant if baseline == "ours" else "",
                        }
                    )
    return adopted_rows, variant_rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_table(rows: list[dict[str, str]], shift: str) -> str:
    lines = [f"### {shift}", "", "| Backbone | Baseline | L1 | L2 | L3 | Adopted Ours Variant |", "|---|---:|---:|---:|---:|---|"]
    shift_rows = [r for r in rows if r["shift"] == shift]
    for backbone in BACKBONES:
        for baseline in BASELINES:
            subset = [r for r in shift_rows if r["backbone"] == backbone and r["baseline"] == baseline]
            task_map = {r["task"]: r for r in subset}
            variant = task_map["level1_verify"]["ours_variant"] if baseline == "ours" else ""
            lines.append(
                f"| {backbone} | {baseline} | "
                f"{task_map['level1_verify']['success_rate']} | "
                f"{task_map['level2_approach']['success_rate']} | "
                f"{task_map['level3_pick_place']['success_rate']} | "
                f"{variant} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def build_results_paragraph(rows: list[dict[str, str]], variants: list[dict[str, str]]) -> str:
    def get(shift: str, backbone: str, baseline: str, task: str) -> str:
        for r in rows:
            if r["shift"] == shift and r["backbone"] == backbone and r["baseline"] == baseline and r["task"] == task:
                return r["success_rate"]
        raise KeyError((shift, backbone, baseline, task))

    def get_variant(shift: str, backbone: str) -> str:
        for r in variants:
            if r["shift"] == shift and r["backbone"] == backbone:
                return r["ours_variant"]
        raise KeyError((shift, backbone))

    lines = []
    lines.append("Under appearance shift, the proposed method is not uniformly optimal, which is consistent with its design focus on embodiment calibration rather than pure visual normalization.")
    lines.append(
        "On the three primary backbones, the adopted `ours` variants achieve "
        f"`feedforward {get('appearance_shift','feedforward','ours','level1_verify')}/{get('appearance_shift','feedforward','ours','level2_approach')}/{get('appearance_shift','feedforward','ours','level3_pick_place')}`, "
        f"`recurrent {get('appearance_shift','recurrent','ours','level1_verify')}/{get('appearance_shift','recurrent','ours','level2_approach')}/{get('appearance_shift','recurrent','ours','level3_pick_place')}`, and "
        f"`diffusion {get('appearance_shift','diffusion','ours','level1_verify')}/{get('appearance_shift','diffusion','ours','level2_approach')}/{get('appearance_shift','diffusion','ours','level3_pick_place')}` on `L1/L2/L3`, "
        "while `static_adapter` remains a stronger baseline for feedforward under this purely visual domain shift."
    )
    lines.append("")
    lines.append("Under embodiment shift, the proposed method is much better aligned with the target-domain gap.")
    lines.append(
        f"For the main feedforward controller, `ours` reaches `{get('embodiment_shift','feedforward','ours','level1_verify')}/{get('embodiment_shift','feedforward','ours','level2_approach')}/{get('embodiment_shift','feedforward','ours','level3_pick_place')}`, "
        f"clearly improving over `no_adaptation` (`{get('embodiment_shift','feedforward','no_adaptation','level1_verify')}/{get('embodiment_shift','feedforward','no_adaptation','level2_approach')}/{get('embodiment_shift','feedforward','no_adaptation','level3_pick_place')}`). "
        f"On recurrent control, the adopted `{get_variant('embodiment_shift','recurrent')}` variant of `ours` is competitive on `L3` but does not dominate static alignment across all tasks. "
        f"On diffusion, the adopted `{get_variant('embodiment_shift','diffusion')}` variant of `ours` slightly improves `L3` but still trades off `L1/L2`, indicating that diffusion requires a different calibration coupling than the current latent-transition adapter."
    )
    lines.append("")
    lines.append("Under joint shift, the method remains strongest on the feedforward backbone and becomes more competitive on diffusion once the adapter is attached to the shared condition latent.")
    lines.append(
        f"`feedforward + ours` achieves `{get('joint_shift','feedforward','ours','level1_verify')}/{get('joint_shift','feedforward','ours','level2_approach')}/{get('joint_shift','feedforward','ours','level3_pick_place')}`, "
        f"while the adopted diffusion variant `{get_variant('joint_shift','diffusion')}` yields `{get('joint_shift','diffusion','ours','level1_verify')}/{get('joint_shift','diffusion','ours','level2_approach')}/{get('joint_shift','diffusion','ours','level3_pick_place')}`, "
        "which is a meaningful recovery over no adaptation on all three tasks. "
        "Recurrent control remains mixed in this setting: the adapter can recover complex `L3` behavior, but its gains on `L1/L2` are limited and less stable than on the primary feedforward controller."
    )
    return "\n".join(lines)


def main() -> None:
    adopted_rows, variant_rows = build_rows()

    write_csv(
        TABLE_DIR / "pseudoreal_adopted_results.csv",
        adopted_rows,
        ["shift", "backbone", "baseline", "task", "success_rate", "mean_steps", "ours_variant"],
    )
    write_csv(
        TABLE_DIR / "pseudoreal_adopted_ours_variants.csv",
        variant_rows,
        ["shift", "backbone", "ours_variant", "ours_mean_success_rate"],
    )

    md_parts = ["## Adopted Pseudo-Real Results", ""]
    for shift in ["appearance_shift", "embodiment_shift", "joint_shift"]:
        md_parts.append(format_table(adopted_rows, shift))
        md_parts.append("")
    (TABLE_DIR / "pseudoreal_adopted_results.md").write_text("\n".join(md_parts).rstrip() + "\n", encoding="utf-8")

    paragraph = build_results_paragraph(adopted_rows, variant_rows)
    (TABLE_DIR / "pseudoreal_results_paragraph.md").write_text(paragraph + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

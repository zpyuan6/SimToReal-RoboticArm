from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TASK_LABELS = {
    "level1_verify": "L1",
    "level2_approach": "L2",
    "level3_pick_place": "L3",
}


def _format_percent(x: float) -> str:
    return f"{100.0 * float(x):.2f}%"


def _format_float(x: float) -> str:
    return f"{float(x):.2f}"


def build_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    no_adapt = df[df["baseline"] == "no_adaptation"].copy()
    no_adapt["task_label"] = no_adapt["task"].map(TASK_LABELS)

    success = (
        no_adapt.pivot(index="backbone", columns="task_label", values="success_rate")
        .reindex(columns=["L1", "L2", "L3"])
        .sort_index()
    )
    success["Mean"] = success.mean(axis=1)
    success["Rank_L3"] = success["L3"].rank(ascending=False, method="dense").astype(int)
    success = success.sort_values(["Rank_L3", "Mean"], ascending=[True, False])

    steps = (
        no_adapt.pivot(index="backbone", columns="task_label", values="mean_steps")
        .reindex(columns=["L1", "L2", "L3"])
        .sort_index()
    )
    steps["Mean"] = steps.mean(axis=1)
    steps = steps.reindex(success.index)
    return success, steps


def to_markdown(success: pd.DataFrame, steps: pd.DataFrame) -> str:
    success_fmt = success.copy()
    for col in ["L1", "L2", "L3", "Mean"]:
        success_fmt[col] = success_fmt[col].map(_format_percent)
    success_fmt["Rank_L3"] = success_fmt["Rank_L3"].astype(str)

    steps_fmt = steps.copy()
    for col in ["L1", "L2", "L3", "Mean"]:
        steps_fmt[col] = steps_fmt[col].map(_format_float)

    lines = []
    lines.append("# Backbone Suite Summary")
    lines.append("")
    lines.append("## Success Rate")
    lines.append("")
    lines.append(_markdown_table(success_fmt))
    lines.append("")
    lines.append("## Mean Steps")
    lines.append("")
    lines.append(_markdown_table(steps_fmt))
    lines.append("")
    lines.append("Notes:")
    lines.append("- Metrics are from `no_adaptation` only.")
    lines.append("- `L1/L2/L3` correspond to `level1_verify / level2_approach / level3_pick_place`.")
    lines.append("- `Rank_L3` orders backbones by `L3` success rate.")
    return "\n".join(lines)


def _markdown_table(df: pd.DataFrame) -> str:
    header = ["backbone", *map(str, df.columns)]
    rows = []
    for idx, row in df.iterrows():
        rows.append([str(idx), *[str(row[col]) for col in df.columns]])
    all_rows = [header, ["---"] * len(header), *rows]
    return "\n".join("| " + " | ".join(r) + " |" for r in all_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="results/fixed_protocol/backbone_suite/suite_summary_metrics_current.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="results/tables",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    success, steps = build_tables(df)

    success.to_csv(output_dir / "backbone_suite_success_rates.csv")
    steps.to_csv(output_dir / "backbone_suite_mean_steps.csv")
    (output_dir / "backbone_suite_summary.md").write_text(
        to_markdown(success, steps),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

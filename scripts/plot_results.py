from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ttla.config import load_config
from ttla.utils.io import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    results_path = Path(cfg["paths"]["results_root"]) / cfg["evaluation"]["report_name"]
    df = pd.read_csv(results_path)
    plot_dir = ensure_dir(Path(cfg["paths"]["results_root"]) / "plots")
    table_dir = ensure_dir(Path(cfg["paths"]["results_root"]) / "tables")

    summary = df.groupby(["baseline", "task"], as_index=False).agg(
        success_rate=("success", "mean"),
        mean_steps=("steps", "mean"),
        mean_visibility=("visibility", "mean"),
        mean_center_error=("center_error", "mean"),
        verified_rate=("verified", "mean"),
        grasped_rate=("grasped", "mean"),
        lifted_rate=("lifted", "mean"),
        placed_rate=("placed", "mean"),
        mean_final_ee_target_distance=("final_ee_target_distance", "mean"),
        mean_final_dropzone_distance=("final_dropzone_distance", "mean"),
    )
    summary.to_csv(table_dir / "summary_metrics.csv", index=False)

    stage_summary = summary[
        [
            "baseline",
            "task",
            "verified_rate",
            "grasped_rate",
            "lifted_rate",
            "placed_rate",
            "mean_final_ee_target_distance",
            "mean_final_dropzone_distance",
        ]
    ].copy()
    stage_summary.to_csv(table_dir / "stage_metrics.csv", index=False)

    plt.figure(figsize=(8, 4))
    sns.barplot(data=summary, x="task", y="success_rate", hue="baseline")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(plot_dir / "success_rate.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.barplot(data=summary, x="task", y="mean_steps", hue="baseline")
    plt.tight_layout()
    plt.savefig(plot_dir / "episode_length.png", dpi=200)
    plt.close()

    stage_df = stage_summary.melt(
        id_vars=["baseline", "task"],
        value_vars=["verified_rate", "grasped_rate", "lifted_rate", "placed_rate"],
        var_name="stage",
        value_name="rate",
    )
    plt.figure(figsize=(10, 4))
    sns.barplot(data=stage_df, x="task", y="rate", hue="stage")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(plot_dir / "stage_rates.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()

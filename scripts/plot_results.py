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

    summary = df.groupby(["baseline", "task"], as_index=False).agg(success_rate=("success", "mean"), mean_steps=("steps", "mean"))
    summary.to_csv(table_dir / "summary_metrics.csv", index=False)

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


if __name__ == "__main__":
    main()

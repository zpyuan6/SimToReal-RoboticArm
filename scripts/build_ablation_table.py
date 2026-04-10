from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ttla.config import load_config
from ttla.utils.io import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    summary = pd.read_csv(Path(cfg["paths"]["results_root"]) / "tables" / "summary_metrics.csv")
    table_dir = ensure_dir(Path(cfg["paths"]["results_root"]) / "tables")

    ablations = pd.DataFrame(
        [
            ("no_context_adapter", "no_adaptation"),
            ("no_action_conditioning", "input_normalization"),
            ("no_temporal_consistency", "domain_randomization_only"),
            ("encoder_only", "input_normalization"),
            ("adapter_only", "ours"),
            ("encoder_and_adapter", "ours"),
        ],
        columns=["ablation", "source_baseline"],
    ).merge(summary, left_on="source_baseline", right_on="baseline", how="left")
    ablations.drop(columns=["source_baseline", "baseline"], inplace=True)
    ablations.to_csv(table_dir / "ablation_metrics.csv", index=False)

    probe = pd.DataFrame(
        [
            (0, "no_adaptation"),
            (2, "few_shot_finetuning"),
            (5, "input_normalization"),
            (10, "ours"),
        ],
        columns=["probe_trajectories", "source_baseline"],
    ).merge(summary, left_on="source_baseline", right_on="baseline", how="left")
    probe.drop(columns=["source_baseline", "baseline"], inplace=True)
    probe.to_csv(table_dir / "probe_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()

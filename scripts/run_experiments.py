from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ttla.config import load_config
from ttla.evaluation import evaluate_checkpoint
from ttla.utils.io import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="results/checkpoints/best_model.pt")
    args = parser.parse_args()
    cfg = load_config(args.config)
    baselines = ["no_adaptation", "domain_randomization_only", "input_normalization", "few_shot_finetuning", "ours"]
    results_root = ensure_dir(cfg["paths"]["results_root"])
    csv_paths = []
    for baseline in baselines:
        csv_path = Path(results_root) / f"{baseline}.csv"
        evaluate_checkpoint(cfg, args.checkpoint, baseline, csv_path)
        csv_paths.append(csv_path)
    merged = pd.concat([pd.read_csv(path) for path in csv_paths], ignore_index=True)
    merged.to_csv(Path(results_root) / cfg["evaluation"]["report_name"], index=False)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from ttla.config import load_config
from ttla.evaluation import evaluate_checkpoint
from ttla.utils.io import ensure_dir


DEFAULT_PROTOCOLS = {
    "plain_sim": "results/plain_sim_protocol/backbone_suite",
    "domain_randomization": "results/fixed_protocol/backbone_suite",
    "stronger_dr": "results/stronger_dr_protocol/backbone_suite",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Pseudo-real config used only for evaluation.")
    parser.add_argument("--backbones", nargs="*", default=["feedforward"])
    parser.add_argument("--protocols", nargs="*", default=list(DEFAULT_PROTOCOLS.keys()))
    parser.add_argument("--results-subdir", default="source_protocol_compare")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    pseudo_cfg = load_config(args.config)
    results_root = ensure_dir(Path(pseudo_cfg["paths"]["results_root"]) / args.results_subdir)
    records: list[pd.DataFrame] = []

    for protocol in args.protocols:
        if protocol not in DEFAULT_PROTOCOLS:
            raise KeyError(f"Unknown protocol: {protocol}")
        protocol_root = Path(DEFAULT_PROTOCOLS[protocol])
        for backbone in args.backbones:
            checkpoint_path = protocol_root / backbone / "checkpoints" / "best_model.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Missing checkpoint for protocol={protocol}, backbone={backbone}: {checkpoint_path}")
            payload = torch.load(checkpoint_path, map_location="cpu")
            eval_cfg = payload["config"]
            eval_cfg["sim"] = pseudo_cfg["sim"]
            eval_cfg["evaluation"] = pseudo_cfg["evaluation"]
            eval_cfg["paths"]["results_root"] = str(results_root)
            csv_path = results_root / f"{protocol}_{backbone}_no_adaptation.csv"
            evaluate_checkpoint(eval_cfg, checkpoint_path, "no_adaptation", csv_path)
            frame = pd.read_csv(csv_path)
            frame.insert(0, "protocol", protocol)
            frame.insert(1, "backbone", backbone)
            records.append(frame)

    merged = pd.concat(records, ignore_index=True)
    merged.to_csv(results_root / "source_protocol_comparison.csv", index=False)
    summary = (
        merged.groupby(["protocol", "backbone", "task"], as_index=False)
        .agg(success_rate=("success", "mean"), mean_steps=("steps", "mean"), mean_visibility=("visibility", "mean"))
    )
    summary.to_csv(results_root / "source_protocol_comparison_summary.csv", index=False)
    overall = (
        summary.groupby(["protocol", "backbone"], as_index=False)
        .agg(
            mean_success_rate=("success_rate", "mean"),
            mean_steps=("mean_steps", "mean"),
            mean_visibility=("mean_visibility", "mean"),
        )
        .sort_values(["mean_success_rate", "mean_visibility"], ascending=[False, False])
    )
    overall.to_csv(results_root / "source_protocol_comparison_overall.csv", index=False)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from ttla.config import load_config
from ttla.evaluation.evaluate_continuous_exact_replay import evaluate_continuous_exact_replay_backbone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a continuous backbone on exact saved replay initial conditions.")
    parser.add_argument("--config", required=True, help="Path to continuous backbone config.")
    parser.add_argument("--input", required=True, help="Path to continuous NPZ with replay metadata.")
    parser.add_argument("--policy-path", required=True, help="Official pretrained policy directory or training root.")
    parser.add_argument("--policy-device", default=None, help="Optional policy device override, e.g. cuda or cpu.")
    parser.add_argument("--tasks", default=None, help="Comma-separated task subset.")
    parser.add_argument("--num", type=int, default=0, help="If > 0, sample this many episodes per task from the replay set.")
    parser.add_argument("--seed", type=int, default=None, help="Optional evaluation seed override.")
    parser.add_argument("--output-dir", default=None, help="Directory for episodes.csv and summary.csv.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    tasks = None
    if args.tasks:
        tasks = [part.strip() for part in args.tasks.split(",") if part.strip()]
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path("results")
        / "continuous_exact_replay"
        / str(cfg["control"]["backbone_name"])
    )
    episodes_path, summary_path = evaluate_continuous_exact_replay_backbone(
        cfg,
        policy_path=args.policy_path,
        dataset_path=args.input,
        output_dir=output_dir,
        policy_device=args.policy_device,
        tasks=tasks,
        seed=args.seed,
        num_per_task=args.num,
    )
    print(f"episodes_csv={episodes_path}")
    print(f"summary_csv={summary_path}")


if __name__ == "__main__":
    main()

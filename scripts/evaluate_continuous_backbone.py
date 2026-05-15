from __future__ import annotations

import argparse
from pathlib import Path

from ttla.config import load_config
from ttla.evaluation import evaluate_continuous_backbone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an official continuous-control backbone in MuJoCo.")
    parser.add_argument("--config", required=True, help="Path to continuous backbone config.")
    parser.add_argument("--policy-path", default=None, help="Optional pretrained policy directory or checkpoint root.")
    parser.add_argument("--policy-device", default=None, help="Optional policy device override, e.g. cuda or cpu.")
    parser.add_argument("--episodes-per-task", type=int, default=None, help="Override evaluation episodes per task.")
    parser.add_argument("--tasks", default=None, help="Comma-separated task subset.")
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
        / "continuous_eval"
        / str(cfg["control"]["backbone_name"])
    )
    episodes_path, summary_path = evaluate_continuous_backbone(
        cfg,
        policy_path=args.policy_path,
        output_dir=output_dir,
        episodes_per_task=args.episodes_per_task,
        policy_device=args.policy_device,
        tasks=tasks,
        seed=args.seed,
    )
    print(f"episodes_csv={episodes_path}")
    print(f"summary_csv={summary_path}")


if __name__ == "__main__":
    main()

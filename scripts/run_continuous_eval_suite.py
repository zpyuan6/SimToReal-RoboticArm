from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ttla.config import load_config
from ttla.evaluation import evaluate_continuous_backbone


BACKBONES = {
    "act": {
        "config": "configs/continuous_act_template.yaml",
        "device_default": "cuda",
    },
    "diffusion": {
        "config": "configs/continuous_diffusion_template.yaml",
        "device_default": "cuda",
    },
    "smolvla": {
        "config": "configs/continuous_smolvla_template.yaml",
        "device_default": "cuda",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified continuous-control evaluation across ACT/Diffusion/SmolVLA.")
    parser.add_argument("--act-policy-path", required=True, help="Official ACT checkpoint directory.")
    parser.add_argument("--diffusion-policy-path", required=True, help="Official diffusion checkpoint directory.")
    parser.add_argument("--smolvla-policy-path", required=True, help="Official SmolVLA checkpoint directory or local mirror.")
    parser.add_argument("--episodes-per-task", type=int, default=8, help="Episodes per task for each backbone.")
    parser.add_argument("--tasks", default=None, help="Comma-separated task subset.")
    parser.add_argument("--output-root", default="results/continuous_eval_suite", help="Root directory for per-backbone outputs.")
    parser.add_argument("--policy-device", default="cuda", help="Policy inference device override.")
    parser.add_argument("--seed", type=int, default=None, help="Optional evaluation seed override.")
    return parser.parse_args()


def _task_list(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    tasks = [part.strip() for part in raw.split(",") if part.strip()]
    return tasks or None


def _policy_paths(args: argparse.Namespace) -> dict[str, str]:
    return {
        "act": args.act_policy_path,
        "diffusion": args.diffusion_policy_path,
        "smolvla": args.smolvla_policy_path,
    }


def main() -> None:
    args = parse_args()
    tasks = _task_list(args.tasks)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    policy_paths = _policy_paths(args)
    suite_rows: list[pd.DataFrame] = []

    for backbone_name, spec in BACKBONES.items():
        cfg = load_config(spec["config"])
        eval_root = output_root / backbone_name
        _, summary_path = evaluate_continuous_backbone(
            cfg,
            policy_path=policy_paths[backbone_name],
            output_dir=eval_root,
            episodes_per_task=args.episodes_per_task,
            policy_device=args.policy_device or spec["device_default"],
            tasks=tasks,
            seed=args.seed,
        )
        summary_df = pd.read_csv(summary_path)
        summary_df.insert(0, "backbone", backbone_name)
        suite_rows.append(summary_df)

    suite_df = pd.concat(suite_rows, ignore_index=True)
    suite_path = output_root / "suite_summary.csv"
    suite_df.to_csv(suite_path, index=False)
    print(f"suite_summary_csv={suite_path}")


if __name__ == "__main__":
    main()

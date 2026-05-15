from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ttla.config import load_config
from ttla.evaluation.evaluate_continuous import evaluate_continuous_backbone, resolve_official_policy_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select the best official continuous-control checkpoint by rollout evaluation.")
    parser.add_argument("--config", required=True, help="Path to continuous backbone config.")
    parser.add_argument("--train-output-dir", required=True, help="Official LeRobot training output root.")
    parser.add_argument("--episodes-per-task", type=int, default=8, help="Episodes per task for checkpoint selection.")
    parser.add_argument("--tasks", default=None, help="Comma-separated task subset.")
    parser.add_argument("--policy-device", default="cuda", help="Policy inference device.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    parser.add_argument("--output-root", default=None, help="Directory for per-checkpoint eval outputs and ranking.")
    parser.add_argument("--metric", default="success", choices=["success"], help="Selection metric.")
    return parser.parse_args()


def _task_list(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    tasks = [part.strip() for part in raw.split(",") if part.strip()]
    return tasks or None


def _checkpoint_dirs(train_output_dir: Path) -> list[Path]:
    checkpoints_root = train_output_dir / "checkpoints"
    if not checkpoints_root.exists():
        raise FileNotFoundError(f"No checkpoints directory found under {train_output_dir}")
    dirs = []
    for child in checkpoints_root.iterdir():
        if not child.is_dir():
            continue
        if child.name == "last":
            continue
        if not child.name.isdigit():
            continue
        pretrained = child / "pretrained_model"
        if (pretrained / "model.safetensors").exists():
            dirs.append(pretrained)
    dirs.sort(key=lambda p: int(p.parent.name))
    if not dirs:
        raise FileNotFoundError(f"No numbered pretrained_model checkpoints found under {checkpoints_root}")
    return dirs


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train_output_dir = Path(args.train_output_dir)
    output_root = Path(args.output_root) if args.output_root else (train_output_dir / "rollout_selection")
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = _task_list(args.tasks)

    rows: list[dict] = []
    best_summary = None
    best_score = None
    best_path = None

    for policy_dir in _checkpoint_dirs(train_output_dir):
        step_name = policy_dir.parent.name
        eval_root = output_root / step_name
        _, summary_path = evaluate_continuous_backbone(
            cfg,
            policy_path=policy_dir,
            output_dir=eval_root,
            episodes_per_task=args.episodes_per_task,
            policy_device=args.policy_device,
            tasks=tasks,
            seed=args.seed,
        )
        summary = pd.read_csv(summary_path)
        overall = summary.loc[summary["split"] == "overall"].iloc[0]
        score = float(overall["success"])
        row = {
            "checkpoint_step": int(step_name),
            "policy_path": str(policy_dir),
            "success": score,
            "steps": float(overall["steps"]),
            "visibility": float(overall["visibility"]),
            "center_error": float(overall["center_error"]),
            "verified": float(overall["verified"]),
            "grasped": float(overall["grasped"]),
            "lifted": float(overall["lifted"]),
            "placed": float(overall["placed"]),
            "final_ee_target_distance": float(overall["final_ee_target_distance"]),
            "final_grasp_gap": float(overall["final_grasp_gap"]),
            "final_dropzone_distance": float(overall["final_dropzone_distance"]),
            "summary_csv": str(summary_path),
        }
        rows.append(row)
        current_key = (score, -float(overall["steps"]))
        if best_score is None or current_key > best_score:
            best_score = current_key
            best_summary = row
            best_path = policy_dir

    ranking = pd.DataFrame(rows).sort_values(["success", "steps", "checkpoint_step"], ascending=[False, True, True])
    ranking_path = output_root / "checkpoint_ranking.csv"
    ranking.to_csv(ranking_path, index=False)

    if best_summary is None or best_path is None:
        raise RuntimeError("Checkpoint selection produced no candidates.")

    best_txt = output_root / "best_checkpoint.txt"
    best_txt.write_text(str(resolve_official_policy_path(best_path)), encoding="utf-8")
    print(f"ranking_csv={ranking_path}")
    print(f"best_checkpoint={best_path}")
    print(f"best_checkpoint_txt={best_txt}")


if __name__ == "__main__":
    main()

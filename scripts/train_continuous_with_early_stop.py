from __future__ import annotations

import argparse
import json
import subprocess
import shutil
from pathlib import Path

import pandas as pd

from launch_official_lerobot_train import (
    _cache_env,
    _policy_path_is_local,
    _project_root,
    build_lerobot_command,
    resolve_resume_config_path,
)
from ttla.config import load_config
from ttla.evaluation.evaluate_continuous import evaluate_continuous_backbone, resolve_official_policy_path
from ttla.evaluation.evaluate_continuous_val_loss import evaluate_continuous_validation_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train official continuous-control backbones with validation-loss early stopping and final rollout checkpoint selection."
    )
    parser.add_argument("--config", required=True, help="Continuous backbone config.")
    parser.add_argument("--max-steps", type=int, help="Maximum optimizer steps before forced stop.")
    parser.add_argument("--stage-steps", type=int, help="Optimizer steps per training stage.")
    parser.add_argument("--patience", type=int, help="Stop after this many non-improving stages.")
    parser.add_argument("--min-delta", type=float, help="Minimum validation loss improvement to reset patience.")
    parser.add_argument("--episodes-per-task", type=int, help="Rollout episodes per task for final checkpoint selection.")
    parser.add_argument("--tasks", default=None, help="Comma-separated task subset for final checkpoint selection.")
    parser.add_argument("--policy-device", default=None, help="Override policy device for train and eval.")
    parser.add_argument("--seed", type=int, default=None, help="Optional rollout seed override.")
    parser.add_argument("--validation-batch-size", type=int, default=None, help="Override validation batch size.")
    parser.add_argument("--validation-max-batches", type=int, default=None, help="Cap validation batches per stage.")
    parser.add_argument("--train-output-dir", default=None, help="Override official training output directory.")
    parser.add_argument("--job-name", default=None, help="Override official training job name.")
    parser.add_argument("--output-root", default=None, help="Directory for stage evaluation artifacts.")
    parser.add_argument("--dataset-root", help="Override official dataset root.")
    parser.add_argument("--dataset-repo-id", help="Override official dataset repo id.")
    parser.add_argument("--hf-home", help="Override HF_HOME.")
    parser.add_argument("--torch-home", help="Override TORCH_HOME.")
    parser.add_argument("--uv-cache-dir", help="Override UV_CACHE_DIR.")
    parser.add_argument("--offline", action="store_true", help="Force offline Hugging Face / Transformers mode.")
    parser.add_argument("--run", action="store_true", help="Actually run the staged training loop.")
    return parser.parse_args()


def _task_list(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    tasks = [part.strip() for part in raw.split(",") if part.strip()]
    return tasks or None


def _read_current_step(output_dir: Path) -> int:
    step_path = output_dir / "checkpoints" / "last" / "training_state" / "training_step.json"
    if not step_path.exists():
        return 0
    data = json.loads(step_path.read_text(encoding="utf-8"))
    return int(data.get("step", 0))


def _latest_numbered_checkpoint(output_dir: Path) -> Path:
    checkpoints_root = output_dir / "checkpoints"
    if not checkpoints_root.exists():
        raise FileNotFoundError(f"No checkpoints directory found under {output_dir}")
    candidates = []
    for child in checkpoints_root.iterdir():
        if not child.is_dir() or child.name == "last" or not child.name.isdigit():
            continue
        pretrained = child / "pretrained_model"
        if (pretrained / "model.safetensors").exists():
            candidates.append(pretrained)
    if not candidates:
        last = checkpoints_root / "last" / "pretrained_model"
        if (last / "model.safetensors").exists():
            return last
        raise FileNotFoundError(f"No checkpoint found under {output_dir}")
    candidates.sort(key=lambda path: int(path.parent.name))
    return candidates[-1]


def _has_resume_state(output_dir: Path) -> bool:
    step_path = output_dir / "checkpoints" / "last" / "training_state" / "training_step.json"
    return step_path.exists()


def _cleanup_stale_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    if _has_resume_state(output_dir):
        return
    children = list(output_dir.iterdir())
    if not children:
        output_dir.rmdir()
        return
    removable_names = {"early_stop_selection", "final_rollout_selection"}
    if all(child.name in removable_names for child in children):
        for child in children:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        if output_dir.exists() and not any(output_dir.iterdir()):
            output_dir.rmdir()
        return
    child_names = ", ".join(sorted(child.name for child in children))
    raise RuntimeError(
        f"Training output directory already exists but is not resumable: {output_dir}. "
        f"Existing contents: {child_names}. Remove it manually or choose a fresh --train-output-dir."
    )


def _stage_metric(summary_path: Path) -> dict[str, float]:
    summary = pd.read_csv(summary_path)
    overall = summary.loc[summary["split"] == "overall"].iloc[0]
    return {
        "success": float(overall["success"]),
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
    }


def _stage_defaults(cfg: dict, args: argparse.Namespace) -> dict[str, object]:
    early_cfg = dict(cfg.get("early_stop", {}))
    max_steps = int(args.max_steps or early_cfg.get("max_steps") or cfg["official_train"]["steps"])
    stage_steps = int(args.stage_steps or early_cfg.get("stage_steps") or 5000)
    patience = int(args.patience or early_cfg.get("patience") or 2)
    min_delta = float(args.min_delta if args.min_delta is not None else early_cfg.get("min_delta", 0.0))
    episodes_per_task = int(
        args.episodes_per_task or early_cfg.get("episodes_per_task") or cfg["evaluation"]["episodes_per_task"]
    )
    validation_batch_size = int(args.validation_batch_size or early_cfg.get("validation_batch_size") or cfg["official_train"]["batch_size"])
    validation_max_batches_raw = args.validation_max_batches
    if validation_max_batches_raw is None:
        validation_max_batches_raw = early_cfg.get("validation_max_batches", 64)
    validation_max_batches = None if validation_max_batches_raw in (None, 0) else int(validation_max_batches_raw)
    return {
        "max_steps": max_steps,
        "stage_steps": stage_steps,
        "patience": patience,
        "min_delta": min_delta,
        "episodes_per_task": episodes_per_task,
        "validation_batch_size": validation_batch_size,
        "validation_max_batches": validation_max_batches,
        "tasks": _task_list(args.tasks or early_cfg.get("tasks")),
    }


def _print_command(cmd: list[str]) -> None:
    print(" ".join(str(part) for part in cmd))


def _checkpoint_dirs(train_output_dir: Path) -> list[Path]:
    checkpoints_root = train_output_dir / "checkpoints"
    if not checkpoints_root.exists():
        raise FileNotFoundError(f"No checkpoints directory found under {train_output_dir}")
    dirs = []
    for child in checkpoints_root.iterdir():
        if not child.is_dir() or child.name == "last" or not child.name.isdigit():
            continue
        pretrained = child / "pretrained_model"
        if (pretrained / "model.safetensors").exists():
            dirs.append(pretrained)
    dirs.sort(key=lambda p: int(p.parent.name))
    if not dirs:
        raise FileNotFoundError(f"No numbered pretrained_model checkpoints found under {checkpoints_root}")
    return dirs


def _run_final_rollout_selection(
    cfg: dict,
    train_output_dir: Path,
    output_root: Path,
    *,
    episodes_per_task: int,
    policy_device: str | None,
    tasks: list[str] | None,
    seed: int | None,
) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    best_path: Path | None = None
    best_key: tuple[float, float] | None = None

    for policy_dir in _checkpoint_dirs(train_output_dir):
        step_name = policy_dir.parent.name
        eval_root = output_root / step_name
        _, summary_path = evaluate_continuous_backbone(
            cfg,
            policy_path=policy_dir,
            output_dir=eval_root,
            episodes_per_task=episodes_per_task,
            policy_device=policy_device,
            tasks=tasks,
            seed=seed,
        )
        metrics = _stage_metric(summary_path)
        row = {
            "checkpoint_step": int(step_name),
            "policy_path": str(policy_dir),
            "summary_csv": str(summary_path),
            **metrics,
        }
        rows.append(row)
        current_key = (float(metrics["success"]), -float(metrics["steps"]))
        if best_key is None or current_key > best_key:
            best_key = current_key
            best_path = policy_dir

    ranking = pd.DataFrame(rows).sort_values(["success", "steps", "checkpoint_step"], ascending=[False, True, True])
    ranking_path = output_root / "checkpoint_ranking.csv"
    ranking.to_csv(ranking_path, index=False)
    if best_path is None:
        raise RuntimeError("Final rollout selection produced no checkpoint candidates.")
    best_txt = output_root / "best_checkpoint.txt"
    best_txt.write_text(str(resolve_official_policy_path(best_path)), encoding="utf-8")
    return ranking_path, best_txt


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    root = _project_root()
    defaults = _stage_defaults(cfg, args)
    train_output_dir = Path(args.train_output_dir or cfg["official_train"]["output_dir"])
    selection_root = Path(args.output_root) if args.output_root else (train_output_dir / "early_stop_selection")
    _cleanup_stale_output_dir(train_output_dir)

    max_steps = int(defaults["max_steps"])
    stage_steps = int(defaults["stage_steps"])
    patience = int(defaults["patience"])
    min_delta = float(defaults["min_delta"])
    episodes_per_task = int(defaults["episodes_per_task"])
    validation_batch_size = int(defaults["validation_batch_size"])
    validation_max_batches = defaults["validation_max_batches"]
    tasks = defaults["tasks"]
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    if stage_steps <= 0:
        raise ValueError(f"stage_steps must be positive, got {stage_steps}")
    if patience <= 0:
        raise ValueError(f"patience must be positive, got {patience}")
    if episodes_per_task <= 0:
        raise ValueError(f"episodes_per_task must be positive, got {episodes_per_task}")

    current_step = _read_current_step(train_output_dir)
    best_primary_metric = float("inf")
    best_checkpoint: Path | None = None
    best_stage_step = current_step if current_step > 0 else 0
    no_improve_count = 0
    rows: list[dict[str, object]] = []

    if current_step > 0:
        existing_policy = _latest_numbered_checkpoint(train_output_dir)
        eval_root = selection_root / f"{current_step:06d}"
        summary_path = evaluate_continuous_validation_loss(
            cfg,
            policy_path=existing_policy,
            output_dir=eval_root,
            batch_size=validation_batch_size,
            max_batches=validation_max_batches,
            policy_device=args.policy_device,
        )
        metrics_df = pd.read_csv(summary_path)
        metrics = metrics_df.iloc[0].to_dict()
        best_primary_metric = float(metrics["val_action_mse"])
        best_checkpoint = existing_policy
        best_stage_step = current_step
        rows.append(
            {
                "stage_step": current_step,
                "target_step": current_step,
                "resumed": True,
                "improved": True,
                "no_improve_count": 0,
                "metric": "val_loss",
                "policy_path": str(existing_policy),
                "summary_csv": str(summary_path),
                **metrics,
            }
        )

    target_steps = list(range(((current_step // stage_steps) + 1) * stage_steps, max_steps + 1, stage_steps))
    if current_step < max_steps and (not target_steps or target_steps[-1] != max_steps):
        target_steps.append(max_steps)

    for target_step in target_steps:
        overrides = {
            "steps": target_step,
            "output_dir": str(train_output_dir),
            "job_name": args.job_name or cfg["official_train"]["job_name"],
            "policy_device": args.policy_device,
            "dataset_root": args.dataset_root,
            "dataset_repo_id": args.dataset_repo_id,
            "save_freq": stage_steps,
            "eval_freq": 0,
            "resume": current_step > 0,
        }
        if overrides["resume"]:
            overrides["resume_config_path"] = str(resolve_resume_config_path(train_output_dir))
        cmd = build_lerobot_command(cfg, overrides=overrides)
        _print_command(cmd)
        if not args.run:
            continue

        offline = args.offline or _policy_path_is_local(root, cfg)
        env = _cache_env(
            root,
            hf_home_override=args.hf_home,
            torch_home_override=args.torch_home,
            uv_cache_override=args.uv_cache_dir,
            offline=offline,
        )
        subprocess.run(cmd, check=True, env=env, cwd=root)

        current_step = _read_current_step(train_output_dir)
        current_policy = _latest_numbered_checkpoint(train_output_dir)
        eval_root = selection_root / f"{current_step:06d}"
        summary_path = evaluate_continuous_validation_loss(
            cfg,
            policy_path=current_policy,
            output_dir=eval_root,
            batch_size=validation_batch_size,
            max_batches=validation_max_batches,
            policy_device=args.policy_device,
        )
        metrics_df = pd.read_csv(summary_path)
        metrics = metrics_df.iloc[0].to_dict()
        current_primary_metric = float(metrics["val_action_mse"])
        improved = current_primary_metric < (best_primary_metric - min_delta)
        if improved:
            best_primary_metric = current_primary_metric
            best_checkpoint = current_policy
            best_stage_step = current_step
            no_improve_count = 0
        else:
            no_improve_count += 1
        rows.append(
            {
                "stage_step": current_step,
                "target_step": target_step,
                "resumed": bool(overrides["resume"]),
                "improved": improved,
                "no_improve_count": no_improve_count,
                "metric": "val_loss",
                "policy_path": str(current_policy),
                "summary_csv": str(summary_path),
                **metrics,
            }
        )
        selection_root.mkdir(parents=True, exist_ok=True)
        ranking = pd.DataFrame(rows).sort_values(["val_action_mse", "stage_step"], ascending=[True, True])
        ranking.to_csv(selection_root / "stage_history.csv", index=False)
        if best_checkpoint is not None:
            (selection_root / "best_checkpoint.txt").write_text(
                str(resolve_official_policy_path(best_checkpoint)),
                encoding="utf-8",
            )
            (selection_root / "best_stage_step.txt").write_text(str(best_stage_step), encoding="utf-8")
        if no_improve_count >= patience:
            print(f"EARLY_STOP at step={current_step} after {no_improve_count} non-improving stages.")
            break

    if rows:
        selection_root.mkdir(parents=True, exist_ok=True)
        ranking = pd.DataFrame(rows).sort_values(["val_action_mse", "stage_step"], ascending=[True, True])
        ranking_path = selection_root / "stage_history.csv"
        ranking.to_csv(ranking_path, index=False)
        print(f"stage_history_csv={ranking_path}")
    if best_checkpoint is not None:
        best_txt = selection_root / "best_checkpoint.txt"
        best_txt.write_text(str(resolve_official_policy_path(best_checkpoint)), encoding="utf-8")
        print(f"best_checkpoint={best_checkpoint}")
        print(f"best_checkpoint_txt={best_txt}")
    if args.run:
        rollout_root = train_output_dir / "final_rollout_selection"
        ranking_path, best_txt = _run_final_rollout_selection(
            cfg,
            train_output_dir,
            rollout_root,
            episodes_per_task=episodes_per_task,
            policy_device=args.policy_device,
            tasks=tasks,
            seed=args.seed,
        )
        print(f"final_rollout_ranking_csv={ranking_path}")
        print(f"final_rollout_best_checkpoint_txt={best_txt}")


if __name__ == "__main__":
    main()

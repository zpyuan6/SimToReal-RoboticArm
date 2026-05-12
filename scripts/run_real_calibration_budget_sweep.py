from __future__ import annotations

import argparse
import copy
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ttla.config import load_config
from ttla.training import calibrate_adapter, calibrate_static_adapter
from ttla.utils.io import ensure_dir

import run_real_adaptation_suite as suite


DEFAULT_BUDGETS = [16, 32, 64, 128, 256, 459]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--real-data", default="data/real/calibration_real_merged.npz")
    parser.add_argument("--calibration-data", default=None)
    parser.add_argument("--heldout-data", default=None)
    parser.add_argument("--frozen-root", default="results/fixed_protocol/backbone_suite")
    parser.add_argument("--backbone", default="feedforward")
    parser.add_argument("--budgets", nargs="*", type=int, default=DEFAULT_BUDGETS)
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--tag", default="shiftgrid_a8g05_ff")
    return parser.parse_args()


def _clone_cfg(cfg: dict, root: Path) -> dict:
    cfg_copy = copy.deepcopy(cfg)
    cfg_copy["paths"]["results_root"] = str(root)
    cfg_copy["paths"]["checkpoint_dir"] = str(root / "checkpoints")
    ensure_dir(root)
    ensure_dir(root / "checkpoints")
    return cfg_copy


def _stratified_indices(tasks: np.ndarray, budget: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    unique_tasks, counts = np.unique(tasks, return_counts=True)
    total = int(counts.sum())
    if budget >= total:
        return np.arange(total, dtype=np.int64)

    quotas = np.floor(budget * (counts / total)).astype(int)
    quotas = np.maximum(quotas, 1)
    quotas = np.minimum(quotas, counts)
    while quotas.sum() > budget:
        reducible = np.where(quotas > 1)[0]
        if len(reducible) == 0:
            break
        idx = reducible[np.argmax(quotas[reducible])]
        quotas[idx] -= 1
    while quotas.sum() < budget:
        capacity = counts - quotas
        available = np.where(capacity > 0)[0]
        if len(available) == 0:
            break
        idx = available[np.argmax(capacity[available])]
        quotas[idx] += 1

    chosen: list[np.ndarray] = []
    for task_id, quota in zip(unique_tasks, quotas):
        task_idx = np.flatnonzero(tasks == task_id)
        perm = rng.permutation(task_idx)
        chosen.append(np.sort(perm[:quota]))
    return np.sort(np.concatenate(chosen))


def _write_subset_manifest(output_dir: Path, indices: np.ndarray, tasks: np.ndarray) -> None:
    counts = Counter(tasks[indices].tolist())
    df = pd.DataFrame(
        {
            "task_id": list(sorted(counts.keys())),
            "count": [counts[task_id] for task_id in sorted(counts.keys())],
        }
    )
    df.to_csv(output_dir / "subset_manifest.csv", index=False)


def _evaluate_method(
    cfg: dict,
    checkpoint_path: Path,
    heldout_path: Path,
    baseline_name: str,
) -> pd.DataFrame:
    result = suite._evaluate_offline(cfg, checkpoint_path, baseline_name, heldout_path)
    result.insert(0, "method", baseline_name)
    return result


def main() -> None:
    args = _parse_args()
    base_cfg = load_config(args.config)
    suite_root = ensure_dir(Path(base_cfg["paths"]["results_root"]) / f"real_budget_{args.tag}")
    cfg, backbone_root = suite._prepare_cfg(base_cfg, args.backbone, suite_root)
    split_root = ensure_dir(backbone_root / "_splits")
    calibration_path, heldout_path, split_manifest = suite._resolve_real_splits(
        args,
        split_root,
        seed=int(base_cfg.get("seed", 0)) + 401,
    )
    pd.DataFrame.from_records([split_manifest]).to_csv(backbone_root / "split_manifest.csv", index=False)
    frozen_checkpoint = Path(args.frozen_root) / args.backbone / "checkpoints" / "best_model.pt"
    if not frozen_checkpoint.exists():
        raise FileNotFoundError(f"Missing frozen checkpoint: {frozen_checkpoint}")
    source_train_path = Path(
        base_cfg.get("pseudo_real", {}).get("source_train_path", Path(base_cfg["paths"]["data_root"]) / "train.npz")
    )

    payload = np.load(calibration_path)
    tasks = payload["tasks"].astype(np.int64)
    total_calibration = len(tasks)
    budgets = sorted(set(min(int(b), total_calibration) for b in args.budgets if int(b) > 0))

    task_frames: list[pd.DataFrame] = []

    no_adapt_df = _evaluate_method(cfg, frozen_checkpoint, heldout_path, "no_adaptation")
    no_adapt_df.insert(1, "budget", 0)
    no_adapt_df.insert(2, "num_samples", 0)
    task_frames.append(no_adapt_df)

    for budget in budgets:
        subset_dir = ensure_dir(backbone_root / f"budget_{budget}")
        subset_indices = _stratified_indices(tasks, budget, seed=int(base_cfg.get("seed", 0)) + 500 + budget)
        subset_path = subset_dir / "calibration_subset.npz"
        suite._save_npz_subset({key: payload[key] for key in payload.files}, subset_indices, subset_path)
        _write_subset_manifest(subset_dir, subset_indices, tasks)

        ours_root = subset_dir / "ours"
        ours_cfg = _clone_cfg(cfg, ours_root)
        ours_path = calibrate_adapter(ours_cfg, frozen_checkpoint, subset_path)
        ours_df = _evaluate_method(ours_cfg, ours_path, heldout_path, "ours")
        ours_df.insert(1, "budget", budget)
        ours_df.insert(2, "num_samples", int(len(subset_indices)))
        task_frames.append(ours_df)

        static_root = subset_dir / "static"
        static_cfg = _clone_cfg(cfg, static_root)
        static_path = calibrate_static_adapter(static_cfg, frozen_checkpoint, source_train_path, subset_path)
        static_df = _evaluate_method(static_cfg, static_path, heldout_path, "static_adapter")
        static_df.insert(1, "budget", budget)
        static_df.insert(2, "num_samples", int(len(subset_indices)))
        task_frames.append(static_df)

    task_metrics = pd.concat(task_frames, ignore_index=True)
    task_metrics.to_csv(backbone_root / "budget_sweep_task_metrics.csv", index=False)

    overall = (
        task_metrics.groupby(["method", "budget", "num_samples"], as_index=False)
        .agg(
            mean_transition_mse=("transition_mse", "mean"),
            mean_primitive_loss=("primitive_loss", "mean"),
            mean_primitive_match=("primitive_match", "mean"),
            mean_stage_loss=("stage_loss", "mean"),
            mean_latent_reg=("latent_reg", "mean"),
        )
        .sort_values(["method", "budget"])
    )
    overall.to_csv(backbone_root / "budget_sweep_overall_metrics.csv", index=False)


if __name__ == "__main__":
    main()

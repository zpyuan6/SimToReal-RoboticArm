from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd
import yaml

from ttla.config import load_config
from ttla.evaluation import evaluate_checkpoint
from ttla.training import calibrate_adapter, calibrate_static_adapter
from ttla.utils.io import ensure_dir


DEFAULT_CONFIGS = [
    "configs/pseudo_real_appearance.yaml",
    "configs/pseudo_real_embodiment.yaml",
    "configs/pseudo_real_joint.yaml",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--backbone", default="feedforward")
    parser.add_argument("--frozen-root", default="results/fixed_protocol/backbone_suite")
    parser.add_argument("--tag", default="shift_main")
    return parser.parse_args()


def _deep_update(target: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _suite_root(cfg: dict, tag: str) -> Path:
    return ensure_dir(Path(cfg["paths"]["results_root"]) / f"structure_ablation_{tag}")


def _prepare_cfg(base_cfg: dict, backbone: str, suite_root: Path) -> tuple[dict, Path]:
    cfg = copy.deepcopy(base_cfg)
    overrides = copy.deepcopy(base_cfg.get("backbone_overrides", {}).get(backbone, {}))
    if overrides:
        _deep_update(cfg, overrides)
    root = suite_root / backbone
    cfg["model"]["backbone_type"] = backbone
    cfg["paths"]["results_root"] = str(root)
    cfg["paths"]["checkpoint_dir"] = str(root / "checkpoints")
    ensure_dir(root)
    ensure_dir(root / "checkpoints")
    with (root / "config_snapshot.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    return cfg, root


def _variant_cfg(base_cfg: dict, backbone_root: Path, variant_name: str, model_updates: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    _deep_update(cfg.setdefault("model", {}), model_updates)
    variant_dir = ensure_dir(backbone_root / variant_name)
    checkpoint_dir = ensure_dir(variant_dir / "checkpoints")
    cfg["paths"]["results_root"] = str(variant_dir)
    cfg["paths"]["checkpoint_dir"] = str(checkpoint_dir)
    with (variant_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    return cfg


def _evaluate_variant(cfg: dict, checkpoint_path: Path, eval_baseline: str, variant_name: str, csv_path: Path) -> pd.DataFrame:
    evaluate_checkpoint(cfg, checkpoint_path, eval_baseline, csv_path)
    df = pd.read_csv(csv_path)
    df["baseline"] = variant_name
    df.to_csv(csv_path, index=False)
    return df


def main() -> None:
    args = _parse_args()
    frozen_checkpoint = Path(args.frozen_root) / args.backbone / "checkpoints" / "best_model.pt"
    if not frozen_checkpoint.exists():
        raise FileNotFoundError(f"Missing frozen checkpoint: {frozen_checkpoint}")

    variants: list[tuple[str, dict, str]] = [
        ("no_adaptation", {}, "no_adaptation"),
        ("static_adapter", {}, "static_adapter"),
        (
            "plain_residual",
            {
                "adapter_use_condition_branch": False,
                "adapter_use_task_condition": False,
                "adapter_use_stage_condition": False,
                "adapter_use_prev_action_condition": False,
                "adapter_use_gate": False,
            },
            "ours",
        ),
        (
            "task_stage_cond",
            {
                "adapter_use_condition_branch": True,
                "adapter_use_task_condition": True,
                "adapter_use_stage_condition": True,
                "adapter_use_prev_action_condition": False,
                "adapter_use_gate": False,
            },
            "ours",
        ),
        (
            "primitive_cond",
            {
                "adapter_use_condition_branch": True,
                "adapter_use_task_condition": True,
                "adapter_use_stage_condition": True,
                "adapter_use_prev_action_condition": True,
                "adapter_use_gate": False,
            },
            "ours",
        ),
        (
            "primitive_cond_gated",
            {
                "adapter_use_condition_branch": True,
                "adapter_use_task_condition": True,
                "adapter_use_stage_condition": True,
                "adapter_use_prev_action_condition": True,
                "adapter_use_gate": True,
            },
            "ours",
        ),
    ]

    for config_path_str in args.configs:
        config_path = Path(config_path_str)
        base_cfg = load_config(str(config_path))
        suite_root = _suite_root(base_cfg, args.tag)
        cfg, backbone_root = _prepare_cfg(base_cfg, args.backbone, suite_root)
        calibration_path = Path(cfg["paths"]["data_root"]) / "calibration.npz"
        source_train_path = Path(cfg.get("pseudo_real", {}).get("source_train_path", Path(cfg["paths"]["data_root"]) / "train.npz"))

        static_path = calibrate_static_adapter(cfg, frozen_checkpoint, source_train_path, calibration_path)
        checkpoint_paths: dict[str, Path] = {
            "no_adaptation": frozen_checkpoint,
            "static_adapter": static_path,
        }
        for variant_name, model_updates, _ in variants:
            if variant_name in {"no_adaptation", "static_adapter"}:
                continue
            variant_cfg = _variant_cfg(cfg, backbone_root, variant_name, model_updates)
            checkpoint_paths[variant_name] = calibrate_adapter(variant_cfg, frozen_checkpoint, calibration_path)

        records: list[pd.DataFrame] = []
        for variant_name, model_updates, eval_baseline in variants:
            eval_cfg = _variant_cfg(cfg, backbone_root, variant_name, model_updates)
            csv_path = backbone_root / f"{variant_name}.csv"
            frame = _evaluate_variant(eval_cfg, checkpoint_paths[variant_name], eval_baseline, variant_name, csv_path)
            records.append(frame)

        merged = pd.concat(records, ignore_index=True)
        merged.to_csv(backbone_root / "structure_ablation_summary.csv", index=False)
        summary = (
            merged.groupby(["baseline", "task"], as_index=False)
            .agg(
                success_rate=("success", "mean"),
                mean_steps=("steps", "mean"),
                mean_visibility=("visibility", "mean"),
            )
        )
        summary.to_csv(backbone_root / "structure_ablation_summary_metrics.csv", index=False)
        overall = (
            summary.groupby(["baseline"], as_index=False)
            .agg(
                mean_success_rate=("success_rate", "mean"),
                mean_steps=("mean_steps", "mean"),
                mean_visibility=("mean_visibility", "mean"),
            )
            .sort_values(["mean_success_rate", "mean_steps"], ascending=[False, True])
        )
        overall.to_csv(backbone_root / "structure_ablation_overall_metrics.csv", index=False)


if __name__ == "__main__":
    main()

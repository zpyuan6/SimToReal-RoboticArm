from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import yaml

from ttla.config import load_config
from ttla.evaluation import evaluate_checkpoint
from ttla.training import calibrate_adapter, calibrate_static_adapter, finetune_few_shot
from ttla.utils.io import ensure_dir


BASELINES = [
    "no_adaptation",
    "few_shot_finetuning",
    "ours",
]


VARIANTS = [
    {
        "name": "progress_base",
        "overrides": {},
    },
    {
        "name": "progress_min_035",
        "overrides": {
            "model": {
                "adapter_progressive_min_scale": 0.35,
                "adapter_progressive_max_scale": 1.0,
            }
        },
    },
    {
        "name": "progress_min_050",
        "overrides": {
            "model": {
                "adapter_progressive_min_scale": 0.50,
                "adapter_progressive_max_scale": 1.0,
            }
        },
    },
    {
        "name": "progress_min_065",
        "overrides": {
            "model": {
                "adapter_progressive_min_scale": 0.65,
                "adapter_progressive_max_scale": 1.0,
            }
        },
    },
    {
        "name": "progress_min_050_policy_001",
        "overrides": {
            "model": {
                "adapter_progressive_min_scale": 0.50,
                "adapter_progressive_max_scale": 1.0,
            },
            "adaptation": {
                "policy_preservation_weight": 0.01,
                "policy_preservation_alpha": 8.0,
            },
        },
    },
]


def _deep_update(target: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _prepare_cfg(base_cfg: dict, variant_name: str) -> tuple[dict, Path]:
    cfg = copy.deepcopy(base_cfg)
    overrides = copy.deepcopy(base_cfg.get("backbone_overrides", {}).get("feedforward", {}))
    if overrides:
        _deep_update(cfg, overrides)
    variant = next(item for item in VARIANTS if item["name"] == variant_name)
    _deep_update(cfg, copy.deepcopy(variant["overrides"]))

    root = ensure_dir(Path(cfg["paths"]["results_root"]) / "joint_progress_sweep" / variant_name / "feedforward")
    cfg["model"]["backbone_type"] = "feedforward"
    cfg["paths"]["results_root"] = str(root)
    cfg["paths"]["checkpoint_dir"] = str(root / "checkpoints")
    ensure_dir(Path(cfg["paths"]["checkpoint_dir"]))
    with (root / "config_snapshot.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    return cfg, root


def main() -> None:
    base_cfg = load_config("configs/pseudo_real_joint.yaml")
    frozen_checkpoint = Path("results/fixed_protocol/backbone_suite/feedforward/checkpoints/best_model.pt")
    if not frozen_checkpoint.exists():
        raise FileNotFoundError(f"Missing frozen checkpoint: {frozen_checkpoint}")
    calibration_path = Path(base_cfg["paths"]["data_root"]) / "calibration.npz"
    source_train_path = Path(base_cfg.get("pseudo_real", {}).get("source_train_path", "data/processed/train.npz"))

    rows: list[dict[str, float | str]] = []
    for variant in VARIANTS:
        cfg, root = _prepare_cfg(base_cfg, str(variant["name"]))
        warm_start_static = bool(cfg.get("adaptation", {}).get("warm_start_static_adapter", False))
        static_adapter_path = calibrate_static_adapter(cfg, frozen_checkpoint, source_train_path, calibration_path)
        adapter_init = static_adapter_path if warm_start_static else frozen_checkpoint
        adapter_path = calibrate_adapter(cfg, adapter_init, calibration_path)
        few_shot_path = finetune_few_shot(cfg, frozen_checkpoint, calibration_path)

        model_paths = {
            "no_adaptation": frozen_checkpoint,
            "few_shot_finetuning": few_shot_path,
            "ours": adapter_path,
        }
        csv_paths: list[Path] = []
        for baseline in BASELINES:
            csv_path = root / f"{baseline}.csv"
            evaluate_checkpoint(cfg, model_paths[baseline], baseline, csv_path)
            csv_paths.append(csv_path)
        merged = pd.concat([pd.read_csv(path) for path in csv_paths], ignore_index=True)
        merged.to_csv(root / "summary.csv", index=False)
        summary = (
            merged.groupby(["baseline", "task"], as_index=False)
            .agg(success_rate=("success", "mean"), mean_steps=("steps", "mean"), mean_visibility=("visibility", "mean"))
        )
        summary.to_csv(root / "summary_metrics.csv", index=False)
        overall = (
            summary.groupby("baseline", as_index=False)
            .agg(
                mean_success_rate=("success_rate", "mean"),
                mean_steps=("mean_steps", "mean"),
                mean_visibility=("mean_visibility", "mean"),
            )
            .sort_values(["mean_success_rate", "mean_visibility"], ascending=[False, False])
        )
        overall.to_csv(root / "overall_metrics.csv", index=False)
        ours = overall.loc[overall["baseline"] == "ours"].iloc[0]
        few = overall.loc[overall["baseline"] == "few_shot_finetuning"].iloc[0]
        rows.append(
            {
                "variant": variant["name"],
                "ours_mean_success_rate": float(ours["mean_success_rate"]),
                "few_shot_mean_success_rate": float(few["mean_success_rate"]),
                "ours_minus_few_shot": float(ours["mean_success_rate"] - few["mean_success_rate"]),
            }
        )

    out_root = ensure_dir(Path(base_cfg["paths"]["results_root"]) / "joint_progress_sweep")
    pd.DataFrame.from_records(rows).sort_values(
        ["ours_mean_success_rate", "ours_minus_few_shot"],
        ascending=[False, False],
    ).to_csv(out_root / "sweep_summary.csv", index=False)


if __name__ == "__main__":
    main()

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import yaml

from ttla.config import load_config
from ttla.evaluation import evaluate_checkpoint
from ttla.training import calibrate_adapter, finetune_few_shot
from ttla.utils.io import ensure_dir


BASELINES = [
    "no_adaptation",
    "few_shot_finetuning",
    "ours",
]


VARIANTS = [
    {"name": "stagepol_base", "overrides": {"adaptation": {"policy_preservation_weight": 0.0}}},
    {
        "name": "stagepol_w001_s2",
        "overrides": {
            "adaptation": {
                "policy_preservation_weight": 0.01,
                "policy_preservation_alpha": 8.0,
                "policy_preservation_stage_max": 2,
            }
        },
    },
    {
        "name": "stagepol_w002_s2",
        "overrides": {
            "adaptation": {
                "policy_preservation_weight": 0.02,
                "policy_preservation_alpha": 8.0,
                "policy_preservation_stage_max": 2,
            }
        },
    },
    {
        "name": "stagepol_w005_s2",
        "overrides": {
            "adaptation": {
                "policy_preservation_weight": 0.05,
                "policy_preservation_alpha": 8.0,
                "policy_preservation_stage_max": 2,
            }
        },
    },
    {
        "name": "stagepol_w002_s3",
        "overrides": {
            "adaptation": {
                "policy_preservation_weight": 0.02,
                "policy_preservation_alpha": 8.0,
                "policy_preservation_stage_max": 3,
            }
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
    root = ensure_dir(Path(cfg["paths"]["results_root"]) / "joint_stage_policy_sweep" / variant_name / "feedforward")
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

    rows: list[dict[str, float | str]] = []
    for variant in VARIANTS:
        cfg, root = _prepare_cfg(base_cfg, str(variant["name"]))
        adapter_path = calibrate_adapter(cfg, frozen_checkpoint, calibration_path)
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

    out_root = ensure_dir(Path(base_cfg["paths"]["results_root"]) / "joint_stage_policy_sweep")
    pd.DataFrame.from_records(rows).sort_values(
        ["ours_mean_success_rate", "ours_minus_few_shot"],
        ascending=[False, False],
    ).to_csv(out_root / "sweep_summary.csv", index=False)


if __name__ == "__main__":
    main()

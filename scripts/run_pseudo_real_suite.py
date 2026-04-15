from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd
import yaml

from ttla.config import load_config
from ttla.evaluation import evaluate_checkpoint
from ttla.training import calibrate_adapter, calibrate_static_adapter, finetune_few_shot, fit_latent_alignment
from ttla.utils.io import ensure_dir


DEFAULT_BACKBONES = ["feedforward", "recurrent", "chunking", "language", "diffusion"]
DEFAULT_BASELINES = [
    "no_adaptation",
    "input_normalization",
    "probe_feature_alignment",
    "static_adapter",
    "few_shot_finetuning",
    "tent_style",
    "ours",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--backbones", nargs="*", default=DEFAULT_BACKBONES)
    parser.add_argument("--baselines", nargs="*", default=DEFAULT_BASELINES)
    parser.add_argument("--frozen-root", default="results/fixed_protocol/backbone_suite")
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def _suite_root(cfg: dict, tag: str = "") -> Path:
    suffix = "backbone_suite" if not tag else f"backbone_suite_{tag}"
    return ensure_dir(Path(cfg["paths"]["results_root"]) / suffix)


def _deep_update(target: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


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


def _summarize(csv_paths: list[Path], backbone: str, output_path: Path) -> pd.DataFrame:
    merged = pd.concat([pd.read_csv(path) for path in csv_paths], ignore_index=True)
    merged.insert(0, "backbone", backbone)
    merged.to_csv(output_path, index=False)
    return merged


def main() -> None:
    args = _parse_args()
    base_cfg = load_config(args.config)
    calibration_path = Path(base_cfg["paths"]["data_root"]) / "calibration.npz"
    source_train_path = Path(base_cfg.get("pseudo_real", {}).get("source_train_path", "data/processed/train.npz"))
    suite_root = _suite_root(base_cfg, args.tag)
    frozen_root = Path(args.frozen_root)
    suite_records: list[pd.DataFrame] = []
    checkpoint_records: list[dict[str, str]] = []

    for backbone in args.backbones:
        cfg, backbone_root = _prepare_cfg(base_cfg, backbone, suite_root)
        frozen_checkpoint = frozen_root / backbone / "checkpoints" / "best_model.pt"
        if not frozen_checkpoint.exists():
            raise FileNotFoundError(f"Missing frozen checkpoint for backbone={backbone}: {frozen_checkpoint}")
        adapter_path: Path | None = None
        static_adapter_path: Path | None = None
        few_shot_path: Path | None = None
        latent_alignment_path: Path | None = None
        if "ours" in args.baselines:
            adapter_path = calibrate_adapter(cfg, frozen_checkpoint, calibration_path)
        if "static_adapter" in args.baselines:
            static_adapter_path = calibrate_static_adapter(cfg, frozen_checkpoint, source_train_path, calibration_path)
        if "few_shot_finetuning" in args.baselines:
            few_shot_path = finetune_few_shot(cfg, frozen_checkpoint, calibration_path)
        if "probe_feature_alignment" in args.baselines:
            latent_alignment_path = fit_latent_alignment(cfg, frozen_checkpoint, source_train_path, calibration_path)
        checkpoint_records.append(
            {
                "backbone": backbone,
                "frozen_checkpoint_path": str(frozen_checkpoint),
                "adapter_path": str(adapter_path) if adapter_path is not None else "",
                "static_adapter_path": str(static_adapter_path) if static_adapter_path is not None else "",
                "few_shot_path": str(few_shot_path) if few_shot_path is not None else "",
                "latent_alignment_path": str(latent_alignment_path) if latent_alignment_path is not None else "",
            }
        )
        csv_paths: list[Path] = []
        for baseline in args.baselines:
            if baseline == "ours":
                model_path = adapter_path
            elif baseline == "static_adapter":
                model_path = static_adapter_path
            elif baseline == "few_shot_finetuning":
                model_path = few_shot_path
            else:
                model_path = frozen_checkpoint
            if model_path is None:
                raise RuntimeError(f"Missing checkpoint for baseline={baseline}, backbone={backbone}")
            csv_path = backbone_root / f"{baseline}.csv"
            artifacts: dict[str, str] = {}
            if baseline == "probe_feature_alignment" and latent_alignment_path is not None:
                artifacts["latent_alignment_path"] = str(latent_alignment_path)
            evaluate_checkpoint(cfg, model_path, baseline, csv_path, baseline_artifacts=artifacts)
            csv_paths.append(csv_path)
        merged = _summarize(csv_paths, backbone, backbone_root / "summary.csv")
        summary = (
            merged.groupby(["backbone", "baseline", "task"], as_index=False)
            .agg(success_rate=("success", "mean"), mean_steps=("steps", "mean"), mean_visibility=("visibility", "mean"))
        )
        summary.to_csv(backbone_root / "summary_metrics.csv", index=False)
        suite_records.append(summary)

    all_summary = pd.concat(suite_records, ignore_index=True)
    all_summary.to_csv(suite_root / "suite_summary_metrics.csv", index=False)
    overall = (
        all_summary.groupby(["backbone", "baseline"], as_index=False)
        .agg(
            mean_success_rate=("success_rate", "mean"),
            mean_steps=("mean_steps", "mean"),
            mean_visibility=("mean_visibility", "mean"),
        )
        .sort_values(["mean_success_rate", "mean_visibility"], ascending=[False, False])
    )
    overall.to_csv(suite_root / "suite_overall_metrics.csv", index=False)
    pd.DataFrame.from_records(checkpoint_records).to_csv(suite_root / "suite_checkpoints.csv", index=False)


if __name__ == "__main__":
    main()

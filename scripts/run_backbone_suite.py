from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd
import yaml

from ttla.config import load_config
from ttla.data import load_split
from ttla.evaluation import evaluate_checkpoint
from ttla.training import calibrate_adapter, train_model
from ttla.utils.io import ensure_dir


DEFAULT_BACKBONES = ["feedforward", "recurrent", "chunking", "language", "diffusion"]
DEFAULT_BASELINES = ["no_adaptation", "input_normalization", "ours"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--backbones", nargs="*", default=DEFAULT_BACKBONES)
    parser.add_argument("--baselines", nargs="*", default=DEFAULT_BASELINES)
    parser.add_argument("--calibration-split", default="val")
    return parser.parse_args()


def _suite_root(cfg: dict) -> Path:
    return ensure_dir(Path(cfg["paths"]["results_root"]) / "backbone_suite")


def _prepare_cfg(base_cfg: dict, backbone: str) -> tuple[dict, Path]:
    cfg = copy.deepcopy(base_cfg)
    root = _suite_root(base_cfg) / backbone
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
    train_path = load_split(base_cfg["paths"]["data_root"], "train")
    val_path = load_split(base_cfg["paths"]["data_root"], "val")
    calibration_path = load_split(base_cfg["paths"]["data_root"], args.calibration_split)
    suite_root = _suite_root(base_cfg)
    suite_records: list[pd.DataFrame] = []

    for backbone in args.backbones:
        cfg, backbone_root = _prepare_cfg(base_cfg, backbone)
        print(f"[suite] backbone={backbone} train_start")
        checkpoint_path = train_model(cfg, train_path, val_path)
        print(f"[suite] backbone={backbone} train_done checkpoint={checkpoint_path}")
        adapter_path = calibrate_adapter(cfg, checkpoint_path, calibration_path)
        print(f"[suite] backbone={backbone} adapter_done checkpoint={adapter_path}")

        csv_paths: list[Path] = []
        for baseline in args.baselines:
            model_path = adapter_path if baseline == "ours" else checkpoint_path
            csv_path = backbone_root / f"{baseline}.csv"
            evaluate_checkpoint(cfg, model_path, baseline, csv_path)
            csv_paths.append(csv_path)
            print(f"[suite] backbone={backbone} baseline={baseline} csv={csv_path}")

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
    print(f"[suite] saved={suite_root}")


if __name__ == "__main__":
    main()

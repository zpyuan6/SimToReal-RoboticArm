from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from ttla.config import load_config
from ttla.data import HistoryTrajectoryDataset, RealCalibrationDataset
from ttla.evaluation.baselines import baseline_overrides
from ttla.models import BaseTTLAModel, load_model_state
from ttla.training import (
    build_model,
    calibrate_adapter,
    calibrate_static_adapter,
    finetune_few_shot,
    fit_latent_alignment,
)
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
    parser.add_argument("--real-data", default="data/real/calibration_real_merged.npz")
    parser.add_argument("--calibration-data", default=None)
    parser.add_argument("--heldout-data", default=None)
    parser.add_argument("--backbones", nargs="*", default=DEFAULT_BACKBONES)
    parser.add_argument("--baselines", nargs="*", default=DEFAULT_BASELINES)
    parser.add_argument("--frozen-root", default="results/fixed_protocol/backbone_suite")
    parser.add_argument("--tag", default="")
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--adapt-policy-supervision-weight", type=float, default=None)
    parser.add_argument("--adapt-inverse-supervision-weight", type=float, default=None)
    parser.add_argument("--adapt-lr", type=float, default=None)
    parser.add_argument("--adapt-epochs", type=int, default=None)
    parser.add_argument("--adapt-batch-size", type=int, default=None)
    parser.add_argument("--adapt-reg-weight", type=float, default=None)
    parser.add_argument("--adapt-stage-weight", type=float, default=None)
    parser.add_argument("--adapt-auto-stage-scaling", action="store_true")
    parser.add_argument("--adapt-auto-stage-scale-min", type=float, default=None)
    parser.add_argument("--adapt-auto-stage-scale-max", type=float, default=None)
    parser.add_argument("--adapt-auto-stage-scale-power", type=float, default=None)
    parser.add_argument("--adapt-auto-non-observation-scaling", action="store_true")
    parser.add_argument("--adapt-auto-non-observation-scale-min", type=float, default=None)
    parser.add_argument("--adapt-auto-non-observation-scale-max", type=float, default=None)
    parser.add_argument("--adapt-auto-non-observation-global-threshold", type=float, default=None)
    parser.add_argument("--adapt-auto-non-observation-global-slope", type=float, default=None)
    parser.add_argument("--adapt-auto-latent-affine-scaling", action="store_true")
    parser.add_argument("--adapt-auto-latent-affine-blend-min", type=float, default=None)
    parser.add_argument("--adapt-auto-latent-affine-blend-max", type=float, default=None)
    parser.add_argument("--adapt-auto-latent-affine-global-threshold", type=float, default=None)
    parser.add_argument("--adapt-auto-latent-affine-global-slope", type=float, default=None)
    parser.add_argument("--adapt-static-alignment-weight", type=float, default=None)
    parser.add_argument("--adapt-warm-start-static-adapter", action="store_true")
    parser.add_argument("--adapt-source-alignment-weight", type=float, default=None)
    parser.add_argument("--adapt-source-primitive-alignment-weight", type=float, default=None)
    parser.add_argument("--adapt-source-delta-alignment-weight", type=float, default=None)
    parser.add_argument("--adapt-source-policy-alignment-weight", type=float, default=None)
    parser.add_argument("--adapt-auto-source-anchor-scaling", action="store_true")
    parser.add_argument("--adapt-auto-source-anchor-scale-min", type=float, default=None)
    parser.add_argument("--adapt-auto-source-anchor-scale-max", type=float, default=None)
    parser.add_argument("--adapt-auto-source-anchor-global-threshold", type=float, default=None)
    parser.add_argument("--adapt-auto-source-anchor-global-slope", type=float, default=None)
    parser.add_argument("--adapt-residual-adaptive-adapt-alpha", type=float, default=None)
    parser.add_argument("--adapt-residual-adaptive-adapt-gamma", type=float, default=None)
    parser.add_argument("--adapt-observation-weight", type=float, default=None)
    parser.add_argument("--adapt-precontact-weight", type=float, default=None)
    parser.add_argument("--adapt-postcontact-weight", type=float, default=None)
    parser.add_argument("--adapt-abort-weight", type=float, default=None)
    parser.add_argument("--adapt-policy-adapter-identity-weight", type=float, default=None)
    parser.add_argument("--model-adapter-scale", type=float, default=None)
    parser.add_argument("--model-policy-residual-adapter-enable", action="store_true")
    parser.add_argument("--model-policy-residual-adapter-disable", action="store_true")
    parser.add_argument("--model-policy-residual-hidden-dim", type=int, default=None)
    parser.add_argument("--model-policy-residual-scale", type=float, default=None)
    parser.add_argument("--model-latent-affine-enable", action="store_true")
    parser.add_argument("--model-latent-affine-disable", action="store_true")
    parser.add_argument("--model-latent-affine-task-conditioned-enable", action="store_true")
    parser.add_argument("--model-latent-affine-task-conditioned-disable", action="store_true")
    parser.add_argument("--model-latent-affine-max-scale", type=float, default=None)
    parser.add_argument("--model-latent-affine-blend", type=float, default=None)
    parser.add_argument("--train-inverse-loss-weight", type=float, default=None)
    return parser.parse_args()


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
    return cfg, root


def _write_cfg_snapshot(cfg: dict, root: Path) -> None:
    with (root / "config_snapshot.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def _suite_root(cfg: dict, tag: str = "") -> Path:
    suffix = "backbone_suite" if not tag else f"backbone_suite_{tag}"
    return ensure_dir(Path(cfg["paths"]["results_root"]) / suffix)


def _save_npz_subset(payload: dict[str, np.ndarray], indices: np.ndarray, output_path: Path) -> None:
    subset = {key: value[indices] for key, value in payload.items()}
    num_samples = len(indices)
    if "success" not in subset:
        subset["success"] = np.zeros(num_samples, dtype=np.float32)
    if "contexts" not in subset:
        subset["contexts"] = np.zeros((num_samples, 8), dtype=np.float32)
    if "episode_ids" not in subset:
        subset["episode_ids"] = np.arange(num_samples, dtype=np.int64)
    if "step_ids" not in subset:
        subset["step_ids"] = np.zeros(num_samples, dtype=np.int64)
    np.savez(output_path, **subset)


def _build_real_splits(
    real_path: Path,
    output_dir: Path,
    split_ratio: float,
    seed: int,
) -> tuple[Path, Path]:
    payload_npz = np.load(real_path)
    payload = {key: payload_npz[key] for key in payload_npz.files}
    tasks = payload["tasks"].astype(np.int64)
    rng = np.random.default_rng(seed)
    calib_indices: list[np.ndarray] = []
    heldout_indices: list[np.ndarray] = []
    for task_id in np.unique(tasks):
        task_indices = np.flatnonzero(tasks == task_id)
        permuted = task_indices[rng.permutation(len(task_indices))]
        split = max(1, int(round(len(permuted) * split_ratio)))
        split = min(split, len(permuted) - 1) if len(permuted) > 1 else len(permuted)
        calib_indices.append(permuted[:split])
        heldout_indices.append(permuted[split:])
    calib_idx = np.sort(np.concatenate(calib_indices))
    heldout_idx = np.sort(np.concatenate(heldout_indices))
    calib_path = output_dir / "calibration_split.npz"
    heldout_path = output_dir / "heldout_split.npz"
    _save_npz_subset(payload, calib_idx, calib_path)
    _save_npz_subset(payload, heldout_idx, heldout_path)
    return calib_path, heldout_path


def _resolve_real_splits(
    args: argparse.Namespace,
    split_root: Path,
    seed: int,
) -> tuple[Path, Path, dict[str, object]]:
    explicit_calibration = args.calibration_data
    explicit_heldout = args.heldout_data
    if bool(explicit_calibration) != bool(explicit_heldout):
        raise ValueError("Provide both --calibration-data and --heldout-data, or neither.")
    if explicit_calibration and explicit_heldout:
        calibration_path = Path(explicit_calibration)
        heldout_path = Path(explicit_heldout)
        if not calibration_path.exists():
            raise FileNotFoundError(f"Missing calibration split: {calibration_path}")
        if not heldout_path.exists():
            raise FileNotFoundError(f"Missing held-out split: {heldout_path}")
        manifest = {
            "split_mode": "explicit",
            "calibration_path": str(calibration_path),
            "heldout_path": str(heldout_path),
        }
        return calibration_path, heldout_path, manifest
    calibration_path, heldout_path = _build_real_splits(
        Path(args.real_data),
        split_root,
        split_ratio=float(args.split_ratio),
        seed=seed,
    )
    manifest = {
        "split_mode": "random_from_merged",
        "real_data": str(args.real_data),
        "split_ratio": float(args.split_ratio),
        "seed": int(seed),
        "calibration_path": str(calibration_path),
        "heldout_path": str(heldout_path),
    }
    return calibration_path, heldout_path, manifest


def _build_dataset(cfg: dict, path: str | Path):
    backbone_type = cfg["model"].get("backbone_type", "feedforward").lower()
    primitive_vocabulary = cfg["model"].get("primitive_vocabulary", "legacy")
    if backbone_type in {"recurrent", "gru", "rnn", "chunking", "chunk", "act"}:
        return HistoryTrajectoryDataset(
            path,
            history_len=cfg["model"].get("history_len", 4),
            chunk_size=cfg["model"].get("chunk_size", 3),
            primitive_vocabulary=primitive_vocabulary,
        )
    return RealCalibrationDataset(path, primitive_vocabulary=primitive_vocabulary)


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _normalize_image_tensor(images: torch.Tensor) -> torch.Tensor:
    mean = images.mean(dim=(-2, -1), keepdim=True)
    std = images.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1.0e-6)
    normalized = (images - mean) / std
    return normalized.clamp_(-3.0, 3.0)


def _normalize_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = dict(batch)
    for key in ("image", "next_image", "history_images", "next_history_images"):
        if key in out:
            out[key] = _normalize_image_tensor(out[key])
    return out


def _load_latent_alignment(path: str | Path | None) -> dict[str, np.ndarray] | None:
    if path is None:
        return None
    payload = np.load(Path(path))
    return {key: payload[key] for key in payload.files}


def _apply_latent_alignment(z: torch.Tensor, task_ids: torch.Tensor, stats: dict[str, np.ndarray]) -> torch.Tensor:
    z_out = z.clone()
    max_scale = float(np.asarray(stats.get("max_scale", np.array(4.0, dtype=np.float32))).item())
    max_scale = max(max_scale, 1.0)
    blend = float(np.asarray(stats.get("blend", np.array(1.0, dtype=np.float32))).item())
    blend = max(0.0, min(blend, 1.0))
    for task_id in task_ids.unique(sorted=True):
        mask = task_ids == task_id
        task_value = int(task_id.item())
        source_mean = torch.from_numpy(stats["source_mean"][task_value]).to(device=z.device, dtype=z.dtype)
        source_std = torch.from_numpy(stats["source_std"][task_value]).to(device=z.device, dtype=z.dtype)
        target_mean = torch.from_numpy(stats["target_mean"][task_value]).to(device=z.device, dtype=z.dtype)
        target_std = torch.from_numpy(stats["target_std"][task_value]).to(device=z.device, dtype=z.dtype)
        scale = (source_std / target_std.clamp_min(1.0e-6)).clamp(min=1.0 / max_scale, max=max_scale)
        aligned = (z_out[mask] - target_mean) * scale + source_mean
        if blend < 1.0:
            aligned = z_out[mask] + blend * (aligned - z_out[mask])
        z_out[mask] = aligned
    return z_out


def _tent_parameters(model: BaseTTLAModel) -> list[torch.nn.Parameter]:
    preferred = {
        "feedforward": ["fusion", "policy_head", "stage_head"],
        "recurrent": ["feature_fusion", "gru", "post_gru", "policy_head", "stage_head"],
        "chunking": ["feature_fusion", "gru", "post_gru", "chunk_head", "stage_head"],
        "language": ["fusion", "policy_head", "stage_head"],
        "diffusion": ["direct_head", "denoiser", "stage_head"],
    }.get(model.backbone_type, ["policy_head", "stage_head"])
    params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(name.startswith(prefix + ".") for prefix in preferred):
            params.append(param)
    return params


def _conditioned_logits(model: BaseTTLAModel, z: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return model.condition_policy_logits(
        model.policy_logits(z),
        z=z,
        task_ids=batch.get("task"),
        state=batch.get("state"),
    )


def _evaluate_offline(
    cfg: dict,
    checkpoint_path: Path,
    baseline: str,
    heldout_path: Path,
    latent_alignment_path: Path | None = None,
) -> pd.DataFrame:
    device = torch.device(cfg["train"]["device"])
    payload = torch.load(checkpoint_path, map_location=device)
    model = build_model(cfg).to(device)
    load_model_state(model, payload["model_state"])
    model.eval()
    dataset = _build_dataset(cfg, heldout_path)
    loader = DataLoader(dataset, batch_size=cfg["adaptation"]["batch_size"], shuffle=False)
    baseline_cfg = baseline_overrides(baseline)
    latent_alignment = _load_latent_alignment(latent_alignment_path)
    tent_optimizer = None
    if baseline == "tent_style":
        params = _tent_parameters(model)
        if params:
            tent_optimizer = torch.optim.AdamW(
                params,
                lr=float(cfg.get("tent", {}).get("lr", 1.0e-4)),
                weight_decay=float(cfg.get("tent", {}).get("weight_decay", 0.0)),
            )
    records: list[dict[str, float | int | str]] = []
    for batch in loader:
        batch = _to_device(batch, device)
        if baseline_cfg.get("input_norm", False):
            batch = _normalize_batch(batch)
        if tent_optimizer is not None:
            model.train()
            tent_optimizer.zero_grad(set_to_none=True)
            z_tmp, _ = model.compute_latents(batch)
            logits_tmp = _conditioned_logits(model, z_tmp, batch)
            probs = torch.softmax(logits_tmp, dim=-1)
            entropy = -(probs * torch.log_softmax(logits_tmp, dim=-1)).sum(dim=-1).mean()
            entropy.backward()
            tent_optimizer.step()
            model.eval()
        with torch.no_grad():
            z, next_z = model.compute_latents(batch)
            if baseline in {"ours", "static_adapter"}:
                z_eval, next_z_eval = model.compute_adapted_latents(batch)
                latent_reg = F.mse_loss(z_eval, z) + 0.5 * F.mse_loss(next_z_eval, next_z)
            else:
                z_eval, next_z_eval = z, next_z
                latent_reg = torch.zeros((), device=device)
                if latent_alignment is not None and baseline_cfg.get("latent_alignment", False):
                    z_eval = _apply_latent_alignment(z_eval, batch["task"].long(), latent_alignment)
                    next_z_eval = _apply_latent_alignment(next_z_eval, batch["task"].long(), latent_alignment)
            pred_next = model.predict_next(z_eval, batch["primitive_id"], batch.get("state"), task_ids=batch.get("task"))
            logits = _conditioned_logits(model, z_eval, batch)
            primitive_loss = F.cross_entropy(logits, batch["primitive_id"])
            primitive_match = (logits.argmax(dim=-1) == batch["primitive_id"]).float().mean()
            transition_mse = F.mse_loss(pred_next, next_z_eval)
            stage_loss = model.compute_stage_loss(batch, z_eval)
            batch_size = int(batch["primitive_id"].size(0))
            task_ids = batch["task"].detach().cpu().numpy()
            for task_id in np.unique(task_ids):
                mask = batch["task"] == int(task_id)
                if not bool(mask.any()):
                    continue
                task_name = {0: "level1_verify", 1: "level2_approach", 2: "level3_pick_place"}[int(task_id)]
                task_logits = logits[mask]
                task_z = z_eval[mask]
                task_next_z = next_z_eval[mask]
                task_pred_next = model.predict_next(task_z, batch["primitive_id"][mask], batch["state"][mask], task_ids=batch["task"][mask])
                record = {
                    "baseline": baseline,
                    "task": task_name,
                    "count": int(mask.sum().item()),
                    "transition_mse": float(F.mse_loss(task_pred_next, task_next_z).item()),
                    "primitive_loss": float(F.cross_entropy(task_logits, batch["primitive_id"][mask]).item()),
                    "primitive_match": float((task_logits.argmax(dim=-1) == batch["primitive_id"][mask]).float().mean().item()),
                    "stage_loss": float(model.compute_stage_loss({k: (v[mask] if isinstance(v, torch.Tensor) and v.shape[0] == batch_size else v) for k, v in batch.items()}, task_z).item()),
                    "latent_reg": float(latent_reg.item()),
                }
                records.append(record)
    return pd.DataFrame.from_records(records)


def _summarize(records: list[pd.DataFrame], suite_root: Path) -> None:
    merged = pd.concat(records, ignore_index=True)
    merged.to_csv(suite_root / "suite_task_metrics.csv", index=False)
    summary = (
        merged.groupby(["backbone", "baseline", "task"], as_index=False)
        .agg(
            transition_mse=("transition_mse", "mean"),
            primitive_loss=("primitive_loss", "mean"),
            primitive_match=("primitive_match", "mean"),
            stage_loss=("stage_loss", "mean"),
            latent_reg=("latent_reg", "mean"),
        )
    )
    summary.to_csv(suite_root / "suite_summary_metrics.csv", index=False)
    overall = (
        summary.groupby(["backbone", "baseline"], as_index=False)
        .agg(
            mean_transition_mse=("transition_mse", "mean"),
            mean_primitive_loss=("primitive_loss", "mean"),
            mean_primitive_match=("primitive_match", "mean"),
            mean_stage_loss=("stage_loss", "mean"),
            mean_latent_reg=("latent_reg", "mean"),
        )
        .sort_values(["mean_transition_mse", "mean_primitive_loss"], ascending=[True, True])
    )
    overall.to_csv(suite_root / "suite_overall_metrics.csv", index=False)


def main() -> None:
    args = _parse_args()
    base_cfg = load_config(args.config)
    real_tag = "real" if not args.tag else f"{args.tag}_real"
    suite_root = _suite_root(base_cfg, real_tag)
    split_root = ensure_dir(suite_root / "_splits")
    calibration_path, heldout_path, split_manifest = _resolve_real_splits(
        args,
        split_root,
        seed=int(base_cfg.get("seed", 0)) + 401,
    )
    with (suite_root / "split_manifest.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(split_manifest, handle, sort_keys=False)
    frozen_root = Path(args.frozen_root)
    source_train_path = Path(
        base_cfg.get("pseudo_real", {}).get("source_train_path", Path(base_cfg["paths"]["data_root"]) / "train.npz")
    )
    records: list[pd.DataFrame] = []
    checkpoint_records: list[dict[str, str]] = []

    for backbone in args.backbones:
        cfg, backbone_root = _prepare_cfg(base_cfg, backbone, suite_root)
        if args.adapt_policy_supervision_weight is not None:
            cfg.setdefault("adaptation", {})["policy_supervision_weight"] = float(
                args.adapt_policy_supervision_weight
            )
        if args.adapt_inverse_supervision_weight is not None:
            cfg.setdefault("adaptation", {})["inverse_supervision_weight"] = float(
                args.adapt_inverse_supervision_weight
            )
        if args.train_inverse_loss_weight is not None:
            cfg.setdefault("train", {})["inverse_loss_weight"] = float(args.train_inverse_loss_weight)
        if args.adapt_lr is not None:
            cfg.setdefault("adaptation", {})["lr"] = float(args.adapt_lr)
        if args.adapt_epochs is not None:
            cfg.setdefault("adaptation", {})["epochs"] = int(args.adapt_epochs)
        if args.adapt_batch_size is not None:
            cfg.setdefault("adaptation", {})["batch_size"] = int(args.adapt_batch_size)
        if args.adapt_reg_weight is not None:
            cfg.setdefault("adaptation", {})["adapter_reg_weight"] = float(args.adapt_reg_weight)
        if args.adapt_stage_weight is not None:
            cfg.setdefault("adaptation", {})["adapter_stage_loss_weight"] = float(args.adapt_stage_weight)
        if args.adapt_auto_stage_scaling:
            cfg.setdefault("adaptation", {})["auto_stage_scaling"] = True
        if args.adapt_auto_stage_scale_min is not None:
            cfg.setdefault("adaptation", {})["auto_stage_scale_min"] = float(args.adapt_auto_stage_scale_min)
        if args.adapt_auto_stage_scale_max is not None:
            cfg.setdefault("adaptation", {})["auto_stage_scale_max"] = float(args.adapt_auto_stage_scale_max)
        if args.adapt_auto_stage_scale_power is not None:
            cfg.setdefault("adaptation", {})["auto_stage_scale_power"] = float(args.adapt_auto_stage_scale_power)
        if args.adapt_auto_non_observation_scaling:
            cfg.setdefault("adaptation", {})["auto_non_observation_scaling"] = True
        if args.adapt_auto_non_observation_scale_min is not None:
            cfg.setdefault("adaptation", {})["auto_non_observation_scale_min"] = float(args.adapt_auto_non_observation_scale_min)
        if args.adapt_auto_non_observation_scale_max is not None:
            cfg.setdefault("adaptation", {})["auto_non_observation_scale_max"] = float(args.adapt_auto_non_observation_scale_max)
        if args.adapt_auto_non_observation_global_threshold is not None:
            cfg.setdefault("adaptation", {})["auto_non_observation_global_threshold"] = float(args.adapt_auto_non_observation_global_threshold)
        if args.adapt_auto_non_observation_global_slope is not None:
            cfg.setdefault("adaptation", {})["auto_non_observation_global_slope"] = float(args.adapt_auto_non_observation_global_slope)
        if args.adapt_auto_latent_affine_scaling:
            cfg.setdefault("adaptation", {})["auto_latent_affine_scaling"] = True
        if args.adapt_auto_latent_affine_blend_min is not None:
            cfg.setdefault("adaptation", {})["auto_latent_affine_blend_min"] = float(args.adapt_auto_latent_affine_blend_min)
        if args.adapt_auto_latent_affine_blend_max is not None:
            cfg.setdefault("adaptation", {})["auto_latent_affine_blend_max"] = float(args.adapt_auto_latent_affine_blend_max)
        if args.adapt_auto_latent_affine_global_threshold is not None:
            cfg.setdefault("adaptation", {})["auto_latent_affine_global_threshold"] = float(args.adapt_auto_latent_affine_global_threshold)
        if args.adapt_auto_latent_affine_global_slope is not None:
            cfg.setdefault("adaptation", {})["auto_latent_affine_global_slope"] = float(args.adapt_auto_latent_affine_global_slope)
        if args.adapt_static_alignment_weight is not None:
            cfg.setdefault("adaptation", {})["static_alignment_weight"] = float(args.adapt_static_alignment_weight)
        if args.adapt_warm_start_static_adapter:
            cfg.setdefault("adaptation", {})["warm_start_static_adapter"] = True
        if args.adapt_source_alignment_weight is not None:
            cfg.setdefault("adaptation", {})["source_alignment_weight"] = float(args.adapt_source_alignment_weight)
        if args.adapt_source_primitive_alignment_weight is not None:
            cfg.setdefault("adaptation", {})["source_primitive_alignment_weight"] = float(args.adapt_source_primitive_alignment_weight)
        if args.adapt_source_delta_alignment_weight is not None:
            cfg.setdefault("adaptation", {})["source_delta_alignment_weight"] = float(args.adapt_source_delta_alignment_weight)
        if args.adapt_source_policy_alignment_weight is not None:
            cfg.setdefault("adaptation", {})["source_policy_alignment_weight"] = float(args.adapt_source_policy_alignment_weight)
        if args.adapt_auto_source_anchor_scaling:
            cfg.setdefault("adaptation", {})["auto_source_anchor_scaling"] = True
        if args.adapt_auto_source_anchor_scale_min is not None:
            cfg.setdefault("adaptation", {})["auto_source_anchor_scale_min"] = float(args.adapt_auto_source_anchor_scale_min)
        if args.adapt_auto_source_anchor_scale_max is not None:
            cfg.setdefault("adaptation", {})["auto_source_anchor_scale_max"] = float(args.adapt_auto_source_anchor_scale_max)
        if args.adapt_auto_source_anchor_global_threshold is not None:
            cfg.setdefault("adaptation", {})["auto_source_anchor_global_threshold"] = float(args.adapt_auto_source_anchor_global_threshold)
        if args.adapt_auto_source_anchor_global_slope is not None:
            cfg.setdefault("adaptation", {})["auto_source_anchor_global_slope"] = float(args.adapt_auto_source_anchor_global_slope)
        if args.adapt_residual_adaptive_adapt_alpha is not None:
            cfg.setdefault("adaptation", {})["residual_adaptive_adapt_alpha"] = float(args.adapt_residual_adaptive_adapt_alpha)
        if args.adapt_residual_adaptive_adapt_gamma is not None:
            cfg.setdefault("adaptation", {})["residual_adaptive_adapt_gamma"] = float(args.adapt_residual_adaptive_adapt_gamma)
        if args.adapt_observation_weight is not None:
            cfg.setdefault("adaptation", {})["observation_adapt_weight"] = float(args.adapt_observation_weight)
        if args.adapt_precontact_weight is not None:
            cfg.setdefault("adaptation", {})["precontact_adapt_weight"] = float(args.adapt_precontact_weight)
        if args.adapt_postcontact_weight is not None:
            cfg.setdefault("adaptation", {})["postcontact_adapt_weight"] = float(args.adapt_postcontact_weight)
        if args.adapt_abort_weight is not None:
            cfg.setdefault("adaptation", {})["abort_adapt_weight"] = float(args.adapt_abort_weight)
        if args.adapt_policy_adapter_identity_weight is not None:
            cfg.setdefault("adaptation", {})["policy_adapter_identity_weight"] = float(
                args.adapt_policy_adapter_identity_weight
            )
        if args.model_adapter_scale is not None:
            cfg.setdefault("model", {})["adapter_scale"] = float(args.model_adapter_scale)
        if args.model_policy_residual_adapter_enable and args.model_policy_residual_adapter_disable:
            raise ValueError("Cannot enable and disable policy residual adapter at the same time.")
        if args.model_policy_residual_adapter_enable:
            cfg.setdefault("model", {})["policy_residual_adapter"] = True
        if args.model_policy_residual_adapter_disable:
            cfg.setdefault("model", {})["policy_residual_adapter"] = False
        if args.model_policy_residual_hidden_dim is not None:
            cfg.setdefault("model", {})["policy_residual_hidden_dim"] = int(args.model_policy_residual_hidden_dim)
        if args.model_policy_residual_scale is not None:
            cfg.setdefault("model", {})["policy_residual_scale"] = float(args.model_policy_residual_scale)
        if args.model_latent_affine_enable and args.model_latent_affine_disable:
            raise ValueError("Cannot enable and disable latent affine alignment at the same time.")
        if args.model_latent_affine_enable:
            cfg.setdefault("model", {})["latent_affine_alignment"] = True
        if args.model_latent_affine_disable:
            cfg.setdefault("model", {})["latent_affine_alignment"] = False
        if (
            args.model_latent_affine_task_conditioned_enable
            and args.model_latent_affine_task_conditioned_disable
        ):
            raise ValueError("Cannot enable and disable task-conditioned latent affine alignment at the same time.")
        if args.model_latent_affine_task_conditioned_enable:
            cfg.setdefault("model", {})["latent_affine_task_conditioned"] = True
        if args.model_latent_affine_task_conditioned_disable:
            cfg.setdefault("model", {})["latent_affine_task_conditioned"] = False
        if args.model_latent_affine_max_scale is not None:
            cfg.setdefault("model", {})["latent_affine_max_scale"] = float(args.model_latent_affine_max_scale)
        if args.model_latent_affine_blend is not None:
            cfg.setdefault("model", {})["latent_affine_blend"] = float(args.model_latent_affine_blend)
        _write_cfg_snapshot(cfg, backbone_root)
        frozen_checkpoint = frozen_root / backbone / "checkpoints" / "best_model.pt"
        if not frozen_checkpoint.exists():
            raise FileNotFoundError(f"Missing frozen checkpoint for backbone={backbone}: {frozen_checkpoint}")
        adapter_path: Path | None = None
        static_adapter_path: Path | None = None
        few_shot_path: Path | None = None
        latent_alignment_path: Path | None = None
        if "static_adapter" in args.baselines:
            static_adapter_path = calibrate_static_adapter(cfg, frozen_checkpoint, source_train_path, calibration_path)
        warm_start_static = bool(cfg.get("adaptation", {}).get("warm_start_static_adapter", False))
        if warm_start_static and static_adapter_path is None:
            static_adapter_path = calibrate_static_adapter(cfg, frozen_checkpoint, source_train_path, calibration_path)
        if "ours" in args.baselines:
            adapter_init_path = static_adapter_path if (warm_start_static and static_adapter_path is not None) else frozen_checkpoint
            adapter_path = calibrate_adapter(
                cfg,
                adapter_init_path,
                calibration_path,
                static_teacher_path=static_adapter_path if (warm_start_static and static_adapter_path is not None) else None,
            )
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
        backbone_frames: list[pd.DataFrame] = []
        for baseline in args.baselines:
            if baseline == "ours":
                model_path = adapter_path
                alignment_path = None
            elif baseline == "static_adapter":
                model_path = static_adapter_path
                alignment_path = None
            elif baseline == "few_shot_finetuning":
                model_path = few_shot_path
                alignment_path = None
            elif baseline == "probe_feature_alignment":
                model_path = frozen_checkpoint
                alignment_path = latent_alignment_path
            else:
                model_path = frozen_checkpoint
                alignment_path = None
            if model_path is None:
                raise RuntimeError(f"Missing model path for backbone={backbone}, baseline={baseline}")
            result = _evaluate_offline(cfg, model_path, baseline, heldout_path, latent_alignment_path=alignment_path)
            result.insert(0, "backbone", backbone)
            result.to_csv(backbone_root / f"{baseline}.csv", index=False)
            backbone_frames.append(result)
        merged = pd.concat(backbone_frames, ignore_index=True)
        merged.to_csv(backbone_root / "summary.csv", index=False)
        summary = (
            merged.groupby(["backbone", "baseline", "task"], as_index=False)
            .agg(
                transition_mse=("transition_mse", "mean"),
                primitive_loss=("primitive_loss", "mean"),
                primitive_match=("primitive_match", "mean"),
                stage_loss=("stage_loss", "mean"),
                latent_reg=("latent_reg", "mean"),
            )
        )
        summary.to_csv(backbone_root / "summary_metrics.csv", index=False)
        records.append(summary)

    _summarize(records, suite_root)
    pd.DataFrame.from_records(checkpoint_records).to_csv(suite_root / "suite_checkpoints.csv", index=False)


if __name__ == "__main__":
    main()

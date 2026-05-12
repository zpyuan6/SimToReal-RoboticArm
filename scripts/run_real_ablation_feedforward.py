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
from tqdm import tqdm

from ttla.config import load_config
from ttla.data import RealCalibrationDataset
from ttla.models import BaseTTLAModel, load_model_state
from ttla.training import build_model, calibrate_static_adapter
from ttla.utils.io import ensure_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--real-data", default="data/real/calibration_real_merged.npz")
    parser.add_argument("--backbone", default="feedforward")
    parser.add_argument("--frozen-root", default="results/fixed_protocol/backbone_suite")
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--tag", default="real")
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
    with (root / "config_snapshot.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    return cfg, root


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


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _conditioned_logits(model: BaseTTLAModel, z: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return model.condition_policy_logits(
        model.policy_logits(z),
        z=z,
        task_ids=batch.get("task"),
        state=batch.get("state"),
    )


def _policy_preservation_loss(
    model: BaseTTLAModel,
    batch: dict[str, torch.Tensor],
    z: torch.Tensor,
    z_prime: torch.Tensor,
    next_z: torch.Tensor,
    *,
    alpha: float,
    temperature: float,
) -> torch.Tensor:
    with torch.no_grad():
        teacher_logits = _conditioned_logits(model, z, batch)
        source_pred_next = model.predict_next(z, batch["primitive_id"], batch.get("state"), task_ids=batch.get("task"))
        transition_residual = F.mse_loss(source_pred_next, next_z, reduction="none").mean(dim=-1)
        weights = torch.exp(-alpha * transition_residual).detach()
        teacher_probs = torch.softmax(teacher_logits / temperature, dim=-1)
    student_logits = _conditioned_logits(model, z_prime, batch)
    student_log_probs = torch.log_softmax(student_logits / temperature, dim=-1)
    policy_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    return (weights * policy_kl).mean() * (temperature ** 2)


def _calibrate_variant(
    cfg: dict,
    checkpoint_path: Path,
    real_path: Path,
    output_path: Path,
    *,
    transition_weight: float,
    reg_weight: float,
    stage_weight: float,
    policy_preservation_weight: float,
    policy_preservation_alpha: float,
    policy_preservation_temperature: float,
) -> Path:
    device = torch.device(cfg["train"]["device"])
    payload = torch.load(checkpoint_path, map_location=device)
    model = build_model(cfg).to(device)
    load_model_state(model, payload["model_state"])
    model.freeze_backbone()
    model.train()
    dataset = RealCalibrationDataset(real_path)
    loader = DataLoader(dataset, batch_size=cfg["adaptation"]["batch_size"], shuffle=True)
    optimizer = torch.optim.AdamW(model.adapter_parameters(), lr=cfg["adaptation"]["lr"])
    epochs = int(cfg["adaptation"].get("epochs", 10))
    for _ in range(epochs):
        for batch in tqdm(loader, desc=f"ablate-{output_path.stem}", leave=False):
            optimizer.zero_grad(set_to_none=True)
            batch = _to_device(batch, device)
            z, next_z = model.compute_latents(batch)
            z = z.detach()
            next_z = next_z.detach()
            z_prime, next_z_prime = model.compute_adapted_latents(batch)
            pred_next = model.predict_next(z_prime, batch["primitive_id"], batch.get("state"), task_ids=batch.get("task"))
            loss_adapt = F.mse_loss(pred_next, next_z_prime)
            loss_reg = F.mse_loss(z_prime, z)
            loss_next_reg = F.mse_loss(next_z_prime, next_z)
            loss_stage = model.compute_stage_loss(batch, z_prime)
            loss_policy_pres = torch.zeros((), device=device)
            if policy_preservation_weight > 0.0:
                loss_policy_pres = _policy_preservation_loss(
                    model,
                    batch,
                    z,
                    z_prime,
                    next_z,
                    alpha=policy_preservation_alpha,
                    temperature=policy_preservation_temperature,
                )
            loss = (
                transition_weight * loss_adapt
                + reg_weight * (loss_reg + 0.5 * loss_next_reg)
                + stage_weight * loss_stage
                + policy_preservation_weight * loss_policy_pres
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.adapter_parameters()), 5.0)
            optimizer.step()
    torch.save({"model_state": model.state_dict(), "config": cfg}, output_path)
    return output_path


def _evaluate_offline(
    cfg: dict,
    checkpoint_path: Path,
    variant_name: str,
    heldout_path: Path,
    *,
    use_adapter: bool,
) -> pd.DataFrame:
    device = torch.device(cfg["train"]["device"])
    payload = torch.load(checkpoint_path, map_location=device)
    model = build_model(cfg).to(device)
    load_model_state(model, payload["model_state"])
    model.eval()
    dataset = RealCalibrationDataset(heldout_path)
    loader = DataLoader(dataset, batch_size=cfg["adaptation"]["batch_size"], shuffle=False)
    records: list[dict[str, float | int | str]] = []
    for batch in loader:
        batch = _to_device(batch, device)
        with torch.no_grad():
            z, next_z = model.compute_latents(batch)
            if use_adapter:
                z_eval, next_z_eval = model.compute_adapted_latents(batch)
                latent_reg = F.mse_loss(z_eval, z) + 0.5 * F.mse_loss(next_z_eval, next_z)
            else:
                z_eval, next_z_eval = z, next_z
                latent_reg = torch.zeros((), device=device)
            pred_next = model.predict_next(z_eval, batch["primitive_id"], batch.get("state"), task_ids=batch.get("task"))
            logits = _conditioned_logits(model, z_eval, batch)
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
                    "variant": variant_name,
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


def main() -> None:
    args = _parse_args()
    base_cfg = load_config(args.config)
    suite_root = ensure_dir(Path(base_cfg["paths"]["results_root"]) / f"ablation_{args.tag}")
    cfg, backbone_root = _prepare_cfg(base_cfg, args.backbone, suite_root)
    split_root = ensure_dir(backbone_root / "_splits")
    calibration_path, heldout_path = _build_real_splits(
        Path(args.real_data),
        split_root,
        split_ratio=float(args.split_ratio),
        seed=int(base_cfg.get("seed", 0)) + 701,
    )
    frozen_checkpoint = Path(args.frozen_root) / args.backbone / "checkpoints" / "best_model.pt"
    if not frozen_checkpoint.exists():
        raise FileNotFoundError(f"Missing frozen checkpoint: {frozen_checkpoint}")

    stage_weight = float(cfg["adaptation"].get("adapter_stage_loss_weight", 0.0))
    reg_weight = float(cfg["adaptation"].get("adapter_reg_weight", 0.1))
    policy_preservation_weight = float(cfg["adaptation"].get("policy_preservation_weight", 0.0))
    policy_preservation_alpha = float(cfg["adaptation"].get("policy_preservation_alpha", 8.0))
    policy_preservation_temperature = float(cfg["adaptation"].get("policy_preservation_temperature", 1.0))

    variant_paths: dict[str, tuple[Path, bool]] = {
        "no_adaptation": (frozen_checkpoint, False),
    }

    source_train_path = Path(
        cfg.get("pseudo_real", {}).get("source_train_path", Path(cfg["paths"]["data_root"]) / "train.npz")
    )
    static_path = calibrate_static_adapter(cfg, frozen_checkpoint, source_train_path, calibration_path)
    variant_paths["static_adapter"] = (static_path, True)

    full_path = _calibrate_variant(
        cfg,
        frozen_checkpoint,
        calibration_path,
        backbone_root / "checkpoints" / "adapter_ablation_full.pt",
        transition_weight=1.0,
        reg_weight=reg_weight,
        stage_weight=stage_weight,
        policy_preservation_weight=policy_preservation_weight,
        policy_preservation_alpha=policy_preservation_alpha,
        policy_preservation_temperature=policy_preservation_temperature,
    )
    variant_paths["ours_full"] = (full_path, True)

    no_transition_path = _calibrate_variant(
        cfg,
        frozen_checkpoint,
        calibration_path,
        backbone_root / "checkpoints" / "adapter_ablation_no_transition.pt",
        transition_weight=0.0,
        reg_weight=reg_weight,
        stage_weight=stage_weight,
        policy_preservation_weight=policy_preservation_weight,
        policy_preservation_alpha=policy_preservation_alpha,
        policy_preservation_temperature=policy_preservation_temperature,
    )
    variant_paths["ours_w_o_transition"] = (no_transition_path, True)

    no_reg_path = _calibrate_variant(
        cfg,
        frozen_checkpoint,
        calibration_path,
        backbone_root / "checkpoints" / "adapter_ablation_no_reg.pt",
        transition_weight=1.0,
        reg_weight=0.0,
        stage_weight=stage_weight,
        policy_preservation_weight=policy_preservation_weight,
        policy_preservation_alpha=policy_preservation_alpha,
        policy_preservation_temperature=policy_preservation_temperature,
    )
    variant_paths["ours_w_o_reg"] = (no_reg_path, True)

    records: list[pd.DataFrame] = []
    for variant_name, (checkpoint_path, use_adapter) in variant_paths.items():
        frame = _evaluate_offline(cfg, checkpoint_path, variant_name, heldout_path, use_adapter=use_adapter)
        frame.to_csv(backbone_root / f"{variant_name}.csv", index=False)
        records.append(frame)

    merged = pd.concat(records, ignore_index=True)
    merged.to_csv(backbone_root / "ablation_summary.csv", index=False)
    summary = (
        merged.groupby(["variant", "task"], as_index=False)
        .agg(
            transition_mse=("transition_mse", "mean"),
            primitive_loss=("primitive_loss", "mean"),
            primitive_match=("primitive_match", "mean"),
            stage_loss=("stage_loss", "mean"),
            latent_reg=("latent_reg", "mean"),
        )
    )
    summary.to_csv(backbone_root / "ablation_summary_metrics.csv", index=False)
    overall = (
        summary.groupby(["variant"], as_index=False)
        .agg(
            mean_transition_mse=("transition_mse", "mean"),
            mean_primitive_loss=("primitive_loss", "mean"),
            mean_primitive_match=("primitive_match", "mean"),
            mean_stage_loss=("stage_loss", "mean"),
            mean_latent_reg=("latent_reg", "mean"),
        )
        .sort_values(["mean_transition_mse", "mean_primitive_loss"], ascending=[True, True])
    )
    overall.to_csv(backbone_root / "ablation_overall_metrics.csv", index=False)


if __name__ == "__main__":
    main()

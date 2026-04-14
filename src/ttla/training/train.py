from __future__ import annotations

from pathlib import Path

import numpy as np
import random
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from ..data import HistoryTrajectoryDataset, RealCalibrationDataset, TrajectoryDataset
from ..models import BaseTTLAModel, build_backbone_model
from ..sim.skills import GRASP_EXECUTE_ID, LIFT_OBJECT_ID, PLACE_OBJECT_ID
from ..utils.io import ensure_dir


def build_model(cfg: dict) -> BaseTTLAModel:
    model_cfg = cfg["model"]
    return build_backbone_model(
        model_cfg.get("backbone_type", "feedforward"),
        state_dim=model_cfg["state_dim"],
        action_dim=model_cfg["action_dim"],
        latent_dim=model_cfg["latent_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        action_embed_dim=model_cfg.get("action_embed_dim", 16),
        task_vocab_size=model_cfg.get("task_vocab_size", 3),
        task_embed_dim=model_cfg.get("task_embed_dim", 16),
        transition_stage_bins=model_cfg.get("transition_stage_bins", 4),
        stage_aux_classes=model_cfg.get("stage_aux_classes", 8),
        adapter_hidden_dim=model_cfg.get("adapter_hidden_dim", 64),
        adapter_scale=model_cfg.get("adapter_scale", 0.1),
        task_prior_scale=model_cfg.get("task_prior_scale", 1.0),
        task_prior_negative_value=model_cfg.get("task_prior_negative_value", 0.0),
        stage_prior_scale=model_cfg.get("stage_prior_scale", 0.0),
        predicted_stage_prior_scale=model_cfg.get("predicted_stage_prior_scale", 0.0),
        task_hard_mask=model_cfg.get("task_hard_mask", False),
        use_prev_action_context=model_cfg.get("use_prev_action_context", False),
        history_len=model_cfg.get("history_len", 4),
        chunk_size=model_cfg.get("chunk_size", 3),
        chunk_future_weight=model_cfg.get("chunk_future_weight", 0.25),
        chunk_temporal_agg=model_cfg.get("chunk_temporal_agg", True),
        chunk_temporal_decay=model_cfg.get("chunk_temporal_decay", 0.35),
        prev_action_dropout=model_cfg.get("prev_action_dropout", 0.35),
        use_prev_action=model_cfg.get("recurrent_use_prev_action", False),
        runtime_horizon=model_cfg.get("recurrent_runtime_horizon"),
        sequence_final_weight=model_cfg.get("recurrent_sequence_final_weight", 1.0),
        language_dim=model_cfg.get("language_dim", 32),
        language_action_prior_scale=model_cfg.get("language_action_prior_scale", 0.8),
        language_state_text_scale=model_cfg.get("language_state_text_scale", 0.0),
        diffusion_steps=model_cfg.get("diffusion_steps", 4),
        diffusion_dim=model_cfg.get("diffusion_dim", 32),
        diffusion_loss_weight=model_cfg.get("diffusion_loss_weight", 0.5),
        diffusion_logit_blend=model_cfg.get("diffusion_logit_blend", 0.5),
    )


def _build_dataset(cfg: dict, path: str | Path, train: bool) -> TrajectoryDataset:
    model_cfg = cfg["model"]
    backbone_type = model_cfg.get("backbone_type", "feedforward").lower()
    if backbone_type in {"recurrent", "gru", "rnn", "chunking", "chunk", "act"}:
        return HistoryTrajectoryDataset(
            path,
            history_len=model_cfg.get("history_len", 4),
            chunk_size=model_cfg.get("chunk_size", 3),
        )
    return TrajectoryDataset(path)


def _build_sampler(cfg: dict, dataset: TrajectoryDataset) -> WeightedRandomSampler | None:
    if not bool(cfg["train"].get("use_weighted_sampler", True)):
        return None
    num_samples = len(dataset)
    if num_samples == 0:
        return None
    task_ids = np.asarray(dataset.tasks)
    success = np.asarray(dataset.success, dtype=np.float32)
    weights = np.ones(num_samples, dtype=np.float32)
    unique_tasks, counts = np.unique(task_ids, return_counts=True)
    task_weight_map = {int(task): float(num_samples) / float(len(unique_tasks) * count) for task, count in zip(unique_tasks, counts)}
    for task, weight in task_weight_map.items():
        weights[task_ids == task] *= weight
    primitive_ids = np.asarray(dataset.primitive_ids)
    stage_ids = np.asarray(getattr(dataset, "stage_ids", np.zeros_like(primitive_ids)))
    primitive_balance_power = float(cfg["train"].get("primitive_balance_power", 0.0))
    if primitive_balance_power > 0.0:
        for task in unique_tasks:
            task_mask = task_ids == task
            task_primitives = primitive_ids[task_mask]
            primitive_vals, primitive_counts = np.unique(task_primitives, return_counts=True)
            if len(primitive_vals) == 0:
                continue
            primitive_weight_map = {
                int(primitive): (float(task_mask.sum()) / float(len(primitive_vals) * count)) ** primitive_balance_power
                for primitive, count in zip(primitive_vals, primitive_counts)
            }
            for primitive, primitive_weight in primitive_weight_map.items():
                mask = task_mask & (primitive_ids == primitive)
                weights[mask] *= primitive_weight
    stage_balance_power = float(cfg["train"].get("stage_balance_power", 0.0))
    if stage_balance_power > 0.0:
        for task in unique_tasks:
            task_mask = task_ids == task
            task_stages = stage_ids[task_mask]
            stage_vals, stage_counts = np.unique(task_stages, return_counts=True)
            if len(stage_vals) == 0:
                continue
            stage_weight_map = {
                int(stage): (float(task_mask.sum()) / float(len(stage_vals) * count)) ** stage_balance_power
                for stage, count in zip(stage_vals, stage_counts)
            }
            for stage, stage_weight in stage_weight_map.items():
                mask = task_mask & (stage_ids == stage)
                weights[mask] *= stage_weight
    success_boost = float(cfg["train"].get("success_sample_boost", 2.0))
    l3_success_boost = float(cfg["train"].get("level3_success_boost", 6.0))
    weights[success > 0.5] *= success_boost
    weights[(task_ids == 2) & (success > 0.5)] *= l3_success_boost
    late_l3_boost = float(cfg["train"].get("level3_late_primitive_boost", 1.0))
    if late_l3_boost > 1.0:
        late_mask = (task_ids == 2) & np.isin(primitive_ids, np.asarray([GRASP_EXECUTE_ID, LIFT_OBJECT_ID, PLACE_OBJECT_ID]))
        weights[late_mask] *= late_l3_boost
    l3_stage_boost = float(cfg["train"].get("level3_transport_stage_boost", 1.0))
    if l3_stage_boost > 1.0:
        transport_mask = (task_ids == 2) & np.isin(stage_ids, np.asarray([5, 6, 7]))
        weights[transport_mask] *= l3_stage_boost
    if bool(cfg["train"].get("normalize_task_mass_after_boost", True)):
        target_mass = 1.0 / max(len(unique_tasks), 1)
        total_weight = float(weights.sum())
        if total_weight > 0.0:
            weights /= total_weight
            for task in unique_tasks:
                mask = task_ids == task
                task_mass = float(weights[mask].sum())
                if task_mass > 0.0:
                    weights[mask] *= target_mass / task_mass
    sampler_seed = int(cfg.get("seed", 0)) + 17
    generator = torch.Generator().manual_seed(sampler_seed)
    return WeightedRandomSampler(
        torch.from_numpy(weights),
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    )


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(cfg: dict, train_path: str | Path, val_path: str | Path) -> Path:
    _set_random_seed(int(cfg.get("seed", 0)))
    device = torch.device(cfg["train"]["device"])
    model = build_model(cfg).to(device)
    train_dataset = _build_dataset(cfg, train_path, train=True)
    val_dataset = _build_dataset(cfg, val_path, train=False)
    sampler = _build_sampler(cfg, train_dataset)
    loader_seed = int(cfg.get("seed", 0)) + 29
    loader_generator = torch.Generator().manual_seed(loader_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=sampler is None,
        sampler=sampler,
        generator=loader_generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg["train"]["batch_size"], shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(cfg["train"]["epochs"]), 1),
        eta_min=float(cfg["train"].get("min_lr", 1.0e-5)),
    )
    best_val = float("inf")
    checkpoint_dir = ensure_dir(cfg["paths"]["checkpoint_dir"])
    best_path = checkpoint_dir / "best_model.pt"
    transition_weight = float(cfg["train"].get("transition_loss_weight", 0.5))
    stage_loss_weight = float(cfg["train"].get("stage_loss_weight", 0.0))
    for _ in range(cfg["train"]["epochs"]):
        model.train()
        for batch in tqdm(train_loader, desc="train", leave=False):
            optimizer.zero_grad(set_to_none=True)
            batch = _to_device(batch, device)
            z, next_z = model.compute_latents(batch)
            pred_next = model.predict_next(z, batch["primitive_id"], batch.get("state"), task_ids=batch.get("task"))
            loss_policy = model.compute_policy_loss(batch, z)
            loss_trans = F.mse_loss(pred_next, next_z.detach())
            loss_stage = model.compute_stage_loss(batch, z)
            loss = loss_policy + transition_weight * loss_trans + stage_loss_weight * loss_stage
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        val_loss = _evaluate_loss(model, val_loader, transition_weight, stage_loss_weight, device)
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(), "config": cfg}, best_path)
        scheduler.step()
    return best_path


def calibrate_adapter(cfg: dict, checkpoint_path: str | Path, real_path: str | Path) -> Path:
    _set_random_seed(int(cfg.get("seed", 0)) + 101)
    device = torch.device(cfg["train"]["device"])
    payload = torch.load(checkpoint_path, map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(payload["model_state"])
    model.freeze_backbone()
    model.train()
    loader = DataLoader(RealCalibrationDataset(real_path), batch_size=cfg["adaptation"]["batch_size"], shuffle=True)
    optimizer = torch.optim.AdamW(model.adapter_parameters(), lr=cfg["adaptation"]["lr"])
    reg_weight = float(cfg["adaptation"].get("adapter_reg_weight", 0.1))
    epochs = int(cfg["adaptation"].get("epochs", 10))
    for _ in range(epochs):
        for batch in tqdm(loader, desc="calibrate", leave=False):
            optimizer.zero_grad(set_to_none=True)
            batch = _to_device(batch, device)
            z = model.encode(batch["image"], batch["state"], task_ids=batch.get("task")).detach()
            next_z = model.encode(batch["next_image"], batch["next_state"], task_ids=batch.get("task")).detach()
            z_prime = model.adapt(z)
            next_z_prime = model.adapt(next_z)
            pred_next = model.predict_next(z_prime, batch["primitive_id"], batch.get("state"), task_ids=batch.get("task"))
            loss_adapt = F.mse_loss(pred_next, next_z_prime)
            loss_reg = F.mse_loss(z_prime, z)
            loss = loss_adapt + reg_weight * loss_reg
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.adapter_parameters()), 5.0)
            optimizer.step()
    checkpoint_dir = ensure_dir(cfg["paths"]["checkpoint_dir"])
    adapter_path = checkpoint_dir / "adapter_calibrated.pt"
    torch.save({"model_state": model.state_dict(), "config": cfg}, adapter_path)
    return adapter_path


@torch.no_grad()
def _evaluate_loss(model: BaseTTLAModel, loader: DataLoader, transition_weight: float, stage_loss_weight: float, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        batch = _to_device(batch, device)
        z, next_z = model.compute_latents(batch)
        pred_next = model.predict_next(z, batch["primitive_id"], batch.get("state"), task_ids=batch.get("task"))
        loss = (
            model.compute_policy_loss(batch, z)
            + transition_weight * F.mse_loss(pred_next, next_z)
            + stage_loss_weight * model.compute_stage_loss(batch, z)
        )
        total += float(loss.item()) * batch["primitive_id"].size(0)
        count += batch["primitive_id"].size(0)
    model.train()
    return total / max(count, 1)


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

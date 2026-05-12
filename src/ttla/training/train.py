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
from ..models import BaseTTLAModel, build_backbone_model, load_model_state
from ..sim.skills import (
    ABORT_ID,
    ABORT_FAMILY_ID,
    APPROACH_COARSE_ID,
    APPROACH_FAMILY_ID,
    APPROACH_FINE_ID,
    CONFIRM_FAMILY_ID,
    GRASP_EXECUTE_ID,
    GRASP_FAMILY_ID,
    HOLD_POSITION_ID,
    HOLD_FAMILY_ID,
    LIFT_OBJECT_ID,
    LIFT_FAMILY_ID,
    OBS_CENTER_ID,
    OBS_LEFT_ID,
    OBS_RIGHT_ID,
    OBSERVE_FAMILY_ID,
    PLACE_FAMILY_ID,
    PLACE_OBJECT_ID,
    PREALIGN_GRASP_ID,
    PREGRASP_SERVO_ID,
    PRIMITIVE_VOCAB_LEGACY,
    REOBSERVE_ID,
    RECOVER_FAMILY_ID,
    RETREAT_ID,
    TRANSPORT_TO_DROPZONE_ID,
    TRANSPORT_FAMILY_ID,
    VERIFY_TARGET_ID,
    family_projected_primitives,
    project_primitive_ids,
    remap_primitive_id,
)
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
        adapter_mode=model_cfg.get("adapter_mode", "full"),
        adapter_use_gate=model_cfg.get("adapter_use_gate", True),
        adapter_use_condition_branch=model_cfg.get("adapter_use_condition_branch", True),
        adapter_use_task_condition=model_cfg.get("adapter_use_task_condition", True),
        adapter_use_stage_condition=model_cfg.get("adapter_use_stage_condition", True),
        adapter_use_prev_action_condition=model_cfg.get("adapter_use_prev_action_condition", True),
        adapter_progressive_min_scale=model_cfg.get("adapter_progressive_min_scale", 1.0),
        adapter_progressive_max_scale=model_cfg.get("adapter_progressive_max_scale", 1.0),
        adapter_stage_scales=model_cfg.get("adapter_stage_scales"),
        adapter_condition_start_stage=model_cfg.get("adapter_condition_start_stage"),
        adapter_condition_end_stage=model_cfg.get("adapter_condition_end_stage"),
        adapter_condition_observation_only=model_cfg.get("adapter_condition_observation_only", False),
        adapter_condition_non_observation_scale=model_cfg.get("adapter_condition_non_observation_scale", 0.0),
        adapter_phase_split=model_cfg.get("adapter_phase_split", False),
        task_prior_scale=model_cfg.get("task_prior_scale", 1.0),
        task_prior_negative_value=model_cfg.get("task_prior_negative_value", 0.0),
        stage_prior_scale=model_cfg.get("stage_prior_scale", 0.0),
        predicted_stage_prior_scale=model_cfg.get("predicted_stage_prior_scale", 0.0),
        task_hard_mask=model_cfg.get("task_hard_mask", False),
        stage_hard_mask=model_cfg.get("stage_hard_mask", False),
        latent_affine_alignment=model_cfg.get("latent_affine_alignment", False),
        latent_affine_task_conditioned=model_cfg.get("latent_affine_task_conditioned", True),
        latent_affine_max_scale=model_cfg.get("latent_affine_max_scale", 4.0),
        latent_affine_blend=model_cfg.get("latent_affine_blend", 1.0),
        transition_action_adapter=model_cfg.get("transition_action_adapter", False),
        transition_action_adapter_scale=model_cfg.get("transition_action_adapter_scale", 1.0),
        transition_residual_adapter=model_cfg.get("transition_residual_adapter", False),
        transition_residual_hidden_dim=model_cfg.get("transition_residual_hidden_dim", 64),
        transition_residual_scale=model_cfg.get("transition_residual_scale", 0.1),
        transition_residual_phase_split=model_cfg.get("transition_residual_phase_split", False),
        transition_residual_observation_scale=model_cfg.get("transition_residual_observation_scale", 1.0),
        transition_residual_non_observation_scale=model_cfg.get("transition_residual_non_observation_scale", 1.0),
        policy_residual_adapter=model_cfg.get("policy_residual_adapter", False),
        policy_residual_hidden_dim=model_cfg.get("policy_residual_hidden_dim", 64),
        policy_residual_scale=model_cfg.get("policy_residual_scale", 0.1),
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
        recurrent_adapter_use_feature=model_cfg.get("recurrent_adapter_use_feature", True),
        recurrent_adapter_use_latent=model_cfg.get("recurrent_adapter_use_latent", True),
        language_dim=model_cfg.get("language_dim", 32),
        language_action_prior_scale=model_cfg.get("language_action_prior_scale", 0.8),
        language_state_text_scale=model_cfg.get("language_state_text_scale", 0.0),
        diffusion_steps=model_cfg.get("diffusion_steps", 4),
        diffusion_dim=model_cfg.get("diffusion_dim", 32),
        diffusion_loss_weight=model_cfg.get("diffusion_loss_weight", 0.5),
        diffusion_logit_blend=model_cfg.get("diffusion_logit_blend", 0.5),
        primitive_vocabulary=model_cfg.get("primitive_vocabulary", PRIMITIVE_VOCAB_LEGACY),
    )


def _build_dataset(cfg: dict, path: str | Path, train: bool) -> TrajectoryDataset:
    model_cfg = cfg["model"]
    backbone_type = model_cfg.get("backbone_type", "feedforward").lower()
    primitive_vocabulary = model_cfg.get("primitive_vocabulary", "legacy")
    if backbone_type in {"recurrent", "gru", "rnn", "chunking", "chunk", "act"}:
        return HistoryTrajectoryDataset(
            path,
            history_len=model_cfg.get("history_len", 4),
            chunk_size=model_cfg.get("chunk_size", 3),
            primitive_vocabulary=primitive_vocabulary,
        )
    return TrajectoryDataset(path, primitive_vocabulary=primitive_vocabulary)


def _build_sampler(cfg: dict, dataset: TrajectoryDataset) -> WeightedRandomSampler | None:
    if not bool(cfg["train"].get("use_weighted_sampler", True)):
        return None
    num_samples = len(dataset)
    if num_samples == 0:
        return None
    task_ids = np.asarray(dataset.tasks)
    success = np.asarray(dataset.success, dtype=np.float32)
    primitive_vocabulary = cfg["model"].get("primitive_vocabulary", PRIMITIVE_VOCAB_LEGACY)
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
        late_ids = np.asarray(
            _projected_ids(
                (GRASP_EXECUTE_ID, LIFT_OBJECT_ID, TRANSPORT_TO_DROPZONE_ID, PLACE_OBJECT_ID),
                primitive_vocabulary=primitive_vocabulary,
            )
        )
        late_mask = (task_ids == 2) & np.isin(primitive_ids, late_ids)
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


def _projected_family_ids(
    family_ids: tuple[int, ...] | list[int] | set[int],
    primitive_vocabulary: str,
) -> tuple[int, ...]:
    return family_projected_primitives(tuple(family_ids), primitive_vocabulary=primitive_vocabulary)


def _projected_ids(
    primitive_ids: tuple[int, ...] | list[int] | set[int],
    primitive_vocabulary: str,
) -> tuple[int, ...]:
    return project_primitive_ids(tuple(primitive_ids), primitive_vocabulary=primitive_vocabulary)


def _build_calibration_dataset(cfg: dict, path: str | Path):
    model_cfg = cfg["model"]
    backbone_type = model_cfg.get("backbone_type", "feedforward").lower()
    primitive_vocabulary = model_cfg.get("primitive_vocabulary", "legacy")
    if backbone_type in {"recurrent", "gru", "rnn", "chunking", "chunk", "act"}:
        return HistoryTrajectoryDataset(
            path,
            history_len=model_cfg.get("history_len", 4),
            chunk_size=model_cfg.get("chunk_size", 3),
            primitive_vocabulary=primitive_vocabulary,
        )
    return RealCalibrationDataset(path, primitive_vocabulary=primitive_vocabulary)


def _build_adaptation_sampler(cfg: dict, dataset) -> WeightedRandomSampler | None:
    if not bool(cfg["adaptation"].get("use_weighted_sampler", True)):
        return None
    num_samples = len(dataset)
    if num_samples == 0:
        return None
    task_ids = np.asarray(dataset.tasks)
    stage_ids = np.asarray(dataset.stage_ids)
    weights = np.ones(num_samples, dtype=np.float32)
    unique_tasks, counts = np.unique(task_ids, return_counts=True)
    task_balance_power = float(cfg["adaptation"].get("task_balance_power", 0.75))
    if task_balance_power > 0.0 and len(unique_tasks) > 0:
        task_weight_map = {
            int(task): (float(num_samples) / float(len(unique_tasks) * count)) ** task_balance_power
            for task, count in zip(unique_tasks, counts)
        }
        for task, weight in task_weight_map.items():
            weights[task_ids == task] *= weight
    stage_balance_power = float(cfg["adaptation"].get("stage_balance_power", 0.75))
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
                weights[task_mask & (stage_ids == stage)] *= stage_weight
    l3_late_stage_boost = float(cfg["adaptation"].get("level3_late_stage_boost", 1.0))
    if l3_late_stage_boost > 1.0:
        late_mask = (task_ids == 2) & np.isin(stage_ids, np.asarray([4, 5, 6, 7]))
        weights[late_mask] *= l3_late_stage_boost
    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        return None
    weights /= total_weight
    sampler_seed = int(cfg.get("seed", 0)) + 211
    generator = torch.Generator().manual_seed(sampler_seed)
    return WeightedRandomSampler(
        torch.from_numpy(weights),
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    )


def _build_adaptation_loader(cfg: dict, dataset) -> DataLoader:
    sampler = _build_adaptation_sampler(cfg, dataset)
    loader_seed = int(cfg.get("seed", 0)) + 223
    loader_generator = torch.Generator().manual_seed(loader_seed)
    return DataLoader(
        dataset,
        batch_size=cfg["adaptation"]["batch_size"],
        shuffle=sampler is None,
        sampler=sampler,
        generator=loader_generator,
    )


def _estimate_mean_transition_residual(
    model: BaseTTLAModel,
    loader: DataLoader,
    device: torch.device,
) -> float:
    residual_sum = 0.0
    residual_count = 0
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            z, next_z = model.compute_latents(batch)
            pred_next = model.predict_next(z, batch["primitive_id"], batch.get("state"), task_ids=batch.get("task"))
            residual = F.mse_loss(pred_next, next_z, reduction="none").mean(dim=-1)
            residual_sum += float(residual.sum().item())
            residual_count += int(residual.numel())
    if residual_count <= 0:
        return 0.0
    return residual_sum / float(residual_count)


def _global_shift_gate(
    mean_transition_residual: float,
    threshold: float,
    slope: float,
) -> float:
    return float(
        torch.sigmoid(
            torch.tensor(
                slope * (mean_transition_residual - threshold),
                dtype=torch.float32,
            )
        ).item()
    )


@torch.no_grad()
def _estimate_adapter_stage_scales(
    model: BaseTTLAModel,
    loader: DataLoader,
    device: torch.device,
    *,
    min_scale: float,
    max_scale: float,
    power: float,
) -> torch.Tensor | None:
    num_stages = int(getattr(model, "stage_aux_classes", 0))
    if num_stages <= 0:
        return None
    residual_sum = torch.zeros(num_stages, device=device, dtype=torch.float32)
    residual_count = torch.zeros(num_stages, device=device, dtype=torch.float32)
    saw_stage = False
    for batch in loader:
        if "stage_id" not in batch:
            continue
        batch = _to_device(batch, device)
        z, next_z = model.compute_latents(batch)
        z_base = model.align_latent(z, task_ids=batch.get("task"), state=batch.get("state"))
        next_z_base = model.align_latent(next_z, task_ids=batch.get("task"), state=batch.get("next_state"))
        pred_next = model.predict_next(z_base, batch["primitive_id"], batch.get("state"), task_ids=batch.get("task"))
        residual = F.mse_loss(pred_next, next_z_base, reduction="none").mean(dim=-1).detach()
        stage_ids = batch["stage_id"].long().clamp(0, num_stages - 1)
        residual_sum.scatter_add_(0, stage_ids, residual.to(dtype=residual_sum.dtype))
        residual_count.scatter_add_(0, stage_ids, torch.ones_like(residual, dtype=residual_count.dtype))
        saw_stage = True
    if not saw_stage:
        return None
    stage_means = torch.where(
        residual_count > 0,
        residual_sum / residual_count.clamp_min(1.0),
        torch.zeros_like(residual_sum),
    )
    valid = residual_count > 0
    if not bool(valid.any()):
        return torch.ones(num_stages, device=device, dtype=torch.float32)
    valid_means = stage_means[valid]
    lo = float(valid_means.min().item())
    hi = float(valid_means.max().item())
    if hi - lo < 1.0e-8:
        return torch.ones(num_stages, device=device, dtype=torch.float32)
    norm = torch.zeros_like(stage_means)
    norm[valid] = ((stage_means[valid] - lo) / max(hi - lo, 1.0e-8)).clamp(0.0, 1.0)
    if power != 1.0:
        norm = norm.pow(float(max(power, 1.0e-6)))
    scale_min = float(min(min_scale, max_scale))
    scale_max = float(max(min_scale, max_scale))
    scales = torch.ones_like(stage_means)
    scales[valid] = scale_min + norm[valid] * (scale_max - scale_min)
    return scales


@torch.no_grad()
def _collect_task_latent_stats(
    model: BaseTTLAModel,
    cfg: dict,
    dataset,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=cfg["adaptation"]["batch_size"], shuffle=False)
    latent_chunks: list[torch.Tensor] = []
    task_chunks: list[torch.Tensor] = []
    for batch in loader:
        batch = _to_device(batch, device)
        z, _ = model.compute_latents(batch)
        latent_chunks.append(z.detach())
        task_chunks.append(batch["task"].detach())
    if not latent_chunks:
        task_vocab_size = int(cfg["model"].get("task_vocab_size", 3))
        latent_dim = int(cfg["model"]["latent_dim"])
        return (
            torch.zeros((task_vocab_size, latent_dim), device=device),
            torch.ones((task_vocab_size, latent_dim), device=device),
        )
    z_all = torch.cat(latent_chunks, dim=0)
    task_all = torch.cat(task_chunks, dim=0)
    task_vocab_size = int(cfg["model"].get("task_vocab_size", 3))
    global_mean = z_all.mean(dim=0)
    global_std = z_all.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    means: list[torch.Tensor] = []
    stds: list[torch.Tensor] = []
    for task_id in range(task_vocab_size):
        mask = task_all == task_id
        if bool(mask.any()):
            task_latents = z_all[mask]
            means.append(task_latents.mean(dim=0))
            stds.append(task_latents.std(dim=0, unbiased=False).clamp_min(1.0e-6))
        else:
            means.append(global_mean)
            stds.append(global_std)
    return torch.stack(means, dim=0), torch.stack(stds, dim=0)


@torch.no_grad()
def _estimate_latent_alignment_stats(
    model: BaseTTLAModel,
    cfg: dict,
    source_path: str | Path,
    target_path: str | Path,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    source_dataset = _build_dataset(cfg, source_path, train=False)
    target_dataset = _build_dataset(cfg, target_path, train=False)
    source_mean, source_std = _collect_task_latent_stats(model, cfg, source_dataset, device)
    target_mean, target_std = _collect_task_latent_stats(model, cfg, target_dataset, device)
    if not bool(cfg["model"].get("latent_affine_task_conditioned", True)):
        source_global_mean = source_mean.mean(dim=0, keepdim=True)
        source_global_std = source_std.mean(dim=0, keepdim=True)
        target_global_mean = target_mean.mean(dim=0, keepdim=True)
        target_global_std = target_std.mean(dim=0, keepdim=True)
        source_mean = source_global_mean.expand_as(source_mean).clone()
        source_std = source_global_std.expand_as(source_std).clone()
        target_mean = target_global_mean.expand_as(target_mean).clone()
        target_std = target_global_std.expand_as(target_std).clone()
    return {
        "source_mean": source_mean.detach(),
        "source_std": source_std.detach(),
        "target_mean": target_mean.detach(),
        "target_std": target_std.detach(),
    }


def _kmeans_assignments(
    points: torch.Tensor,
    num_clusters: int,
    *,
    num_iters: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = int(points.shape[0])
    if count <= 0:
        raise ValueError("k-means requires at least one point")
    num_clusters = max(1, min(int(num_clusters), count))
    if num_clusters == 1:
        centroid = points.mean(dim=0, keepdim=True)
        return centroid, torch.zeros((count,), dtype=torch.long, device=points.device)
    step = max(1, count // num_clusters)
    seed_indices = torch.arange(0, count, step, device=points.device)[:num_clusters]
    if int(seed_indices.numel()) < num_clusters:
        pad = torch.arange(num_clusters - int(seed_indices.numel()), device=points.device) % count
        seed_indices = torch.cat([seed_indices, pad], dim=0)
    centroids = points.index_select(0, seed_indices).clone()
    assignments = torch.zeros((count,), dtype=torch.long, device=points.device)
    for _ in range(max(1, int(num_iters))):
        distances = torch.cdist(points, centroids)
        assignments = distances.argmin(dim=1)
        updated = centroids.clone()
        for cluster_idx in range(num_clusters):
            mask = assignments == cluster_idx
            if bool(mask.any()):
                updated[cluster_idx] = points[mask].mean(dim=0)
        if torch.allclose(updated, centroids, atol=1.0e-5, rtol=1.0e-5):
            centroids = updated
            break
        centroids = updated
    return centroids, assignments


def _estimate_source_primitive_anchors(
    model: BaseTTLAModel,
    cfg: dict,
    source_path: str | Path,
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    source_dataset = _build_dataset(cfg, source_path, train=False)
    if len(source_dataset) == 0:
        return None
    source_loader = DataLoader(source_dataset, batch_size=cfg["adaptation"]["batch_size"], shuffle=False)
    action_dim = int(cfg["model"]["action_dim"])
    latent_dim = int(cfg["model"]["latent_dim"])
    num_prototypes = max(1, int(cfg["adaptation"].get("source_local_prototypes_per_primitive", 1)))
    kmeans_iters = max(1, int(cfg["adaptation"].get("source_local_prototype_iters", 12)))
    primitive_z: list[list[torch.Tensor]] = [[] for _ in range(action_dim)]
    primitive_delta: list[list[torch.Tensor]] = [[] for _ in range(action_dim)]
    primitive_prob: list[list[torch.Tensor]] = [[] for _ in range(action_dim)]
    global_z_chunks: list[torch.Tensor] = []
    global_delta_chunks: list[torch.Tensor] = []
    global_prob_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in source_loader:
            batch = _to_device(batch, device)
            z, next_z = model.compute_latents(batch)
            logits = model.condition_policy_logits(
                model.policy_logits(z),
                z=z,
                task_ids=batch.get("task"),
                state=batch.get("state"),
            )
            probs = torch.softmax(logits, dim=-1)
            deltas = next_z - z
            primitive_ids = batch["primitive_id"].long()
            global_z_chunks.append(z.detach())
            global_delta_chunks.append(deltas.detach())
            global_prob_chunks.append(probs.detach())
            for primitive_id in primitive_ids.unique(sorted=True):
                mask = primitive_ids == primitive_id
                primitive_idx = int(primitive_id.item())
                primitive_z[primitive_idx].append(z[mask].detach())
                primitive_delta[primitive_idx].append(deltas[mask].detach())
                primitive_prob[primitive_idx].append(probs[mask].detach())
    if not global_z_chunks:
        return None
    global_z = torch.cat(global_z_chunks, dim=0)
    global_delta = torch.cat(global_delta_chunks, dim=0)
    global_prob = torch.cat(global_prob_chunks, dim=0)
    global_mean = global_z.mean(dim=0)
    global_std = global_z.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    global_delta_mean = global_delta.mean(dim=0)
    global_policy = global_prob.mean(dim=0)
    means = torch.zeros((action_dim, num_prototypes, latent_dim), device=device)
    stds = torch.zeros((action_dim, num_prototypes, latent_dim), device=device)
    deltas = torch.zeros((action_dim, num_prototypes, latent_dim), device=device)
    policies = torch.zeros((action_dim, num_prototypes, action_dim), device=device)
    counts = torch.zeros((action_dim, num_prototypes), device=device)
    for primitive_idx in range(action_dim):
        if primitive_z[primitive_idx]:
            z_cat = torch.cat(primitive_z[primitive_idx], dim=0)
            delta_cat = torch.cat(primitive_delta[primitive_idx], dim=0)
            prob_cat = torch.cat(primitive_prob[primitive_idx], dim=0)
            centroids, assignments = _kmeans_assignments(
                z_cat,
                num_prototypes,
                num_iters=kmeans_iters,
            )
            actual_clusters = int(centroids.shape[0])
            for cluster_idx in range(actual_clusters):
                mask = assignments == cluster_idx
                if not bool(mask.any()):
                    continue
                z_sel = z_cat[mask]
                delta_sel = delta_cat[mask]
                prob_sel = prob_cat[mask]
                means[primitive_idx, cluster_idx] = centroids[cluster_idx]
                stds[primitive_idx, cluster_idx] = z_sel.std(dim=0, unbiased=False).clamp_min(1.0e-6)
                deltas[primitive_idx, cluster_idx] = delta_sel.mean(dim=0)
                policies[primitive_idx, cluster_idx] = prob_sel.mean(dim=0)
                counts[primitive_idx, cluster_idx] = float(mask.sum().item())
            if actual_clusters < num_prototypes:
                means[primitive_idx, actual_clusters:] = global_mean
                stds[primitive_idx, actual_clusters:] = global_std
                deltas[primitive_idx, actual_clusters:] = global_delta_mean
                policies[primitive_idx, actual_clusters:] = global_policy
                counts[primitive_idx, actual_clusters:] = 1.0
        else:
            means[primitive_idx] = global_mean
            stds[primitive_idx] = global_std
            deltas[primitive_idx] = global_delta_mean
            policies[primitive_idx] = global_policy
            counts[primitive_idx] = 1.0
    policies = policies / policies.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
    return {
        "means": means.detach(),
        "stds": stds.detach(),
        "deltas": deltas.detach(),
        "policies": policies.detach(),
        "counts": counts.detach(),
    }


def _policy_preservation_loss(
    model: BaseTTLAModel,
    batch: dict[str, torch.Tensor],
    z: torch.Tensor,
    z_prime: torch.Tensor,
    next_z: torch.Tensor,
    *,
    alpha: float,
    temperature: float,
    mode: str = "full",
    stage_max_id: int | None = None,
    observation_only: bool = False,
    observation_scale: float | None = None,
    non_observation_scale: float = 1.0,
) -> torch.Tensor:
    mode = str(mode).lower()
    primitive_vocabulary = getattr(model, "primitive_vocabulary", PRIMITIVE_VOCAB_LEGACY)
    with torch.no_grad():
        teacher_logits = model.condition_policy_logits(
            model.policy_logits(z),
            z=z,
            task_ids=batch.get("task"),
            state=batch.get("state"),
        )
        source_pred_next = model.predict_next(
            z,
            batch["primitive_id"],
            batch.get("state"),
            task_ids=batch.get("task"),
        )
        transition_residual = F.mse_loss(source_pred_next, next_z, reduction="none").mean(dim=-1)
        weights = torch.exp(-alpha * transition_residual).detach()
    student_logits = model.condition_policy_logits(
        model.policy_logits(z_prime),
        z=z_prime,
        task_ids=batch.get("task"),
        state=batch.get("state"),
    )
    if mode == "family":
        family_groups = (
            _projected_family_ids(
                (OBSERVE_FAMILY_ID, CONFIRM_FAMILY_ID, HOLD_FAMILY_ID),
                primitive_vocabulary=primitive_vocabulary,
            ),
            _projected_family_ids(
                (APPROACH_FAMILY_ID, RECOVER_FAMILY_ID),
                primitive_vocabulary=primitive_vocabulary,
            ),
            _projected_family_ids(
                (GRASP_FAMILY_ID, LIFT_FAMILY_ID, TRANSPORT_FAMILY_ID, PLACE_FAMILY_ID),
                primitive_vocabulary=primitive_vocabulary,
            ),
            _projected_family_ids((ABORT_FAMILY_ID,), primitive_vocabulary=primitive_vocabulary),
        )
        teacher_family_logits = torch.stack(
            [torch.logsumexp(teacher_logits[:, group], dim=-1) for group in family_groups],
            dim=-1,
        )
        student_family_logits = torch.stack(
            [torch.logsumexp(student_logits[:, group], dim=-1) for group in family_groups],
            dim=-1,
        )
        teacher_probs = torch.softmax(teacher_family_logits / temperature, dim=-1)
        student_log_probs = torch.log_softmax(student_family_logits / temperature, dim=-1)
    else:
        teacher_probs = torch.softmax(teacher_logits / temperature, dim=-1)
        student_log_probs = torch.log_softmax(student_logits / temperature, dim=-1)
    policy_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    weighted = weights * policy_kl
    if observation_scale is None:
        observation_scale = 1.0 if observation_only else 1.0
    if "primitive_id" in batch and (
        observation_only
        or float(observation_scale) != 1.0
        or float(non_observation_scale) != 1.0
    ):
        primitive_ids = batch["primitive_id"].long()
        family_scale = _branch_scale_vector(
            primitive_ids,
            primitive_vocabulary=primitive_vocabulary,
            observation_scale=float(observation_scale),
            non_observation_scale=float(non_observation_scale),
        )
        weighted = weighted * family_scale
    if stage_max_id is not None and "stage_id" in batch:
        stage_mask = (batch["stage_id"].long() <= int(stage_max_id)).to(dtype=weighted.dtype)
        denom = stage_mask.sum().clamp_min(1.0)
        return ((weighted * stage_mask).sum() / denom) * (temperature ** 2)
    return weighted.mean() * (temperature ** 2)


def _residual_adaptive_reg_weights(
    source_transition_residual: torch.Tensor,
    *,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    weights = 1.0 + gamma * torch.exp(-alpha * source_transition_residual.detach())
    return weights / weights.mean().clamp_min(1.0e-6)


def _residual_adaptive_adapt_weights(
    source_transition_residual: torch.Tensor,
    *,
    alpha: float,
    gamma: float,
    center: float = 0.0,
    min_weight: float = 1.0,
    normalize: bool = True,
) -> torch.Tensor:
    detached = source_transition_residual.detach()
    if center > 0.0 or min_weight < 1.0:
        max_weight = max(1.0 + gamma, min_weight)
        weights = min_weight + (max_weight - min_weight) * torch.sigmoid(alpha * (detached - center))
    else:
        weights = 1.0 + gamma * (1.0 - torch.exp(-alpha * detached))
    if normalize:
        weights = weights / weights.mean().clamp_min(1.0e-6)
    return weights


def _primitive_family_adaptation_weights(
    primitive_ids: torch.Tensor,
    *,
    primitive_vocabulary: str = PRIMITIVE_VOCAB_LEGACY,
    observation_weight: float,
    precontact_weight: float,
    postcontact_weight: float,
    abort_weight: float,
) -> torch.Tensor:
    weights = torch.full_like(primitive_ids, float(precontact_weight), dtype=torch.float32)
    observation_ids = set(
        _projected_family_ids(
            (OBSERVE_FAMILY_ID, CONFIRM_FAMILY_ID, HOLD_FAMILY_ID),
            primitive_vocabulary=primitive_vocabulary,
        )
    )
    postcontact_ids = set(
        _projected_family_ids(
            (GRASP_FAMILY_ID, LIFT_FAMILY_ID, TRANSPORT_FAMILY_ID, PLACE_FAMILY_ID),
            primitive_vocabulary=primitive_vocabulary,
        )
    )
    abort_ids = set(_projected_family_ids((ABORT_FAMILY_ID,), primitive_vocabulary=primitive_vocabulary))
    if observation_ids:
        observation_mask = torch.zeros_like(primitive_ids, dtype=torch.bool)
        for primitive_id in observation_ids:
            observation_mask |= primitive_ids == primitive_id
        weights = torch.where(
            observation_mask,
            torch.full_like(weights, float(observation_weight)),
            weights,
        )
    if postcontact_ids:
        postcontact_mask = torch.zeros_like(primitive_ids, dtype=torch.bool)
        for primitive_id in postcontact_ids:
            postcontact_mask |= primitive_ids == primitive_id
        weights = torch.where(
            postcontact_mask,
            torch.full_like(weights, float(postcontact_weight)),
            weights,
        )
    if abort_ids:
        abort_mask = torch.zeros_like(primitive_ids, dtype=torch.bool)
        for primitive_id in abort_ids:
            abort_mask |= primitive_ids == primitive_id
        weights = torch.where(
            abort_mask,
            torch.full_like(weights, float(abort_weight)),
            weights,
        )
    return weights / weights.mean().clamp_min(1.0e-6)


def _observation_family_mask(
    primitive_ids: torch.Tensor,
    *,
    primitive_vocabulary: str = PRIMITIVE_VOCAB_LEGACY,
) -> torch.Tensor:
    observation_ids = set(
        _projected_family_ids(
            (OBSERVE_FAMILY_ID, CONFIRM_FAMILY_ID, HOLD_FAMILY_ID),
            primitive_vocabulary=primitive_vocabulary,
        )
    )
    mask = torch.zeros_like(primitive_ids, dtype=torch.bool)
    for primitive_id in observation_ids:
        mask |= primitive_ids == primitive_id
    return mask


def _branch_scale_vector(
    primitive_ids: torch.Tensor,
    *,
    primitive_vocabulary: str = PRIMITIVE_VOCAB_LEGACY,
    observation_scale: float = 1.0,
    non_observation_scale: float = 1.0,
) -> torch.Tensor:
    observation_mask = _observation_family_mask(
        primitive_ids,
        primitive_vocabulary=primitive_vocabulary,
    )
    return torch.where(
        observation_mask,
        torch.full_like(primitive_ids, float(observation_scale), dtype=torch.float32),
        torch.full_like(primitive_ids, float(non_observation_scale), dtype=torch.float32),
    )


def _weighted_branch_average(
    per_sample_loss: torch.Tensor,
    sample_weights: torch.Tensor,
    branch_scales: torch.Tensor,
) -> torch.Tensor:
    effective = sample_weights * branch_scales
    return (effective * per_sample_loss).sum() / effective.sum().clamp_min(1.0e-6)


def _estimate_transition_branch_stats(
    model: BaseTTLAModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float | None, float | None]:
    primitive_vocabulary = getattr(model, "primitive_vocabulary", PRIMITIVE_VOCAB_LEGACY)
    residual_sum = 0.0
    residual_count = 0
    obs_sum = 0.0
    obs_count = 0
    nonobs_sum = 0.0
    nonobs_count = 0
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            z, next_z = model.compute_latents(batch)
            pred_next = model.predict_next(
                z,
                batch["primitive_id"],
                batch.get("state"),
                task_ids=batch.get("task"),
            )
            residual = F.mse_loss(pred_next, next_z, reduction="none").mean(dim=-1)
            obs_mask = _observation_family_mask(
                batch["primitive_id"],
                primitive_vocabulary=primitive_vocabulary,
            )
            residual_sum += float(residual.sum().item())
            residual_count += int(residual.numel())
            if bool(obs_mask.any()):
                obs_sum += float(residual[obs_mask].sum().item())
                obs_count += int(obs_mask.sum().item())
            nonobs_mask = ~obs_mask
            if bool(nonobs_mask.any()):
                nonobs_sum += float(residual[nonobs_mask].sum().item())
                nonobs_count += int(nonobs_mask.sum().item())
    if residual_count == 0:
        return None, None
    obs_mean = (obs_sum / obs_count) if obs_count > 0 else None
    nonobs_mean = (nonobs_sum / nonobs_count) if nonobs_count > 0 else None
    return obs_mean, nonobs_mean


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
    inverse_loss_weight = float(cfg["train"].get("inverse_loss_weight", 0.0))
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
            loss_inverse = model.compute_inverse_loss(batch, z, next_z)
            loss_stage = model.compute_stage_loss(batch, z)
            loss = (
                loss_policy
                + transition_weight * loss_trans
                + inverse_loss_weight * loss_inverse
                + stage_loss_weight * loss_stage
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        val_loss = _evaluate_loss(model, val_loader, transition_weight, inverse_loss_weight, stage_loss_weight, device)
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(), "config": cfg}, best_path)
        scheduler.step()
    return best_path


def calibrate_adapter(
    cfg: dict,
    checkpoint_path: str | Path,
    real_path: str | Path,
    *,
    static_teacher_path: str | Path | None = None,
) -> Path:
    _set_random_seed(int(cfg.get("seed", 0)) + 101)
    device = torch.device(cfg["train"]["device"])
    payload = torch.load(checkpoint_path, map_location=device)
    model = build_model(cfg).to(device)
    load_model_state(model, payload["model_state"])
    model.freeze_backbone()
    model.train()
    static_teacher: BaseTTLAModel | None = None
    if static_teacher_path is not None:
        static_payload = torch.load(static_teacher_path, map_location=device)
        static_teacher = build_model(cfg).to(device)
        load_model_state(static_teacher, static_payload["model_state"])
        static_teacher.freeze_backbone()
        static_teacher.eval()
    loader = _build_adaptation_loader(cfg, _build_calibration_dataset(cfg, real_path))
    optimizer = torch.optim.AdamW(model.adapter_parameters(), lr=cfg["adaptation"]["lr"])
    reg_weight = float(cfg["adaptation"].get("adapter_reg_weight", 0.1))
    static_alignment_weight = float(cfg["adaptation"].get("static_alignment_weight", 0.0))
    stage_weight = float(cfg["adaptation"].get("adapter_stage_loss_weight", 0.0))
    auto_stage_scaling = bool(cfg["adaptation"].get("auto_stage_scaling", False))
    auto_stage_scale_min = float(cfg["adaptation"].get("auto_stage_scale_min", 1.0))
    auto_stage_scale_max = float(cfg["adaptation"].get("auto_stage_scale_max", 1.0))
    auto_stage_scale_power = float(cfg["adaptation"].get("auto_stage_scale_power", 1.0))
    auto_non_observation_scaling = bool(cfg["adaptation"].get("auto_non_observation_scaling", False))
    auto_non_observation_scale_min = float(cfg["adaptation"].get("auto_non_observation_scale_min", 0.0))
    auto_non_observation_scale_max = float(cfg["adaptation"].get("auto_non_observation_scale_max", 1.0))
    auto_non_observation_global_threshold = float(cfg["adaptation"].get("auto_non_observation_global_threshold", 0.3))
    auto_non_observation_global_slope = float(cfg["adaptation"].get("auto_non_observation_global_slope", 20.0))
    auto_latent_affine_scaling = bool(cfg["adaptation"].get("auto_latent_affine_scaling", False))
    auto_latent_affine_blend_min = float(cfg["adaptation"].get("auto_latent_affine_blend_min", 0.0))
    auto_latent_affine_blend_max = float(cfg["adaptation"].get("auto_latent_affine_blend_max", 1.0))
    auto_latent_affine_global_threshold = float(cfg["adaptation"].get("auto_latent_affine_global_threshold", 0.3))
    auto_latent_affine_global_slope = float(cfg["adaptation"].get("auto_latent_affine_global_slope", 20.0))
    source_alignment_weight = float(cfg["adaptation"].get("source_alignment_weight", 0.0))
    source_primitive_alignment_weight = float(cfg["adaptation"].get("source_primitive_alignment_weight", 0.0))
    source_delta_alignment_weight = float(cfg["adaptation"].get("source_delta_alignment_weight", 0.0))
    source_policy_alignment_weight = float(cfg["adaptation"].get("source_policy_alignment_weight", 0.0))
    source_anchor_observation_scale = float(cfg["adaptation"].get("source_anchor_observation_scale", 1.0))
    source_anchor_non_observation_scale = float(cfg["adaptation"].get("source_anchor_non_observation_scale", 1.0))
    auto_source_anchor_observation_scale = bool(
        cfg["adaptation"].get("auto_source_anchor_observation_scale", False)
    )
    auto_source_anchor_non_observation_balance = bool(
        cfg["adaptation"].get("auto_source_anchor_non_observation_balance", False)
    )
    auto_source_anchor_balance_min = float(cfg["adaptation"].get("auto_source_anchor_balance_min", 0.0))
    auto_source_anchor_balance_max = float(cfg["adaptation"].get("auto_source_anchor_balance_max", 1.0))
    auto_source_anchor_balance_threshold = float(
        cfg["adaptation"].get("auto_source_anchor_balance_threshold", 0.55)
    )
    auto_source_anchor_balance_slope = float(
        cfg["adaptation"].get("auto_source_anchor_balance_slope", 12.0)
    )
    auto_source_anchor_scaling = bool(cfg["adaptation"].get("auto_source_anchor_scaling", False))
    auto_source_anchor_scale_min = float(cfg["adaptation"].get("auto_source_anchor_scale_min", 0.0))
    auto_source_anchor_scale_max = float(cfg["adaptation"].get("auto_source_anchor_scale_max", 1.0))
    auto_source_anchor_global_threshold = float(cfg["adaptation"].get("auto_source_anchor_global_threshold", 0.3))
    auto_source_anchor_global_slope = float(cfg["adaptation"].get("auto_source_anchor_global_slope", 20.0))
    source_alignment_temperature = float(cfg["adaptation"].get("source_alignment_temperature", 1.0))
    source_alignment_use_std_norm = bool(cfg["adaptation"].get("source_alignment_use_std_norm", True))
    use_latent_affine_alignment = bool(cfg["model"].get("latent_affine_alignment", False))
    policy_preservation_weight = float(cfg["adaptation"].get("policy_preservation_weight", 0.0))
    policy_preservation_alpha = float(cfg["adaptation"].get("policy_preservation_alpha", 8.0))
    policy_preservation_temperature = float(cfg["adaptation"].get("policy_preservation_temperature", 1.0))
    policy_preservation_mode = str(cfg["adaptation"].get("policy_preservation_mode", "full"))
    policy_preservation_global_gate = bool(cfg["adaptation"].get("policy_preservation_global_gate", False))
    policy_preservation_global_threshold = float(cfg["adaptation"].get("policy_preservation_global_threshold", 0.22))
    policy_preservation_global_slope = float(cfg["adaptation"].get("policy_preservation_global_slope", 30.0))
    policy_preservation_stage_max = cfg["adaptation"].get("policy_preservation_stage_max")
    policy_preservation_observation_only = bool(cfg["adaptation"].get("policy_preservation_observation_only", False))
    policy_preservation_observation_scale = cfg["adaptation"].get("policy_preservation_observation_scale")
    policy_preservation_non_observation_scale = float(
        cfg["adaptation"].get("policy_preservation_non_observation_scale", 1.0)
    )
    if policy_preservation_observation_scale is None:
        policy_preservation_observation_scale = 1.0
    policy_preservation_observation_scale = float(policy_preservation_observation_scale)
    policy_supervision_weight = float(cfg["adaptation"].get("policy_supervision_weight", 0.0))
    inverse_supervision_weight = float(cfg["adaptation"].get("inverse_supervision_weight", 0.0))
    gate_identity_weight = float(cfg["adaptation"].get("gate_identity_weight", 0.0))
    policy_adapter_identity_weight = float(cfg["adaptation"].get("policy_adapter_identity_weight", 0.0))
    transition_adapter_identity_weight = float(
        cfg["adaptation"].get("transition_adapter_identity_weight", 0.0)
    )
    residual_adaptive_reg_alpha = float(cfg["adaptation"].get("residual_adaptive_reg_alpha", 0.0))
    residual_adaptive_reg_gamma = float(cfg["adaptation"].get("residual_adaptive_reg_gamma", 0.0))
    residual_adaptive_adapt_alpha = float(cfg["adaptation"].get("residual_adaptive_adapt_alpha", 0.0))
    residual_adaptive_adapt_gamma = float(cfg["adaptation"].get("residual_adaptive_adapt_gamma", 0.0))
    residual_adaptive_adapt_center = float(cfg["adaptation"].get("residual_adaptive_adapt_center", 0.0))
    residual_adaptive_adapt_min_weight = float(cfg["adaptation"].get("residual_adaptive_adapt_min_weight", 1.0))
    residual_adaptive_adapt_normalize = bool(cfg["adaptation"].get("residual_adaptive_adapt_normalize", True))
    observation_adapt_weight = float(cfg["adaptation"].get("observation_adapt_weight", 1.0))
    precontact_adapt_weight = float(cfg["adaptation"].get("precontact_adapt_weight", 1.0))
    postcontact_adapt_weight = float(cfg["adaptation"].get("postcontact_adapt_weight", 1.0))
    abort_adapt_weight = float(cfg["adaptation"].get("abort_adapt_weight", 1.0))
    epochs = int(cfg["adaptation"].get("epochs", 10))
    policy_preservation_scale = 1.0
    mean_transition_residual: float | None = None
    auto_non_observation_scale_value: float | None = None
    auto_latent_affine_blend_value: float | None = None
    auto_source_anchor_scale_value: float | None = None
    auto_source_anchor_balance_value: float | None = None
    effective_source_anchor_observation_scale = source_anchor_observation_scale
    effective_source_anchor_non_observation_scale = source_anchor_non_observation_scale
    observation_transition_residual: float | None = None
    non_observation_transition_residual: float | None = None
    if (
        auto_non_observation_scaling
        or auto_latent_affine_scaling
        or auto_source_anchor_scaling
        or auto_source_anchor_non_observation_balance
        or (policy_preservation_weight > 0.0 and policy_preservation_global_gate)
    ):
        mean_transition_residual = _estimate_mean_transition_residual(model, loader, device)
    if auto_source_anchor_non_observation_balance:
        (
            observation_transition_residual,
            non_observation_transition_residual,
        ) = _estimate_transition_branch_stats(model, loader, device)
    if auto_non_observation_scaling and bool(getattr(model, "adapter_condition_observation_only", False)):
        gate = _global_shift_gate(
            mean_transition_residual,
            auto_non_observation_global_threshold,
            auto_non_observation_global_slope,
        )
        auto_scale = float(
            auto_non_observation_scale_min
            + gate * (auto_non_observation_scale_max - auto_non_observation_scale_min)
        )
        model.adapter_condition_non_observation_scale = auto_scale
        cfg.setdefault("model", {})["adapter_condition_non_observation_scale"] = auto_scale
        auto_non_observation_scale_value = auto_scale
    if auto_latent_affine_scaling and use_latent_affine_alignment:
        gate = _global_shift_gate(
            mean_transition_residual,
            auto_latent_affine_global_threshold,
            auto_latent_affine_global_slope,
        )
        auto_blend = float(
            auto_latent_affine_blend_min
            + gate * (auto_latent_affine_blend_max - auto_latent_affine_blend_min)
        )
        model.latent_affine_blend = auto_blend
        cfg.setdefault("model", {})["latent_affine_blend"] = auto_blend
        auto_latent_affine_blend_value = auto_blend
    if auto_source_anchor_scaling:
        gate = _global_shift_gate(
            mean_transition_residual,
            auto_source_anchor_global_threshold,
            auto_source_anchor_global_slope,
        )
        source_scale = float(
            auto_source_anchor_scale_min
            + gate * (auto_source_anchor_scale_max - auto_source_anchor_scale_min)
        )
        source_alignment_weight *= source_scale
        source_primitive_alignment_weight *= source_scale
        source_delta_alignment_weight *= source_scale
        source_policy_alignment_weight *= source_scale
        auto_source_anchor_scale_value = source_scale
        if auto_source_anchor_observation_scale:
            effective_source_anchor_observation_scale = source_scale
    if auto_source_anchor_non_observation_balance:
        obs_mean = observation_transition_residual
        nonobs_mean = non_observation_transition_residual
        if obs_mean is not None and nonobs_mean is not None:
            obs_share = float(obs_mean / max(obs_mean + nonobs_mean, 1.0e-6))
            balance_gate = _global_shift_gate(
                obs_share,
                auto_source_anchor_balance_threshold,
                auto_source_anchor_balance_slope,
            )
            auto_source_anchor_balance_value = float(
                auto_source_anchor_balance_min
                + balance_gate * (auto_source_anchor_balance_max - auto_source_anchor_balance_min)
            )
            effective_source_anchor_non_observation_scale *= auto_source_anchor_balance_value
    if auto_stage_scaling:
        stage_scales = _estimate_adapter_stage_scales(
            model,
            loader,
            device,
            min_scale=auto_stage_scale_min,
            max_scale=auto_stage_scale_max,
            power=auto_stage_scale_power,
        )
        if stage_scales is not None:
            model.set_adapter_stage_scales(stage_scales)
    if policy_preservation_weight > 0.0 and policy_preservation_global_gate:
        shift_gate = _global_shift_gate(
            mean_transition_residual,
            policy_preservation_global_threshold,
            policy_preservation_global_slope,
        )
        policy_preservation_scale = float(1.0 - shift_gate)
    source_means = None
    source_stds = None
    source_primitive_anchors = None
    source_path = cfg.get("pseudo_real", {}).get("source_train_path")
    if source_path:
        if use_latent_affine_alignment:
            latent_alignment_stats = _estimate_latent_alignment_stats(model, cfg, source_path, real_path, device)
            model.set_latent_alignment_stats(
                latent_alignment_stats["source_mean"],
                latent_alignment_stats["source_std"],
                latent_alignment_stats["target_mean"],
                latent_alignment_stats["target_std"],
            )
        if source_alignment_weight > 0.0:
            source_dataset = _build_dataset(cfg, source_path, train=False)
            source_loader = DataLoader(source_dataset, batch_size=cfg["adaptation"]["batch_size"], shuffle=False)
            task_vocab_size = int(cfg["model"].get("task_vocab_size", 3))
            latent_chunks: list[torch.Tensor] = []
            task_chunks: list[torch.Tensor] = []
            with torch.no_grad():
                for source_batch in source_loader:
                    source_batch = _to_device(source_batch, device)
                    source_z, _ = model.compute_latents(source_batch)
                    latent_chunks.append(source_z.detach())
                    task_chunks.append(source_batch["task"].detach())
            if latent_chunks:
                z_all = torch.cat(latent_chunks, dim=0)
                task_all = torch.cat(task_chunks, dim=0)
                global_mean = z_all.mean(dim=0)
                global_std = z_all.std(dim=0, unbiased=False).clamp_min(1.0e-6)
                source_mean_list = []
                source_std_list = []
                for task_id in range(task_vocab_size):
                    mask = task_all == task_id
                    if bool(mask.any()):
                        task_latents = z_all[mask]
                        source_mean_list.append(task_latents.mean(dim=0))
                        source_std_list.append(task_latents.std(dim=0, unbiased=False).clamp_min(1.0e-6))
                    else:
                        source_mean_list.append(global_mean)
                        source_std_list.append(global_std)
                source_means = torch.stack(source_mean_list, dim=0)
                source_stds = torch.stack(source_std_list, dim=0)
        if (
            source_primitive_alignment_weight > 0.0
            or source_delta_alignment_weight > 0.0
            or source_policy_alignment_weight > 0.0
        ):
            source_primitive_anchors = _estimate_source_primitive_anchors(model, cfg, source_path, device)
    for _ in range(epochs):
        for batch in tqdm(loader, desc="calibrate", leave=False):
            optimizer.zero_grad(set_to_none=True)
            batch = _to_device(batch, device)
            z, next_z = model.compute_latents(batch)
            z = z.detach()
            next_z = next_z.detach()
            z_base = model.align_latent(z, task_ids=batch.get("task"), state=batch.get("state")).detach()
            next_z_base = model.align_latent(next_z, task_ids=batch.get("task"), state=batch.get("next_state")).detach()
            z_prime, next_z_prime = model.compute_adapted_latents(batch)
            source_pred_next = model.predict_next(z_base, batch["primitive_id"], batch.get("state"), task_ids=batch.get("task"))
            pred_next = model.predict_next(z_prime, batch["primitive_id"], batch.get("state"), task_ids=batch.get("task"))
            source_transition_residual = F.mse_loss(source_pred_next, next_z_base, reduction="none").mean(dim=-1)
            adaptation_weights = _primitive_family_adaptation_weights(
                batch["primitive_id"],
                observation_weight=observation_adapt_weight,
                precontact_weight=precontact_adapt_weight,
                postcontact_weight=postcontact_adapt_weight,
                abort_weight=abort_adapt_weight,
            )
            if residual_adaptive_adapt_alpha > 0.0 and residual_adaptive_adapt_gamma > 0.0:
                adaptation_weights = adaptation_weights * _residual_adaptive_adapt_weights(
                    source_transition_residual,
                    alpha=residual_adaptive_adapt_alpha,
                    gamma=residual_adaptive_adapt_gamma,
                    center=residual_adaptive_adapt_center,
                    min_weight=residual_adaptive_adapt_min_weight,
                    normalize=residual_adaptive_adapt_normalize,
                )
            loss_adapt = (
                adaptation_weights * F.mse_loss(pred_next, next_z_prime, reduction="none").mean(dim=-1)
            ).mean()
            if residual_adaptive_reg_alpha > 0.0 and residual_adaptive_reg_gamma > 0.0:
                residual_weights = _residual_adaptive_reg_weights(
                    source_transition_residual,
                    alpha=residual_adaptive_reg_alpha,
                    gamma=residual_adaptive_reg_gamma,
                )
                loss_reg = (
                    residual_weights * F.mse_loss(z_prime, z_base, reduction="none").mean(dim=-1)
                ).mean()
                loss_next_reg = (
                    residual_weights * F.mse_loss(next_z_prime, next_z_base, reduction="none").mean(dim=-1)
                ).mean()
            else:
                loss_reg = F.mse_loss(z_prime, z_base)
                loss_next_reg = F.mse_loss(next_z_prime, next_z_base)
            loss_stage = model.compute_stage_loss(batch, z_prime)
            loss_source = torch.zeros((), device=device)
            loss_source_primitive = torch.zeros((), device=device)
            loss_source_delta = torch.zeros((), device=device)
            loss_source_policy = torch.zeros((), device=device)
            loss_policy_pres = torch.zeros((), device=device)
            loss_policy_sup = torch.zeros((), device=device)
            loss_inverse_sup = torch.zeros((), device=device)
            loss_gate_identity = torch.zeros((), device=device)
            loss_policy_adapter_identity = torch.zeros((), device=device)
            loss_transition_identity = torch.zeros((), device=device)
            loss_static_teacher = torch.zeros((), device=device)
            primitive_vocabulary = getattr(model, "primitive_vocabulary", PRIMITIVE_VOCAB_LEGACY)
            primitive_ids = batch["primitive_id"].long()
            source_anchor_branch_scales = _branch_scale_vector(
                primitive_ids,
                primitive_vocabulary=primitive_vocabulary,
                observation_scale=effective_source_anchor_observation_scale,
                non_observation_scale=effective_source_anchor_non_observation_scale,
            )
            if static_teacher is not None and static_alignment_weight > 0.0:
                with torch.no_grad():
                    z_static, next_z_static = static_teacher.compute_adapted_latents(batch)
                loss_static_teacher = (
                    adaptation_weights * F.mse_loss(z_prime, z_static.detach(), reduction="none").mean(dim=-1)
                ).mean()
                loss_static_teacher = loss_static_teacher + 0.5 * (
                    adaptation_weights
                    * F.mse_loss(next_z_prime, next_z_static.detach(), reduction="none").mean(dim=-1)
                ).mean()
            if source_alignment_weight > 0.0 and source_means is not None and source_stds is not None:
                task_ids = batch["task"].long()
                batch_means = []
                batch_stds = []
                target_means = []
                target_stds = []
                for task_id in task_ids.unique(sorted=True):
                    mask = task_ids == task_id
                    task_latents = z_prime[mask]
                    batch_means.append(task_latents.mean(dim=0))
                    batch_stds.append(task_latents.std(dim=0, unbiased=False).clamp_min(1.0e-6))
                    target_means.append(source_means[int(task_id)])
                    target_stds.append(source_stds[int(task_id)])
                loss_source = F.mse_loss(torch.stack(batch_means), torch.stack(target_means))
                loss_source = loss_source + F.mse_loss(torch.stack(batch_stds), torch.stack(target_stds))
            if source_primitive_anchors is not None:
                anchor_means_all = source_primitive_anchors["means"].index_select(0, primitive_ids)
                anchor_stds_all = source_primitive_anchors["stds"].index_select(0, primitive_ids)
                anchor_deltas_all = source_primitive_anchors["deltas"].index_select(0, primitive_ids)
                anchor_policies_all = source_primitive_anchors["policies"].index_select(0, primitive_ids)
                selector_latents = z_base
                prototype_distances = ((selector_latents.unsqueeze(1) - anchor_means_all) ** 2).mean(dim=-1)
                nearest_proto = prototype_distances.argmin(dim=1)
                gather_index_latent = nearest_proto.view(-1, 1, 1).expand(-1, 1, anchor_means_all.shape[-1])
                gather_index_policy = nearest_proto.view(-1, 1, 1).expand(-1, 1, anchor_policies_all.shape[-1])
                anchor_means = anchor_means_all.gather(1, gather_index_latent).squeeze(1)
                anchor_stds = anchor_stds_all.gather(1, gather_index_latent).squeeze(1)
                anchor_deltas = anchor_deltas_all.gather(1, gather_index_latent).squeeze(1)
                anchor_policies = anchor_policies_all.gather(1, gather_index_policy).squeeze(1)
                if source_primitive_alignment_weight > 0.0:
                    if source_alignment_use_std_norm:
                        primitive_residual = ((z_prime - anchor_means) / anchor_stds.clamp_min(1.0e-6)) ** 2
                    else:
                        primitive_residual = (z_prime - anchor_means) ** 2
                    loss_source_primitive = _weighted_branch_average(
                        primitive_residual.mean(dim=-1),
                        adaptation_weights,
                        source_anchor_branch_scales,
                    )
                if source_delta_alignment_weight > 0.0:
                    delta_residual = ((next_z_prime - z_prime) - anchor_deltas) ** 2
                    loss_source_delta = _weighted_branch_average(
                        delta_residual.mean(dim=-1),
                        adaptation_weights,
                        source_anchor_branch_scales,
                    )
                if source_policy_alignment_weight > 0.0:
                    student_logits = model.condition_policy_logits(
                        model.policy_logits(z_prime),
                        z=z_prime,
                        task_ids=batch.get("task"),
                        state=batch.get("state"),
                    )
                    student_log_probs = torch.log_softmax(student_logits / source_alignment_temperature, dim=-1)
                    source_policy_kl = F.kl_div(
                        student_log_probs,
                        anchor_policies,
                        reduction="none",
                    ).sum(dim=-1)
                    loss_source_policy = _weighted_branch_average(
                        source_policy_kl,
                        adaptation_weights,
                        source_anchor_branch_scales,
                    ) * (source_alignment_temperature ** 2)
            if policy_preservation_weight > 0.0:
                loss_policy_pres = _policy_preservation_loss(
                    model,
                    batch,
                    z_base,
                    z_prime,
                    next_z_base,
                    alpha=policy_preservation_alpha,
                    temperature=policy_preservation_temperature,
                    mode=policy_preservation_mode,
                    stage_max_id=None if policy_preservation_stage_max is None else int(policy_preservation_stage_max),
                    observation_only=policy_preservation_observation_only,
                    observation_scale=policy_preservation_observation_scale,
                    non_observation_scale=policy_preservation_non_observation_scale,
                )
            if policy_supervision_weight > 0.0:
                student_logits = model.condition_policy_logits(
                    model.policy_logits(z_prime),
                    z=z_prime,
                    task_ids=batch.get("task"),
                    state=batch.get("state"),
                )
                policy_sup_ce = F.cross_entropy(
                    student_logits,
                    primitive_ids,
                    reduction="none",
                )
                loss_policy_sup = adaptation_weights.mul(policy_sup_ce).mean()
            if inverse_supervision_weight > 0.0:
                loss_inverse_sup = model.compute_inverse_loss(
                    batch,
                    z_prime,
                    next_z_prime,
                    weights=adaptation_weights,
                )
            if gate_identity_weight > 0.0:
                loss_gate_identity = model.gate_identity_penalty()
            if policy_adapter_identity_weight > 0.0:
                loss_policy_adapter_identity = model.policy_adapter_identity_penalty()
            if transition_adapter_identity_weight > 0.0:
                loss_transition_identity = model.transition_adapter_identity_penalty()
            if "next_stage_id" in batch:
                next_stage_batch = dict(batch)
                next_stage_batch["stage_id"] = batch["next_stage_id"]
                loss_stage = 0.5 * (loss_stage + model.compute_stage_loss(next_stage_batch, next_z_prime))
            loss = (
                loss_adapt
                + reg_weight * (loss_reg + 0.5 * loss_next_reg)
                + static_alignment_weight * loss_static_teacher
                + stage_weight * loss_stage
                + source_alignment_weight * loss_source
                + source_primitive_alignment_weight * loss_source_primitive
                + source_delta_alignment_weight * loss_source_delta
                + source_policy_alignment_weight * loss_source_policy
                + (policy_preservation_weight * policy_preservation_scale) * loss_policy_pres
                + policy_supervision_weight * loss_policy_sup
                + inverse_supervision_weight * loss_inverse_sup
                + gate_identity_weight * loss_gate_identity
                + policy_adapter_identity_weight * loss_policy_adapter_identity
                + transition_adapter_identity_weight * loss_transition_identity
            )
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.adapter_parameters()), 5.0)
            optimizer.step()
    checkpoint_dir = ensure_dir(cfg["paths"]["checkpoint_dir"])
    adapter_path = checkpoint_dir / "adapter_calibrated.pt"
    if mean_transition_residual is not None:
        cfg.setdefault("adaptation", {})["last_mean_transition_residual"] = float(mean_transition_residual)
    if auto_non_observation_scale_value is not None:
        cfg.setdefault("adaptation", {})["last_auto_non_observation_scale"] = float(auto_non_observation_scale_value)
    if auto_latent_affine_blend_value is not None:
        cfg.setdefault("adaptation", {})["last_auto_latent_affine_blend"] = float(auto_latent_affine_blend_value)
    if auto_source_anchor_scale_value is not None:
        cfg.setdefault("adaptation", {})["last_auto_source_anchor_scale"] = float(auto_source_anchor_scale_value)
    if auto_source_anchor_balance_value is not None:
        cfg.setdefault("adaptation", {})["last_auto_source_anchor_balance"] = float(
            auto_source_anchor_balance_value
        )
    if observation_transition_residual is not None:
        cfg.setdefault("adaptation", {})["last_observation_transition_residual"] = float(
            observation_transition_residual
        )
    if non_observation_transition_residual is not None:
        cfg.setdefault("adaptation", {})["last_non_observation_transition_residual"] = float(
            non_observation_transition_residual
        )
    torch.save({"model_state": model.state_dict(), "config": cfg}, adapter_path)
    return adapter_path


def calibrate_static_adapter(
    cfg: dict,
    checkpoint_path: str | Path,
    source_path: str | Path,
    target_path: str | Path,
) -> Path:
    _set_random_seed(int(cfg.get("seed", 0)) + 131)
    device = torch.device(cfg["train"]["device"])
    payload = torch.load(checkpoint_path, map_location=device)
    model = build_model(cfg).to(device)
    load_model_state(model, payload["model_state"])
    model.freeze_backbone()
    model.train()

    source_dataset = _build_dataset(cfg, source_path, train=False)
    target_dataset = _build_calibration_dataset(cfg, target_path)
    source_loader = DataLoader(source_dataset, batch_size=cfg["adaptation"]["batch_size"], shuffle=False)
    target_loader = _build_adaptation_loader(cfg, target_dataset)

    task_vocab_size = int(cfg["model"].get("task_vocab_size", 3))

    @torch.no_grad()
    def _source_stats() -> tuple[torch.Tensor, torch.Tensor]:
        latent_chunks: list[torch.Tensor] = []
        task_chunks: list[torch.Tensor] = []
        for batch in source_loader:
            batch = _to_device(batch, device)
            z, _ = model.compute_latents(batch)
            latent_chunks.append(z.detach())
            task_chunks.append(batch["task"].detach())
        z_all = torch.cat(latent_chunks, dim=0)
        task_all = torch.cat(task_chunks, dim=0)
        global_mean = z_all.mean(dim=0)
        global_std = z_all.std(dim=0, unbiased=False).clamp_min(1.0e-6)
        task_means = []
        task_stds = []
        for task_id in range(task_vocab_size):
            mask = task_all == task_id
            if bool(mask.any()):
                task_latents = z_all[mask]
                task_means.append(task_latents.mean(dim=0))
                task_stds.append(task_latents.std(dim=0, unbiased=False).clamp_min(1.0e-6))
            else:
                task_means.append(global_mean)
                task_stds.append(global_std)
        return torch.stack(task_means, dim=0), torch.stack(task_stds, dim=0)

    source_means, source_stds = _source_stats()
    optimizer = torch.optim.AdamW(model.adapter_parameters(), lr=cfg["adaptation"]["lr"])
    reg_weight = float(cfg["adaptation"].get("adapter_reg_weight", 0.1))
    stats_weight = float(cfg["adaptation"].get("static_alignment_weight", 1.0))
    stage_weight = float(cfg["adaptation"].get("adapter_stage_loss_weight", 0.0))
    epochs = int(cfg["adaptation"].get("epochs", 10))

    for _ in range(epochs):
        for batch in tqdm(target_loader, desc="static-adapt", leave=False):
            optimizer.zero_grad(set_to_none=True)
            batch = _to_device(batch, device)
            z, next_z = model.compute_latents(batch)
            z = z.detach()
            next_z = next_z.detach()
            z_prime, next_z_prime = model.compute_adapted_latents(batch)
            task_ids = batch["task"].long()
            loss_reg = F.mse_loss(z_prime, z)
            batch_means = []
            batch_stds = []
            target_means = []
            target_stds = []
            for task_id in task_ids.unique(sorted=True):
                mask = task_ids == task_id
                task_latents = z_prime[mask]
                batch_means.append(task_latents.mean(dim=0))
                batch_stds.append(task_latents.std(dim=0, unbiased=False).clamp_min(1.0e-6))
                target_means.append(source_means[int(task_id)])
                target_stds.append(source_stds[int(task_id)])
            loss_stats = F.mse_loss(torch.stack(batch_means), torch.stack(target_means))
            loss_stats = loss_stats + F.mse_loss(torch.stack(batch_stds), torch.stack(target_stds))
            loss_stage = model.compute_stage_loss(batch, z_prime)
            if "next_stage_id" in batch:
                next_stage_batch = dict(batch)
                next_stage_batch["stage_id"] = batch["next_stage_id"]
                loss_stage = 0.5 * (loss_stage + model.compute_stage_loss(next_stage_batch, next_z_prime))
            loss_next_reg = F.mse_loss(next_z_prime, next_z)
            loss = stats_weight * loss_stats + reg_weight * (loss_reg + 0.5 * loss_next_reg) + stage_weight * loss_stage
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.adapter_parameters()), 5.0)
            optimizer.step()

    checkpoint_dir = ensure_dir(cfg["paths"]["checkpoint_dir"])
    output_path = checkpoint_dir / "static_adapter_calibrated.pt"
    torch.save({"model_state": model.state_dict(), "config": cfg}, output_path)
    return output_path


def finetune_few_shot(cfg: dict, checkpoint_path: str | Path, real_path: str | Path) -> Path:
    _set_random_seed(int(cfg.get("seed", 0)) + 151)
    device = torch.device(cfg["train"]["device"])
    payload = torch.load(checkpoint_path, map_location=device)
    model = build_model(cfg).to(device)
    load_model_state(model, payload["model_state"])
    model.train()
    dataset = _build_dataset(cfg, real_path, train=True)
    loader_seed = int(cfg.get("seed", 0)) + 157
    loader_generator = torch.Generator().manual_seed(loader_seed)
    few_shot_cfg = cfg.get("few_shot", {})
    loader = DataLoader(
        dataset,
        batch_size=int(few_shot_cfg.get("batch_size", cfg["adaptation"].get("batch_size", 8))),
        shuffle=True,
        generator=loader_generator,
    )
    trainable_modules = set(few_shot_cfg.get("trainable_modules", ["image_encoder", "state_encoder", "fusion", "policy_head", "feature_fusion", "gru", "post_gru", "chunk_head", "direct_head", "denoiser", "stage_head"]))
    for name, param in model.named_parameters():
        if name.startswith("adapter."):
            param.requires_grad_(False)
            continue
        module_name = name.split(".", 1)[0]
        param.requires_grad_(module_name in trainable_modules)
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("few_shot_finetuning selected no trainable parameters")
    optimizer = torch.optim.AdamW(
        params,
        lr=float(few_shot_cfg.get("lr", 2.0e-4)),
        weight_decay=float(few_shot_cfg.get("weight_decay", cfg["train"].get("weight_decay", 1.0e-4))),
    )
    transition_weight = float(few_shot_cfg.get("transition_loss_weight", cfg["train"].get("transition_loss_weight", 0.5)))
    stage_loss_weight = float(few_shot_cfg.get("stage_loss_weight", cfg["train"].get("stage_loss_weight", 0.0)))
    epochs = int(few_shot_cfg.get("epochs", 6))
    for _ in range(epochs):
        for batch in tqdm(loader, desc="few-shot", leave=False):
            optimizer.zero_grad(set_to_none=True)
            batch = _to_device(batch, device)
            z, next_z = model.compute_latents(batch)
            pred_next = model.predict_next(z, batch["primitive_id"], batch.get("state"), task_ids=batch.get("task"))
            loss_policy = model.compute_policy_loss(batch, z)
            loss_trans = F.mse_loss(pred_next, next_z.detach())
            loss_stage = model.compute_stage_loss(batch, z)
            loss = loss_policy + transition_weight * loss_trans + stage_loss_weight * loss_stage
            loss.backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()
    checkpoint_dir = ensure_dir(cfg["paths"]["checkpoint_dir"])
    few_shot_path = checkpoint_dir / "few_shot_finetuned.pt"
    torch.save({"model_state": model.state_dict(), "config": cfg}, few_shot_path)
    return few_shot_path


@torch.no_grad()
def fit_latent_alignment(
    cfg: dict,
    checkpoint_path: str | Path,
    source_path: str | Path,
    target_path: str | Path,
) -> Path:
    device = torch.device(cfg["train"]["device"])
    payload = torch.load(checkpoint_path, map_location=device)
    model = build_model(cfg).to(device)
    load_model_state(model, payload["model_state"])
    model.eval()
    latent_alignment_stats = _estimate_latent_alignment_stats(model, cfg, source_path, target_path, device)
    source_mean = latent_alignment_stats["source_mean"].detach().cpu().numpy().astype(np.float32)
    source_std = latent_alignment_stats["source_std"].detach().cpu().numpy().astype(np.float32)
    target_mean = latent_alignment_stats["target_mean"].detach().cpu().numpy().astype(np.float32)
    target_std = latent_alignment_stats["target_std"].detach().cpu().numpy().astype(np.float32)
    source_global_mean = source_mean.mean(axis=0).astype(np.float32)
    checkpoint_dir = ensure_dir(cfg["paths"]["checkpoint_dir"])
    output_path = checkpoint_dir / "latent_alignment_stats.npz"
    np.savez(
        output_path,
        source_mean=source_mean,
        source_std=source_std,
        target_mean=target_mean,
        target_std=target_std,
        source_global_mean=source_global_mean,
        max_scale=np.array(float(cfg["model"].get("latent_affine_max_scale", 4.0)), dtype=np.float32),
        blend=np.array(float(cfg["model"].get("latent_affine_blend", 1.0)), dtype=np.float32),
    )
    return output_path


@torch.no_grad()
def _evaluate_loss(
    model: BaseTTLAModel,
    loader: DataLoader,
    transition_weight: float,
    inverse_loss_weight: float,
    stage_loss_weight: float,
    device: torch.device,
) -> float:
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
            + inverse_loss_weight * model.compute_inverse_loss(batch, z, next_z)
            + stage_loss_weight * model.compute_stage_loss(batch, z)
        )
        total += float(loss.item()) * batch["primitive_id"].size(0)
        count += batch["primitive_id"].size(0)
    model.train()
    return total / max(count, 1)


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

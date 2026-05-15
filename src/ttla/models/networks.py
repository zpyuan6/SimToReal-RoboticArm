from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..sim.skills import (
    ABORT_ID,
    ABORT_FAMILY_ID,
    APPROACH_FAMILY_ID,
    GRASP_EXECUTE_ID,
    GRASP_FAMILY_ID,
    LIFT_OBJECT_ID,
    LIFT_FAMILY_ID,
    OBS_CENTER_ID,
    OBS_LEFT_ID,
    OBS_RIGHT_ID,
    OBSERVE_FAMILY_ID,
    PLACE_OBJECT_ID,
    PLACE_FAMILY_ID,
    PREGRASP_SERVO_ID,
    PRIMITIVE_VOCAB_LEGACY,
    RECOVER_FAMILY_ID,
    RETREAT_ID,
    TRANSPORT_TO_DROPZONE_ID,
    TRANSPORT_FAMILY_ID,
    family_projected_primitives,
    project_primitive_ids,
)
from ..sim.task_defs import TASK_SPECS, primitive_instruction, task_action_hint, task_allowed_primitives, task_instruction, task_primary_primitives
from ..sim.task_defs import NUM_SUPERVISION_STAGES

OBSERVATION_FAMILY_IDS = (OBSERVE_FAMILY_ID,)


def _hash_text_embedding(text: str, dim: int) -> torch.Tensor:
    vec = torch.zeros(dim, dtype=torch.float32)
    normalized = text.lower().replace("_", " ")
    for i, ch in enumerate(normalized):
        bucket = (ord(ch) + 31 * i) % dim
        vec[bucket] += 1.0
        if i + 1 < len(normalized):
            pair_bucket = (ord(ch) * 17 + ord(normalized[i + 1]) * 13 + i) % dim
            vec[pair_bucket] += 0.5
    norm = vec.norm(p=2).clamp_min(1.0)
    return vec / norm


def _fixed_prompt_embedding_table(
    task_vocab_size: int,
    language_dim: int,
    primitive_vocabulary: str = PRIMITIVE_VOCAB_LEGACY,
) -> torch.Tensor:
    table = torch.zeros(task_vocab_size, language_dim, dtype=torch.float32)
    task_specs = sorted(TASK_SPECS.values(), key=lambda spec: spec.task_id)
    for spec in task_specs:
        if spec.task_id >= task_vocab_size:
            continue
        table[spec.task_id] = _hash_text_embedding(
            task_instruction(spec.task_id, primitive_vocabulary=primitive_vocabulary),
            language_dim,
        )
    return table


def _fixed_task_action_hint_table(
    task_vocab_size: int,
    language_dim: int,
    primitive_vocabulary: str = PRIMITIVE_VOCAB_LEGACY,
) -> torch.Tensor:
    table = torch.zeros(task_vocab_size, language_dim, dtype=torch.float32)
    task_specs = sorted(TASK_SPECS.values(), key=lambda spec: spec.task_id)
    for spec in task_specs:
        if spec.task_id >= task_vocab_size:
            continue
        table[spec.task_id] = _hash_text_embedding(
            task_action_hint(spec.task_id, primitive_vocabulary=primitive_vocabulary),
            language_dim,
        )
    return table


def _valid_projected_primitives(
    primitive_ids: tuple[int, ...] | list[int] | set[int],
    action_dim: int,
    primitive_vocabulary: str,
) -> list[int]:
    return [
        primitive_id
        for primitive_id in project_primitive_ids(primitive_ids, primitive_vocabulary=primitive_vocabulary)
        if 0 <= primitive_id < action_dim
    ]


def _valid_family_primitives(
    family_ids: tuple[int, ...] | list[int] | set[int],
    action_dim: int,
    primitive_vocabulary: str,
) -> list[int]:
    return [
        primitive_id
        for primitive_id in family_projected_primitives(family_ids, primitive_vocabulary=primitive_vocabulary)
        if 0 <= primitive_id < action_dim
    ]


def _fixed_primitive_text_embedding_table(
    action_dim: int,
    language_dim: int,
    primitive_vocabulary: str = PRIMITIVE_VOCAB_LEGACY,
) -> torch.Tensor:
    table = torch.zeros(action_dim, language_dim, dtype=torch.float32)
    for primitive_id in range(action_dim):
        table[primitive_id] = _hash_text_embedding(
            primitive_instruction(primitive_id, primitive_vocabulary=primitive_vocabulary),
            language_dim,
        )
    return table


def _fixed_task_action_prior_table(
    task_vocab_size: int,
    action_dim: int,
    negative_value: float = 0.0,
    primitive_vocabulary: str = PRIMITIVE_VOCAB_LEGACY,
) -> torch.Tensor:
    table = torch.full((task_vocab_size, action_dim), negative_value, dtype=torch.float32)
    task_specs = sorted(TASK_SPECS.values(), key=lambda spec: spec.task_id)
    for spec in task_specs:
        if spec.task_id >= task_vocab_size:
            continue
        for primitive_id in task_allowed_primitives(spec.task_id, primitive_vocabulary=primitive_vocabulary):
            if 0 <= primitive_id < action_dim:
                table[spec.task_id, primitive_id] = 0.45
        for primitive_id in task_primary_primitives(spec.task_id, primitive_vocabulary=primitive_vocabulary):
            if 0 <= primitive_id < action_dim:
                table[spec.task_id, primitive_id] = 1.0
        for primitive_id in _valid_family_primitives((ABORT_FAMILY_ID,), action_dim, primitive_vocabulary):
            table[spec.task_id, primitive_id] = max(table[spec.task_id, primitive_id], 0.35)
    return table


def _fixed_task_action_mask_table(
    task_vocab_size: int,
    action_dim: int,
    primitive_vocabulary: str = PRIMITIVE_VOCAB_LEGACY,
) -> torch.Tensor:
    table = torch.zeros((task_vocab_size, action_dim), dtype=torch.float32)
    task_specs = sorted(TASK_SPECS.values(), key=lambda spec: spec.task_id)
    for spec in task_specs:
        if spec.task_id >= task_vocab_size:
            continue
        for primitive_id in task_allowed_primitives(spec.task_id, primitive_vocabulary=primitive_vocabulary):
            if 0 <= primitive_id < action_dim:
                table[spec.task_id, primitive_id] = 1.0
    fallback = table.sum(dim=-1) == 0
    if fallback.any():
        table[fallback] = 1.0
    return table


def _fixed_stage_action_prior_table(
    stage_aux_classes: int,
    action_dim: int,
    primitive_vocabulary: str = PRIMITIVE_VOCAB_LEGACY,
) -> torch.Tensor:
    table = torch.full((stage_aux_classes, action_dim), -0.25, dtype=torch.float32)

    def _set(stage_id: int, indices: list[int], value: float = 1.0) -> None:
        if stage_id >= stage_aux_classes:
            return
        for primitive_id in indices:
            if 0 <= primitive_id < action_dim:
                table[stage_id, primitive_id] = value

    observe_ids = _valid_family_primitives((OBSERVE_FAMILY_ID,), action_dim, primitive_vocabulary)
    approach_ids = _valid_family_primitives((APPROACH_FAMILY_ID,), action_dim, primitive_vocabulary)
    recover_ids = _valid_family_primitives((RECOVER_FAMILY_ID,), action_dim, primitive_vocabulary)
    grasp_ids = _valid_family_primitives((GRASP_FAMILY_ID,), action_dim, primitive_vocabulary)
    lift_ids = _valid_family_primitives((LIFT_FAMILY_ID,), action_dim, primitive_vocabulary)
    transport_ids = _valid_family_primitives((TRANSPORT_FAMILY_ID,), action_dim, primitive_vocabulary)
    place_ids = _valid_family_primitives((PLACE_FAMILY_ID,), action_dim, primitive_vocabulary)
    abort_ids = _valid_family_primitives((ABORT_FAMILY_ID,), action_dim, primitive_vocabulary)

    # Observe / search.
    _set(0, observe_ids, 1.0)
    _set(0, recover_ids, 0.35)
    _set(0, abort_ids, 0.15)
    # Reserved / compatibility stage.
    _set(1, observe_ids, 0.55)
    _set(1, abort_ids, 0.2)
    # Approach / recover.
    _set(2, approach_ids, 1.0)
    _set(2, recover_ids, 0.75)
    _set(2, observe_ids, 0.30)
    _set(2, abort_ids, 0.2)
    # Grasp.
    _set(3, grasp_ids, 1.0)
    _set(3, approach_ids, 0.55)
    _set(3, lift_ids, 0.50)
    _set(3, recover_ids + abort_ids, 0.2)
    # Lift.
    _set(4, lift_ids, 1.0)
    _set(4, transport_ids, 0.65)
    _set(4, recover_ids + abort_ids, 0.2)
    # Transport.
    _set(5, transport_ids, 1.0)
    _set(5, place_ids, 0.70)
    _set(5, recover_ids + abort_ids, 0.2)
    # Place.
    _set(6, place_ids, 1.0)
    _set(6, transport_ids, 0.40)
    _set(6, abort_ids, 0.2)
    # Terminal.
    _set(7, place_ids + abort_ids, 1.0)
    _set(7, recover_ids, 0.2)
    return table


def _extract_task_ids(state: torch.Tensor, task_vocab_size: int) -> torch.Tensor:
    if state.ndim == 1:
        state = state.unsqueeze(0)
    if state.shape[-1] < 2:
        return torch.zeros(state.shape[0], dtype=torch.long, device=state.device)
    task_id = torch.round(state[..., -2]).long()
    return task_id.clamp(0, max(task_vocab_size - 1, 0))


def _resolve_task_ids(
    task_ids: torch.Tensor | None,
    state: torch.Tensor | None,
    task_vocab_size: int,
) -> torch.Tensor:
    if task_ids is not None:
        if task_ids.ndim > 1:
            task_ids = task_ids.squeeze(-1)
        return task_ids.long().clamp(0, max(task_vocab_size - 1, 0))
    if state is None:
        raise ValueError("Either task_ids or state must be provided.")
    return _extract_task_ids(state, task_vocab_size)


def _strip_task_feature(state: torch.Tensor) -> torch.Tensor:
    if state.shape[-1] < 2:
        return state
    return torch.cat([state[..., :-2], state[..., -1:]], dim=-1)


def _extract_progress_bins(state: torch.Tensor, num_bins: int) -> torch.Tensor:
    if state.ndim == 1:
        state = state.unsqueeze(0)
    if state.shape[-1] < 1:
        return torch.zeros(state.shape[0], dtype=torch.long, device=state.device)
    progress = state[..., -1].clamp(0.0, 1.0)
    scaled = torch.floor(progress * num_bins).long()
    return scaled.clamp(0, max(num_bins - 1, 0))


def _progressive_scale_from_stage_ids(
    stage_ids: torch.Tensor,
    num_bins: int,
    min_scale: float,
    max_scale: float,
    *,
    start_stage: int | None = None,
    end_stage: int | None = None,
    dtype: torch.dtype,
) -> torch.Tensor:
    if stage_ids.ndim == 0:
        stage_ids = stage_ids.unsqueeze(0)
    if start_stage is not None or end_stage is not None:
        active = torch.ones(stage_ids.shape, device=stage_ids.device, dtype=torch.bool)
        if start_stage is not None:
            active = active & (stage_ids >= int(start_stage))
        if end_stage is not None:
            active = active & (stage_ids <= int(end_stage))
        low_scale = float(min_scale if start_stage is not None else 0.0)
        high_scale = float(max_scale)
        scale = torch.where(
            active,
            torch.full(stage_ids.shape, high_scale, device=stage_ids.device, dtype=dtype),
            torch.full(stage_ids.shape, low_scale, device=stage_ids.device, dtype=dtype),
        )
        return scale.unsqueeze(-1)
    if start_stage is not None:
        scale = torch.where(
            stage_ids < int(start_stage),
            torch.full(stage_ids.shape, min_scale, device=stage_ids.device, dtype=dtype),
            torch.full(stage_ids.shape, max_scale, device=stage_ids.device, dtype=dtype),
        )
        return scale.unsqueeze(-1)
    if num_bins <= 1:
        return torch.ones((*stage_ids.shape, 1), device=stage_ids.device, dtype=dtype) * float(max_scale)
    if abs(max_scale - min_scale) < 1.0e-8:
        return torch.ones((*stage_ids.shape, 1), device=stage_ids.device, dtype=dtype) * float(max_scale)
    stage_ratio = stage_ids.float() / float(max(num_bins - 1, 1))
    scale = min_scale + (max_scale - min_scale) * stage_ratio
    return scale.unsqueeze(-1).to(dtype=dtype)


def _predicted_stage_ids_from_latent(z: torch.Tensor, stage_head: nn.Module, stage_aux_classes: int) -> torch.Tensor:
    if z.ndim == 1:
        z = z.unsqueeze(0)
    flat = z.reshape(-1, z.shape[-1])
    stage_logits = stage_head(flat.detach())
    stage_ids = stage_logits.argmax(dim=-1)
    return stage_ids.clamp(0, max(stage_aux_classes - 1, 0)).view(*z.shape[:-1])


def _stage_action_mask(
    task_ids: torch.Tensor,
    state: torch.Tensor,
    action_dim: int,
    primitive_vocabulary: str = PRIMITIVE_VOCAB_LEGACY,
) -> torch.Tensor:
    if state.ndim == 1:
        state = state.unsqueeze(0)
    device = state.device
    mask = torch.zeros((state.shape[0], action_dim), device=device, dtype=state.dtype)
    # Route B uses only observable runtime state (qpos, qvel, task id, progress)
    # rather than simulator-only task flags. Fall back to a task-level mask with
    # coarse progress-based phasing instead of reading hidden booleans from the
    # tail of the state vector.
    progress = state[..., -1].clamp(0.0, 1.0) if state.shape[-1] >= 1 else torch.zeros(state.shape[0], device=device, dtype=state.dtype)
    attached = torch.zeros(state.shape[0], dtype=torch.bool, device=device)
    lifted = torch.zeros(state.shape[0], dtype=torch.bool, device=device)
    placed = torch.zeros(state.shape[0], dtype=torch.bool, device=device)

    def _set_rows(rows: torch.Tensor, indices: list[int]) -> None:
        if rows.any():
            row_idx = torch.nonzero(rows, as_tuple=False).squeeze(-1)
            col_idx = torch.tensor(indices, device=device, dtype=torch.long)
            mask[row_idx.unsqueeze(1), col_idx.unsqueeze(0)] = 1.0

    # Generic manipulation phases. Task-specific differences are handled later
    # by intersecting this phase mask with the task-level allowed-action mask.
    place_ids = _valid_family_primitives((PLACE_FAMILY_ID,), action_dim, primitive_vocabulary)
    abort_ids = _valid_family_primitives((ABORT_FAMILY_ID,), action_dim, primitive_vocabulary)
    lift_ids = _valid_family_primitives((LIFT_FAMILY_ID,), action_dim, primitive_vocabulary)
    recover_ids = _valid_family_primitives((RECOVER_FAMILY_ID,), action_dim, primitive_vocabulary)
    transport_ids = _valid_family_primitives((TRANSPORT_FAMILY_ID,), action_dim, primitive_vocabulary)
    observe_ids = _valid_family_primitives((OBSERVE_FAMILY_ID,), action_dim, primitive_vocabulary)
    approach_ids = _valid_family_primitives((APPROACH_FAMILY_ID,), action_dim, primitive_vocabulary)
    grasp_ids = _valid_family_primitives((GRASP_FAMILY_ID,), action_dim, primitive_vocabulary)

    rows = progress < 0.20
    _set_rows(rows, observe_ids + approach_ids + recover_ids + abort_ids)

    rows = (progress >= 0.20) & (progress < 0.55)
    _set_rows(rows, observe_ids + approach_ids + grasp_ids + recover_ids + abort_ids)

    rows = (progress >= 0.55) & (progress < 0.75)
    _set_rows(rows, grasp_ids + lift_ids + recover_ids + abort_ids)

    rows = (progress >= 0.75) & (progress < 0.90)
    _set_rows(rows, lift_ids + transport_ids + place_ids + recover_ids + abort_ids)

    rows = progress >= 0.90
    _set_rows(rows, transport_ids + place_ids + abort_ids)

    fallback = mask.sum(dim=-1) == 0
    if fallback.any():
        mask[fallback] = 1.0

    # Intersect generic phase mask with task-level allowed sets so the same
    # phase logic can be reused across tasks without enabling out-of-task
    # primitives.
    task_mask = torch.zeros_like(mask)
    for spec in TASK_SPECS.values():
        row_mask = task_ids == int(spec.task_id)
        if not row_mask.any():
            continue
        allowed = [
            primitive_id
            for primitive_id in task_allowed_primitives(spec.task_id, primitive_vocabulary=primitive_vocabulary)
            if 0 <= primitive_id < action_dim
        ]
        if not allowed:
            continue
        row_idx = torch.nonzero(row_mask, as_tuple=False).squeeze(-1)
        col_idx = torch.tensor(allowed, device=device, dtype=torch.long)
        task_mask[row_idx.unsqueeze(1), col_idx.unsqueeze(0)] = 1.0
    mask = mask * task_mask
    fallback = mask.sum(dim=-1) == 0
    if fallback.any():
        mask[fallback] = task_mask[fallback]
    fallback = mask.sum(dim=-1) == 0
    if fallback.any():
        mask[fallback] = 1.0
    return mask


def _mask_policy_logits(
    logits: torch.Tensor,
    state: torch.Tensor,
    task_ids: torch.Tensor | None,
    task_vocab_size: int,
    primitive_vocabulary: str = PRIMITIVE_VOCAB_LEGACY,
) -> torch.Tensor:
    if logits.ndim == 2:
        resolved_task_ids = _resolve_task_ids(task_ids, state, task_vocab_size)
        action_mask = _stage_action_mask(
            resolved_task_ids,
            state,
            logits.shape[-1],
            primitive_vocabulary=primitive_vocabulary,
        ).to(dtype=logits.dtype)
        return logits.masked_fill(action_mask < 0.5, -1.0e9)
    if logits.ndim == 3:
        batch, steps, action_dim = logits.shape
        if state.ndim == 2:
            resolved_task_ids = _resolve_task_ids(task_ids, state, task_vocab_size)
            action_mask = _stage_action_mask(
                resolved_task_ids,
                state,
                action_dim,
                primitive_vocabulary=primitive_vocabulary,
            ).to(dtype=logits.dtype)
            return logits.masked_fill(action_mask.unsqueeze(1) < 0.5, -1.0e9)
        if state.ndim == 3:
            flat_state = state.reshape(batch * steps, state.shape[-1])
            if task_ids is None:
                flat_task_ids = None
            else:
                flat_task_ids = task_ids.unsqueeze(1).expand(batch, steps).reshape(-1)
            resolved_task_ids = _resolve_task_ids(flat_task_ids, flat_state, task_vocab_size)
            action_mask = _stage_action_mask(
                resolved_task_ids,
                flat_state,
                action_dim,
                primitive_vocabulary=primitive_vocabulary,
            ).to(dtype=logits.dtype)
            return logits.masked_fill(action_mask.view(batch, steps, action_dim) < 0.5, -1.0e9)
    raise ValueError(f"Unsupported logits rank for action masking: {logits.ndim}")


def _stage_action_prior(
    task_ids: torch.Tensor,
    state: torch.Tensor,
    action_dim: int,
    positive: float = 1.0,
    negative: float = -1.0,
    primitive_vocabulary: str = PRIMITIVE_VOCAB_LEGACY,
) -> torch.Tensor:
    mask = _stage_action_mask(task_ids, state, action_dim, primitive_vocabulary=primitive_vocabulary)
    return torch.where(mask > 0.5, torch.full_like(mask, positive), torch.full_like(mask, negative))


class ImageEncoder(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((11, 11)),
            nn.Flatten(),
            nn.Linear(64 * 11 * 11, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class BaseTTLAModel(nn.Module, ABC):
    uses_history = False

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
        hidden_dim: int,
        action_embed_dim: int = 16,
        task_vocab_size: int = 3,
        task_embed_dim: int = 16,
        transition_stage_bins: int = 4,
        stage_aux_classes: int = NUM_SUPERVISION_STAGES,
        adapter_hidden_dim: int = 64,
        adapter_scale: float = 0.1,
        adapter_mode: str = "full",
        adapter_use_gate: bool = True,
        adapter_use_condition_branch: bool = True,
        adapter_use_task_condition: bool = True,
        adapter_use_stage_condition: bool = True,
        adapter_use_prev_action_condition: bool = True,
        adapter_progressive_min_scale: float = 1.0,
        adapter_progressive_max_scale: float = 1.0,
        adapter_stage_scales: list[float] | tuple[float, ...] | None = None,
        adapter_condition_start_stage: int | None = None,
        adapter_condition_end_stage: int | None = None,
        adapter_condition_observation_only: bool = False,
        adapter_condition_non_observation_scale: float = 0.0,
        adapter_phase_split: bool = False,
        task_prior_scale: float = 1.0,
        task_prior_negative_value: float = 0.0,
        stage_prior_scale: float = 0.0,
        predicted_stage_prior_scale: float = 0.0,
        task_hard_mask: bool = False,
        stage_hard_mask: bool = False,
        latent_affine_alignment: bool = False,
        latent_affine_task_conditioned: bool = True,
        latent_affine_max_scale: float = 4.0,
        latent_affine_blend: float = 1.0,
        transition_action_adapter: bool = False,
        transition_action_adapter_scale: float = 1.0,
        transition_residual_adapter: bool = False,
        transition_residual_hidden_dim: int = 64,
        transition_residual_scale: float = 0.1,
        transition_residual_phase_split: bool = False,
        transition_residual_observation_scale: float = 1.0,
        transition_residual_non_observation_scale: float = 1.0,
        policy_residual_adapter: bool = False,
        policy_residual_hidden_dim: int = 64,
        policy_residual_scale: float = 0.1,
        primitive_vocabulary: str = PRIMITIVE_VOCAB_LEGACY,
        **_unused: Any,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_embed_dim = action_embed_dim
        self.task_vocab_size = task_vocab_size
        self.task_embed_dim = task_embed_dim
        self.transition_stage_bins = transition_stage_bins
        self.stage_aux_classes = stage_aux_classes
        self.adapter_scale = adapter_scale
        self.adapter_mode = str(adapter_mode).lower()
        self.adapter_use_gate = adapter_use_gate
        self.adapter_use_condition_branch = adapter_use_condition_branch
        self.adapter_use_task_condition = adapter_use_task_condition
        self.adapter_use_stage_condition = adapter_use_stage_condition
        self.adapter_use_prev_action_condition = adapter_use_prev_action_condition
        self.adapter_progressive_min_scale = adapter_progressive_min_scale
        self.adapter_progressive_max_scale = adapter_progressive_max_scale
        self.adapter_condition_start_stage = adapter_condition_start_stage
        self.adapter_condition_end_stage = adapter_condition_end_stage
        self.adapter_condition_observation_only = adapter_condition_observation_only
        self.adapter_condition_non_observation_scale = adapter_condition_non_observation_scale
        self.adapter_phase_split = bool(adapter_phase_split)
        self.task_prior_scale = task_prior_scale
        self.task_prior_negative_value = task_prior_negative_value
        self.stage_prior_scale = stage_prior_scale
        self.predicted_stage_prior_scale = predicted_stage_prior_scale
        self.task_hard_mask = task_hard_mask
        self.stage_hard_mask = stage_hard_mask
        self.latent_affine_alignment = bool(latent_affine_alignment)
        self.latent_affine_task_conditioned = bool(latent_affine_task_conditioned)
        self.latent_affine_max_scale = float(max(latent_affine_max_scale, 1.0))
        self.latent_affine_blend = float(max(0.0, min(latent_affine_blend, 1.0)))
        self.use_transition_action_adapter = bool(transition_action_adapter)
        self.transition_action_adapter_scale = float(transition_action_adapter_scale)
        self.use_transition_residual_adapter = bool(transition_residual_adapter)
        self.transition_residual_scale = float(transition_residual_scale)
        self.transition_residual_phase_split = bool(transition_residual_phase_split)
        self.transition_residual_observation_scale = float(transition_residual_observation_scale)
        self.transition_residual_non_observation_scale = float(transition_residual_non_observation_scale)
        self.use_policy_residual_adapter = bool(policy_residual_adapter)
        self.policy_residual_scale = float(policy_residual_scale)
        self.primitive_vocabulary = str(primitive_vocabulary).lower()
        self.proprio_dim = max(state_dim - 1, 1)
        self.context_init = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.action_embedding = nn.Embedding(action_dim, action_embed_dim)
        self.register_buffer(
            "adapter_stage_scale_vector",
            torch.ones(stage_aux_classes, dtype=torch.float32),
        )
        self.register_buffer(
            "task_prompt_embedding_table",
            _fixed_prompt_embedding_table(
                task_vocab_size,
                task_embed_dim,
                primitive_vocabulary=self.primitive_vocabulary,
            ),
        )
        self.register_buffer(
            "task_action_prior_table",
            _fixed_task_action_prior_table(
                task_vocab_size,
                action_dim,
                negative_value=task_prior_negative_value,
                primitive_vocabulary=self.primitive_vocabulary,
            ),
        )
        self.register_buffer(
            "task_action_mask_table",
            _fixed_task_action_mask_table(
                task_vocab_size,
                action_dim,
                primitive_vocabulary=self.primitive_vocabulary,
            ),
        )
        self.register_buffer(
            "stage_action_prior_table",
            _fixed_stage_action_prior_table(
                stage_aux_classes,
                action_dim,
                primitive_vocabulary=self.primitive_vocabulary,
            ),
        )
        self.register_buffer("latent_align_source_mean", torch.zeros(task_vocab_size, latent_dim, dtype=torch.float32))
        self.register_buffer("latent_align_source_std", torch.ones(task_vocab_size, latent_dim, dtype=torch.float32))
        self.register_buffer("latent_align_target_mean", torch.zeros(task_vocab_size, latent_dim, dtype=torch.float32))
        self.register_buffer("latent_align_target_std", torch.ones(task_vocab_size, latent_dim, dtype=torch.float32))
        self.transition_task_embedding = nn.Embedding(task_vocab_size, action_embed_dim)
        self.transition_stage_embedding = nn.Embedding(transition_stage_bins, action_embed_dim)
        self.transition_action_adapter_embedding = nn.Embedding(action_dim, action_embed_dim)
        self.transition = nn.Sequential(
            nn.Linear(latent_dim + action_embed_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.transition_residual_adapter = nn.Sequential(
            nn.Linear(latent_dim + action_embed_dim * 3, transition_residual_hidden_dim),
            nn.ReLU(),
            nn.Linear(transition_residual_hidden_dim, latent_dim),
        )
        self.transition_non_observation_residual_adapter = nn.Sequential(
            nn.Linear(latent_dim + action_embed_dim * 3, transition_residual_hidden_dim),
            nn.ReLU(),
            nn.Linear(transition_residual_hidden_dim, latent_dim),
        )
        self.policy_residual_adapter = nn.Sequential(
            nn.Linear(latent_dim, policy_residual_hidden_dim),
            nn.ReLU(),
            nn.Linear(policy_residual_hidden_dim, action_dim),
        )
        self.inverse_head = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.stage_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, stage_aux_classes),
        )
        self.adapter = nn.Sequential(
            nn.Linear(latent_dim, adapter_hidden_dim),
            nn.ReLU(),
            nn.Linear(adapter_hidden_dim, latent_dim),
            nn.Tanh(),
        )
        self.adapter_task_embedding = nn.Embedding(task_vocab_size, task_embed_dim)
        self.adapter_stage_embedding = nn.Embedding(stage_aux_classes, task_embed_dim)
        self.adapter_prev_action_embedding = nn.Embedding(action_dim + 1, action_embed_dim)
        self.adapter_condition = nn.Sequential(
            nn.Linear(latent_dim + task_embed_dim * 2 + action_embed_dim, adapter_hidden_dim),
            nn.ReLU(),
            nn.Linear(adapter_hidden_dim, latent_dim),
            nn.Tanh(),
        )
        self.adapter_gate = nn.Sequential(
            nn.Linear(latent_dim + task_embed_dim * 2 + action_embed_dim, adapter_hidden_dim),
            nn.ReLU(),
            nn.Linear(adapter_hidden_dim, latent_dim),
        )
        self.adapter_non_observation = nn.Sequential(
            nn.Linear(latent_dim, adapter_hidden_dim),
            nn.ReLU(),
            nn.Linear(adapter_hidden_dim, latent_dim),
            nn.Tanh(),
        )
        self.adapter_non_observation_condition = nn.Sequential(
            nn.Linear(latent_dim + task_embed_dim * 2 + action_embed_dim, adapter_hidden_dim),
            nn.ReLU(),
            nn.Linear(adapter_hidden_dim, latent_dim),
            nn.Tanh(),
        )
        self.adapter_non_observation_gate = nn.Sequential(
            nn.Linear(latent_dim + task_embed_dim * 2 + action_embed_dim, adapter_hidden_dim),
            nn.ReLU(),
            nn.Linear(adapter_hidden_dim, latent_dim),
        )
        nn.init.zeros_(self.adapter[2].weight)
        nn.init.zeros_(self.adapter[2].bias)
        nn.init.zeros_(self.adapter_condition[2].weight)
        nn.init.zeros_(self.adapter_condition[2].bias)
        nn.init.zeros_(self.adapter_gate[2].weight)
        nn.init.zeros_(self.adapter_gate[2].bias)
        nn.init.zeros_(self.adapter_non_observation[2].weight)
        nn.init.zeros_(self.adapter_non_observation[2].bias)
        nn.init.zeros_(self.adapter_non_observation_condition[2].weight)
        nn.init.zeros_(self.adapter_non_observation_condition[2].bias)
        nn.init.zeros_(self.adapter_non_observation_gate[2].weight)
        nn.init.zeros_(self.adapter_non_observation_gate[2].bias)
        nn.init.zeros_(self.transition_action_adapter_embedding.weight)
        nn.init.zeros_(self.transition_residual_adapter[2].weight)
        nn.init.zeros_(self.transition_residual_adapter[2].bias)
        nn.init.zeros_(self.transition_non_observation_residual_adapter[2].weight)
        nn.init.zeros_(self.transition_non_observation_residual_adapter[2].bias)
        nn.init.zeros_(self.policy_residual_adapter[2].weight)
        nn.init.zeros_(self.policy_residual_adapter[2].bias)
        if adapter_stage_scales is not None:
            self.set_adapter_stage_scales(adapter_stage_scales)
        if self.adapter_mode not in {"full", "legacy_prevprim"}:
            raise ValueError(f"Unsupported adapter_mode: {self.adapter_mode}")

    @property
    def backbone_type(self) -> str:
        return getattr(self, "_backbone_type", "feedforward")

    def set_adapter_stage_scales(self, stage_scales: list[float] | tuple[float, ...] | torch.Tensor) -> None:
        scale_tensor = torch.as_tensor(
            stage_scales,
            dtype=self.adapter_stage_scale_vector.dtype,
            device=self.adapter_stage_scale_vector.device,
        )
        if scale_tensor.ndim != 1:
            raise ValueError("adapter stage scales must be a 1D sequence or tensor")
        if scale_tensor.numel() < self.stage_aux_classes:
            pad = torch.ones(
                self.stage_aux_classes - scale_tensor.numel(),
                dtype=scale_tensor.dtype,
                device=scale_tensor.device,
            )
            scale_tensor = torch.cat([scale_tensor, pad], dim=0)
        elif scale_tensor.numel() > self.stage_aux_classes:
            scale_tensor = scale_tensor[: self.stage_aux_classes]
        self.adapter_stage_scale_vector.copy_(scale_tensor.clamp_min(0.0))

    def _task_embedding(self, task_ids: torch.Tensor | None, state: torch.Tensor | None) -> torch.Tensor:
        resolved_task_ids = _resolve_task_ids(task_ids, state, self.task_vocab_size)
        return self.task_prompt_embedding_table.index_select(0, resolved_task_ids)

    def _task_action_prior(self, task_ids: torch.Tensor | None, state: torch.Tensor | None = None) -> torch.Tensor:
        resolved_task_ids = _resolve_task_ids(task_ids, state, self.task_vocab_size)
        return self.task_action_prior_table.index_select(0, resolved_task_ids) * self.task_prior_scale

    def _task_action_mask(self, task_ids: torch.Tensor | None, state: torch.Tensor | None = None) -> torch.Tensor:
        resolved_task_ids = _resolve_task_ids(task_ids, state, self.task_vocab_size)
        return self.task_action_mask_table.index_select(0, resolved_task_ids)

    def _predicted_stage_prior(self, z: torch.Tensor) -> torch.Tensor:
        stage_probs = torch.softmax(self.stage_head(z.detach()), dim=-1)
        return stage_probs @ self.stage_action_prior_table.to(dtype=z.dtype, device=z.device)

    def _proprio_state(self, state: torch.Tensor) -> torch.Tensor:
        return _strip_task_feature(state)

    def _transition_context_embeddings(
        self,
        action_index: torch.Tensor,
        state: torch.Tensor | None = None,
        task_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        resolved_action_index = action_index.long()
        action_emb = self.action_embedding(resolved_action_index)
        if self.use_transition_action_adapter:
            action_emb = action_emb + self.transition_action_adapter_scale * self.transition_action_adapter_embedding(
                resolved_action_index
            )
        if state is None:
            if task_ids is None:
                task_emb = torch.zeros_like(action_emb)
            else:
                resolved_task_ids = _resolve_task_ids(task_ids, None, self.task_vocab_size)
                task_emb = self.transition_task_embedding(resolved_task_ids)
            stage_emb = torch.zeros_like(action_emb)
        else:
            resolved_task_ids = _resolve_task_ids(task_ids, state, self.task_vocab_size)
            stage_ids = _extract_progress_bins(state, self.transition_stage_bins)
            task_emb = self.transition_task_embedding(resolved_task_ids)
            stage_emb = self.transition_stage_embedding(stage_ids)
        return action_emb, task_emb, stage_emb

    def transition_adapter_delta(
        self,
        z: torch.Tensor,
        action_index: torch.Tensor,
        state: torch.Tensor | None = None,
        task_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.use_transition_residual_adapter:
            return torch.zeros_like(z)
        action_emb, task_emb, stage_emb = self._transition_context_embeddings(
            action_index,
            state=state,
            task_ids=task_ids,
        )
        transition_input = torch.cat([z, action_emb, task_emb, stage_emb], dim=-1)
        if self.transition_residual_phase_split:
            observation_mask = self._primitive_observation_mask(action_index.long())
            obs_delta = self.transition_residual_adapter(transition_input)
            non_obs_delta = self.transition_non_observation_residual_adapter(transition_input)
            obs_scale = torch.full(
                (z.shape[0], 1),
                self.transition_residual_observation_scale,
                device=z.device,
                dtype=z.dtype,
            )
            non_obs_scale = torch.full(
                (z.shape[0], 1),
                self.transition_residual_non_observation_scale,
                device=z.device,
                dtype=z.dtype,
            )
            residual = torch.where(
                observation_mask.unsqueeze(-1),
                obs_scale * obs_delta,
                non_obs_scale * non_obs_delta,
            )
        else:
            base_delta = self.transition_residual_adapter(transition_input)
            if abs(self.transition_residual_observation_scale - self.transition_residual_non_observation_scale) > 1.0e-8:
                observation_mask = self._primitive_observation_mask(action_index.long())
                scale = torch.where(
                    observation_mask,
                    torch.full_like(observation_mask, self.transition_residual_observation_scale, dtype=z.dtype),
                    torch.full_like(observation_mask, self.transition_residual_non_observation_scale, dtype=z.dtype),
                ).unsqueeze(-1)
                residual = scale * base_delta
            else:
                residual = self.transition_residual_non_observation_scale * base_delta
        return self.transition_residual_scale * residual

    def transition_adapter_identity_penalty(self) -> torch.Tensor:
        penalties: list[torch.Tensor] = []
        for module in (
            self.transition_residual_adapter,
            self.transition_non_observation_residual_adapter,
        ):
            if module is None:
                continue
            for param in module.parameters():
                penalties.append(param.pow(2).mean())
        if not penalties:
            return torch.zeros((), device=next(self.parameters()).device)
        return torch.stack(penalties).sum()

    def policy_adapter_identity_penalty(self) -> torch.Tensor:
        penalties: list[torch.Tensor] = []
        for param in self.policy_residual_adapter.parameters():
            penalties.append(param.pow(2).mean())
        if not penalties:
            return torch.zeros((), device=next(self.parameters()).device)
        return torch.stack(penalties).sum()

    def predict_next(
        self,
        z: torch.Tensor,
        action_index: torch.Tensor,
        state: torch.Tensor | None = None,
        task_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        action_emb, task_emb, stage_emb = self._transition_context_embeddings(
            action_index,
            state=state,
            task_ids=task_ids,
        )
        delta = self.transition(torch.cat([z, action_emb, task_emb, stage_emb], dim=-1))
        if self.use_transition_residual_adapter:
            delta = delta + self.transition_adapter_delta(
                z,
                action_index,
                state=state,
                task_ids=task_ids,
            )
        return z + delta

    def _resolve_adapter_task_ids(
        self,
        z: torch.Tensor,
        task_ids: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if task_ids is None and state is None:
            return torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
        return _resolve_task_ids(task_ids, state, self.task_vocab_size).to(device=z.device)

    def _resolve_adapter_stage_ids(
        self,
        z: torch.Tensor,
        stage_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if stage_ids is not None:
            if stage_ids.ndim > 1:
                stage_ids = stage_ids.squeeze(-1)
            return stage_ids.long().clamp(0, max(self.stage_aux_classes - 1, 0)).to(device=z.device)
        return _predicted_stage_ids_from_latent(z, self.stage_head, self.stage_aux_classes).reshape(-1)

    def _resolve_adapter_prev_primitives(
        self,
        z: torch.Tensor,
        prev_primitives: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if prev_primitives is None:
            return torch.full(
                (z.shape[0],),
                self.action_dim,
                dtype=torch.long,
                device=z.device,
            )
        if prev_primitives.ndim > 1:
            prev_primitives = prev_primitives.squeeze(-1)
        return prev_primitives.long().clamp(0, self.action_dim).to(device=z.device)

    def _adapter_observation_mask(self, resolved_prev_primitives: torch.Tensor) -> torch.Tensor:
        is_observation = self._primitive_observation_mask(resolved_prev_primitives)
        is_start_token = resolved_prev_primitives == self.action_dim
        return is_observation | is_start_token

    def _primitive_observation_mask(self, primitive_ids: torch.Tensor) -> torch.Tensor:
        observation_ids = torch.tensor(
            sorted(
                _valid_family_primitives(
                    OBSERVATION_FAMILY_IDS,
                    self.action_dim,
                    self.primitive_vocabulary,
                )
            ),
            device=primitive_ids.device,
            dtype=primitive_ids.dtype,
        )
        return (primitive_ids.unsqueeze(-1) == observation_ids.unsqueeze(0)).any(dim=-1)

    def set_latent_alignment_stats(
        self,
        source_mean: torch.Tensor,
        source_std: torch.Tensor,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
    ) -> None:
        self.latent_align_source_mean.copy_(source_mean.to(device=self.latent_align_source_mean.device, dtype=self.latent_align_source_mean.dtype))
        self.latent_align_source_std.copy_(source_std.to(device=self.latent_align_source_std.device, dtype=self.latent_align_source_std.dtype))
        self.latent_align_target_mean.copy_(target_mean.to(device=self.latent_align_target_mean.device, dtype=self.latent_align_target_mean.dtype))
        self.latent_align_target_std.copy_(target_std.to(device=self.latent_align_target_std.device, dtype=self.latent_align_target_std.dtype))

    def align_latent(
        self,
        z: torch.Tensor,
        task_ids: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.latent_affine_alignment:
            return z
        if z.ndim == 1:
            z = z.unsqueeze(0)
        resolved_task_ids = _resolve_task_ids(task_ids, state, self.task_vocab_size).to(device=z.device)
        if not self.latent_affine_task_conditioned:
            resolved_task_ids = torch.zeros_like(resolved_task_ids)
        source_mean = self.latent_align_source_mean.index_select(0, resolved_task_ids).to(dtype=z.dtype)
        source_std = self.latent_align_source_std.index_select(0, resolved_task_ids).to(dtype=z.dtype)
        target_mean = self.latent_align_target_mean.index_select(0, resolved_task_ids).to(dtype=z.dtype)
        target_std = self.latent_align_target_std.index_select(0, resolved_task_ids).to(dtype=z.dtype).clamp_min(1.0e-6)
        scale = (source_std / target_std).clamp(
            min=1.0 / self.latent_affine_max_scale,
            max=self.latent_affine_max_scale,
        )
        aligned = (z - target_mean) * scale + source_mean
        if self.latent_affine_blend >= 1.0:
            return aligned
        return z + self.latent_affine_blend * (aligned - z)

    def adapt(
        self,
        z: torch.Tensor,
        task_ids: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
        stage_ids: torch.Tensor | None = None,
        prev_primitives: torch.Tensor | None = None,
    ) -> torch.Tensor:
        original_shape = z.shape
        if z.ndim == 1:
            z = z.unsqueeze(0)
        flat_z = z.reshape(-1, z.shape[-1])
        if state is not None and state.ndim == z.ndim:
            flat_state = state.reshape(-1, state.shape[-1])
        else:
            flat_state = state
        if task_ids is not None and task_ids.ndim == z.ndim - 1:
            flat_task_ids = task_ids.reshape(-1)
        else:
            flat_task_ids = task_ids
        if stage_ids is not None and stage_ids.ndim == z.ndim - 1:
            flat_stage_ids = stage_ids.reshape(-1)
        else:
            flat_stage_ids = stage_ids
        if prev_primitives is not None and prev_primitives.ndim == z.ndim - 1:
            flat_prev_primitives = prev_primitives.reshape(-1)
        else:
            flat_prev_primitives = prev_primitives
        resolved_task_ids = self._resolve_adapter_task_ids(flat_z, task_ids=flat_task_ids, state=flat_state)
        resolved_stage_ids = self._resolve_adapter_stage_ids(flat_z, stage_ids=flat_stage_ids)
        resolved_prev_primitives = self._resolve_adapter_prev_primitives(flat_z, prev_primitives=flat_prev_primitives)
        aligned_z = self.align_latent(flat_z, task_ids=resolved_task_ids)
        if self.adapter_use_task_condition:
            task_emb = self.adapter_task_embedding(resolved_task_ids)
        else:
            task_emb = torch.zeros(
                (aligned_z.shape[0], self.task_embed_dim),
                device=aligned_z.device,
                dtype=aligned_z.dtype,
            )
        if self.adapter_use_stage_condition:
            stage_emb = self.adapter_stage_embedding(resolved_stage_ids)
        else:
            stage_emb = torch.zeros(
                (aligned_z.shape[0], self.task_embed_dim),
                device=aligned_z.device,
                dtype=aligned_z.dtype,
            )
        if self.adapter_use_prev_action_condition:
            prev_action_emb = self.adapter_prev_action_embedding(resolved_prev_primitives)
        else:
            prev_action_emb = torch.zeros(
                (aligned_z.shape[0], self.action_embed_dim),
                device=aligned_z.device,
                dtype=aligned_z.dtype,
            )
        adapter_input = torch.cat([aligned_z, task_emb, stage_emb, prev_action_emb], dim=-1)
        progress_scale = _progressive_scale_from_stage_ids(
            resolved_stage_ids,
            self.transition_stage_bins,
            self.adapter_progressive_min_scale,
            self.adapter_progressive_max_scale,
            start_stage=self.adapter_condition_start_stage,
            end_stage=self.adapter_condition_end_stage,
            dtype=flat_z.dtype,
        )
        observation_mask = self._adapter_observation_mask(resolved_prev_primitives)
        if self.adapter_phase_split:
            obs_base_delta = self.adapter(aligned_z)
            if self.adapter_use_condition_branch:
                obs_cond_delta = self.adapter_condition(adapter_input)
            else:
                obs_cond_delta = torch.zeros_like(obs_base_delta)
            if not self.adapter_use_condition_branch or self.adapter_mode == "legacy_prevprim":
                obs_gate = torch.ones_like(obs_cond_delta)
            elif self.adapter_use_gate:
                obs_gate = 2.0 * torch.sigmoid(self.adapter_gate(adapter_input))
            else:
                obs_gate = torch.ones_like(obs_cond_delta)

            non_obs_base_delta = self.adapter_non_observation(aligned_z)
            if self.adapter_use_condition_branch:
                non_obs_cond_delta = self.adapter_non_observation_condition(adapter_input)
            else:
                non_obs_cond_delta = torch.zeros_like(non_obs_base_delta)
            if not self.adapter_use_condition_branch or self.adapter_mode == "legacy_prevprim":
                non_obs_gate = torch.ones_like(non_obs_cond_delta)
            elif self.adapter_use_gate:
                non_obs_gate = 2.0 * torch.sigmoid(self.adapter_non_observation_gate(adapter_input))
            else:
                non_obs_gate = torch.ones_like(non_obs_cond_delta)

            obs_residual = (obs_base_delta + obs_gate * obs_cond_delta) * self.adapter_scale
            non_obs_residual = (non_obs_base_delta + non_obs_gate * non_obs_cond_delta) * self.adapter_scale
            non_observation_scale_value = (
                self.adapter_condition_non_observation_scale
                if self.adapter_condition_observation_only
                else 1.0
            )
            non_observation_scale = torch.full(
                (aligned_z.shape[0], 1),
                float(non_observation_scale_value),
                device=aligned_z.device,
                dtype=aligned_z.dtype,
            )
            residual = torch.where(
                observation_mask.unsqueeze(-1),
                obs_residual,
                non_observation_scale * non_obs_residual,
            )
        else:
            base_delta = self.adapter(aligned_z)
            if self.adapter_use_condition_branch:
                cond_delta = self.adapter_condition(adapter_input)
            else:
                cond_delta = torch.zeros_like(base_delta)
            if not self.adapter_use_condition_branch:
                gate = torch.ones_like(cond_delta)
            elif self.adapter_mode == "legacy_prevprim":
                gate = torch.ones_like(cond_delta)
            elif self.adapter_use_gate:
                gate = 2.0 * torch.sigmoid(self.adapter_gate(adapter_input))
            else:
                gate = torch.ones_like(cond_delta)
            if self.adapter_condition_observation_only:
                observation_scale = torch.where(
                    observation_mask,
                    torch.ones_like(observation_mask, dtype=aligned_z.dtype),
                    torch.full_like(observation_mask, self.adapter_condition_non_observation_scale, dtype=aligned_z.dtype),
                ).unsqueeze(-1)
                progress_scale = progress_scale * observation_scale
            residual = (base_delta + gate * cond_delta) * self.adapter_scale
        stage_scale = self.adapter_stage_scale_vector.index_select(0, resolved_stage_ids).to(dtype=aligned_z.dtype).unsqueeze(-1)
        progress_scale = progress_scale * stage_scale
        adapted = aligned_z + progress_scale * residual
        return adapted.view(*original_shape)

    def encode_adapted(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        task_ids: torch.Tensor | None = None,
        prev_primitives: torch.Tensor | None = None,
        stage_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.adapt(
            self.encode(image, state, task_ids=task_ids, prev_primitives=prev_primitives),
            task_ids=task_ids,
            state=state,
            stage_ids=stage_ids,
            prev_primitives=prev_primitives,
        )

    def encode_step_adapted(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        runtime_state: Any = None,
        task_ids: torch.Tensor | None = None,
        stage_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Any]:
        z, next_runtime_state = self.encode_step(image, state, runtime_state, task_ids=task_ids)
        prev_primitive = None
        if isinstance(runtime_state, dict) and "prev_primitive" in runtime_state:
            prev_primitive = runtime_state["prev_primitive"]
        return self.adapt(
            z,
            task_ids=task_ids,
            state=state,
            stage_ids=stage_ids,
            prev_primitives=prev_primitive,
        ), next_runtime_state

    def freeze_backbone(self) -> None:
        for name, param in self.named_parameters():
            if not (
                name.startswith("adapter.")
                or name.startswith("adapter_task_embedding.")
                or name.startswith("adapter_stage_embedding.")
                or name.startswith("adapter_prev_action_embedding.")
                or name.startswith("adapter_condition.")
                or name.startswith("adapter_gate.")
                or name.startswith("adapter_non_observation.")
                or name.startswith("adapter_non_observation_condition.")
                or name.startswith("adapter_non_observation_gate.")
                or (self.use_transition_action_adapter and name.startswith("transition_action_adapter_embedding."))
                or (self.use_transition_residual_adapter and name.startswith("transition_residual_adapter."))
                or (
                    self.use_transition_residual_adapter
                    and self.transition_residual_phase_split
                    and name.startswith("transition_non_observation_residual_adapter.")
                )
                or (self.use_policy_residual_adapter and name.startswith("policy_residual_adapter."))
            ):
                param.requires_grad_(False)

    def adapter_parameters(self):
        modules = [
            self.adapter.parameters(),
            self.adapter_task_embedding.parameters(),
            self.adapter_stage_embedding.parameters(),
            self.adapter_prev_action_embedding.parameters(),
            self.adapter_condition.parameters(),
        ]
        if self.adapter_mode != "legacy_prevprim":
            modules.append(self.adapter_gate.parameters())
        if self.adapter_phase_split:
            modules.extend(
                [
                    self.adapter_non_observation.parameters(),
                    self.adapter_non_observation_condition.parameters(),
                ]
            )
            if self.adapter_mode != "legacy_prevprim":
                modules.append(self.adapter_non_observation_gate.parameters())
        if self.use_transition_action_adapter:
            modules.append(self.transition_action_adapter_embedding.parameters())
        if self.use_transition_residual_adapter:
            modules.append(self.transition_residual_adapter.parameters())
            if self.transition_residual_phase_split:
                modules.append(self.transition_non_observation_residual_adapter.parameters())
        if self.use_policy_residual_adapter:
            modules.append(self.policy_residual_adapter.parameters())
        for params in modules:
            yield from params

    def gate_identity_penalty(self) -> torch.Tensor:
        penalties: list[torch.Tensor] = []
        for attr_name in (
            "adapter_gate",
            "adapter_non_observation_gate",
            "feature_adapter_gate",
            "condition_adapter_gate",
        ):
            module = getattr(self, attr_name, None)
            if module is None:
                continue
            for param in module.parameters():
                penalties.append(param.pow(2).mean())
        if not penalties:
            return torch.zeros((), device=next(self.parameters()).device)
        return torch.stack(penalties).sum()

    def init_runtime_state(self, batch_size: int = 1, device: torch.device | None = None) -> Any:
        target_device = device or next(self.parameters()).device
        return {
            "prev_primitive": torch.full((batch_size,), self.action_dim, dtype=torch.long, device=target_device),
        }

    @abstractmethod
    def encode(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        task_ids: torch.Tensor | None = None,
        prev_primitives: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def policy_logits(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def condition_policy_logits(
        self,
        logits: torch.Tensor,
        z: torch.Tensor | None = None,
        task_ids: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_policy_residual_adapter and z is not None:
            logits = logits + self.policy_residual_scale * self.policy_residual_adapter(z)
        prior = self._task_action_prior(task_ids, state)
        if logits.ndim == 2:
            conditioned = logits + prior
            if state is not None and self.stage_prior_scale > 0.0:
                resolved_task_ids = _resolve_task_ids(task_ids, state, self.task_vocab_size)
                conditioned = conditioned + self.stage_prior_scale * _stage_action_prior(
                    resolved_task_ids,
                    state,
                    logits.shape[-1],
                    positive=0.8,
                    negative=-0.8,
                    primitive_vocabulary=self.primitive_vocabulary,
                )
            if self.predicted_stage_prior_scale > 0.0 and z is not None:
                conditioned = conditioned + self.predicted_stage_prior_scale * self._predicted_stage_prior(z)
            if self.stage_hard_mask and state is not None:
                resolved_task_ids = _resolve_task_ids(task_ids, state, self.task_vocab_size)
                action_mask = _stage_action_mask(
                    resolved_task_ids,
                    state,
                    conditioned.shape[-1],
                    primitive_vocabulary=self.primitive_vocabulary,
                ).to(dtype=conditioned.dtype)
                conditioned = conditioned.masked_fill(action_mask < 0.5, -1.0e9)
            elif self.task_hard_mask:
                action_mask = self._task_action_mask(task_ids, state).to(dtype=conditioned.dtype)
                conditioned = conditioned.masked_fill(action_mask < 0.5, -1.0e9)
            return conditioned
        if logits.ndim == 3:
            conditioned = logits + prior.unsqueeze(1)
            if state is not None and self.stage_prior_scale > 0.0:
                if state.ndim == 2:
                    resolved_task_ids = _resolve_task_ids(task_ids, state, self.task_vocab_size)
                    stage_prior = _stage_action_prior(
                        resolved_task_ids,
                        state,
                        logits.shape[-1],
                        positive=0.8,
                        negative=-0.8,
                        primitive_vocabulary=self.primitive_vocabulary,
                    ).unsqueeze(1)
                else:
                    batch, steps, _ = state.shape
                    flat_state = state.reshape(batch * steps, state.shape[-1])
                    if task_ids is None:
                        flat_task_ids = None
                    else:
                        flat_task_ids = task_ids.unsqueeze(1).expand(batch, steps).reshape(-1)
                    resolved_task_ids = _resolve_task_ids(flat_task_ids, flat_state, self.task_vocab_size)
                    stage_prior = _stage_action_prior(
                        resolved_task_ids,
                        flat_state,
                        logits.shape[-1],
                        positive=0.8,
                        negative=-0.8,
                        primitive_vocabulary=self.primitive_vocabulary,
                    ).view(batch, steps, logits.shape[-1])
                conditioned = conditioned + self.stage_prior_scale * stage_prior
            if self.predicted_stage_prior_scale > 0.0 and z is not None:
                if z.ndim == 2:
                    predicted_prior = self._predicted_stage_prior(z).unsqueeze(1)
                else:
                    predicted_prior = self._predicted_stage_prior(z.reshape(-1, z.shape[-1])).view(*z.shape[:-1], -1)
                conditioned = conditioned + self.predicted_stage_prior_scale * predicted_prior
            if self.stage_hard_mask and state is not None:
                if state.ndim == 2:
                    resolved_task_ids = _resolve_task_ids(task_ids, state, self.task_vocab_size)
                    action_mask = _stage_action_mask(
                        resolved_task_ids,
                        state,
                        conditioned.shape[-1],
                        primitive_vocabulary=self.primitive_vocabulary,
                    ).to(dtype=conditioned.dtype)
                    conditioned = conditioned.masked_fill(action_mask.unsqueeze(1) < 0.5, -1.0e9)
                    return conditioned
                if state.ndim == 3:
                    batch, steps, _ = state.shape
                    flat_state = state.reshape(batch * steps, state.shape[-1])
                    if task_ids is None:
                        flat_task_ids = None
                    else:
                        flat_task_ids = task_ids.unsqueeze(1).expand(batch, steps).reshape(-1)
                    resolved_task_ids = _resolve_task_ids(flat_task_ids, flat_state, self.task_vocab_size)
                    action_mask = _stage_action_mask(
                        resolved_task_ids,
                        flat_state,
                        conditioned.shape[-1],
                        primitive_vocabulary=self.primitive_vocabulary,
                    ).to(dtype=conditioned.dtype)
                    conditioned = conditioned.masked_fill(action_mask.view(batch, steps, conditioned.shape[-1]) < 0.5, -1.0e9)
                    return conditioned
            if self.task_hard_mask:
                if state is not None:
                    action_mask = self._task_action_mask(task_ids, state).to(dtype=conditioned.dtype)
                else:
                    action_mask = self._task_action_mask(task_ids, None).to(dtype=conditioned.dtype)
                conditioned = conditioned.masked_fill(action_mask.unsqueeze(1) < 0.5, -1.0e9)
            return conditioned
        raise ValueError(f"Unsupported logits rank for task conditioning: {logits.ndim}")

    def encode_step(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        runtime_state: Any = None,
        task_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Any]:
        prev_primitive = None
        if isinstance(runtime_state, dict) and "prev_primitive" in runtime_state:
            prev_primitive = runtime_state["prev_primitive"]
        return self.encode(image, state, task_ids=task_ids, prev_primitives=prev_primitive), runtime_state

    def compute_latents(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        task_ids = batch.get("task")
        z = self.encode(
            batch["image"],
            batch["state"],
            task_ids=task_ids,
            prev_primitives=batch.get("prev_primitive_id"),
        )
        next_z = self.encode(
            batch["next_image"],
            batch["next_state"],
            task_ids=task_ids,
            prev_primitives=batch.get("next_prev_primitive_id"),
        )
        return z, next_z

    def compute_adapted_latents(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        task_ids = batch.get("task")
        z = self.encode_adapted(
            batch["image"],
            batch["state"],
            task_ids=task_ids,
            prev_primitives=batch.get("prev_primitive_id"),
            stage_ids=batch.get("stage_id"),
        )
        next_z = self.encode_adapted(
            batch["next_image"],
            batch["next_state"],
            task_ids=task_ids,
            prev_primitives=batch.get("next_prev_primitive_id"),
            stage_ids=batch.get("next_stage_id"),
        )
        return z, next_z

    def compute_policy_loss(self, batch: dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        logits = self.condition_policy_logits(self.policy_logits(z), z=z, task_ids=batch.get("task"), state=batch.get("state"))
        return F.cross_entropy(logits, batch["primitive_id"])

    def inverse_logits(self, z: torch.Tensor, next_z: torch.Tensor) -> torch.Tensor:
        delta = next_z - z
        return self.inverse_head(torch.cat([z, delta], dim=-1))

    def compute_inverse_loss(
        self,
        batch: dict[str, torch.Tensor],
        z: torch.Tensor,
        next_z: torch.Tensor,
        *,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        losses = F.cross_entropy(self.inverse_logits(z, next_z), batch["primitive_id"], reduction="none")
        if weights is None:
            return losses.mean()
        return weights.mul(losses).mean()

    def compute_stage_loss(self, batch: dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        if "stage_id" not in batch:
            return torch.zeros((), device=z.device, dtype=z.dtype)
        return F.cross_entropy(self.stage_head(z), batch["stage_id"].long())

    def act(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        runtime_state: Any = None,
        use_adapter: bool = False,
        task_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Any, torch.Tensor]:
        if runtime_state is None:
            runtime_state = self.init_runtime_state(batch_size=image.shape[0], device=image.device)
        if use_adapter:
            z_used, next_runtime_state = self.encode_step_adapted(image, state, runtime_state, task_ids=task_ids)
        else:
            z_used, next_runtime_state = self.encode_step(image, state, runtime_state, task_ids=task_ids)
        logits = self.condition_policy_logits(self.policy_logits(z_used), z=z_used, task_ids=task_ids, state=state)
        primitive = logits.argmax(dim=-1)
        if isinstance(next_runtime_state, dict):
            next_runtime_state["prev_primitive"] = primitive.detach()
        return primitive, next_runtime_state, z_used

    # Backward-compatible wrappers for older viewer/debug scripts.
    def adapted_latent(self, z: torch.Tensor, *_unused) -> torch.Tensor:
        return self.adapt(z)

    def predict_action(self, z: torch.Tensor, *_unused) -> torch.Tensor:
        return self.policy_logits(z)

    def predict_residual(self, z: torch.Tensor, *_unused) -> torch.Tensor:
        return torch.zeros(z.shape[0], 6, device=z.device, dtype=z.dtype)

    def action_features(self, action_index: torch.Tensor, action_residual: torch.Tensor | None = None) -> torch.Tensor:
        action_one_hot = torch.nn.functional.one_hot(action_index.long(), num_classes=self.action_dim).float()
        if action_residual is None:
            return action_one_hot
        return torch.cat([action_one_hot, action_residual.float()], dim=-1)


class FeedForwardTTLAModel(BaseTTLAModel):
    def __init__(self, use_prev_action_context: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._backbone_type = "feedforward"
        self.use_prev_action_context = use_prev_action_context
        self.image_encoder = ImageEncoder(self.hidden_dim)
        self.state_encoder = nn.Sequential(
            nn.Linear(self.proprio_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.prev_action_embedding = nn.Embedding(self.action_dim + 1, self.action_embed_dim)
        self.fusion = nn.Sequential(
            nn.Linear(
                self.hidden_dim * 2 + self.task_embed_dim + (self.action_embed_dim if use_prev_action_context else 0),
                self.latent_dim,
            ),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

    def encode(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        task_ids: torch.Tensor | None = None,
        prev_primitives: torch.Tensor | None = None,
    ) -> torch.Tensor:
        visual = self.image_encoder(image)
        proprio = self.state_encoder(self._proprio_state(state))
        task_feat = self._task_embedding(task_ids, state)
        parts = [visual, proprio, task_feat]
        if self.use_prev_action_context:
            if prev_primitives is None:
                prev_primitives = torch.full((image.shape[0],), self.action_dim, dtype=torch.long, device=image.device)
            parts.append(self.prev_action_embedding(prev_primitives.long().clamp(0, self.action_dim)))
        return self.fusion(torch.cat(parts, dim=-1))

    def policy_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.policy_head(z)


class RecurrentTTLAModel(BaseTTLAModel):
    uses_history = True

    def __init__(
        self,
        history_len: int = 4,
        prev_action_dropout: float = 0.35,
        use_prev_action: bool = False,
        runtime_horizon: int | None = None,
        sequence_final_weight: float = 1.0,
        recurrent_adapter_use_feature: bool = True,
        recurrent_adapter_use_latent: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._backbone_type = "recurrent"
        self.history_len = history_len
        self.prev_action_dropout = prev_action_dropout
        self.use_prev_action = use_prev_action
        self.runtime_horizon = runtime_horizon or history_len
        self.sequence_final_weight = sequence_final_weight
        self.recurrent_adapter_use_feature = recurrent_adapter_use_feature
        self.recurrent_adapter_use_latent = recurrent_adapter_use_latent
        self.image_encoder = ImageEncoder(self.hidden_dim)
        self.state_encoder = nn.Sequential(
            nn.Linear(self.proprio_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.feature_fusion = nn.Sequential(
            nn.Linear(
                self.hidden_dim * 2 + self.task_embed_dim + (self.action_embed_dim if use_prev_action else 0),
                self.hidden_dim,
            ),
            nn.ReLU(),
        )
        self.gru = nn.GRU(self.hidden_dim, self.latent_dim, batch_first=True)
        self.policy_head = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )
        self.prev_action_embedding = nn.Embedding(self.action_dim + 1, self.action_embed_dim)
        self.feature_adapter = nn.Sequential(
            nn.Linear(self.hidden_dim, kwargs.get("adapter_hidden_dim", 64)),
            nn.ReLU(),
            nn.Linear(kwargs.get("adapter_hidden_dim", 64), self.hidden_dim),
            nn.Tanh(),
        )
        self.feature_adapter_task_embedding = nn.Embedding(self.task_vocab_size, self.task_embed_dim)
        self.feature_adapter_progress_embedding = nn.Embedding(self.transition_stage_bins, self.task_embed_dim)
        self.feature_adapter_prev_action_embedding = nn.Embedding(self.action_dim + 1, self.action_embed_dim)
        self.feature_adapter_condition = nn.Sequential(
            nn.Linear(
                self.hidden_dim + self.task_embed_dim * 2 + self.action_embed_dim,
                kwargs.get("adapter_hidden_dim", 64),
            ),
            nn.ReLU(),
            nn.Linear(kwargs.get("adapter_hidden_dim", 64), self.hidden_dim),
            nn.Tanh(),
        )
        self.feature_adapter_gate = nn.Sequential(
            nn.Linear(
                self.hidden_dim + self.task_embed_dim * 2 + self.action_embed_dim,
                kwargs.get("adapter_hidden_dim", 64),
            ),
            nn.ReLU(),
            nn.Linear(kwargs.get("adapter_hidden_dim", 64), self.hidden_dim),
        )
        nn.init.zeros_(self.feature_adapter[2].weight)
        nn.init.zeros_(self.feature_adapter[2].bias)
        nn.init.zeros_(self.feature_adapter_condition[2].weight)
        nn.init.zeros_(self.feature_adapter_condition[2].bias)
        nn.init.zeros_(self.feature_adapter_gate[2].weight)
        nn.init.zeros_(self.feature_adapter_gate[2].bias)

    def _feature_step(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        task_ids: torch.Tensor | None = None,
        prev_primitives: torch.Tensor | None = None,
    ) -> torch.Tensor:
        visual = self.image_encoder(image)
        proprio = self.state_encoder(self._proprio_state(state))
        task_feat = self._task_embedding(task_ids, state)
        parts = [visual, proprio, task_feat]
        if self.use_prev_action:
            if prev_primitives is None:
                prev_primitives = torch.full(
                    (image.shape[0],),
                    self.action_dim,
                    dtype=torch.long,
                    device=image.device,
                )
            elif self.training and self.prev_action_dropout > 0.0:
                dropout_mask = torch.rand(prev_primitives.shape, device=prev_primitives.device) < self.prev_action_dropout
                prev_primitives = prev_primitives.clone()
                prev_primitives[dropout_mask] = self.action_dim
            prev_feat = self.prev_action_embedding(prev_primitives.long().clamp(0, self.action_dim))
            parts.append(prev_feat)
        return self.feature_fusion(torch.cat(parts, dim=-1))

    def init_runtime_state(self, batch_size: int = 1, device: torch.device | None = None) -> dict[str, Any]:
        target_device = device or next(self.parameters()).device
        return {
            "hidden_state": None,
            "prev_primitive": torch.full((batch_size,), self.action_dim, dtype=torch.long, device=target_device),
            "steps": 0,
        }

    def _adapt_feature(
        self,
        feature: torch.Tensor,
        state: torch.Tensor,
        task_ids: torch.Tensor | None = None,
        prev_primitives: torch.Tensor | None = None,
    ) -> torch.Tensor:
        resolved_task_ids = _resolve_task_ids(task_ids, state, self.task_vocab_size).to(device=feature.device)
        progress_bins = _extract_progress_bins(state, self.transition_stage_bins).to(device=feature.device)
        task_emb = self.feature_adapter_task_embedding(resolved_task_ids)
        progress_emb = self.feature_adapter_progress_embedding(progress_bins)
        prev_ids = self._resolve_adapter_prev_primitives(feature, prev_primitives=prev_primitives)
        prev_emb = self.feature_adapter_prev_action_embedding(prev_ids)
        adapter_input = torch.cat([feature, task_emb, progress_emb, prev_emb], dim=-1)
        base_delta = self.feature_adapter(feature)
        cond_delta = self.feature_adapter_condition(adapter_input)
        if self.adapter_mode == "legacy_prevprim":
            gate = torch.ones_like(cond_delta)
        else:
            gate = 2.0 * torch.sigmoid(self.feature_adapter_gate(adapter_input))
        progress_scale = _progressive_scale_from_stage_ids(
            progress_bins,
            self.transition_stage_bins,
            self.adapter_progressive_min_scale,
            self.adapter_progressive_max_scale,
            start_stage=self.adapter_condition_start_stage,
            end_stage=self.adapter_condition_end_stage,
            dtype=feature.dtype,
        )
        residual = (base_delta + gate * cond_delta) * self.adapter_scale
        return feature + progress_scale * residual

    def _encode_step_internal(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        runtime_state: Any = None,
        task_ids: torch.Tensor | None = None,
        use_adapter: bool = False,
    ) -> tuple[torch.Tensor, Any]:
        if runtime_state is None:
            runtime_state = self.init_runtime_state(batch_size=image.shape[0], device=image.device)
        prev_primitive = runtime_state["prev_primitive"]
        hidden_state = runtime_state["hidden_state"]
        steps = int(runtime_state.get("steps", 0))
        if hidden_state is None or (self.runtime_horizon > 0 and steps % self.runtime_horizon == 0):
            hidden_state = torch.zeros((1, image.shape[0], self.latent_dim), device=image.device, dtype=image.dtype)
        feature = self._feature_step(image, state, task_ids=task_ids, prev_primitives=prev_primitive)
        if use_adapter:
            feature = self._adapt_feature(feature, state, task_ids=task_ids, prev_primitives=prev_primitive)
        feature = feature.unsqueeze(1)
        output, next_hidden_state = self.gru(feature, hidden_state)
        z = output[:, -1]
        next_state = {
            "hidden_state": next_hidden_state.detach(),
            "prev_primitive": prev_primitive,
            "steps": steps + 1,
        }
        return z, next_state

    def encode_step(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        runtime_state: Any = None,
        task_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Any]:
        return self._encode_step_internal(image, state, runtime_state, task_ids=task_ids, use_adapter=False)

    def encode_step_adapted(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        runtime_state: Any = None,
        task_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Any]:
        z, next_runtime_state = self._encode_step_internal(
            image,
            state,
            runtime_state,
            task_ids=task_ids,
            use_adapter=self.recurrent_adapter_use_feature,
        )
        prev_primitive = None
        if runtime_state is not None:
            prev_primitive = runtime_state.get("prev_primitive")
        adapted_z = (
            self.adapt(z, task_ids=task_ids, state=state, prev_primitives=prev_primitive)
            if self.recurrent_adapter_use_latent
            else z
        )
        return adapted_z, next_runtime_state

    def encode(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        task_ids: torch.Tensor | None = None,
        prev_primitives: torch.Tensor | None = None,
    ) -> torch.Tensor:
        runtime_state = self.init_runtime_state(batch_size=image.shape[0], device=image.device)
        if prev_primitives is not None:
            runtime_state["prev_primitive"] = prev_primitives.long().clamp(0, self.action_dim)
        z, _ = self.encode_step(image, state, runtime_state, task_ids=task_ids)
        return z

    def encode_adapted(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        task_ids: torch.Tensor | None = None,
        prev_primitives: torch.Tensor | None = None,
    ) -> torch.Tensor:
        runtime_state = self.init_runtime_state(batch_size=image.shape[0], device=image.device)
        if prev_primitives is not None:
            runtime_state["prev_primitive"] = prev_primitives.long().clamp(0, self.action_dim)
        z, _ = self.encode_step_adapted(image, state, runtime_state, task_ids=task_ids)
        return z

    def encode_history(
        self,
        images: torch.Tensor,
        states: torch.Tensor,
        prev_primitives: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        task_ids: torch.Tensor | None = None,
        return_sequence: bool = False,
        use_adapter: bool = False,
    ) -> torch.Tensor:
        batch_size, history_len = images.shape[:2]
        flat_images = images.reshape(batch_size * history_len, *images.shape[2:])
        flat_states = states.reshape(batch_size * history_len, states.shape[-1])
        if self.use_prev_action:
            if prev_primitives is None:
                flat_prev_primitives = torch.full(
                    (batch_size * history_len,),
                    self.action_dim,
                    dtype=torch.long,
                    device=images.device,
                )
            else:
                flat_prev_primitives = prev_primitives.reshape(batch_size * history_len)
        else:
            flat_prev_primitives = None
        if task_ids is not None:
            expanded_task_ids = task_ids.unsqueeze(1).expand(-1, history_len).reshape(batch_size * history_len)
        else:
            expanded_task_ids = None
        fused = self._feature_step(
            flat_images,
            flat_states,
            task_ids=expanded_task_ids,
            prev_primitives=flat_prev_primitives,
        )
        if use_adapter and self.recurrent_adapter_use_feature:
            fused = self._adapt_feature(
                fused,
                flat_states,
                task_ids=expanded_task_ids,
                prev_primitives=flat_prev_primitives,
            )
        fused = fused.reshape(batch_size, history_len, self.hidden_dim)
        output, _ = self.gru(fused)
        if return_sequence:
            return (
                self.adapt(output, task_ids=task_ids, state=states, prev_primitives=prev_primitives)
                if use_adapter and self.recurrent_adapter_use_latent
                else output
            )
        if mask is None:
            z = output[:, -1]
            current_prev = None if prev_primitives is None else prev_primitives[:, -1]
            return (
                self.adapt(z, task_ids=task_ids, state=states[:, -1], prev_primitives=current_prev)
                if use_adapter and self.recurrent_adapter_use_latent
                else z
            )
        lengths = mask.long().sum(dim=1).clamp_min(1)
        gather_idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, output.shape[-1])
        z = output.gather(1, gather_idx).squeeze(1)
        if not use_adapter or not self.recurrent_adapter_use_latent:
            return z
        batch_idx = torch.arange(states.shape[0], device=states.device)
        final_states = states[batch_idx, lengths - 1]
        final_prev = None if prev_primitives is None else prev_primitives[batch_idx, lengths - 1]
        return self.adapt(z, task_ids=task_ids, state=final_states, prev_primitives=final_prev)

    def compute_latents(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if "history_images" not in batch:
            return super().compute_latents(batch)
        task_ids = batch.get("task")
        z = self.encode_history(
            batch["history_images"],
            batch["history_states"],
            batch.get("history_prev_primitives"),
            batch.get("history_mask"),
            task_ids=task_ids,
        )
        next_z = self.encode_history(
            batch["next_history_images"],
            batch["next_history_states"],
            batch.get("next_history_prev_primitives"),
            batch.get("next_history_mask"),
            task_ids=task_ids,
        )
        return z, next_z

    def compute_adapted_latents(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if "history_images" not in batch:
            return super().compute_adapted_latents(batch)
        task_ids = batch.get("task")
        z = self.encode_history(
            batch["history_images"],
            batch["history_states"],
            batch.get("history_prev_primitives"),
            batch.get("history_mask"),
            task_ids=task_ids,
            use_adapter=True,
        )
        next_z = self.encode_history(
            batch["next_history_images"],
            batch["next_history_states"],
            batch.get("next_history_prev_primitives"),
            batch.get("next_history_mask"),
            task_ids=task_ids,
            use_adapter=True,
        )
        return z, next_z

    def policy_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.policy_head(z)

    def compute_policy_loss(self, batch: dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        if "history_images" not in batch or "history_primitive_ids" not in batch:
            return super().compute_policy_loss(batch, z)
        task_ids = batch.get("task")
        seq_latents = self.encode_history(
            batch["history_images"],
            batch["history_states"],
            batch.get("history_prev_primitives"),
            batch.get("history_mask"),
            task_ids=task_ids,
            return_sequence=True,
        )
        seq_logits = self.condition_policy_logits(
            self.policy_head(seq_latents),
            z=seq_latents,
            task_ids=task_ids,
            state=batch.get("history_states"),
        )
        targets = batch["history_primitive_ids"]
        mask = batch.get("history_mask")
        losses = F.cross_entropy(
            seq_logits.reshape(-1, self.action_dim),
            targets.reshape(-1),
            reduction="none",
        ).view_as(targets)
        if mask is None:
            return losses.mean()
        if self.sequence_final_weight > 1.0:
            lengths = mask.long().sum(dim=1).clamp_min(1)
            batch_index = torch.arange(losses.shape[0], device=losses.device)
            losses = losses.clone()
            losses[batch_index, lengths - 1] *= self.sequence_final_weight
        weighted_mask = mask.clone()
        if self.sequence_final_weight > 1.0:
            lengths = mask.long().sum(dim=1).clamp_min(1)
            batch_index = torch.arange(weighted_mask.shape[0], device=weighted_mask.device)
            weighted_mask[batch_index, lengths - 1] *= self.sequence_final_weight
        return (losses * mask).sum() / weighted_mask.sum().clamp_min(1.0)

    def act(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        runtime_state: Any = None,
        use_adapter: bool = False,
        task_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Any, torch.Tensor]:
        if use_adapter:
            z_used, next_runtime_state = self.encode_step_adapted(image, state, runtime_state, task_ids=task_ids)
        else:
            z_used, next_runtime_state = self.encode_step(image, state, runtime_state, task_ids=task_ids)
        logits = self.condition_policy_logits(self.policy_logits(z_used), z=z_used, task_ids=task_ids, state=state)
        primitive = logits.argmax(dim=-1)
        next_runtime_state["prev_primitive"] = primitive.detach()
        return primitive, next_runtime_state, z_used

    def freeze_backbone(self) -> None:
        allowed_prefixes = {
            "adapter.",
            "adapter_task_embedding.",
            "adapter_stage_embedding.",
            "adapter_condition.",
            "adapter_non_observation.",
            "adapter_non_observation_condition.",
        }
        if self.adapter_mode != "legacy_prevprim":
            allowed_prefixes.add("adapter_gate.")
            allowed_prefixes.add("adapter_non_observation_gate.")
        if self.recurrent_adapter_use_feature:
            allowed_prefixes.update(
                {
                    "feature_adapter.",
                    "feature_adapter_task_embedding.",
                    "feature_adapter_progress_embedding.",
                    "feature_adapter_prev_action_embedding.",
                    "feature_adapter_condition.",
                }
            )
            if self.adapter_mode != "legacy_prevprim":
                allowed_prefixes.add("feature_adapter_gate.")
        if self.use_transition_action_adapter:
            allowed_prefixes.add("transition_action_adapter_embedding.")
        if self.use_transition_residual_adapter:
            allowed_prefixes.add("transition_residual_adapter.")
            if self.transition_residual_phase_split:
                allowed_prefixes.add("transition_non_observation_residual_adapter.")
        if self.use_policy_residual_adapter:
            allowed_prefixes.add("policy_residual_adapter.")
        for name, param in self.named_parameters():
            if not any(name.startswith(prefix) for prefix in allowed_prefixes):
                param.requires_grad_(False)

    def adapter_parameters(self):
        modules = []
        if self.recurrent_adapter_use_latent:
            modules.extend(
                [
                    self.adapter.parameters(),
                    self.adapter_task_embedding.parameters(),
                    self.adapter_stage_embedding.parameters(),
                    self.adapter_condition.parameters(),
                ]
            )
            if self.adapter_mode != "legacy_prevprim":
                modules.append(self.adapter_gate.parameters())
            if self.adapter_phase_split:
                modules.extend(
                    [
                        self.adapter_non_observation.parameters(),
                        self.adapter_non_observation_condition.parameters(),
                    ]
                )
                if self.adapter_mode != "legacy_prevprim":
                    modules.append(self.adapter_non_observation_gate.parameters())
        if self.recurrent_adapter_use_feature:
            modules.extend(
                [
                    self.feature_adapter.parameters(),
                    self.feature_adapter_task_embedding.parameters(),
                    self.feature_adapter_progress_embedding.parameters(),
                    self.feature_adapter_prev_action_embedding.parameters(),
                    self.feature_adapter_condition.parameters(),
                ]
            )
            if self.adapter_mode != "legacy_prevprim":
                modules.append(self.feature_adapter_gate.parameters())
        if self.use_transition_action_adapter:
            modules.append(self.transition_action_adapter_embedding.parameters())
        if self.use_transition_residual_adapter:
            modules.append(self.transition_residual_adapter.parameters())
            if self.transition_residual_phase_split:
                modules.append(self.transition_non_observation_residual_adapter.parameters())
        if self.use_policy_residual_adapter:
            modules.append(self.policy_residual_adapter.parameters())
        for params in modules:
            yield from params


class ChunkingTTLAModel(BaseTTLAModel):
    uses_history = True

    def __init__(
        self,
        chunk_size: int = 3,
        history_len: int = 4,
        chunk_future_weight: float = 0.25,
        chunk_temporal_agg: bool = True,
        chunk_temporal_decay: float = 0.35,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._backbone_type = "chunking"
        self.chunk_size = chunk_size
        self.history_len = history_len
        self.chunk_future_weight = chunk_future_weight
        self.chunk_temporal_agg = chunk_temporal_agg
        self.chunk_temporal_decay = chunk_temporal_decay
        self.image_encoder = ImageEncoder(self.hidden_dim)
        self.state_encoder = nn.Sequential(
            nn.Linear(self.proprio_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.prev_action_embedding = nn.Embedding(self.action_dim + 1, self.action_embed_dim)
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + self.task_embed_dim + self.action_embed_dim, self.latent_dim),
            nn.ReLU(),
        )
        self.chunk_head = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim * self.chunk_size),
        )

    def encode(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        task_ids: torch.Tensor | None = None,
        prev_primitives: torch.Tensor | None = None,
    ) -> torch.Tensor:
        visual = self.image_encoder(image)
        proprio = self.state_encoder(self._proprio_state(state))
        task_feat = self._task_embedding(task_ids, state)
        if prev_primitives is None:
            prev_primitives = torch.full((image.shape[0],), self.action_dim, dtype=torch.long, device=image.device)
        prev_feat = self.prev_action_embedding(prev_primitives.long().clamp(0, self.action_dim))
        return self.fusion(torch.cat([visual, proprio, task_feat, prev_feat], dim=-1))

    def encode_history(
        self,
        images: torch.Tensor,
        states: torch.Tensor,
        prev_primitives: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        task_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is None:
            current_prev = None if prev_primitives is None else prev_primitives[:, -1]
            return self.encode(images[:, -1], states[:, -1], task_ids=task_ids, prev_primitives=current_prev)
        valid_index = (mask.long().sum(dim=1).clamp_min(1) - 1)
        batch_idx = torch.arange(images.shape[0], device=images.device)
        current_prev = None if prev_primitives is None else prev_primitives[batch_idx, valid_index]
        return self.encode(
            images[batch_idx, valid_index],
            states[batch_idx, valid_index],
            task_ids=task_ids,
            prev_primitives=current_prev,
        )

    def compute_latents(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if "history_images" not in batch:
            return super().compute_latents(batch)
        task_ids = batch.get("task")
        z = self.encode_history(
            batch["history_images"],
            batch["history_states"],
            batch.get("history_prev_primitives"),
            batch.get("history_mask"),
            task_ids=task_ids,
        )
        next_z = self.encode_history(
            batch["next_history_images"],
            batch["next_history_states"],
            batch.get("next_history_prev_primitives"),
            batch.get("next_history_mask"),
            task_ids=task_ids,
        )
        return z, next_z

    def policy_chunk_logits(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.chunk_head(z)
        return logits.view(z.shape[0], self.chunk_size, self.action_dim)

    def policy_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.policy_chunk_logits(z)[:, 0]

    def compute_policy_loss(self, batch: dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        chunk_logits = self.policy_chunk_logits(z)
        if "chunk_primitive_ids" not in batch:
            step0_logits = self.condition_policy_logits(
                chunk_logits[:, 0],
                z=z,
                task_ids=batch.get("task"),
                state=batch.get("state"),
            )
            return F.cross_entropy(step0_logits, batch["primitive_id"])
        targets = batch["chunk_primitive_ids"]
        mask = batch.get("chunk_mask")
        conditioned_logits = self.condition_policy_logits(
            chunk_logits,
            z=z.unsqueeze(1).expand(-1, self.chunk_size, -1),
            task_ids=batch.get("task"),
            state=batch.get("state"),
        )
        flat_logits = conditioned_logits.reshape(-1, self.action_dim)
        flat_targets = targets.reshape(-1)
        losses = F.cross_entropy(flat_logits, flat_targets, reduction="none").view_as(targets)
        step_weights = torch.ones(self.chunk_size, device=losses.device, dtype=losses.dtype)
        if self.chunk_size > 1:
            step_weights[1:] = self.chunk_future_weight
        losses = losses * step_weights.unsqueeze(0)
        if mask is None:
            return losses.mean()
        denom = mask.sum().clamp_min(1.0)
        return (losses * mask).sum() / denom

    def init_runtime_state(self, batch_size: int = 1, device: torch.device | None = None) -> dict[str, Any]:
        _target_device = device or next(self.parameters()).device
        return {
            "images": [],
            "states": [],
            "prev_primitives": [],
            "time": 0,
            "predicted_chunks": [],
        }

    def _encode_runtime_history(
        self,
        history_images: list[torch.Tensor],
        history_states: list[torch.Tensor],
        history_prev_primitives: list[torch.Tensor] | None = None,
        task_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        valid_len = min(len(history_images), self.history_len)
        history_images = history_images[-self.history_len :]
        history_states = history_states[-self.history_len :]
        image_shape = history_images[-1].shape[1:]
        state_dim = history_states[-1].shape[-1]
        batch_size = history_images[-1].shape[0]
        device = history_images[-1].device
        dtype = history_images[-1].dtype
        aligned_prev_primitives: list[torch.Tensor] | None = None
        if history_prev_primitives is not None:
            start_token = torch.full((batch_size,), self.action_dim, device=device, dtype=torch.long)
            aligned_prev_primitives = list(history_prev_primitives)
            if len(aligned_prev_primitives) < len(history_images):
                pad_needed = len(history_images) - len(aligned_prev_primitives)
                aligned_prev_primitives = [start_token.clone() for _ in range(pad_needed)] + aligned_prev_primitives
            aligned_prev_primitives = aligned_prev_primitives[-len(history_images) :]
        pad_len = self.history_len - valid_len
        if pad_len > 0:
            zero_image = torch.zeros((batch_size, *image_shape), device=device, dtype=dtype)
            zero_state = torch.zeros((batch_size, state_dim), device=device, dtype=history_states[-1].dtype)
            history_images = [zero_image.clone() for _ in range(pad_len)] + history_images
            history_states = [zero_state.clone() for _ in range(pad_len)] + history_states
            if aligned_prev_primitives is not None:
                start_token = torch.full((batch_size,), self.action_dim, device=device, dtype=torch.long)
                aligned_prev_primitives = [start_token.clone() for _ in range(pad_len)] + aligned_prev_primitives
        images = torch.stack(history_images, dim=1)
        states = torch.stack(history_states, dim=1)
        prev_primitives = None if aligned_prev_primitives is None else torch.stack(aligned_prev_primitives, dim=1)
        mask = torch.zeros((batch_size, self.history_len), device=device, dtype=states.dtype)
        mask[:, -valid_len:] = 1.0
        return self.encode_history(images, states, prev_primitives, mask, task_ids=task_ids)

    def act(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        runtime_state: Any = None,
        use_adapter: bool = False,
        task_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Any, torch.Tensor]:
        runtime_state = runtime_state or self.init_runtime_state(batch_size=image.shape[0], device=image.device)
        history_images = list(runtime_state["images"]) + [image]
        history_states = list(runtime_state["states"]) + [state]
        prev_history = list(runtime_state["prev_primitives"])
        z = self._encode_runtime_history(history_images, history_states, prev_history, task_ids=task_ids)
        z_used = self.adapt(z, task_ids=task_ids, state=state) if use_adapter else z
        chunk_logits = self.policy_chunk_logits(z_used)
        current_time = int(runtime_state.get("time", 0))
        predicted_chunks = list(runtime_state.get("predicted_chunks", []))
        predicted_chunks.append((current_time, chunk_logits.detach()))
        predicted_chunks = [
            (start, logits)
            for start, logits in predicted_chunks
            if current_time - start < self.chunk_size
        ]
        if self.chunk_temporal_agg:
            aggregated_probs: list[torch.Tensor] = []
            for start, logits in predicted_chunks:
                offset = current_time - start
                if offset < 0 or offset >= self.chunk_size:
                    continue
                aggregated_probs.append(F.softmax(logits[:, offset], dim=-1))
            if aggregated_probs:
                stacked_probs = torch.stack(aggregated_probs, dim=0)
                order = torch.arange(stacked_probs.shape[0], device=stacked_probs.device, dtype=stacked_probs.dtype)
                weights = torch.exp(-self.chunk_temporal_decay * order)
                weights = weights / weights.sum().clamp_min(1e-6)
                primitive_probs = (stacked_probs * weights.view(-1, 1, 1)).sum(dim=0)
                primitive = primitive_probs.argmax(dim=-1)
            else:
                primitive = self.condition_policy_logits(chunk_logits[:, 0], z=z_used, task_ids=task_ids, state=state).argmax(dim=-1)
        else:
            primitive = self.condition_policy_logits(chunk_logits[:, 0], z=z_used, task_ids=task_ids, state=state).argmax(dim=-1)
        next_state = {
            "images": history_images[-self.history_len :],
            "states": history_states[-self.history_len :],
            "prev_primitives": (prev_history + [primitive.detach()])[-self.history_len :],
            "time": current_time + 1,
            "predicted_chunks": predicted_chunks,
        }
        return primitive, next_state, z_used


class LanguageConditionedTTLAModel(BaseTTLAModel):
    def __init__(
        self,
        task_vocab_size: int = 3,
        language_dim: int = 32,
        use_prev_action_context: bool = False,
        language_action_prior_scale: float = 0.8,
        language_state_text_scale: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._backbone_type = "language"
        self.task_vocab_size = task_vocab_size
        self.language_dim = language_dim
        self.use_prev_action_context = use_prev_action_context
        self.language_action_prior_scale = language_action_prior_scale
        self.language_state_text_scale = language_state_text_scale
        self.image_encoder = ImageEncoder(self.hidden_dim)
        self.state_encoder = nn.Sequential(
            nn.Linear(self.proprio_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.register_buffer(
            "prompt_embedding_table",
            _fixed_prompt_embedding_table(
                task_vocab_size,
                language_dim,
                primitive_vocabulary=self.primitive_vocabulary,
            ),
        )
        self.register_buffer(
            "action_hint_embedding_table",
            _fixed_task_action_hint_table(
                task_vocab_size,
                language_dim,
                primitive_vocabulary=self.primitive_vocabulary,
            ),
        )
        self.register_buffer(
            "primitive_prompt_embedding_table",
            _fixed_primitive_text_embedding_table(
                self.action_dim,
                language_dim,
                primitive_vocabulary=self.primitive_vocabulary,
            ),
        )
        self.language_encoder = nn.Sequential(
            nn.Linear(language_dim * 2, language_dim),
            nn.ReLU(),
        )
        self.primitive_text_projector = nn.Linear(language_dim, self.latent_dim, bias=False)
        self.prev_action_embedding = nn.Embedding(self.action_dim + 1, self.action_embed_dim)
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + language_dim + (self.action_embed_dim if use_prev_action_context else 0), self.latent_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

    def encode(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        task_ids: torch.Tensor | None = None,
        prev_primitives: torch.Tensor | None = None,
    ) -> torch.Tensor:
        visual = self.image_encoder(image)
        proprio = self.state_encoder(self._proprio_state(state))
        resolved_task_ids = _resolve_task_ids(task_ids, state, self.task_vocab_size)
        prompt = self.prompt_embedding_table.index_select(0, resolved_task_ids)
        action_hint = self.action_hint_embedding_table.index_select(0, resolved_task_ids)
        language = self.language_encoder(torch.cat([prompt, action_hint], dim=-1))
        parts = [visual, proprio, language]
        if self.use_prev_action_context:
            if prev_primitives is None:
                prev_primitives = torch.full((image.shape[0],), self.action_dim, dtype=torch.long, device=image.device)
            parts.append(self.prev_action_embedding(prev_primitives.long().clamp(0, self.action_dim)))
        return self.fusion(torch.cat(parts, dim=-1))

    def policy_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.policy_head(z)

    def condition_policy_logits(
        self,
        logits: torch.Tensor,
        z: torch.Tensor | None = None,
        task_ids: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        conditioned = super().condition_policy_logits(logits, z=z, task_ids=task_ids, state=state)
        resolved_task_ids = _resolve_task_ids(task_ids, state, self.task_vocab_size)
        prompt = self.action_hint_embedding_table.index_select(0, resolved_task_ids)
        primitive_emb = self.primitive_prompt_embedding_table
        prompt = F.normalize(prompt, dim=-1)
        primitive_emb = F.normalize(primitive_emb, dim=-1)
        prior = prompt @ primitive_emb.t()
        prior = prior - prior.mean(dim=-1, keepdim=True)
        if z is not None and self.language_state_text_scale > 0.0:
            primitive_proto = F.normalize(self.primitive_text_projector(primitive_emb), dim=-1)
            semantic = F.normalize(z, dim=-1) @ primitive_proto.t()
            semantic = semantic - semantic.mean(dim=-1, keepdim=True)
            prior = prior + self.language_state_text_scale * semantic
        if conditioned.ndim == 3:
            prior = prior.unsqueeze(1)
        return conditioned + self.language_action_prior_scale * prior


class DiffusionPrimitiveTTLAModel(BaseTTLAModel):
    def __init__(
        self,
        diffusion_steps: int = 4,
        diffusion_dim: int = 32,
        diffusion_loss_weight: float = 0.5,
        diffusion_logit_blend: float = 0.5,
        use_prev_action_context: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._backbone_type = "diffusion"
        self.diffusion_steps = diffusion_steps
        self.diffusion_dim = diffusion_dim
        self.diffusion_loss_weight = diffusion_loss_weight
        self.diffusion_logit_blend = diffusion_logit_blend
        self.use_prev_action_context = use_prev_action_context
        self.image_encoder = ImageEncoder(self.hidden_dim)
        self.state_encoder = nn.Sequential(
            nn.Linear(self.proprio_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.prev_action_embedding = nn.Embedding(self.action_dim + 1, self.action_embed_dim)
        self.fusion = nn.Sequential(
            nn.Linear(
                self.hidden_dim * 2 + self.task_embed_dim + (self.action_embed_dim if use_prev_action_context else 0),
                self.latent_dim,
            ),
            nn.ReLU(),
        )
        self.action_codebook = nn.Embedding(self.action_dim, diffusion_dim)
        self.time_embedding = nn.Embedding(diffusion_steps, diffusion_dim)
        self.denoiser = nn.Sequential(
            nn.Linear(self.latent_dim + diffusion_dim + diffusion_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, diffusion_dim),
        )
        self.direct_head = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )
        self.policy_head = self.direct_head
        self.condition_adapter = nn.Sequential(
            nn.Linear(self.latent_dim, kwargs.get("adapter_hidden_dim", 64)),
            nn.ReLU(),
            nn.Linear(kwargs.get("adapter_hidden_dim", 64), self.latent_dim),
            nn.Tanh(),
        )
        self.condition_adapter_task_embedding = nn.Embedding(self.task_vocab_size, self.task_embed_dim)
        self.condition_adapter_stage_embedding = nn.Embedding(self.stage_aux_classes, self.task_embed_dim)
        self.condition_adapter_condition = nn.Sequential(
            nn.Linear(self.latent_dim + self.task_embed_dim * 2, kwargs.get("adapter_hidden_dim", 64)),
            nn.ReLU(),
            nn.Linear(kwargs.get("adapter_hidden_dim", 64), self.latent_dim),
            nn.Tanh(),
        )
        self.condition_adapter_gate = nn.Sequential(
            nn.Linear(self.latent_dim + self.task_embed_dim * 2, kwargs.get("adapter_hidden_dim", 64)),
            nn.ReLU(),
            nn.Linear(kwargs.get("adapter_hidden_dim", 64), self.latent_dim),
        )
        nn.init.zeros_(self.condition_adapter[2].weight)
        nn.init.zeros_(self.condition_adapter[2].bias)
        nn.init.zeros_(self.condition_adapter_condition[2].weight)
        nn.init.zeros_(self.condition_adapter_condition[2].bias)
        nn.init.zeros_(self.condition_adapter_gate[2].weight)
        nn.init.zeros_(self.condition_adapter_gate[2].bias)

    def encode(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        task_ids: torch.Tensor | None = None,
        prev_primitives: torch.Tensor | None = None,
    ) -> torch.Tensor:
        visual = self.image_encoder(image)
        proprio = self.state_encoder(self._proprio_state(state))
        task_feat = self._task_embedding(task_ids, state)
        parts = [visual, proprio, task_feat]
        if self.use_prev_action_context:
            if prev_primitives is None:
                prev_primitives = torch.full((image.shape[0],), self.action_dim, dtype=torch.long, device=image.device)
            parts.append(self.prev_action_embedding(prev_primitives.long().clamp(0, self.action_dim)))
        return self.fusion(torch.cat(parts, dim=-1))

    def _denoise(self, z: torch.Tensor, noisy_action: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embedding(timestep.long())
        return self.denoiser(torch.cat([z, noisy_action, t_emb], dim=-1))

    def _condition_latent(
        self,
        z: torch.Tensor,
        task_ids: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
        stage_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z.ndim == 1:
            z = z.unsqueeze(0)
        resolved_task_ids = self._resolve_adapter_task_ids(z, task_ids=task_ids, state=state)
        resolved_stage_ids = self._resolve_adapter_stage_ids(z, stage_ids=stage_ids)
        task_emb = self.condition_adapter_task_embedding(resolved_task_ids)
        stage_emb = self.condition_adapter_stage_embedding(resolved_stage_ids)
        adapter_input = torch.cat([z, task_emb, stage_emb], dim=-1)
        base_delta = self.condition_adapter(z)
        cond_delta = self.condition_adapter_condition(adapter_input)
        if self.adapter_mode == "legacy_prevprim":
            gate = torch.ones_like(cond_delta)
        else:
            gate = 2.0 * torch.sigmoid(self.condition_adapter_gate(adapter_input))
        progress_scale = _progressive_scale_from_stage_ids(
            resolved_stage_ids,
            self.transition_stage_bins,
            self.adapter_progressive_min_scale,
            self.adapter_progressive_max_scale,
            start_stage=self.adapter_condition_start_stage,
            end_stage=self.adapter_condition_end_stage,
            dtype=z.dtype,
        )
        residual = (base_delta + gate * cond_delta) * self.adapter_scale
        return z + progress_scale * residual

    def _action_logits_from_embed(self, action_embed: torch.Tensor) -> torch.Tensor:
        return action_embed @ self.action_codebook.weight.t()

    def policy_logits(self, z: torch.Tensor) -> torch.Tensor:
        action_embed = torch.zeros(z.shape[0], self.diffusion_dim, device=z.device, dtype=z.dtype)
        for step in reversed(range(self.diffusion_steps)):
            timestep = torch.full((z.shape[0],), step, device=z.device, dtype=torch.long)
            action_embed = self._denoise(z, action_embed, timestep)
        diffusion_logits = self._action_logits_from_embed(action_embed)
        direct_logits = self.direct_head(z)
        return direct_logits + self.diffusion_logit_blend * diffusion_logits

    def compute_policy_loss(self, batch: dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        primitive_id = batch["primitive_id"]
        target_embed = self.action_codebook(primitive_id)
        timestep = torch.randint(0, self.diffusion_steps, (z.shape[0],), device=z.device)
        alpha = 1.0 - timestep.float().unsqueeze(-1) / max(self.diffusion_steps, 1)
        noise = torch.randn_like(target_embed)
        noisy = alpha * target_embed + (1.0 - alpha) * noise
        pred_embed = self._denoise(z, noisy, timestep)
        loss_diffusion = F.mse_loss(pred_embed, target_embed)
        diffusion_logits = self._action_logits_from_embed(pred_embed)
        direct_logits = self.direct_head(z)
        logits = self.condition_policy_logits(
            direct_logits + self.diffusion_logit_blend * diffusion_logits,
            z=z,
            task_ids=batch.get("task"),
            state=batch.get("state"),
        )
        loss_ce = F.cross_entropy(logits, primitive_id)
        return loss_ce + self.diffusion_loss_weight * loss_diffusion

    def encode_adapted(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        task_ids: torch.Tensor | None = None,
        prev_primitives: torch.Tensor | None = None,
        stage_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z = super().encode_adapted(
            image,
            state,
            task_ids=task_ids,
            prev_primitives=prev_primitives,
            stage_ids=stage_ids,
        )
        return self._condition_latent(z, task_ids=task_ids, state=state, stage_ids=stage_ids)

    def encode_step_adapted(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        runtime_state: Any = None,
        task_ids: torch.Tensor | None = None,
        stage_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Any]:
        z, next_runtime_state = super().encode_step_adapted(image, state, runtime_state, task_ids=task_ids)
        return self._condition_latent(z, task_ids=task_ids, state=state, stage_ids=stage_ids), next_runtime_state

    def compute_adapted_latents(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        task_ids = batch.get("task")
        z = self.encode_adapted(
            batch["image"],
            batch["state"],
            task_ids=task_ids,
            prev_primitives=batch.get("prev_primitive_id"),
            stage_ids=batch.get("stage_id"),
        )
        next_z = self.encode_adapted(
            batch["next_image"],
            batch["next_state"],
            task_ids=task_ids,
            prev_primitives=batch.get("next_prev_primitive_id"),
            stage_ids=batch.get("next_stage_id"),
        )
        return z, next_z

    def freeze_backbone(self) -> None:
        for name, param in self.named_parameters():
            if not (
                name.startswith("adapter.")
                or name.startswith("adapter_task_embedding.")
                or name.startswith("adapter_stage_embedding.")
                or name.startswith("adapter_condition.")
                or name.startswith("adapter_gate.")
                or name.startswith("adapter_non_observation.")
                or name.startswith("adapter_non_observation_condition.")
                or name.startswith("adapter_non_observation_gate.")
                or name.startswith("condition_adapter.")
                or name.startswith("condition_adapter_task_embedding.")
                or name.startswith("condition_adapter_stage_embedding.")
                or name.startswith("condition_adapter_condition.")
                or name.startswith("condition_adapter_gate.")
                or (self.use_transition_action_adapter and name.startswith("transition_action_adapter_embedding."))
                or (self.use_transition_residual_adapter and name.startswith("transition_residual_adapter."))
                or (
                    self.use_transition_residual_adapter
                    and self.transition_residual_phase_split
                    and name.startswith("transition_non_observation_residual_adapter.")
                )
                or (self.use_policy_residual_adapter and name.startswith("policy_residual_adapter."))
            ):
                param.requires_grad_(False)

    def adapter_parameters(self):
        modules = [
            self.adapter.parameters(),
            self.adapter_task_embedding.parameters(),
            self.adapter_stage_embedding.parameters(),
            self.adapter_condition.parameters(),
            self.condition_adapter.parameters(),
            self.condition_adapter_task_embedding.parameters(),
            self.condition_adapter_stage_embedding.parameters(),
            self.condition_adapter_condition.parameters(),
        ]
        if self.adapter_phase_split:
            modules.extend(
                [
                    self.adapter_non_observation.parameters(),
                    self.adapter_non_observation_condition.parameters(),
                ]
            )
        if self.adapter_mode != "legacy_prevprim":
            modules.extend(
                [
                    self.adapter_gate.parameters(),
                    self.adapter_non_observation_gate.parameters(),
                    self.condition_adapter_gate.parameters(),
                ]
            )
        if self.use_transition_action_adapter:
            modules.append(self.transition_action_adapter_embedding.parameters())
        if self.use_transition_residual_adapter:
            modules.append(self.transition_residual_adapter.parameters())
            if self.transition_residual_phase_split:
                modules.append(self.transition_non_observation_residual_adapter.parameters())
        if self.use_policy_residual_adapter:
            modules.append(self.policy_residual_adapter.parameters())
        for params in modules:
            yield from params


TTLAModel = FeedForwardTTLAModel


def build_backbone_model(backbone_type: str, **model_kwargs: Any) -> BaseTTLAModel:
    normalized = backbone_type.lower()
    if normalized in {"feedforward", "ff", "mlp"}:
        return FeedForwardTTLAModel(**model_kwargs)
    if normalized in {"recurrent", "gru", "rnn"}:
        return RecurrentTTLAModel(**model_kwargs)
    if normalized in {"chunking", "chunk", "act"}:
        return ChunkingTTLAModel(**model_kwargs)
    if normalized in {"language", "language_conditioned", "tiny_vla"}:
        return LanguageConditionedTTLAModel(**model_kwargs)
    if normalized in {"diffusion", "primitive_diffusion"}:
        return DiffusionPrimitiveTTLAModel(**model_kwargs)
    raise KeyError(f"Unsupported backbone_type: {backbone_type}")

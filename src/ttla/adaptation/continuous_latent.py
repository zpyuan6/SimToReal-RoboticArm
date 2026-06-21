from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ttla.control import ControlObservationBatch
from ttla.sim.task_defs import ID_TO_TASK


@dataclass(frozen=True)
class ContinuousLatentPairDataset:
    z: torch.Tensor
    next_z: torch.Tensor
    action: torch.Tensor
    action_chunk: torch.Tensor
    task: torch.Tensor
    condition: torch.Tensor
    next_condition: torch.Tensor
    latent_shape: tuple[int, ...] | None = None

    @property
    def latent_dim(self) -> int:
        return int(self.z.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.action.shape[1])

    @property
    def condition_dim(self) -> int:
        return int(self.condition.shape[1])


class ContinuousLatentTransition(nn.Module):
    """Frozen source latent dynamics used to calibrate target-domain latents."""

    def __init__(self, latent_dim: int, action_dim: int, condition_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(latent_dim + action_dim + condition_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(latent_dim)),
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        delta = self.net(torch.cat([z, action, condition], dim=-1))
        return z + delta


class ContinuousLatentActionDecoder(nn.Module):
    """Frozen source latent-action decoder used as an action-consistency critic."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        condition_dim: int,
        *,
        task_count: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.task_count = int(task_count)
        self.net = nn.Sequential(
            nn.Linear(int(latent_dim + condition_dim + task_count), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(action_dim)),
        )

    def forward(self, z: torch.Tensor, condition: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        task = task.long().reshape(-1).clamp(0, self.task_count - 1)
        one_hot = F.one_hot(task, num_classes=self.task_count).to(dtype=z.dtype, device=z.device)
        return self.net(torch.cat([z, condition, one_hot], dim=-1))


class ContinuousLatentAdapter(nn.Module):
    """Task-conditioned latent residual adapter for continuous-control policies.

    The adapter keeps the early TTLA/PLICA idea intact: it changes the policy
    latent, not the final action.  Conditions are continuous-control state
    features instead of primitive stage labels.
    """

    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        *,
        task_count: int = 3,
        hidden_dim: int = 64,
        scale: float = 0.1,
        alignment_blend: float = 0.25,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.condition_dim = int(condition_dim)
        self.task_count = int(task_count)
        self.scale = float(scale)
        self.alignment_blend = float(max(0.0, min(float(alignment_blend), 1.0)))
        self.adapter = nn.Sequential(
            nn.Linear(self.latent_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), self.latent_dim),
            nn.Tanh(),
        )
        cond_input_dim = self.latent_dim + self.condition_dim + self.task_count
        self.adapter_condition = nn.Sequential(
            nn.Linear(cond_input_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), self.latent_dim),
            nn.Tanh(),
        )
        self.adapter_gate = nn.Sequential(
            nn.Linear(cond_input_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), self.latent_dim),
        )
        nn.init.zeros_(self.adapter[2].weight)
        nn.init.zeros_(self.adapter[2].bias)
        nn.init.zeros_(self.adapter_condition[2].weight)
        nn.init.zeros_(self.adapter_condition[2].bias)
        nn.init.zeros_(self.adapter_gate[2].weight)
        nn.init.zeros_(self.adapter_gate[2].bias)
        self.register_buffer("source_mean", torch.zeros(self.task_count, self.latent_dim))
        self.register_buffer("source_std", torch.ones(self.task_count, self.latent_dim))
        self.register_buffer("target_mean", torch.zeros(self.task_count, self.latent_dim))
        self.register_buffer("target_std", torch.ones(self.task_count, self.latent_dim))
        self.register_buffer("condition_mean", torch.zeros(self.condition_dim))
        self.register_buffer("condition_std", torch.ones(self.condition_dim))

    def set_alignment_stats(
        self,
        *,
        source_mean: torch.Tensor,
        source_std: torch.Tensor,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
        condition_mean: torch.Tensor,
        condition_std: torch.Tensor,
    ) -> None:
        self.source_mean.copy_(source_mean.to(device=self.source_mean.device, dtype=self.source_mean.dtype))
        self.source_std.copy_(source_std.to(device=self.source_std.device, dtype=self.source_std.dtype).clamp_min(1.0e-6))
        self.target_mean.copy_(target_mean.to(device=self.target_mean.device, dtype=self.target_mean.dtype))
        self.target_std.copy_(target_std.to(device=self.target_std.device, dtype=self.target_std.dtype).clamp_min(1.0e-6))
        self.condition_mean.copy_(condition_mean.to(device=self.condition_mean.device, dtype=self.condition_mean.dtype))
        self.condition_std.copy_(
            condition_std.to(device=self.condition_std.device, dtype=self.condition_std.dtype).clamp_min(1.0e-6)
        )

    def _task_one_hot(self, task: torch.Tensor) -> torch.Tensor:
        task = task.long().reshape(-1).clamp(0, self.task_count - 1)
        return F.one_hot(task, num_classes=self.task_count).to(dtype=self.source_mean.dtype, device=task.device)

    def align_latent(self, z: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        task = task.long().reshape(-1).clamp(0, self.task_count - 1)
        source_mean = self.source_mean.index_select(0, task).to(dtype=z.dtype, device=z.device)
        source_std = self.source_std.index_select(0, task).to(dtype=z.dtype, device=z.device)
        target_mean = self.target_mean.index_select(0, task).to(dtype=z.dtype, device=z.device)
        target_std = self.target_std.index_select(0, task).to(dtype=z.dtype, device=z.device).clamp_min(1.0e-6)
        stat_aligned = ((z - target_mean) / target_std) * source_std + source_mean
        if self.alignment_blend <= 0.0:
            return z
        if self.alignment_blend >= 1.0:
            return stat_aligned
        return z + float(self.alignment_blend) * (stat_aligned - z)

    def forward(self, z: torch.Tensor, task: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        z_shape = z.shape
        z_flat = z.reshape(z.shape[0], -1)
        task = task.long().reshape(-1).to(device=z_flat.device)
        if task.numel() == 1 and z_flat.shape[0] > 1:
            task = task.expand(z_flat.shape[0])
        condition = condition.reshape(condition.shape[0], -1).to(device=z_flat.device, dtype=z_flat.dtype)
        if condition.shape[0] == 1 and z_flat.shape[0] > 1:
            condition = condition.expand(z_flat.shape[0], -1)
        condition = (condition - self.condition_mean.to(device=z_flat.device, dtype=z_flat.dtype)) / self.condition_std.to(
            device=z_flat.device,
            dtype=z_flat.dtype,
        ).clamp_min(1.0e-6)
        aligned = self.align_latent(z_flat, task)
        one_hot = self._task_one_hot(task).to(dtype=z_flat.dtype, device=z_flat.device)
        adapter_input = torch.cat([aligned, condition, one_hot], dim=-1)
        base_delta = self.adapter(aligned)
        cond_delta = self.adapter_condition(adapter_input)
        gate = 2.0 * torch.sigmoid(self.adapter_gate(adapter_input))
        adapted = aligned + float(self.scale) * (base_delta + gate * cond_delta)
        return adapted.view(*z_shape)

    def adapt_tensor(
        self,
        latent: torch.Tensor,
        *,
        task_id: torch.Tensor | int | None = None,
        proprio: torch.Tensor | None = None,
    ) -> torch.Tensor:
        original_shape = latent.shape
        flat = latent.reshape(latent.shape[0], -1)
        if task_id is None:
            task = torch.zeros((flat.shape[0],), dtype=torch.long, device=latent.device)
        else:
            task = torch.as_tensor(task_id, dtype=torch.long, device=latent.device).reshape(-1)
            if task.numel() == 1 and flat.shape[0] > 1:
                task = task.expand(flat.shape[0])
        if proprio is None:
            condition = torch.zeros((flat.shape[0], self.condition_dim), dtype=flat.dtype, device=flat.device)
        else:
            condition = proprio.to(device=flat.device, dtype=flat.dtype)
            if condition.ndim == 3:
                condition = condition[:, -1]
            condition = condition.reshape(condition.shape[0], -1)
            if condition.shape[-1] < self.condition_dim:
                pad = torch.zeros(
                    (condition.shape[0], self.condition_dim - condition.shape[-1]),
                    dtype=condition.dtype,
                    device=condition.device,
                )
                condition = torch.cat([condition, pad], dim=-1)
            elif condition.shape[-1] > self.condition_dim:
                condition = condition[:, : self.condition_dim]
        adapted = self.forward(flat, task, condition)
        return adapted.view(*original_shape)


def _task_text(payload: dict[str, np.ndarray], index: int, task_id: int) -> str:
    task_text = payload.get("task_text")
    if task_text is None:
        return ID_TO_TASK[int(task_id)].name
    value = task_text[index]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def _history_indices(indices: np.ndarray, end_pos: int, history_len: int) -> np.ndarray:
    return np.asarray([indices[max(0, end_pos - offset)] for offset in range(history_len - 1, -1, -1)], dtype=np.int64)


def _build_batch(
    payload: dict[str, np.ndarray],
    episode_indices: np.ndarray,
    pos: int,
    *,
    history_len: int,
    uses_language: bool,
) -> ControlObservationBatch:
    index = int(episode_indices[pos])
    hist = _history_indices(episode_indices, pos, history_len)
    images = torch.from_numpy(payload["images"][hist]).permute(0, 3, 1, 2).unsqueeze(0).float() / 255.0
    proprio = torch.from_numpy(payload["proprio"][hist]).unsqueeze(0).float()
    task_id = int(payload["tasks"][index])
    return ControlObservationBatch(
        images=images,
        proprio=proprio,
        task_text=[_task_text(payload, index, task_id)] if uses_language else None,
        task_id=task_id,
    )


def _load_payload(path: str | Path) -> dict[str, np.ndarray]:
    payload = np.load(Path(path), allow_pickle=True)
    return {key: payload[key] for key in payload.files}


def collect_continuous_latent_pairs(
    backbone: Any,
    cfg: dict,
    dataset_path: str | Path,
    *,
    max_pairs: int = 0,
    seed: int = 0,
    action_source: str = "teacher",
) -> ContinuousLatentPairDataset:
    payload = _load_payload(dataset_path)
    history_len = int(cfg.get("data_continuous", {}).get("history_len", 1))
    episode_ids = payload["episode_ids"].astype(np.int64)
    step_ids = payload["step_ids"].astype(np.int64)
    unique_episodes = sorted(int(v) for v in np.unique(episode_ids))
    rng = np.random.default_rng(int(seed))
    unique_episodes = [unique_episodes[int(i)] for i in rng.permutation(len(unique_episodes))]
    z_rows: list[np.ndarray] = []
    next_z_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    action_chunk_rows: list[np.ndarray] = []
    task_rows: list[int] = []
    condition_rows: list[np.ndarray] = []
    next_condition_rows: list[np.ndarray] = []
    max_pairs = int(max_pairs)
    action_source_key = str(action_source).strip().lower()
    if action_source_key not in {"policy", "teacher"}:
        raise ValueError("action_source must be either 'policy' or 'teacher'.")

    set_adapter = getattr(backbone, "set_latent_adapter", None)
    if set_adapter is not None:
        set_adapter(None)

    latent_shape: tuple[int, ...] | None = None
    for episode_id in unique_episodes:
        indices = np.flatnonzero(episode_ids == episode_id)
        indices = indices[np.argsort(step_ids[indices])]
        if len(indices) < 2:
            continue
        latents: list[np.ndarray] = []
        policy_actions: list[np.ndarray] = []
        for pos, index in enumerate(indices):
            # Calibration needs a stable observation latent for every sample.
            # Chunk policies may otherwise serve queued actions without running
            # the model, leaving no fresh latent for that observation.
            backbone.reset_policy_state()
            batch = _build_batch(
                payload,
                indices,
                pos,
                history_len=history_len,
                uses_language=bool(backbone.uses_language),
            )
            with torch.no_grad():
                output = backbone.forward_policy(batch)
            raw_latent = getattr(backbone, "_last_policy_latent", None)
            if torch.is_tensor(raw_latent) and raw_latent.ndim >= 2:
                latent_shape = tuple(int(v) for v in raw_latent.shape[1:])
                latent_tensor = raw_latent[0]
            else:
                latent_tensor = output.latent[0]
            latents.append(latent_tensor.detach().cpu().numpy().astype(np.float32).reshape(-1))
            policy_actions.append(output.actions[0, 0].detach().cpu().numpy().astype(np.float32).reshape(-1))
        for pos in range(len(indices) - 1):
            index = int(indices[pos])
            next_index = int(indices[pos + 1])
            z_rows.append(latents[pos])
            next_z_rows.append(latents[pos + 1])
            chunk_horizon = int(latent_shape[0]) if latent_shape and len(latent_shape) >= 2 else 1
            chunk_actions = []
            for offset in range(chunk_horizon):
                future_pos = min(pos + offset, len(indices) - 1)
                future_index = int(indices[future_pos])
                if action_source_key == "policy":
                    chunk_actions.append(policy_actions[future_pos])
                else:
                    chunk_actions.append(np.asarray(payload["actions"][future_index], dtype=np.float32).reshape(-1))
            action_chunk_rows.append(np.asarray(chunk_actions, dtype=np.float32))
            if action_source_key == "policy":
                action_rows.append(policy_actions[pos])
            else:
                action_rows.append(np.asarray(payload["actions"][index], dtype=np.float32).reshape(-1))
            task_rows.append(int(payload["tasks"][index]))
            condition_rows.append(np.asarray(payload["proprio"][index], dtype=np.float32).reshape(-1))
            next_condition_rows.append(np.asarray(payload["proprio"][next_index], dtype=np.float32).reshape(-1))
            if max_pairs > 0 and len(z_rows) >= max_pairs:
                break
        if max_pairs > 0 and len(z_rows) >= max_pairs:
            break
    if not z_rows:
        raise RuntimeError(f"No latent transition pairs collected from {dataset_path}.")
    return ContinuousLatentPairDataset(
        z=torch.from_numpy(np.asarray(z_rows, dtype=np.float32)),
        next_z=torch.from_numpy(np.asarray(next_z_rows, dtype=np.float32)),
        action=torch.from_numpy(np.asarray(action_rows, dtype=np.float32)),
        action_chunk=torch.from_numpy(np.asarray(action_chunk_rows, dtype=np.float32)),
        task=torch.from_numpy(np.asarray(task_rows, dtype=np.int64)),
        condition=torch.from_numpy(np.asarray(condition_rows, dtype=np.float32)),
        next_condition=torch.from_numpy(np.asarray(next_condition_rows, dtype=np.float32)),
        latent_shape=latent_shape,
    )


def _task_stats(values: torch.Tensor, tasks: torch.Tensor, *, task_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    global_mean = values.mean(dim=0)
    global_std = values.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    means = []
    stds = []
    for task_id in range(int(task_count)):
        mask = tasks.long() == task_id
        if bool(mask.any()):
            selected = values[mask]
            means.append(selected.mean(dim=0))
            stds.append(selected.std(dim=0, unbiased=False).clamp_min(1.0e-6))
        else:
            means.append(global_mean)
            stds.append(global_std)
    return torch.stack(means, dim=0), torch.stack(stds, dim=0)


def _loader(*tensors: torch.Tensor, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator if shuffle else None,
    )


def train_source_transition(
    source: ContinuousLatentPairDataset,
    *,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str | torch.device,
    seed: int,
) -> ContinuousLatentTransition:
    torch.manual_seed(int(seed))
    device = torch.device(device)
    transition = ContinuousLatentTransition(
        source.latent_dim,
        source.action_dim,
        source.condition_dim,
        hidden_dim=int(hidden_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(transition.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    data = _loader(
        source.z,
        source.next_z,
        source.action,
        source.condition,
        batch_size=int(batch_size),
        shuffle=True,
        seed=int(seed) + 17,
    )
    transition.train()
    for _ in range(int(epochs)):
        for z, next_z, action, condition in data:
            z = z.to(device)
            next_z = next_z.to(device)
            action = action.to(device)
            condition = condition.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(transition(z, action, condition), next_z)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transition.parameters(), 5.0)
            optimizer.step()
    transition.eval()
    return transition


def train_source_action_decoder(
    source: ContinuousLatentPairDataset,
    *,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str | torch.device,
    seed: int,
    task_count: int,
) -> ContinuousLatentActionDecoder:
    torch.manual_seed(int(seed))
    device = torch.device(device)
    decoder = ContinuousLatentActionDecoder(
        source.latent_dim,
        source.action_dim,
        source.condition_dim,
        task_count=int(task_count),
        hidden_dim=int(hidden_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    data = _loader(
        source.z,
        source.action,
        source.condition,
        source.task,
        batch_size=int(batch_size),
        shuffle=True,
        seed=int(seed) + 19,
    )
    decoder.train()
    for _ in range(int(epochs)):
        for z, action, condition, task in data:
            z = z.to(device)
            action = action.to(device)
            condition = condition.to(device)
            task = task.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(decoder(z, condition, task), action)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
            optimizer.step()
    decoder.eval()
    return decoder


def _policy_action_normalization_stats(
    backbone: Any,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    postprocessor = getattr(backbone, "_policy_postprocessor", None)
    steps = getattr(postprocessor, "steps", None) or getattr(postprocessor, "_steps", None) or []
    for step in steps:
        stats = getattr(step, "stats", None) or getattr(step, "_tensor_stats", None)
        if not isinstance(stats, dict) or "action" not in stats:
            continue
        action_stats = stats["action"]
        if not isinstance(action_stats, dict) or "mean" not in action_stats or "std" not in action_stats:
            continue
        mean = torch.as_tensor(action_stats["mean"], dtype=torch.float32, device=device).reshape(1, -1)
        std = torch.as_tensor(action_stats["std"], dtype=torch.float32, device=device).reshape(1, -1).clamp_min(1.0e-6)
        return mean, std
    return None


def fit_continuous_latent_adapter(
    backbone: Any,
    cfg: dict,
    calibration_path: str | Path,
    output_dir: str | Path,
    *,
    source_path: str | Path | None = None,
    device: str | torch.device | None = None,
    seed: int = 0,
    source_max_pairs: int = 4096,
    calibration_max_pairs: int = 2048,
    hidden_dim: int = 64,
    transition_hidden_dim: int = 128,
    action_decoder_hidden_dim: int = 128,
    epochs: int = 60,
    transition_epochs: int = 40,
    action_decoder_epochs: int = 40,
    batch_size: int = 128,
    lr: float = 1.0e-3,
    transition_lr: float = 1.0e-3,
    action_decoder_lr: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
    reg_weight: float = 0.1,
    action_loss_weight: float = 1.0,
    chunk_action_loss_weight: float = 0.0,
    action_loss_backend: str = "source_decoder",
    source_stat_weight: float = 0.02,
    scale: float = 0.1,
    alignment_blend: float = 0.25,
    hyperparam_gate: str = "none",
    action_source: str = "teacher",
    task_count: int = 3,
) -> tuple[ContinuousLatentAdapter, Path]:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    action_loss_backend = str(action_loss_backend).strip().lower()
    if action_loss_backend not in {"source_decoder", "policy_head"}:
        raise ValueError("action_loss_backend must be either 'source_decoder' or 'policy_head'.")
    hyperparam_gate = str(hyperparam_gate).strip().lower()
    if hyperparam_gate not in {"none", "latent_shift"}:
        raise ValueError("hyperparam_gate must be either 'none' or 'latent_shift'.")
    calibration_pairs = collect_continuous_latent_pairs(
        backbone,
        cfg,
        calibration_path,
        max_pairs=int(calibration_max_pairs),
        seed=int(seed) + 101,
        action_source=action_source,
    )
    source_pairs = collect_continuous_latent_pairs(
        backbone,
        cfg,
        source_path or calibration_path,
        max_pairs=int(source_max_pairs),
        seed=int(seed) + 151,
        action_source=action_source,
    )
    transition = train_source_transition(
        source_pairs,
        hidden_dim=int(transition_hidden_dim),
        epochs=int(transition_epochs),
        batch_size=int(batch_size),
        lr=float(transition_lr),
        weight_decay=float(weight_decay),
        device=device,
        seed=int(seed) + 211,
    )
    for param in transition.parameters():
        param.requires_grad_(False)
    action_decoder: ContinuousLatentActionDecoder | None = None
    action_head: nn.Module | None = None
    action_head_latent_shape: tuple[int, ...] | None = None
    action_raw_mean: torch.Tensor | None = None
    action_raw_std: torch.Tensor | None = None
    if action_loss_backend == "source_decoder":
        action_decoder = train_source_action_decoder(
            source_pairs,
            hidden_dim=int(action_decoder_hidden_dim),
            epochs=int(action_decoder_epochs),
            batch_size=int(batch_size),
            lr=float(action_decoder_lr),
            weight_decay=float(weight_decay),
            device=device,
            seed=int(seed) + 223,
            task_count=int(task_count),
        )
        for param in action_decoder.parameters():
            param.requires_grad_(False)
    else:
        find_module = getattr(backbone, "_find_policy_module", None)
        action_head = find_module("model.action_head") if callable(find_module) else None
        action_head_latent_shape = calibration_pairs.latent_shape or source_pairs.latent_shape
        if action_head is None or action_head_latent_shape is None:
            raise RuntimeError("policy_head action loss requires an ACT backbone with model.action_head capture.")
        raw_stats = _policy_action_normalization_stats(backbone, device=device)
        if raw_stats is None:
            raise RuntimeError("policy_head action loss requires LeRobot action mean/std stats.")
        action_raw_mean, action_raw_std = raw_stats
        if int(np.prod(action_head_latent_shape)) != int(calibration_pairs.latent_dim):
            raise RuntimeError(
                "Captured latent shape is incompatible with flattened latent dim: "
                f"shape={action_head_latent_shape}, latent_dim={calibration_pairs.latent_dim}."
            )
        action_head.to(device)
        action_head.eval()
        for param in action_head.parameters():
            param.requires_grad_(False)

    source_mean, source_std = _task_stats(source_pairs.z, source_pairs.task, task_count=int(task_count))
    target_mean, target_std = _task_stats(calibration_pairs.z, calibration_pairs.task, task_count=int(task_count))
    latent_mean_shift = torch.sqrt(torch.mean(((target_mean - source_mean) / source_std.clamp_min(1.0e-6)) ** 2)).item()
    latent_std_shift = torch.mean(torch.abs((target_std - source_std) / source_std.clamp_min(1.0e-6))).item()
    gate_bucket = "fixed"
    effective_scale = float(scale)
    effective_reg_weight = float(reg_weight)
    effective_action_loss_weight = float(action_loss_weight)
    chunk_action_loss_weight = float(chunk_action_loss_weight)
    if hyperparam_gate == "latent_shift":
        # The gate is driven only by source/calibration latent statistics.  It
        # adjusts the same residual adapter's strength before training; it does
        # not evaluate or select among held-out policies.
        if latent_std_shift > 0.34:
            gate_bucket = "geometry_visual_joint"
            effective_scale = 0.035
            effective_reg_weight = 0.75
            effective_action_loss_weight = 0.15
        elif latent_mean_shift < 0.62:
            gate_bucket = "geometry_dominant"
            effective_scale = 0.05
            effective_reg_weight = 0.5
            effective_action_loss_weight = 0.25
        else:
            gate_bucket = "appearance_dominant"
            effective_scale = 0.025
            effective_reg_weight = 1.0
            effective_action_loss_weight = 0.1

    adapter = ContinuousLatentAdapter(
        calibration_pairs.latent_dim,
        calibration_pairs.condition_dim,
        task_count=int(task_count),
        hidden_dim=int(hidden_dim),
        scale=float(effective_scale),
        alignment_blend=float(alignment_blend),
    ).to(device)
    adapter.set_alignment_stats(
        source_mean=source_mean,
        source_std=source_std,
        target_mean=target_mean,
        target_std=target_std,
        condition_mean=calibration_pairs.condition.mean(dim=0),
        condition_std=calibration_pairs.condition.std(dim=0, unbiased=False).clamp_min(1.0e-6),
    )
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    data = _loader(
        calibration_pairs.z,
        calibration_pairs.next_z,
        calibration_pairs.action,
        calibration_pairs.action_chunk,
        calibration_pairs.task,
        calibration_pairs.condition,
        calibration_pairs.next_condition,
        batch_size=int(batch_size),
        shuffle=True,
        seed=int(seed) + 257,
    )
    source_mean_device = adapter.source_mean
    source_std_device = adapter.source_std
    adapter.train()
    for _ in range(int(epochs)):
        for z, next_z, action, action_chunk, task, condition, next_condition in data:
            z = z.to(device)
            next_z = next_z.to(device)
            action = action.to(device)
            action_chunk = action_chunk.to(device)
            task = task.to(device)
            condition = condition.to(device)
            next_condition = next_condition.to(device)
            optimizer.zero_grad(set_to_none=True)
            z_base = adapter.align_latent(z, task).detach()
            next_z_base = adapter.align_latent(next_z, task).detach()
            z_prime = adapter(z, task, condition)
            next_z_prime = adapter(next_z, task, next_condition)
            pred_next = transition(z_prime, action, condition)
            loss_dyn = F.mse_loss(pred_next, next_z_prime)
            if action_loss_backend == "policy_head":
                assert action_head is not None
                assert action_head_latent_shape is not None
                assert action_raw_mean is not None
                assert action_raw_std is not None
                z_for_head = z_prime.reshape(z_prime.shape[0], *action_head_latent_shape)
                pred_action_chunk = action_head(z_for_head)
                pred_action = pred_action_chunk[:, 0] if pred_action_chunk.ndim == 3 else pred_action_chunk
                mean = action_raw_mean.to(dtype=action.dtype)
                std = action_raw_std.to(dtype=action.dtype)
                action_target = (action - mean) / std
                if pred_action_chunk.ndim == 3:
                    chunk_horizon = min(int(pred_action_chunk.shape[1]), int(action_chunk.shape[1]))
                    loss_action = F.mse_loss(pred_action, action_target)
                    if chunk_action_loss_weight > 0.0 and chunk_horizon > 1:
                        chunk_target = (action_chunk[:, :chunk_horizon] - mean.unsqueeze(1)) / std.unsqueeze(1)
                        chunk_pred = pred_action_chunk[:, :chunk_horizon]
                        loss_action = loss_action + float(chunk_action_loss_weight) * F.mse_loss(
                            chunk_pred,
                            chunk_target.to(dtype=chunk_pred.dtype),
                        )
                else:
                    loss_action = F.mse_loss(pred_action, action_target)
            else:
                assert action_decoder is not None
                pred_action = action_decoder(z_prime, condition, task)
                action_target = action
                loss_action = F.mse_loss(pred_action, action_target)
            loss_reg = F.mse_loss(z_prime, z_base) + 0.5 * F.mse_loss(next_z_prime, next_z_base)
            task_clamped = task.long().clamp(0, int(task_count) - 1)
            task_source_mean = source_mean_device.index_select(0, task_clamped).to(dtype=z_prime.dtype)
            task_source_std = source_std_device.index_select(0, task_clamped).to(dtype=z_prime.dtype)
            loss_stat = F.mse_loss(z_prime, task_source_mean) + 0.25 * F.mse_loss(
                (z_prime - task_source_mean).abs(),
                task_source_std,
            )
            loss = (
                loss_dyn
                + float(effective_action_loss_weight) * loss_action
                + float(effective_reg_weight) * loss_reg
                + float(source_stat_weight) * loss_stat
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
            optimizer.step()
    adapter.eval()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "continuous_latent_adapter.pt"
    torch.save(
        {
            "adapter_state": adapter.state_dict(),
            "transition_state": transition.state_dict(),
            "action_decoder_state": action_decoder.state_dict() if action_decoder is not None else None,
            "latent_dim": int(calibration_pairs.latent_dim),
            "action_dim": int(calibration_pairs.action_dim),
            "condition_dim": int(calibration_pairs.condition_dim),
            "task_count": int(task_count),
            "hyperparameters": {
                "hidden_dim": int(hidden_dim),
                "transition_hidden_dim": int(transition_hidden_dim),
                "action_decoder_hidden_dim": int(action_decoder_hidden_dim),
                "epochs": int(epochs),
                "transition_epochs": int(transition_epochs),
                "action_decoder_epochs": int(action_decoder_epochs),
                "batch_size": int(batch_size),
                "lr": float(lr),
                "transition_lr": float(transition_lr),
                "action_decoder_lr": float(action_decoder_lr),
                "weight_decay": float(weight_decay),
                "reg_weight": float(effective_reg_weight),
                "action_loss_weight": float(effective_action_loss_weight),
                "chunk_action_loss_weight": float(chunk_action_loss_weight),
                "action_loss_backend": str(action_loss_backend),
                "source_stat_weight": float(source_stat_weight),
                "scale": float(effective_scale),
                "alignment_blend": float(alignment_blend),
                "requested_reg_weight": float(reg_weight),
                "requested_action_loss_weight": float(action_loss_weight),
                "requested_scale": float(scale),
                "hyperparam_gate": str(hyperparam_gate),
                "hyperparam_gate_bucket": str(gate_bucket),
                "latent_mean_shift": float(latent_mean_shift),
                "latent_std_shift": float(latent_std_shift),
                "action_source": str(action_source),
                "source_path": str(source_path or calibration_path),
                "calibration_path": str(calibration_path),
                "source_max_pairs": int(source_max_pairs),
                "calibration_max_pairs": int(calibration_max_pairs),
            },
        },
        output_path,
    )
    return adapter, output_path

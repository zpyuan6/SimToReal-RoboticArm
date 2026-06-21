from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .continuous_latent import _build_batch, _load_payload


@dataclass(frozen=True)
class ContinuousTrajectoryPairDataset:
    plan: torch.Tensor
    target_plan: torch.Tensor
    task: torch.Tensor
    condition: torch.Tensor

    @property
    def horizon(self) -> int:
        return int(self.plan.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.plan.shape[2])

    @property
    def plan_dim(self) -> int:
        return int(self.horizon * self.action_dim)

    @property
    def condition_dim(self) -> int:
        return int(self.condition.shape[1])


class ContinuousTrajectoryAdapter(nn.Module):
    """Task-conditioned residual adapter on an action-trajectory representation."""

    def __init__(
        self,
        *,
        horizon: int,
        action_dim: int,
        condition_dim: int,
        task_count: int = 3,
        hidden_dim: int = 64,
        scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.condition_dim = int(condition_dim)
        self.task_count = int(task_count)
        self.plan_dim = int(self.horizon * self.action_dim)
        self.scale = float(scale)
        input_dim = self.plan_dim + self.condition_dim + self.task_count
        self.net = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), self.plan_dim),
            nn.Tanh(),
        )
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)
        self.register_buffer("condition_mean", torch.zeros(self.condition_dim))
        self.register_buffer("condition_std", torch.ones(self.condition_dim))

    def set_condition_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.condition_mean.copy_(mean.to(device=self.condition_mean.device, dtype=self.condition_mean.dtype))
        self.condition_std.copy_(std.to(device=self.condition_std.device, dtype=self.condition_std.dtype).clamp_min(1.0e-6))

    def _task_one_hot(self, task: torch.Tensor) -> torch.Tensor:
        task = task.long().reshape(-1).clamp(0, self.task_count - 1)
        return F.one_hot(task, num_classes=self.task_count).to(dtype=self.condition_mean.dtype, device=task.device)

    def _normalize_condition(self, condition: torch.Tensor) -> torch.Tensor:
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
        mean = self.condition_mean.to(device=condition.device, dtype=condition.dtype)
        std = self.condition_std.to(device=condition.device, dtype=condition.dtype).clamp_min(1.0e-6)
        return (condition - mean) / std

    def _canonical_plan(self, plan: torch.Tensor) -> tuple[torch.Tensor, int]:
        if plan.ndim == 2:
            plan = plan.unsqueeze(1)
        if plan.ndim != 3:
            raise ValueError(f"Expected plan with shape [B,H,A] or [B,A], got {tuple(plan.shape)}.")
        runtime_horizon = int(plan.shape[1])
        if int(plan.shape[2]) != self.action_dim:
            raise ValueError(f"Expected action_dim={self.action_dim}, got {int(plan.shape[2])}.")
        if runtime_horizon < self.horizon:
            pad = plan[:, -1:].expand(plan.shape[0], self.horizon - runtime_horizon, self.action_dim)
            plan = torch.cat([plan, pad], dim=1)
        elif runtime_horizon > self.horizon:
            plan = plan[:, : self.horizon]
        return plan, runtime_horizon

    def forward(self, plan: torch.Tensor, task: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        plan, _runtime_horizon = self._canonical_plan(plan)
        flat = plan.reshape(plan.shape[0], -1)
        task = task.long().reshape(-1).to(device=flat.device)
        if task.numel() == 1 and flat.shape[0] > 1:
            task = task.expand(flat.shape[0])
        condition = self._normalize_condition(condition.to(device=flat.device, dtype=flat.dtype))
        if condition.shape[0] == 1 and flat.shape[0] > 1:
            condition = condition.expand(flat.shape[0], -1)
        one_hot = self._task_one_hot(task).to(dtype=flat.dtype, device=flat.device)
        residual = self.net(torch.cat([flat, condition, one_hot], dim=-1))
        adapted = flat + float(self.scale) * residual
        return adapted.reshape(plan.shape[0], self.horizon, self.action_dim)

    def adapt_tensor(
        self,
        plan: torch.Tensor,
        *,
        task_id: torch.Tensor | int | None = None,
        proprio: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if plan.ndim == 2:
            original_ndim = 2
            plan_3d = plan.unsqueeze(1)
        else:
            original_ndim = plan.ndim
            plan_3d = plan
        runtime_horizon = int(plan_3d.shape[1])
        if task_id is None:
            task = torch.zeros((plan_3d.shape[0],), dtype=torch.long, device=plan.device)
        else:
            task = torch.as_tensor(task_id, dtype=torch.long, device=plan.device).reshape(-1)
            if task.numel() == 1 and plan_3d.shape[0] > 1:
                task = task.expand(plan_3d.shape[0])
        if proprio is None:
            condition = torch.zeros((plan_3d.shape[0], self.condition_dim), dtype=plan.dtype, device=plan.device)
        else:
            condition = proprio.to(device=plan.device, dtype=plan.dtype)
            if condition.ndim == 3:
                condition = condition[:, -1]
        adapted = self.forward(plan_3d, task, condition)
        if runtime_horizon > self.horizon:
            adapted = torch.cat([adapted, plan_3d[:, self.horizon :]], dim=1)
        else:
            adapted = adapted[:, :runtime_horizon]
        if original_ndim == 2:
            return adapted[:, 0]
        return adapted


def _resize_plan(plan: np.ndarray, *, horizon: int, action_dim: int) -> np.ndarray:
    plan = np.asarray(plan, dtype=np.float32)
    if plan.ndim == 1:
        plan = plan.reshape(1, -1)
    if plan.shape[-1] != action_dim:
        plan = np.resize(plan, (plan.shape[0], action_dim)).astype(np.float32)
    if plan.shape[0] < horizon:
        pad = np.repeat(plan[-1:], horizon - plan.shape[0], axis=0)
        plan = np.concatenate([plan, pad], axis=0)
    elif plan.shape[0] > horizon:
        plan = plan[:horizon]
    return plan.astype(np.float32)


def _target_plan(payload: dict[str, np.ndarray], indices: np.ndarray, pos: int, *, horizon: int) -> np.ndarray:
    rows = []
    for offset in range(int(horizon)):
        future_pos = min(int(pos) + int(offset), len(indices) - 1)
        rows.append(np.asarray(payload["actions"][int(indices[future_pos])], dtype=np.float32).reshape(-1))
    return np.stack(rows, axis=0).astype(np.float32)


def collect_continuous_trajectory_pairs(
    backbone: Any,
    cfg: dict,
    dataset_path: str | Path,
    *,
    max_pairs: int = 0,
    seed: int = 0,
) -> ContinuousTrajectoryPairDataset:
    payload = _load_payload(dataset_path)
    history_len = int(cfg.get("data_continuous", {}).get("history_len", 1))
    episode_ids = payload["episode_ids"].astype(np.int64)
    step_ids = payload["step_ids"].astype(np.int64)
    unique_episodes = sorted(int(v) for v in np.unique(episode_ids))
    rng = np.random.default_rng(int(seed))
    unique_episodes = [unique_episodes[int(i)] for i in rng.permutation(len(unique_episodes))]
    max_pairs = int(max_pairs)

    set_latent_adapter = getattr(backbone, "set_latent_adapter", None)
    if set_latent_adapter is not None:
        set_latent_adapter(None)
    set_trajectory_adapter = getattr(backbone, "set_trajectory_adapter", None)
    if set_trajectory_adapter is not None:
        set_trajectory_adapter(None)

    plan_rows: list[np.ndarray] = []
    target_plan_rows: list[np.ndarray] = []
    task_rows: list[int] = []
    condition_rows: list[np.ndarray] = []
    horizon: int | None = None
    action_dim: int | None = None

    for episode_id in unique_episodes:
        indices = np.flatnonzero(episode_ids == episode_id)
        indices = indices[np.argsort(step_ids[indices])]
        for pos, index in enumerate(indices):
            # A fresh chunk gives the adapter a full action-trajectory representation
            # for the current observation. Runtime also handles shorter queued plans.
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
            planned = output.aux.get("planned_actions", output.actions)
            plan = planned[0].detach().cpu().numpy().astype(np.float32)
            if plan.ndim == 1:
                plan = plan.reshape(1, -1)
            if horizon is None:
                horizon = int(plan.shape[0])
                action_dim = int(plan.shape[1])
            assert horizon is not None and action_dim is not None
            plan = _resize_plan(plan, horizon=horizon, action_dim=action_dim)
            plan_rows.append(plan)
            target_plan_rows.append(_target_plan(payload, indices, pos, horizon=horizon))
            task_rows.append(int(payload["tasks"][int(index)]))
            condition_rows.append(np.asarray(payload["proprio"][int(index)], dtype=np.float32).reshape(-1))
            if max_pairs > 0 and len(plan_rows) >= max_pairs:
                break
        if max_pairs > 0 and len(plan_rows) >= max_pairs:
            break
    if not plan_rows:
        raise RuntimeError(f"No trajectory pairs collected from {dataset_path}.")
    return ContinuousTrajectoryPairDataset(
        plan=torch.from_numpy(np.asarray(plan_rows, dtype=np.float32)),
        target_plan=torch.from_numpy(np.asarray(target_plan_rows, dtype=np.float32)),
        task=torch.from_numpy(np.asarray(task_rows, dtype=np.int64)),
        condition=torch.from_numpy(np.asarray(condition_rows, dtype=np.float32)),
    )


def _loader(*tensors: torch.Tensor, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator if shuffle else None,
    )


def fit_continuous_trajectory_adapter(
    backbone: Any,
    cfg: dict,
    calibration_path: str | Path,
    output_dir: str | Path,
    *,
    device: str | torch.device | None = None,
    seed: int = 0,
    max_pairs: int = 2048,
    hidden_dim: int = 64,
    epochs: int = 80,
    batch_size: int = 128,
    lr: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
    scale: float = 0.25,
    first_action_weight: float = 2.0,
    plan_loss_weight: float = 1.0,
    smooth_loss_weight: float = 0.1,
    reg_weight: float = 0.02,
    task_count: int = 3,
) -> tuple[ContinuousTrajectoryAdapter, Path]:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    pairs = collect_continuous_trajectory_pairs(
        backbone,
        cfg,
        calibration_path,
        max_pairs=int(max_pairs),
        seed=int(seed) + 101,
    )
    torch.manual_seed(int(seed))
    adapter = ContinuousTrajectoryAdapter(
        horizon=pairs.horizon,
        action_dim=pairs.action_dim,
        condition_dim=pairs.condition_dim,
        task_count=int(task_count),
        hidden_dim=int(hidden_dim),
        scale=float(scale),
    ).to(device)
    adapter.set_condition_stats(
        pairs.condition.mean(dim=0),
        pairs.condition.std(dim=0, unbiased=False).clamp_min(1.0e-6),
    )
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    data = _loader(
        pairs.plan,
        pairs.target_plan,
        pairs.task,
        pairs.condition,
        batch_size=int(batch_size),
        shuffle=True,
        seed=int(seed) + 257,
    )
    adapter.train()
    for _ in range(int(epochs)):
        for plan, target_plan, task, condition in data:
            plan = plan.to(device)
            target_plan = target_plan.to(device)
            task = task.to(device)
            condition = condition.to(device)
            optimizer.zero_grad(set_to_none=True)
            adapted = adapter(plan, task, condition)
            loss_first = F.mse_loss(adapted[:, 0], target_plan[:, 0])
            loss_plan = F.mse_loss(adapted, target_plan)
            if adapted.shape[1] > 1:
                loss_smooth = F.mse_loss(adapted[:, 1:] - adapted[:, :-1], target_plan[:, 1:] - target_plan[:, :-1])
            else:
                loss_smooth = torch.zeros((), dtype=adapted.dtype, device=adapted.device)
            loss_reg = F.mse_loss(adapted, plan)
            loss = (
                float(first_action_weight) * loss_first
                + float(plan_loss_weight) * loss_plan
                + float(smooth_loss_weight) * loss_smooth
                + float(reg_weight) * loss_reg
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
            optimizer.step()
    adapter.eval()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "continuous_trajectory_adapter.pt"
    torch.save(
        {
            "adapter_state": adapter.state_dict(),
            "horizon": int(pairs.horizon),
            "action_dim": int(pairs.action_dim),
            "condition_dim": int(pairs.condition_dim),
            "task_count": int(task_count),
            "hyperparameters": {
                "hidden_dim": int(hidden_dim),
                "epochs": int(epochs),
                "batch_size": int(batch_size),
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "scale": float(scale),
                "first_action_weight": float(first_action_weight),
                "plan_loss_weight": float(plan_loss_weight),
                "smooth_loss_weight": float(smooth_loss_weight),
                "reg_weight": float(reg_weight),
                "calibration_path": str(calibration_path),
                "max_pairs": int(max_pairs),
            },
        },
        output_path,
    )
    return adapter, output_path

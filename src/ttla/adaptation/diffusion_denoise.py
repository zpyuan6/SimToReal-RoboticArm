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
class DiffusionDenoiseDataset:
    global_cond: torch.Tensor
    clean_trajectory: torch.Tensor
    task: torch.Tensor
    condition: torch.Tensor

    @property
    def horizon(self) -> int:
        return int(self.clean_trajectory.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.clean_trajectory.shape[2])

    @property
    def condition_dim(self) -> int:
        return int(self.condition.shape[1])


class DiffusionDenoiseAdapter(nn.Module):
    """Residual adapter for the denoiser output in a frozen diffusion policy."""

    def __init__(
        self,
        *,
        horizon: int,
        action_dim: int,
        condition_dim: int,
        task_count: int = 3,
        hidden_dim: int = 64,
        scale: float = 0.1,
        task_scales: list[float] | tuple[float, ...] | np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.condition_dim = int(condition_dim)
        self.task_count = int(task_count)
        self.plan_dim = int(self.horizon * self.action_dim)
        self.scale = float(scale)
        if task_scales is None:
            task_scale_values = torch.full((self.task_count,), float(scale), dtype=torch.float32)
        else:
            task_scale_array = np.asarray(task_scales, dtype=np.float32).reshape(-1)
            if task_scale_array.shape[0] != self.task_count:
                raise ValueError(f"task_scales expects {self.task_count} values, got {task_scale_array.shape[0]}.")
            task_scale_values = torch.from_numpy(task_scale_array)
        input_dim = self.plan_dim * 2 + self.condition_dim + self.task_count + 1
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
        self.register_buffer("timestep_scale", torch.ones(()))
        self.register_buffer("task_scales", task_scale_values)

    def set_stats(self, *, condition_mean: torch.Tensor, condition_std: torch.Tensor, timestep_scale: float) -> None:
        self.condition_mean.copy_(condition_mean.to(device=self.condition_mean.device, dtype=self.condition_mean.dtype))
        self.condition_std.copy_(
            condition_std.to(device=self.condition_std.device, dtype=self.condition_std.dtype).clamp_min(1.0e-6)
        )
        self.timestep_scale.copy_(torch.as_tensor(float(max(timestep_scale, 1.0)), dtype=self.timestep_scale.dtype))

    def _task_one_hot(self, task: torch.Tensor) -> torch.Tensor:
        task = task.long().reshape(-1).clamp(0, self.task_count - 1)
        return F.one_hot(task, num_classes=self.task_count).to(dtype=self.condition_mean.dtype, device=task.device)

    def _condition(self, condition: torch.Tensor) -> torch.Tensor:
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

    def forward(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        task: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if model_output.ndim != 3 or sample.ndim != 3:
            raise ValueError("DiffusionDenoiseAdapter expects [B,H,A] model_output and sample tensors.")
        if int(model_output.shape[1]) != self.horizon or int(model_output.shape[2]) != self.action_dim:
            raise ValueError(
                f"Expected denoise tensors with [H,A]=[{self.horizon},{self.action_dim}], "
                f"got {tuple(model_output.shape[1:])}."
            )
        flat_output = model_output.reshape(model_output.shape[0], -1)
        flat_sample = sample.reshape(sample.shape[0], -1)
        task = task.long().reshape(-1).to(device=model_output.device)
        if task.numel() == 1 and model_output.shape[0] > 1:
            task = task.expand(model_output.shape[0])
        condition = self._condition(condition.to(device=model_output.device, dtype=model_output.dtype))
        if condition.shape[0] == 1 and model_output.shape[0] > 1:
            condition = condition.expand(model_output.shape[0], -1)
        timestep = timestep.reshape(-1, 1).to(device=model_output.device, dtype=model_output.dtype)
        timestep = timestep / self.timestep_scale.to(device=model_output.device, dtype=model_output.dtype).clamp_min(1.0)
        if timestep.shape[0] == 1 and model_output.shape[0] > 1:
            timestep = timestep.expand(model_output.shape[0], -1)
        one_hot = self._task_one_hot(task).to(dtype=model_output.dtype, device=model_output.device)
        residual = self.net(torch.cat([flat_output, flat_sample, condition, one_hot, timestep], dim=-1))
        task_scale = self.task_scales.to(device=model_output.device, dtype=model_output.dtype)[task].reshape(-1, 1)
        adapted = flat_output + task_scale * residual
        return adapted.reshape_as(model_output)

    def adapt_denoise(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        *,
        task_id: torch.Tensor | int | None = None,
        proprio: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if task_id is None:
            task = torch.zeros((model_output.shape[0],), dtype=torch.long, device=model_output.device)
        else:
            task = torch.as_tensor(task_id, dtype=torch.long, device=model_output.device).reshape(-1)
            if task.numel() == 1 and model_output.shape[0] > 1:
                task = task.expand(model_output.shape[0])
        if proprio is None:
            condition = torch.zeros(
                (model_output.shape[0], self.condition_dim),
                dtype=model_output.dtype,
                device=model_output.device,
            )
        else:
            condition = proprio.to(device=model_output.device, dtype=model_output.dtype)
            if condition.ndim == 3:
                condition = condition[:, -1]
        return self.forward(model_output, sample, timestep, task, condition)


def _diffusion_target_trajectory(
    payload: dict[str, np.ndarray],
    indices: np.ndarray,
    pos: int,
    *,
    horizon: int,
    n_obs_steps: int,
) -> np.ndarray:
    start_offset = int(n_obs_steps) - 1
    rows = []
    for horizon_pos in range(int(horizon)):
        source_pos = int(pos) + int(horizon_pos) - start_offset
        source_pos = min(max(source_pos, 0), len(indices) - 1)
        rows.append(np.asarray(payload["actions"][int(indices[source_pos])], dtype=np.float32).reshape(-1))
    return np.stack(rows, axis=0).astype(np.float32)


def collect_diffusion_denoise_dataset(
    backbone: Any,
    cfg: dict,
    dataset_path: str | Path,
    *,
    max_pairs: int = 0,
    seed: int = 0,
) -> DiffusionDenoiseDataset:
    ensure_loaded = getattr(backbone, "_ensure_policy_loaded", None)
    if ensure_loaded is not None:
        ensure_loaded()
    policy = getattr(backbone, "_policy_impl", None)
    diffusion = getattr(policy, "diffusion", None)
    if diffusion is None:
        raise RuntimeError("Diffusion denoise adapter requires a loaded LeRobot diffusion policy.")
    horizon = int(diffusion.config.horizon)
    n_obs_steps = int(diffusion.config.n_obs_steps)

    payload = _load_payload(dataset_path)
    history_len = int(cfg.get("data_continuous", {}).get("history_len", 1))
    episode_ids = payload["episode_ids"].astype(np.int64)
    step_ids = payload["step_ids"].astype(np.int64)
    unique_episodes = sorted(int(v) for v in np.unique(episode_ids))
    rng = np.random.default_rng(int(seed))
    unique_episodes = [unique_episodes[int(i)] for i in rng.permutation(len(unique_episodes))]

    set_latent_adapter = getattr(backbone, "set_latent_adapter", None)
    if set_latent_adapter is not None:
        set_latent_adapter(None)
    set_trajectory_adapter = getattr(backbone, "set_trajectory_adapter", None)
    if set_trajectory_adapter is not None:
        set_trajectory_adapter(None)
    set_denoise_adapter = getattr(backbone, "set_diffusion_denoise_adapter", None)
    if set_denoise_adapter is not None:
        set_denoise_adapter(None)

    global_rows: list[np.ndarray] = []
    clean_rows: list[np.ndarray] = []
    task_rows: list[int] = []
    condition_rows: list[np.ndarray] = []
    max_pairs = int(max_pairs)

    for episode_id in unique_episodes:
        indices = np.flatnonzero(episode_ids == episode_id)
        indices = indices[np.argsort(step_ids[indices])]
        for pos, index in enumerate(indices):
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
            global_cond = output.latent[0].detach().cpu().numpy().astype(np.float32).reshape(-1)
            global_rows.append(global_cond)
            clean_rows.append(
                _diffusion_target_trajectory(
                    payload,
                    indices,
                    pos,
                    horizon=horizon,
                    n_obs_steps=n_obs_steps,
                )
            )
            task_rows.append(int(payload["tasks"][int(index)]))
            condition_rows.append(np.asarray(payload["proprio"][int(index)], dtype=np.float32).reshape(-1))
            if max_pairs > 0 and len(global_rows) >= max_pairs:
                break
        if max_pairs > 0 and len(global_rows) >= max_pairs:
            break
    if not global_rows:
        raise RuntimeError(f"No denoise calibration pairs collected from {dataset_path}.")
    return DiffusionDenoiseDataset(
        global_cond=torch.from_numpy(np.asarray(global_rows, dtype=np.float32)),
        clean_trajectory=torch.from_numpy(np.asarray(clean_rows, dtype=np.float32)),
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


def _predict_clean_trajectory(
    *,
    noisy: torch.Tensor,
    model_output: torch.Tensor,
    timesteps: torch.Tensor,
    noise_scheduler: Any,
    prediction_type: str,
) -> torch.Tensor:
    if prediction_type == "sample":
        return model_output
    if prediction_type != "epsilon":
        raise ValueError(f"Unsupported prediction_type: {prediction_type}")
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=noisy.device, dtype=noisy.dtype)
    timestep_ids = timesteps.long().reshape(-1).clamp(0, int(alphas_cumprod.shape[0]) - 1)
    alpha_prod = alphas_cumprod[timestep_ids].reshape(-1, 1, 1)
    beta_prod = (1.0 - alpha_prod).clamp_min(0.0)
    return (noisy - beta_prod.sqrt() * model_output) / alpha_prod.sqrt().clamp_min(1.0e-8)


def _clean_prediction_weight(
    *,
    reference: torch.Tensor,
    timesteps: torch.Tensor,
    noise_scheduler: Any,
    prediction_type: str,
) -> torch.Tensor:
    if prediction_type == "sample":
        return torch.ones((reference.shape[0], 1, 1), dtype=reference.dtype, device=reference.device)
    if prediction_type != "epsilon":
        raise ValueError(f"Unsupported prediction_type: {prediction_type}")
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=reference.device, dtype=reference.dtype)
    timestep_ids = timesteps.long().reshape(-1).clamp(0, int(alphas_cumprod.shape[0]) - 1)
    return alphas_cumprod[timestep_ids].reshape(-1, 1, 1).clamp(0.0, 1.0)


def _weighted_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return ((pred - target).square() * weight).mean()


def _sample_training_timesteps(
    diffusion: Any,
    *,
    batch_size: int,
    device: torch.device,
    mode: str,
) -> torch.Tensor:
    mode = str(mode)
    train_steps = int(diffusion.noise_scheduler.config.num_train_timesteps)
    if mode == "train":
        return torch.randint(low=0, high=train_steps, size=(int(batch_size),), device=device).long()
    if mode == "inference":
        diffusion.noise_scheduler.set_timesteps(int(diffusion.num_inference_steps))
        inference_steps = diffusion.noise_scheduler.timesteps.to(device=device, dtype=torch.long)
        ids = torch.randint(low=0, high=int(inference_steps.numel()), size=(int(batch_size),), device=device)
        return inference_steps[ids].long()
    raise ValueError(f"Unsupported timestep_sampling mode: {mode}")


def fit_diffusion_denoise_adapter(
    backbone: Any,
    cfg: dict,
    calibration_path: str | Path,
    output_dir: str | Path,
    *,
    device: str | torch.device | None = None,
    seed: int = 0,
    max_pairs: int = 1024,
    hidden_dim: int = 64,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
    scale: float = 0.1,
    task_scales: list[float] | tuple[float, ...] | np.ndarray | None = None,
    trajectory_loss_weight: float = 0.0,
    first_action_loss_weight: float = 0.0,
    action_window_loss_weight: float = 0.0,
    reg_weight: float = 0.02,
    timestep_sampling: str = "train",
    task_count: int = 3,
) -> tuple[DiffusionDenoiseAdapter, Path]:
    ensure_loaded = getattr(backbone, "_ensure_policy_loaded", None)
    if ensure_loaded is not None:
        ensure_loaded()
    policy = getattr(backbone, "_policy_impl", None)
    diffusion = getattr(policy, "diffusion", None)
    if diffusion is None:
        raise RuntimeError("Diffusion denoise adapter requires a loaded LeRobot diffusion policy.")
    first_action_index = max(0, int(diffusion.config.n_obs_steps) - 1)
    action_window_end = min(
        int(diffusion.config.horizon),
        first_action_index + int(diffusion.config.n_action_steps),
    )
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    diffusion.to(device)
    diffusion.eval()
    for param in diffusion.parameters():
        param.requires_grad_(False)

    pairs = collect_diffusion_denoise_dataset(
        backbone,
        cfg,
        calibration_path,
        max_pairs=int(max_pairs),
        seed=int(seed) + 101,
    )
    torch.manual_seed(int(seed))
    adapter = DiffusionDenoiseAdapter(
        horizon=pairs.horizon,
        action_dim=pairs.action_dim,
        condition_dim=pairs.condition_dim,
        task_count=int(task_count),
        hidden_dim=int(hidden_dim),
        scale=float(scale),
        task_scales=task_scales,
    ).to(device)
    adapter.set_stats(
        condition_mean=pairs.condition.mean(dim=0),
        condition_std=pairs.condition.std(dim=0, unbiased=False).clamp_min(1.0e-6),
        timestep_scale=float(diffusion.noise_scheduler.config.num_train_timesteps),
    )
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    data = _loader(
        pairs.global_cond,
        pairs.clean_trajectory,
        pairs.task,
        pairs.condition,
        batch_size=int(batch_size),
        shuffle=True,
        seed=int(seed) + 257,
    )
    adapter.train()
    for _ in range(int(epochs)):
        for global_cond, clean, task, condition in data:
            global_cond = global_cond.to(device)
            clean = clean.to(device)
            task = task.to(device)
            condition = condition.to(device)
            eps = torch.randn_like(clean)
            timesteps = _sample_training_timesteps(
                diffusion,
                batch_size=int(clean.shape[0]),
                device=device,
                mode=str(timestep_sampling),
            )
            noisy = diffusion.noise_scheduler.add_noise(clean, eps, timesteps)
            with torch.no_grad():
                pred = diffusion.unet(noisy, timesteps, global_cond=global_cond)
                if diffusion.config.prediction_type == "epsilon":
                    target = eps
                elif diffusion.config.prediction_type == "sample":
                    target = clean
                else:
                    raise ValueError(f"Unsupported prediction_type: {diffusion.config.prediction_type}")
            optimizer.zero_grad(set_to_none=True)
            adapted = adapter(pred, noisy, timesteps, task, condition)
            loss_denoise = F.mse_loss(adapted, target)
            loss_reg = F.mse_loss(adapted, pred)
            loss = loss_denoise + float(reg_weight) * loss_reg
            if (
                float(trajectory_loss_weight) > 0.0
                or float(first_action_loss_weight) > 0.0
                or float(action_window_loss_weight) > 0.0
            ):
                clean_pred = _predict_clean_trajectory(
                    noisy=noisy,
                    model_output=adapted,
                    timesteps=timesteps,
                    noise_scheduler=diffusion.noise_scheduler,
                    prediction_type=str(diffusion.config.prediction_type),
                )
                clean_weight = _clean_prediction_weight(
                    reference=clean_pred,
                    timesteps=timesteps,
                    noise_scheduler=diffusion.noise_scheduler,
                    prediction_type=str(diffusion.config.prediction_type),
                )
                if float(trajectory_loss_weight) > 0.0:
                    loss = loss + float(trajectory_loss_weight) * _weighted_mse(clean_pred, clean, clean_weight)
                if float(first_action_loss_weight) > 0.0:
                    action_index = min(first_action_index, int(clean_pred.shape[1]) - 1)
                    loss = loss + float(first_action_loss_weight) * _weighted_mse(
                        clean_pred[:, action_index],
                        clean[:, action_index],
                        clean_weight.reshape(-1, 1),
                    )
                if float(action_window_loss_weight) > 0.0:
                    start = min(first_action_index, int(clean_pred.shape[1]) - 1)
                    end = max(start + 1, min(action_window_end, int(clean_pred.shape[1])))
                    loss = loss + float(action_window_loss_weight) * _weighted_mse(
                        clean_pred[:, start:end],
                        clean[:, start:end],
                        clean_weight,
                    )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
            optimizer.step()
    adapter.eval()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "diffusion_denoise_adapter.pt"
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
                "task_scales": (
                    np.asarray(task_scales, dtype=np.float32).reshape(-1).tolist()
                    if task_scales is not None
                    else None
                ),
                "trajectory_loss_weight": float(trajectory_loss_weight),
                "first_action_loss_weight": float(first_action_loss_weight),
                "action_window_loss_weight": float(action_window_loss_weight),
                "reg_weight": float(reg_weight),
                "timestep_sampling": str(timestep_sampling),
                "calibration_path": str(calibration_path),
                "max_pairs": int(max_pairs),
            },
        },
        output_path,
    )
    return adapter, output_path

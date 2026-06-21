from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import yaml

from generate_continuous_sim_data import _expert_kwargs, _save_bundle, collect_split
from run_continuous_sim2real_stress import PROFILE_CONTEXTS
from ttla.config import load_config
from ttla.control import ControlObservationBatch, build_control_backbone
from ttla.evaluation.evaluate_continuous import (
    _action_clamps_from_cfg,
    _build_env,
    _build_interface_spec,
    _merge_official_eval_cfg,
    _summarize_records,
    resolve_official_policy_path,
)
from ttla.evaluation.evaluate_continuous_exact_replay import _load_records, _restore_episode, _sample_records_per_task
from ttla.sim import ContinuousRoArmSimEnv, ContinuousWaypointExpert
from ttla.sim.task_defs import ID_TO_TASK, TASK_TO_ID


@dataclass(frozen=True)
class PredictionBundle:
    predicted: np.ndarray
    target: np.ndarray
    tasks: np.ndarray
    latents: np.ndarray | None = None


@dataclass(frozen=True)
class ContinuousActionAdapter:
    mode: str
    blend: float
    task_bias: np.ndarray
    task_blend: np.ndarray | None = None
    task_matrix: np.ndarray | None = None
    task_offset: np.ndarray | None = None
    latent_mean: np.ndarray | None = None
    latent_components: np.ndarray | None = None
    feature_mean: np.ndarray | None = None
    feature_scale: np.ndarray | None = None
    ridge_coef: np.ndarray | None = None
    mlp_w1: np.ndarray | None = None
    mlp_b1: np.ndarray | None = None
    mlp_w2: np.ndarray | None = None
    mlp_b2: np.ndarray | None = None

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    def _blend_for_task(self, task: int) -> float:
        if self.task_blend is None:
            return float(self.blend)
        return float(self.task_blend[int(np.clip(task, 0, len(self.task_blend) - 1))])

    @staticmethod
    def identity(action_dim: int, task_count: int = 3) -> "ContinuousActionAdapter":
        return ContinuousActionAdapter(
            mode="none",
            blend=0.0,
            task_bias=np.zeros((task_count, action_dim), dtype=np.float32),
        )

    def apply(self, action: np.ndarray, task_id: int, latent: np.ndarray | None = None) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        task = int(np.clip(task_id, 0, self.task_bias.shape[0] - 1))
        blend = self._blend_for_task(task)
        if self.mode == "none":
            return action
        if self.mode == "task_bias":
            return (action + blend * self.task_bias[task]).astype(np.float32)
        if self.mode == "task_affine":
            if self.task_matrix is None or self.task_offset is None:
                raise RuntimeError("task_affine adapter is missing fitted matrix/offset.")
            affine = self.task_matrix[task] @ action + self.task_offset[task]
            return (action + blend * (affine - action)).astype(np.float32)
        if self.mode in {"task_moment", "task_regularized_affine"}:
            if self.task_matrix is None or self.task_offset is None:
                raise RuntimeError(f"{self.mode} adapter is missing fitted matrix/offset.")
            aligned = self.task_matrix[task] @ action + self.task_offset[task]
            return (action + blend * (aligned - action)).astype(np.float32)
        if self.mode == "residual_mlp":
            if self.mlp_w1 is None or self.mlp_b1 is None or self.mlp_w2 is None or self.mlp_b2 is None:
                raise RuntimeError("residual_mlp adapter is missing fitted weights.")
            one_hot = np.zeros(self.task_bias.shape[0], dtype=np.float32)
            one_hot[task] = 1.0
            features = np.concatenate([action, one_hot], axis=0).astype(np.float32)
            hidden = np.tanh(features @ self.mlp_w1 + self.mlp_b1)
            residual = hidden @ self.mlp_w2 + self.mlp_b2
            return (action + blend * residual).astype(np.float32)
        if self.mode == "latent_residual_ridge":
            if (
                self.latent_mean is None
                or self.latent_components is None
                or self.feature_mean is None
                or self.feature_scale is None
                or self.ridge_coef is None
            ):
                raise RuntimeError("latent_residual_ridge adapter is missing fitted statistics.")
            if latent is None:
                latent_vec = np.zeros_like(self.latent_mean, dtype=np.float32)
            else:
                latent_vec = np.asarray(latent, dtype=np.float32).reshape(-1)
                if latent_vec.shape[0] != self.latent_mean.shape[0]:
                    latent_vec = np.resize(latent_vec, self.latent_mean.shape[0]).astype(np.float32)
            latent_proj = (latent_vec - self.latent_mean) @ self.latent_components
            one_hot = np.zeros(self.task_bias.shape[0], dtype=np.float32)
            one_hot[task] = 1.0
            features = np.concatenate([action, one_hot, latent_proj.astype(np.float32)], axis=0)
            features = (features - self.feature_mean) / self.feature_scale
            residual = np.concatenate([features, np.ones(1, dtype=np.float32)], axis=0) @ self.ridge_coef
            return (action + blend * residual).astype(np.float32)
        raise KeyError(f"Unsupported continuous adapter mode: {self.mode}")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": np.asarray(self.mode),
            "blend": np.asarray(self.blend, dtype=np.float32),
            "task_bias": self.task_bias.astype(np.float32),
        }
        if self.task_blend is not None:
            payload["task_blend"] = self.task_blend.astype(np.float32)
        if self.task_matrix is not None:
            payload["task_matrix"] = self.task_matrix.astype(np.float32)
        if self.task_offset is not None:
            payload["task_offset"] = self.task_offset.astype(np.float32)
        if self.latent_mean is not None:
            payload["latent_mean"] = self.latent_mean.astype(np.float32)
        if self.latent_components is not None:
            payload["latent_components"] = self.latent_components.astype(np.float32)
        if self.feature_mean is not None:
            payload["feature_mean"] = self.feature_mean.astype(np.float32)
        if self.feature_scale is not None:
            payload["feature_scale"] = self.feature_scale.astype(np.float32)
        if self.ridge_coef is not None:
            payload["ridge_coef"] = self.ridge_coef.astype(np.float32)
        if self.mlp_w1 is not None:
            payload["mlp_w1"] = self.mlp_w1.astype(np.float32)
        if self.mlp_b1 is not None:
            payload["mlp_b1"] = self.mlp_b1.astype(np.float32)
        if self.mlp_w2 is not None:
            payload["mlp_w2"] = self.mlp_w2.astype(np.float32)
        if self.mlp_b2 is not None:
            payload["mlp_b2"] = self.mlp_b2.astype(np.float32)
        np.savez(path, **payload)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continuous-control sim-to-real bridge: generate explicit pseudo-real calibration/heldout "
            "splits, fit a small action/chunk adapter, and evaluate transition plus rollout metrics."
        )
    )
    parser.add_argument("--config", required=True, help="Frozen continuous policy config.")
    parser.add_argument("--policy-path", default=None, help="Policy path. Defaults to frozen_baseline.policy_path.")
    parser.add_argument("--policy-device", default=None, help="Policy device override.")
    parser.add_argument("--profile", default="combined_mild", choices=sorted(PROFILE_CONTEXTS))
    parser.add_argument("--output-dir", default="results/continuous_sim2real_bridge/act_frozen_combined_mild")
    parser.add_argument("--calibration-data", default=None, help="Reuse an existing calibration NPZ instead of generating one.")
    parser.add_argument("--heldout-data", default=None, help="Reuse an existing heldout NPZ instead of generating one.")
    parser.add_argument("--calibration-episodes", type=int, default=6)
    parser.add_argument("--heldout-episodes", type=int, default=12)
    parser.add_argument("--fresh-episodes-per-task", type=int, default=4)
    parser.add_argument("--exact-num-per-task", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument(
        "--adapter",
        choices=(
            "none",
            "task_bias",
            "task_affine",
            "task_moment",
            "task_regularized_affine",
            "latent_residual_ridge",
        ),
        default="task_bias",
    )
    parser.add_argument("--baseline-name", default=None, help="Optional label for the adapted baseline.")
    parser.add_argument("--adapter-blend", type=float, default=0.25)
    parser.add_argument(
        "--adapter-task-blends",
        default=None,
        help="Optional comma-separated per-task blend for level1,level2,level3; overrides --adapter-blend per task.",
    )
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    parser.add_argument("--latent-components", type=int, default=16)
    parser.add_argument("--input-normalization", action="store_true", help="Apply the legacy per-image input normalization baseline.")
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--max-attempts-per-episode", type=int, default=40)
    parser.add_argument("--l1-terminal-hold-steps", type=int, default=0)
    return parser.parse_args()


def _parse_task_blends(raw: str | None, task_count: int = 3) -> np.ndarray | None:
    if raw is None or not str(raw).strip():
        return None
    values = [float(part.strip()) for part in str(raw).split(",") if part.strip()]
    if len(values) != int(task_count):
        raise ValueError(f"--adapter-task-blends expects {task_count} values, got {len(values)}.")
    return np.asarray(values, dtype=np.float32)


def _fit_calibration_task_blend(
    predictions: PredictionBundle,
    adapter: ContinuousActionAdapter,
    *,
    max_blend: float,
    ridge: float,
    task_count: int = 3,
) -> np.ndarray:
    blends = np.zeros((int(task_count),), dtype=np.float32)
    for task_id in range(int(task_count)):
        mask = predictions.tasks == task_id
        if not bool(mask.any()):
            continue
        actions = predictions.predicted[mask].astype(np.float32)
        targets = predictions.target[mask].astype(np.float32)
        full = np.stack([adapter.apply(action, task_id) for action in actions], axis=0).astype(np.float32)
        delta = full - actions
        residual = targets - actions
        denom = float(np.square(delta).sum()) + float(ridge)
        if denom <= 0.0:
            continue
        alpha = float((delta * residual).sum()) / denom
        blends[task_id] = float(np.clip(alpha, 0.0, float(max_blend)))
    return blends


def _policy_path(cfg: dict, override: str | None) -> str:
    if override:
        return str(override)
    frozen = cfg.get("frozen_baseline", {})
    policy_path = frozen.get("policy_path")
    if not policy_path:
        raise ValueError("No --policy-path was provided and config has no frozen_baseline.policy_path.")
    return str(policy_path)


def _cfg_for_profile(base_cfg: dict, profile_name: str) -> dict:
    cfg = deepcopy(base_cfg)
    cfg["sim"]["context"] = dict(PROFILE_CONTEXTS[profile_name])
    cfg["sim"]["task_context_rescaling"] = False
    return cfg


def _build_generation_env(cfg: dict, seed: int) -> ContinuousRoArmSimEnv:
    action_cfg = cfg["control"]["action"]
    low, high = _action_clamps_from_cfg(action_cfg)
    return ContinuousRoArmSimEnv(
        cfg["sim"],
        seed=int(seed),
        action_low=low,
        action_high=high,
        control_mode=str(action_cfg.get("control_mode", "joint_delta")),
    )


def _generate_split(
    cfg: dict,
    output_path: Path,
    *,
    split_name: str,
    episodes: int,
    seed: int,
    force: bool,
    max_attempts_per_episode: int,
    l1_terminal_hold_steps: int,
) -> None:
    if output_path.exists() and not force:
        print(f"[{split_name}] using_existing={output_path}", flush=True)
        return
    env = _build_generation_env(cfg, seed=seed)
    expert = ContinuousWaypointExpert(**_expert_kwargs(cfg))
    try:
        payload = collect_split(
            env,
            expert,
            episodes=int(episodes),
            split_name=split_name,
            log_every=max(1, int(episodes)),
            success_only=True,
            max_attempts_per_episode=int(max_attempts_per_episode),
            context_mode="random",
            l1_terminal_hold_steps=int(l1_terminal_hold_steps),
        )
        _save_bundle(output_path, payload, compression="compressed")
    finally:
        env.close()


def _history_indices(indices: np.ndarray, end_pos: int, history_len: int) -> np.ndarray:
    out: list[int] = []
    for offset in range(history_len - 1, -1, -1):
        pos = max(0, end_pos - offset)
        out.append(int(indices[pos]))
    return np.asarray(out, dtype=np.int64)


def _task_text(payload: dict[str, np.ndarray], index: int, task_id: int) -> str:
    task_text = payload.get("task_text")
    if task_text is None:
        return ID_TO_TASK[int(task_id)].name
    value = task_text[index]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def _normalize_uint8_image(image: np.ndarray) -> np.ndarray:
    x = image.astype(np.float32)
    x = (x - x.mean()) / max(float(x.std()), 1.0)
    x = np.clip(x * 48.0 + 127.0, 0.0, 255.0)
    return x.astype(np.uint8)


def _normalize_uint8_stack(images: np.ndarray) -> np.ndarray:
    return np.stack([_normalize_uint8_image(image) for image in images], axis=0)


def _build_batch_from_payload(
    payload: dict[str, np.ndarray],
    episode_indices: np.ndarray,
    pos: int,
    *,
    history_len: int,
    uses_language: bool,
    input_normalization: bool = False,
) -> ControlObservationBatch:
    index = int(episode_indices[pos])
    hist = _history_indices(episode_indices, pos, history_len)
    image_stack = payload["images"][hist]
    if input_normalization:
        image_stack = _normalize_uint8_stack(image_stack)
    proprio_stack = payload["proprio"][hist]
    images = torch.from_numpy(image_stack).permute(0, 3, 1, 2).unsqueeze(0).float() / 255.0
    proprio = torch.from_numpy(proprio_stack).unsqueeze(0).float()
    task_id = int(payload["tasks"][index])
    text = _task_text(payload, index, task_id)
    return ControlObservationBatch(
        images=images,
        proprio=proprio,
        task_text=[text] if uses_language else None,
        task_id=task_id,
    )


def _load_payload(path: str | Path) -> dict[str, np.ndarray]:
    npz = np.load(Path(path), allow_pickle=True)
    return {key: npz[key] for key in npz.files}


def _predict_dataset(
    backbone,
    cfg: dict,
    dataset_path: str | Path,
    *,
    input_normalization: bool = False,
) -> PredictionBundle:
    payload = _load_payload(dataset_path)
    history_len = int(cfg.get("data_continuous", {}).get("history_len", 1))
    episode_ids = payload["episode_ids"].astype(np.int64)
    step_ids = payload["step_ids"].astype(np.int64)
    predicted: list[np.ndarray] = []
    target: list[np.ndarray] = []
    tasks: list[int] = []
    latents: list[np.ndarray] = []

    for episode_id in sorted(int(v) for v in np.unique(episode_ids)):
        indices = np.flatnonzero(episode_ids == episode_id)
        indices = indices[np.argsort(step_ids[indices])]
        backbone.reset_policy_state()
        for pos, index in enumerate(indices):
            task_id = int(payload["tasks"][index])
            batch = _build_batch_from_payload(
                payload,
                indices,
                pos,
                history_len=history_len,
                uses_language=bool(backbone.uses_language),
                input_normalization=bool(input_normalization),
            )
            with torch.no_grad():
                policy_output = _forward_policy_for_task(backbone, batch, task_id)
            predicted.append(policy_output.actions[0, 0].detach().cpu().numpy().astype(np.float32))
            latents.append(policy_output.latent[0].detach().cpu().numpy().astype(np.float32).reshape(-1))
            target.append(np.asarray(payload["actions"][index], dtype=np.float32))
            tasks.append(task_id)
    return PredictionBundle(
        predicted=np.asarray(predicted, dtype=np.float32),
        target=np.asarray(target, dtype=np.float32),
        tasks=np.asarray(tasks, dtype=np.int64),
        latents=np.asarray(latents, dtype=np.float32) if latents else None,
    )


def _fit_adapter(
    predictions: PredictionBundle,
    *,
    mode: str,
    blend: float,
    ridge: float,
    latent_components: int = 16,
    task_blend: np.ndarray | None = None,
    fit_task_blend: bool = False,
    task_blend_max: float = 0.75,
    task_count: int = 3,
) -> ContinuousActionAdapter:
    action_dim = int(predictions.predicted.shape[1])
    if mode == "none":
        return ContinuousActionAdapter.identity(action_dim=action_dim, task_count=task_count)
    bias = np.zeros((task_count, action_dim), dtype=np.float32)
    for task_id in range(task_count):
        mask = predictions.tasks == task_id
        if bool(mask.any()):
            bias[task_id] = (predictions.target[mask] - predictions.predicted[mask]).mean(axis=0)
    if mode == "task_bias":
        adapter = ContinuousActionAdapter(mode=mode, blend=1.0, task_bias=bias, task_blend=None)
        if bool(fit_task_blend):
            task_blend = _fit_calibration_task_blend(
                predictions,
                adapter,
                max_blend=float(task_blend_max),
                ridge=float(ridge),
                task_count=int(task_count),
            )
        return ContinuousActionAdapter(mode=mode, blend=float(blend), task_bias=bias, task_blend=task_blend)
    if mode == "task_moment":
        matrices = np.stack([np.eye(action_dim, dtype=np.float32) for _ in range(task_count)], axis=0)
        offsets = np.zeros((task_count, action_dim), dtype=np.float32)
        for task_id in range(task_count):
            mask = predictions.tasks == task_id
            if not bool(mask.any()):
                offsets[task_id] = bias[task_id]
                continue
            pred = predictions.predicted[mask]
            target = predictions.target[mask]
            pred_mean = pred.mean(axis=0)
            target_mean = target.mean(axis=0)
            pred_std = pred.std(axis=0).clip(min=1.0e-4)
            target_std = target.std(axis=0).clip(min=1.0e-4)
            scale = np.clip(target_std / pred_std, 0.25, 4.0).astype(np.float32)
            matrices[task_id] = np.diag(scale).astype(np.float32)
            offsets[task_id] = (target_mean - scale * pred_mean).astype(np.float32)
        adapter = ContinuousActionAdapter(
            mode=mode,
            blend=1.0,
            task_bias=bias,
            task_blend=None,
            task_matrix=matrices,
            task_offset=offsets,
        )
        if bool(fit_task_blend):
            task_blend = _fit_calibration_task_blend(
                predictions,
                adapter,
                max_blend=float(task_blend_max),
                ridge=float(ridge),
                task_count=int(task_count),
            )
        return ContinuousActionAdapter(
            mode=mode,
            blend=float(blend),
            task_bias=bias,
            task_blend=task_blend,
            task_matrix=matrices,
            task_offset=offsets,
        )
    if mode == "task_regularized_affine":
        matrices = np.stack([np.eye(action_dim, dtype=np.float32) for _ in range(task_count)], axis=0)
        offsets = np.zeros((task_count, action_dim), dtype=np.float32)
        for task_id in range(task_count):
            mask = predictions.tasks == task_id
            if int(mask.sum()) < 2:
                offsets[task_id] = bias[task_id]
                continue
            pred = predictions.predicted[mask]
            target = predictions.target[mask]
            pred_mean = pred.mean(axis=0).astype(np.float32)
            target_mean = target.mean(axis=0).astype(np.float32)
            pred_var = pred.var(axis=0).astype(np.float32).clip(min=1.0e-8)
            pred_std = np.sqrt(pred_var).clip(min=1.0e-4)
            target_std = target.std(axis=0).astype(np.float32).clip(min=1.0e-4)
            raw_scale = np.clip(target_std / pred_std, 0.25, 4.0).astype(np.float32)
            shrink = (pred_var / (pred_var + float(ridge))).astype(np.float32)
            scale = (1.0 + shrink * (raw_scale - 1.0)).astype(np.float32)
            matrices[task_id] = np.diag(scale).astype(np.float32)
            offsets[task_id] = (target_mean - scale * pred_mean).astype(np.float32)
        adapter = ContinuousActionAdapter(
            mode=mode,
            blend=1.0,
            task_bias=bias,
            task_blend=None,
            task_matrix=matrices,
            task_offset=offsets,
        )
        if bool(fit_task_blend):
            task_blend = _fit_calibration_task_blend(
                predictions,
                adapter,
                max_blend=float(task_blend_max),
                ridge=float(ridge),
                task_count=int(task_count),
            )
        return ContinuousActionAdapter(
            mode=mode,
            blend=float(blend),
            task_bias=bias,
            task_blend=task_blend,
            task_matrix=matrices,
            task_offset=offsets,
        )
    if mode == "latent_residual_ridge":
        if predictions.latents is None:
            raise ValueError("latent_residual_ridge requires PredictionBundle.latents.")
        latents = np.asarray(predictions.latents, dtype=np.float32)
        if latents.ndim != 2:
            raise ValueError(f"Expected 2-D latents, got shape {latents.shape}.")
        latent_mean = latents.mean(axis=0).astype(np.float32)
        centered = latents - latent_mean
        max_components = max(1, min(int(latent_components), centered.shape[0] - 1, centered.shape[1]))
        if centered.shape[0] <= 1:
            components = np.zeros((centered.shape[1], 1), dtype=np.float32)
        else:
            _, _s, vt = np.linalg.svd(centered, full_matrices=False)
            components = vt[:max_components].T.astype(np.float32)
        latent_proj = centered @ components
        one_hot = np.zeros((len(predictions.tasks), task_count), dtype=np.float32)
        one_hot[np.arange(len(predictions.tasks)), np.clip(predictions.tasks, 0, task_count - 1)] = 1.0
        features = np.concatenate([predictions.predicted, one_hot, latent_proj], axis=1).astype(np.float32)
        feature_mean = features.mean(axis=0).astype(np.float32)
        feature_scale = features.std(axis=0).clip(min=1.0e-4).astype(np.float32)
        x = (features - feature_mean) / feature_scale
        x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
        y = (predictions.target - predictions.predicted).astype(np.float32)
        reg = float(ridge) * np.eye(x_aug.shape[1], dtype=np.float32)
        reg[-1, -1] = 0.0
        coef = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y).astype(np.float32)
        return ContinuousActionAdapter(
            mode=mode,
            blend=float(blend),
            task_bias=bias,
            task_blend=task_blend,
            latent_mean=latent_mean,
            latent_components=components,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            ridge_coef=coef,
        )
    if mode != "task_affine":
        raise KeyError(f"Unsupported adapter mode: {mode}")

    matrices = np.stack([np.eye(action_dim, dtype=np.float32) for _ in range(task_count)], axis=0)
    offsets = np.zeros((task_count, action_dim), dtype=np.float32)
    for task_id in range(task_count):
        mask = predictions.tasks == task_id
        if int(mask.sum()) < action_dim + 1:
            offsets[task_id] = bias[task_id]
            continue
        x = np.concatenate(
            [predictions.predicted[mask], np.ones((int(mask.sum()), 1), dtype=np.float32)],
            axis=1,
        )
        y = predictions.target[mask]
        reg = float(ridge) * np.eye(action_dim + 1, dtype=np.float32)
        reg[-1, -1] = 0.0
        coef = np.linalg.solve(x.T @ x + reg, x.T @ y)
        matrices[task_id] = coef[:-1].T.astype(np.float32)
        offsets[task_id] = coef[-1].astype(np.float32)
    return ContinuousActionAdapter(
        mode=mode,
        blend=float(blend),
        task_bias=bias,
        task_blend=task_blend,
        task_matrix=matrices,
        task_offset=offsets,
    )


def _transition_metrics(
    predictions: PredictionBundle,
    adapter: ContinuousActionAdapter,
    *,
    baseline: str,
) -> pd.DataFrame:
    latents = predictions.latents
    if latents is None:
        latents_iter = [None] * len(predictions.predicted)
    else:
        latents_iter = list(latents)
    adapted = np.stack(
        [
            adapter.apply(action, int(task_id), latent)
            for action, task_id, latent in zip(predictions.predicted, predictions.tasks, latents_iter)
        ],
        axis=0,
    )
    sample_mse = ((adapted - predictions.target) ** 2).mean(axis=1)
    rows: list[dict[str, float | int | str]] = []
    for task_id in sorted(int(v) for v in np.unique(predictions.tasks)):
        mask = predictions.tasks == task_id
        rows.append(
            {
                "baseline": baseline,
                "split": "task",
                "task": ID_TO_TASK[task_id].name,
                "count": int(mask.sum()),
                "action_mse": float(sample_mse[mask].mean()),
                "action_mae": float(np.abs(adapted[mask] - predictions.target[mask]).mean()),
            }
        )
    rows.append(
        {
            "baseline": baseline,
            "split": "overall",
            "task": "overall",
            "count": int(len(sample_mse)),
            "action_mse": float(sample_mse.mean()),
            "action_mae": float(np.abs(adapted - predictions.target).mean()),
        }
    )
    return pd.DataFrame.from_records(rows)


def _forward_policy_for_task(backbone, batch: ControlObservationBatch, task_id: int):
    task_forward = getattr(backbone, "forward_policy_for_task", None)
    if task_forward is not None:
        return task_forward(batch, int(task_id))
    return backbone.forward_policy(batch)


def _apply_policy_action(backbone, batch: ControlObservationBatch, adapter: ContinuousActionAdapter, task_id: int) -> np.ndarray:
    with torch.no_grad():
        policy_output = _forward_policy_for_task(backbone, batch, task_id)
    action = policy_output.actions[0, 0].detach().cpu().numpy().astype(np.float32)
    latent = policy_output.latent[0].detach().cpu().numpy().astype(np.float32)
    return adapter.apply(action, task_id, latent)


def _write_rollout_outputs(rows: list[dict[str, float | int | str]], output_dir: Path, baseline: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = output_dir / "episodes.csv"
    summary_path = output_dir / "summary.csv"
    df = pd.DataFrame.from_records(rows)
    df.insert(0, "baseline", baseline)
    df.to_csv(episodes_path, index=False)
    summary = _summarize_records(df.drop(columns=["baseline"]))
    summary.insert(0, "baseline", baseline)
    summary.to_csv(summary_path, index=False)
    return episodes_path, summary_path


def _evaluate_exact_replay(
    backbone,
    cfg: dict,
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    adapter: ContinuousActionAdapter,
    baseline: str,
    num_per_task: int,
    seed: int,
    tasks: Iterable[str] | None = None,
    input_normalization: bool = False,
) -> tuple[Path, Path]:
    payload = np.load(Path(dataset_path), allow_pickle=True)
    task_filter = set(tasks or [])
    rng = np.random.default_rng(int(seed))
    records = _sample_records_per_task(
        _load_records(payload, task_filter),
        num_per_task=int(num_per_task),
        rng=rng,
    )
    env_seed = int(seed)
    history_len = int(cfg.get("data_continuous", {}).get("history_len", 1))
    sim_horizon = int(cfg["sim"]["episode_horizon"])
    rows: list[dict[str, float | int | str]] = []
    for record_idx, record in enumerate(records):
        env = _build_env(cfg, seed=env_seed + 303 + int(record_idx))
        try:
            _restore_episode(env, record)
            obs = env.observe()
            obs_history = [obs]
            backbone.reset_policy_state()
            total_reward = 0.0
            info: dict = {
                "visibility": 0.0,
                "center_error": 0.0,
                "verified": 0,
                "grasped": 0,
                "lifted": 0,
                "placed": 0,
                "ee_ear_center_distance": float("nan"),
                "ee_target_distance": float("nan"),
                "grasp_gap": float("nan"),
                "dropzone_distance": float("nan"),
            }
            success = 0
            max_steps = max(sim_horizon, len(record.indices))
            for step in range(max_steps):
                batch = _build_observation_batch_for_rollout(
                    obs_history,
                    history_len=history_len,
                    task_text=record.task_text,
                    uses_language=bool(backbone.uses_language),
                    input_normalization=bool(input_normalization),
                )
                action = _apply_policy_action(backbone, batch, adapter, int(record.task_id))
                next_obs, reward, done, info = env.step_action(action)
                total_reward += float(reward)
                obs_history.append(next_obs)
                success = int(info["success"])
                if done:
                    break
            rows.append(
                {
                    "backbone": cfg["control"]["backbone_name"],
                    "task": record.task_name,
                    "episode": record.episode_id,
                    "stored_episode_success": record.episode_success,
                    "stored_steps": len(record.indices),
                    "success": success,
                    "steps": step + 1,
                    "reward": total_reward,
                    "visibility": float(info.get("visibility", 0.0)),
                    "center_error": float(info.get("center_error", 0.0)),
                    "verified": int(info.get("verified", 0)),
                    "grasped": int(info.get("grasped", 0)),
                    "lifted": int(info.get("lifted", 0)),
                    "placed": int(info.get("placed", 0)),
                    "final_ee_ear_center_distance": float(info.get("ee_ear_center_distance", float("nan"))),
                    "final_ee_target_distance": float(info.get("ee_target_distance", float("nan"))),
                    "final_grasp_gap": float(info.get("grasp_gap", float("nan"))),
                    "final_dropzone_distance": float(info.get("dropzone_distance", float("nan"))),
                }
            )
        finally:
            env.close()
    return _write_rollout_outputs(rows, Path(output_dir), baseline)


def _build_observation_batch_for_rollout(
    obs_history: list[dict[str, np.ndarray]],
    *,
    history_len: int,
    task_text: str | None,
    uses_language: bool,
    input_normalization: bool = False,
) -> ControlObservationBatch:
    end_pos = max(0, len(obs_history) - 1)
    indices = [max(0, end_pos - offset) for offset in range(history_len - 1, -1, -1)]
    image_stack = np.stack([obs_history[idx]["image"] for idx in indices], axis=0)
    if input_normalization:
        image_stack = _normalize_uint8_stack(image_stack)
    proprio_stack = np.stack([obs_history[idx]["state"] for idx in indices], axis=0)
    images = torch.from_numpy(image_stack).permute(0, 3, 1, 2).unsqueeze(0).float() / 255.0
    proprio = torch.from_numpy(proprio_stack).unsqueeze(0).float()
    return ControlObservationBatch(
        images=images,
        proprio=proprio,
        task_text=[str(task_text or "")] if uses_language else None,
        task_id=int(np.argmax(proprio_stack[-1, 12:15])) if proprio_stack.shape[-1] >= 15 else None,
    )


def _evaluate_fresh_rollout(
    backbone,
    cfg: dict,
    output_dir: str | Path,
    *,
    adapter: ContinuousActionAdapter,
    baseline: str,
    episodes_per_task: int,
    seed: int,
    input_normalization: bool = False,
) -> tuple[Path, Path]:
    history_len = int(cfg.get("data_continuous", {}).get("history_len", 1))
    rows: list[dict[str, float | int | str]] = []
    for task_name in cfg["sim"]["tasks"]:
        task_id = int(TASK_TO_ID[str(task_name)])
        for episode in range(int(episodes_per_task)):
            episode_seed = int(seed) + 101 + task_id * 1009 + int(episode)
            env = _build_env(cfg, seed=episode_seed)
            try:
                obs = env.reset(task_name=str(task_name))
                obs_history = [obs]
                backbone.reset_policy_state()
                total_reward = 0.0
                info: dict = {
                    "visibility": 0.0,
                    "center_error": 0.0,
                    "verified": 0,
                    "grasped": 0,
                    "lifted": 0,
                    "placed": 0,
                    "ee_ear_center_distance": float("nan"),
                    "ee_target_distance": float("nan"),
                    "grasp_gap": float("nan"),
                    "dropzone_distance": float("nan"),
                }
                success = 0
                for step in range(int(cfg["sim"]["episode_horizon"])):
                    batch = _build_observation_batch_for_rollout(
                        obs_history,
                        history_len=history_len,
                        task_text=env.task_text(),
                        uses_language=bool(backbone.uses_language),
                        input_normalization=bool(input_normalization),
                    )
                    action = _apply_policy_action(backbone, batch, adapter, task_id)
                    next_obs, reward, done, info = env.step_action(action)
                    total_reward += float(reward)
                    obs_history.append(next_obs)
                    success = int(info["success"])
                    if done:
                        break
                rows.append(
                    {
                        "backbone": cfg["control"]["backbone_name"],
                        "task": str(task_name),
                        "episode": episode,
                        "success": success,
                        "steps": step + 1,
                        "reward": total_reward,
                        "visibility": float(info.get("visibility", 0.0)),
                        "center_error": float(info.get("center_error", 0.0)),
                        "verified": int(info.get("verified", 0)),
                        "grasped": int(info.get("grasped", 0)),
                        "lifted": int(info.get("lifted", 0)),
                        "placed": int(info.get("placed", 0)),
                        "final_ee_ear_center_distance": float(info.get("ee_ear_center_distance", float("nan"))),
                        "final_ee_target_distance": float(info.get("ee_target_distance", float("nan"))),
                        "final_grasp_gap": float(info.get("grasp_gap", float("nan"))),
                        "final_dropzone_distance": float(info.get("dropzone_distance", float("nan"))),
                    }
                )
            finally:
                env.close()
    return _write_rollout_outputs(rows, Path(output_dir), baseline)


def _build_backbone(cfg: dict, policy_path: str, policy_device: str | None):
    interface_spec = _build_interface_spec(cfg)
    resolved_policy_path = resolve_official_policy_path(policy_path)
    official_cfg = _merge_official_eval_cfg(cfg, resolved_policy_path, policy_device)
    backbone = build_control_backbone(cfg["control"]["backbone_name"], interface_spec, official_cfg=official_cfg)
    backbone.eval()
    return backbone


def _flatten_success_summary(eval_type: str, summary_path: Path, *, profile: str) -> list[dict[str, object]]:
    df = pd.read_csv(summary_path)
    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        rows.append(
            {
                "eval_type": eval_type,
                "profile": profile,
                "baseline": row["baseline"],
                "split": row["split"],
                "task": row["task"],
                "success": float(row["success"]),
                "steps": float(row["steps"]),
                "summary_csv": str(summary_path),
            }
        )
    return rows


def main() -> None:
    args = _parse_args()
    base_cfg = load_config(args.config)
    cfg = _cfg_for_profile(base_cfg, args.profile)
    policy_path = _policy_path(base_cfg, args.policy_path)
    output_root = Path(args.output_dir)
    split_root = output_root / "splits"
    split_root.mkdir(parents=True, exist_ok=True)
    if bool(args.calibration_data) != bool(args.heldout_data):
        raise ValueError("Provide both --calibration-data and --heldout-data, or neither.")
    if args.calibration_data and args.heldout_data:
        calibration_path = Path(args.calibration_data)
        heldout_path = Path(args.heldout_data)
        if not calibration_path.exists():
            raise FileNotFoundError(f"Missing calibration split: {calibration_path}")
        if not heldout_path.exists():
            raise FileNotFoundError(f"Missing heldout split: {heldout_path}")
    else:
        calibration_path = split_root / "calibration.npz"
        heldout_path = split_root / "heldout.npz"
        _generate_split(
            cfg,
            calibration_path,
            split_name=f"{args.profile}_calibration",
            episodes=int(args.calibration_episodes),
            seed=int(args.seed) + 11,
            force=bool(args.force_regenerate),
            max_attempts_per_episode=int(args.max_attempts_per_episode),
            l1_terminal_hold_steps=int(args.l1_terminal_hold_steps),
        )
        _generate_split(
            cfg,
            heldout_path,
            split_name=f"{args.profile}_heldout",
            episodes=int(args.heldout_episodes),
            seed=int(args.seed) + 29,
            force=bool(args.force_regenerate),
            max_attempts_per_episode=int(args.max_attempts_per_episode),
            l1_terminal_hold_steps=int(args.l1_terminal_hold_steps),
        )

    backbone = _build_backbone(cfg, policy_path=policy_path, policy_device=args.policy_device)
    calibration_predictions = _predict_dataset(
        backbone,
        cfg,
        calibration_path,
        input_normalization=bool(args.input_normalization),
    )
    task_blend = _parse_task_blends(args.adapter_task_blends)
    adapter = _fit_adapter(
        calibration_predictions,
        mode=str(args.adapter),
        blend=float(args.adapter_blend),
        ridge=float(args.ridge),
        latent_components=int(args.latent_components),
        task_blend=task_blend,
    )
    adapter_path = output_root / "adapter.npz"
    adapter.save(adapter_path)

    heldout_predictions = _predict_dataset(
        backbone,
        cfg,
        heldout_path,
        input_normalization=bool(args.input_normalization),
    )
    identity = ContinuousActionAdapter.identity(action_dim=int(heldout_predictions.predicted.shape[1]))
    identity_baseline = "input_normalization" if args.input_normalization else "no_adaptation"
    adapted_baseline = str(args.baseline_name or args.adapter)
    transition_frames = [_transition_metrics(heldout_predictions, identity, baseline=identity_baseline)]
    if str(args.adapter) != "none":
        transition_frames.append(_transition_metrics(heldout_predictions, adapter, baseline=adapted_baseline))
    transition_df = pd.concat(transition_frames, ignore_index=True)
    transition_path = output_root / "heldout_transition_metrics.csv"
    transition_df.to_csv(transition_path, index=False)

    success_rows: list[dict[str, object]] = []
    baselines: list[tuple[str, ContinuousActionAdapter]] = [(identity_baseline, identity)]
    if str(args.adapter) != "none":
        baselines.append((adapted_baseline, adapter))
    for baseline, baseline_adapter in baselines:
        exact_summary = _evaluate_exact_replay(
            backbone,
            cfg,
            heldout_path,
            output_root / "exact_replay" / baseline,
            adapter=baseline_adapter,
            baseline=baseline,
            num_per_task=int(args.exact_num_per_task),
            seed=int(args.seed) + 401,
            input_normalization=bool(args.input_normalization),
        )[1]
        success_rows.extend(_flatten_success_summary("heldout_exact_replay", exact_summary, profile=str(args.profile)))
        fresh_summary = _evaluate_fresh_rollout(
            backbone,
            cfg,
            output_root / "fresh_rollout" / baseline,
            adapter=baseline_adapter,
            baseline=baseline,
            episodes_per_task=int(args.fresh_episodes_per_task),
            seed=int(args.seed) + 503,
            input_normalization=bool(args.input_normalization),
        )[1]
        success_rows.extend(_flatten_success_summary("fresh_rollout", fresh_summary, profile=str(args.profile)))

    combined_success_path = output_root / "combined_success_summary.csv"
    pd.DataFrame.from_records(success_rows).to_csv(combined_success_path, index=False)
    manifest = {
        "config": str(args.config),
        "policy_path": str(policy_path),
        "profile": str(args.profile),
        "profile_context": PROFILE_CONTEXTS[str(args.profile)],
        "calibration_path": str(calibration_path),
        "heldout_path": str(heldout_path),
        "adapter": str(args.adapter),
        "adapter_baseline": adapted_baseline,
        "adapter_path": str(adapter_path),
        "adapter_task_blends": task_blend.tolist() if task_blend is not None else None,
        "latent_components": int(args.latent_components),
        "input_normalization": bool(args.input_normalization),
        "transition_metrics_csv": str(transition_path),
        "combined_success_summary_csv": str(combined_success_path),
        "calibration_episodes": int(args.calibration_episodes),
        "heldout_episodes": int(args.heldout_episodes),
        "fresh_episodes_per_task": int(args.fresh_episodes_per_task),
        "exact_num_per_task": int(args.exact_num_per_task),
        "seed": int(args.seed),
    }
    with (output_root / "manifest.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)

    print(f"calibration_npz={calibration_path}")
    print(f"heldout_npz={heldout_path}")
    print(f"adapter_npz={adapter_path}")
    print(f"heldout_transition_metrics_csv={transition_path}")
    print(f"combined_success_summary_csv={combined_success_path}")


if __name__ == "__main__":
    main()

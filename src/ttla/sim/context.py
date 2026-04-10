from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ContextConfig:
    camera_jitter: float
    fov_jitter: float
    light_jitter: float
    texture_jitter: float
    blur_sigma: float
    noise_std: float
    action_gain_low: float
    action_gain_high: float
    action_delay_max: int
    joint_bias: float


def sample_context(cfg: ContextConfig, rng: np.random.Generator) -> dict[str, float]:
    return {
        "cam_x": float(rng.uniform(-cfg.camera_jitter, cfg.camera_jitter)),
        "cam_y": float(rng.uniform(-cfg.camera_jitter, cfg.camera_jitter)),
        "cam_z": float(rng.uniform(-cfg.camera_jitter, cfg.camera_jitter)),
        "cam_roll": float(rng.uniform(-0.08, 0.08)),
        "cam_pitch": float(rng.uniform(-0.08, 0.08)),
        "cam_yaw": float(rng.uniform(-0.08, 0.08)),
        "fov_bias": float(rng.uniform(-cfg.fov_jitter, cfg.fov_jitter)),
        "light_gain": float(rng.uniform(1.0 - cfg.light_jitter, 1.0 + cfg.light_jitter)),
        "texture_shift": float(rng.uniform(-cfg.texture_jitter, cfg.texture_jitter)),
        "blur_sigma": float(rng.uniform(0.0, cfg.blur_sigma)),
        "noise_std": float(rng.uniform(0.0, cfg.noise_std)),
        "action_gain": float(rng.uniform(cfg.action_gain_low, cfg.action_gain_high)),
        "action_delay": int(rng.integers(0, cfg.action_delay_max + 1)),
        "joint_bias": float(rng.uniform(-cfg.joint_bias, cfg.joint_bias)),
        "link_scale": float(rng.uniform(0.97, 1.03)),
        "deadzone": float(rng.uniform(0.0, 0.03)),
    }


def context_vector(context: dict[str, float]) -> np.ndarray:
    keys = [
        "cam_x",
        "cam_y",
        "cam_z",
        "cam_pitch",
        "cam_yaw",
        "light_gain",
        "action_gain",
        "joint_bias",
    ]
    return np.asarray([context[k] for k in keys], dtype=np.float32)

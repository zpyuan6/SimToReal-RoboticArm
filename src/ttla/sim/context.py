from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ContextConfig:
    camera_translation_jitter: float = 0.0
    camera_rotation_jitter: float = 0.0
    fov_jitter: float = 0.0
    light_jitter: float = 0.0
    blur_sigma: float = 0.0
    noise_std: float = 0.0
    action_gain_low: float = 1.0
    action_gain_high: float = 1.0
    action_delay_max: int = 0
    joint_bias: float = 0.0
    # Legacy compatibility fields. Keep them optional so older experiment
    # configs still load, but do not use them as hidden extra shift channels
    # once the factorized settings are provided explicitly.
    camera_jitter: float = 0.0
    texture_jitter: float = 0.0


def sample_context(cfg: ContextConfig, rng: np.random.Generator) -> dict[str, float]:
    translation_jitter = float(cfg.camera_translation_jitter or cfg.camera_jitter)
    rotation_jitter = float(cfg.camera_rotation_jitter or cfg.camera_jitter)
    return {
        "cam_x": float(rng.uniform(-translation_jitter, translation_jitter)),
        "cam_y": float(rng.uniform(-translation_jitter, translation_jitter)),
        "cam_z": float(rng.uniform(-translation_jitter, translation_jitter)),
        "cam_roll": float(rng.uniform(-rotation_jitter, rotation_jitter)),
        "cam_pitch": float(rng.uniform(-rotation_jitter, rotation_jitter)),
        "cam_yaw": float(rng.uniform(-rotation_jitter, rotation_jitter)),
        "fov_bias": float(rng.uniform(-cfg.fov_jitter, cfg.fov_jitter)),
        "light_gain": float(rng.uniform(1.0 - cfg.light_jitter, 1.0 + cfg.light_jitter)),
        "blur_sigma": float(rng.uniform(0.0, cfg.blur_sigma)),
        "noise_std": float(rng.uniform(0.0, cfg.noise_std)),
        "action_gain": float(rng.uniform(cfg.action_gain_low, cfg.action_gain_high)),
        "action_delay": int(rng.integers(0, cfg.action_delay_max + 1)),
        "joint_bias": float(rng.uniform(-cfg.joint_bias, cfg.joint_bias)),
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

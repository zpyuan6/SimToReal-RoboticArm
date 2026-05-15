from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch


ControlMode = Literal["joint_delta", "joint_target"]


@dataclass(frozen=True)
class ContinuousActionSpec:
    action_dim: int
    horizon: int
    control_mode: ControlMode = "joint_delta"
    clamp_low: tuple[float, ...] | None = None
    clamp_high: tuple[float, ...] | None = None


@dataclass(frozen=True)
class ControlInterfaceSpec:
    image_shape: tuple[int, int, int]
    proprio_dim: int
    action_spec: ContinuousActionSpec
    uses_language: bool = False


@dataclass
class ControlObservationBatch:
    images: torch.Tensor
    proprio: torch.Tensor
    task_text: list[str] | None = None
    attention_mask: torch.Tensor | None = None


@dataclass
class ControlPolicyOutput:
    actions: torch.Tensor
    latent: torch.Tensor
    aux: dict[str, torch.Tensor] = field(default_factory=dict)


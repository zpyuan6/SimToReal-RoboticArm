from __future__ import annotations

from .base import BaseControlBackbone
from .backbones import ACTBackbone, DiffusionPolicyBackbone, LeRobotOfficialBackbone, LeRobotPolicySpec, SmolVLABackbone
from .types import (
    ContinuousActionSpec,
    ControlInterfaceSpec,
    ControlObservationBatch,
    ControlPolicyOutput,
)


def build_control_backbone(
    backbone_name: str,
    interface_spec: ControlInterfaceSpec,
    official_cfg: dict | None = None,
) -> BaseControlBackbone:
    key = str(backbone_name).strip().lower()
    if key == "act":
        return ACTBackbone(interface_spec, official_cfg=official_cfg)
    if key in {"diffusion", "diffusion_policy"}:
        return DiffusionPolicyBackbone(interface_spec, official_cfg=official_cfg)
    if key == "smolvla":
        return SmolVLABackbone(interface_spec, official_cfg=official_cfg)
    raise KeyError(f"Unsupported control backbone scaffold: {backbone_name}")


__all__ = [
    "ACTBackbone",
    "BaseControlBackbone",
    "ContinuousActionSpec",
    "ControlInterfaceSpec",
    "ControlObservationBatch",
    "ControlPolicyOutput",
    "DiffusionPolicyBackbone",
    "LeRobotOfficialBackbone",
    "LeRobotPolicySpec",
    "SmolVLABackbone",
    "build_control_backbone",
]

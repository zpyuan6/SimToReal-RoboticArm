from __future__ import annotations

from .lerobot_base import LeRobotOfficialBackbone, LeRobotPolicySpec
from ..types import ControlInterfaceSpec


class DiffusionPolicyBackbone(LeRobotOfficialBackbone):
    """Official LeRobot diffusion-style policy wrapper."""

    def __init__(self, interface_spec: ControlInterfaceSpec, official_cfg: dict | None = None) -> None:
        policy_path = None
        if official_cfg is not None:
            policy_path = official_cfg.get("policy_path") or None
        super().__init__(
            interface_spec,
            policy_spec=LeRobotPolicySpec(
                policy_type="diffusion",
                policy_path=policy_path,
                latent_name="official_diffusion_condition_latent",
                family="generative_diffusion_control",
            ),
            official_cfg=official_cfg,
        )

from __future__ import annotations

from .lerobot_base import LeRobotOfficialBackbone, LeRobotPolicySpec
from ..types import ControlInterfaceSpec


class SmolVLABackbone(LeRobotOfficialBackbone):
    """Official SmolVLA wrapper.

    SmolVLA requires the official LeRobot package plus SmolVLA extras.
    """

    def __init__(self, interface_spec: ControlInterfaceSpec, official_cfg: dict | None = None) -> None:
        policy_path = "lerobot/smolvla_base"
        if official_cfg is not None:
            policy_path = official_cfg.get("policy_path") or policy_path
        super().__init__(
            interface_spec,
            policy_spec=LeRobotPolicySpec(
                policy_type="smolvla",
                policy_path=policy_path,
                latent_name="official_smolvla_multimodal_latent",
                family="lightweight_vla",
                extra_dependencies=("smolvla",),
            ),
            official_cfg=official_cfg,
        )

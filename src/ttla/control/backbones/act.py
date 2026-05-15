from __future__ import annotations

from .lerobot_base import LeRobotOfficialBackbone, LeRobotPolicySpec
from ..types import ControlInterfaceSpec


class ACTBackbone(LeRobotOfficialBackbone):
    """Official LeRobot ACT wrapper.

    Formal baseline policy must come from the official LeRobot ACT
    implementation, not from a local reimplementation.
    """

    def __init__(self, interface_spec: ControlInterfaceSpec, official_cfg: dict | None = None) -> None:
        policy_path = None
        if official_cfg is not None:
            policy_path = official_cfg.get("policy_path") or None
        super().__init__(
            interface_spec,
            policy_spec=LeRobotPolicySpec(
                policy_type="act",
                policy_path=policy_path,
                latent_name="official_act_latent",
                family="action_chunking_transformer",
            ),
            official_cfg=official_cfg,
        )

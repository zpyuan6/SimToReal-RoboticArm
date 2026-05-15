from __future__ import annotations

from abc import ABC, abstractmethod

from torch import nn

from .types import ControlInterfaceSpec, ControlObservationBatch, ControlPolicyOutput


class BaseControlBackbone(nn.Module, ABC):
    """Unified host-controller interface for continuous/chunk control backbones.

    The refactor target is to keep PLICA independent from any one policy family.
    Each concrete backbone wrapper must expose the same control interface and a
    latent tensor suitable for adapter-based calibration.
    """

    backbone_name: str
    backbone_family: str

    def __init__(self, interface_spec: ControlInterfaceSpec) -> None:
        super().__init__()
        self.interface_spec = interface_spec

    @property
    def action_spec(self):
        return self.interface_spec.action_spec

    @property
    def proprio_dim(self) -> int:
        return int(self.interface_spec.proprio_dim)

    @property
    def uses_language(self) -> bool:
        return bool(self.interface_spec.uses_language)

    @abstractmethod
    def forward_policy(self, batch: ControlObservationBatch) -> ControlPolicyOutput:
        """Run the host controller and return actions plus adaptation latent."""

    @abstractmethod
    def latent_target_name(self) -> str:
        """Human-readable name of the latent feature adapted by PLICA."""

    def forward(self, batch: ControlObservationBatch) -> ControlPolicyOutput:
        return self.forward_policy(batch)

    def reset_policy_state(self) -> None:
        """Reset any policy-internal rollout state such as action queues."""

from __future__ import annotations

from typing import Any

import torch

from ..base import BaseControlBackbone
from ..types import ControlInterfaceSpec, ControlObservationBatch, ControlPolicyOutput
from .act import ACTBackbone
from .diffusion import DiffusionPolicyBackbone
from .smolvla import SmolVLABackbone


class TaskPolicySelectorBackbone(BaseControlBackbone):
    """Route each task to a configured policy checkpoint.

    The staged continuous tasks already encode task id as a one-hot vector in
    the proprioceptive state, so this selector can be used by rollout and
    deployment code without requiring language conditioning.
    """

    backbone_name = "task_policy_selector"
    backbone_family = "task_policy_selector"

    def __init__(self, interface_spec: ControlInterfaceSpec, official_cfg: dict | None = None) -> None:
        super().__init__(interface_spec)
        self.official_cfg = dict(official_cfg or {})
        self.child_backbone_name = str(self.official_cfg.get("selector_backbone_name") or "act").strip().lower()
        self.task_backbones = self._build_task_backbones()

    def _child_backbone(self, official_cfg: dict[str, Any]) -> BaseControlBackbone:
        if self.child_backbone_name == "act":
            return ACTBackbone(self.interface_spec, official_cfg=official_cfg)
        if self.child_backbone_name in {"diffusion", "diffusion_policy"}:
            return DiffusionPolicyBackbone(self.interface_spec, official_cfg=official_cfg)
        if self.child_backbone_name == "smolvla":
            return SmolVLABackbone(self.interface_spec, official_cfg=official_cfg)
        raise KeyError(f"Unsupported selector child backbone: {self.child_backbone_name}")

    @staticmethod
    def _normalize_task_policy_paths(raw: Any) -> dict[int, str]:
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return {int(key): str(value) for key, value in raw.items()}
        if isinstance(raw, (list, tuple)):
            return {int(index): str(value) for index, value in enumerate(raw) if value}
        raise TypeError("control.official.task_policy_paths must be a dict or list when provided.")

    def _build_task_backbones(self) -> dict[int, BaseControlBackbone]:
        default_path = self.official_cfg.get("policy_path")
        task_paths = self._normalize_task_policy_paths(self.official_cfg.get("task_policy_paths"))
        if default_path is None and not task_paths:
            raise ValueError("task_policy_selector requires control.official.policy_path or task_policy_paths.")
        backbones: dict[int, BaseControlBackbone] = {}
        for task_id in range(3):
            policy_path = task_paths.get(task_id, default_path)
            if policy_path is None:
                raise ValueError(f"No policy path configured for task id {task_id}.")
            child_cfg = dict(self.official_cfg)
            child_cfg.pop("task_policy_paths", None)
            child_cfg.pop("selector_backbone_name", None)
            child_cfg["policy_path"] = str(policy_path)
            backbones[task_id] = self._child_backbone(child_cfg)
        return backbones

    def _infer_task_id(self, batch: ControlObservationBatch) -> int:
        if batch.task_id is not None:
            task_tensor = torch.as_tensor(batch.task_id)
            return int(task_tensor.reshape(-1)[0].item())
        proprio = batch.proprio
        if proprio.ndim == 3:
            state = proprio[0, -1]
        elif proprio.ndim == 2:
            state = proprio[0]
        else:
            state = proprio.reshape(-1)
        if state.numel() < 15:
            raise ValueError("Cannot infer task id: proprio state is missing task one-hot entries.")
        return int(torch.argmax(state[12:15]).item())

    def forward_policy(self, batch: ControlObservationBatch) -> ControlPolicyOutput:
        task_id = self._infer_task_id(batch)
        if task_id not in self.task_backbones:
            raise KeyError(f"No policy configured for inferred task id {task_id}.")
        return self.task_backbones[task_id].forward_policy(batch)

    def latent_target_name(self) -> str:
        return "task_policy_selector_latent"

    def set_latent_adapter(self, adapter) -> None:
        for backbone in self.task_backbones.values():
            set_adapter = getattr(backbone, "set_latent_adapter", None)
            if set_adapter is not None:
                set_adapter(adapter)

    def set_trajectory_adapter(self, adapter) -> None:
        for backbone in self.task_backbones.values():
            set_adapter = getattr(backbone, "set_trajectory_adapter", None)
            if set_adapter is not None:
                set_adapter(adapter)

    def set_diffusion_denoise_adapter(self, adapter) -> None:
        for backbone in self.task_backbones.values():
            set_adapter = getattr(backbone, "set_diffusion_denoise_adapter", None)
            if set_adapter is not None:
                set_adapter(adapter)

    def reset_policy_state(self) -> None:
        for backbone in self.task_backbones.values():
            backbone.reset_policy_state()

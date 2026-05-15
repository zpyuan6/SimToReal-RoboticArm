from __future__ import annotations

import os
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ..base import BaseControlBackbone
from ..types import ControlInterfaceSpec, ControlObservationBatch, ControlPolicyOutput


@dataclass(frozen=True)
class LeRobotPolicySpec:
    policy_type: str
    policy_path: str | None = None
    extra_dependencies: tuple[str, ...] = ()
    latent_name: str = "policy_latent"
    family: str = "official_lerobot_policy"


class LeRobotOfficialBackbone(BaseControlBackbone):
    """Official-policy wrapper contract.

    This project no longer treats local reimplementations as formal baselines.
    Concrete backbones must resolve to official LeRobot/Hugging Face policy
    implementations. If the package is not installed, construction is still
    allowed so the integration layer can exist in-repo, but actual forward use
    will fail with a clear installation message.
    """

    def __init__(
        self,
        interface_spec: ControlInterfaceSpec,
        policy_spec: LeRobotPolicySpec,
        official_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(interface_spec)
        self.policy_spec = policy_spec
        self.official_cfg = dict(official_cfg or {})
        self.backbone_name = str(policy_spec.policy_type)
        self.backbone_family = str(policy_spec.family)
        self._policy_impl: Any | None = None
        self._availability_error: str | None = None
        self._lazy_pretrained_load = bool(policy_spec.policy_path)
        self._requested_device = self.official_cfg.get("policy_device") or self.official_cfg.get("device") or "cpu"
        self._set_workspace_caches()
        self._try_resolve_policy()

    @staticmethod
    def _workspace_root() -> Path:
        return Path(__file__).resolve().parents[4]

    def _set_workspace_caches(self) -> None:
        root = self._workspace_root()
        os.environ.setdefault("HF_HOME", str(root / ".hf-home"))
        os.environ.setdefault("TORCH_HOME", str(root / ".torch-home"))
        os.environ.setdefault("UV_CACHE_DIR", str(root / ".uv-cache"))
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    @staticmethod
    def _feature_shape_hwc_to_chw(shape: tuple[int, int, int]) -> tuple[int, int, int]:
        height, width, channels = (int(v) for v in shape)
        return channels, height, width

    def _resolve_policy_api(self) -> tuple[type[Any], type[Any], type[Any], type[Any]]:
        from lerobot.configs.types import FeatureType, PolicyFeature

        if self.policy_spec.policy_type == "act":
            from lerobot.policies.act.configuration_act import ACTConfig
            from lerobot.policies.act.modeling_act import ACTPolicy

            return ACTPolicy, ACTConfig, PolicyFeature, FeatureType
        if self.policy_spec.policy_type == "diffusion":
            from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
            from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

            return DiffusionPolicy, DiffusionConfig, PolicyFeature, FeatureType
        if self.policy_spec.policy_type == "smolvla":
            from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

            return SmolVLAPolicy, SmolVLAConfig, PolicyFeature, FeatureType
        raise KeyError(f"Unsupported official policy type: {self.policy_spec.policy_type}")

    def _build_policy_config(self) -> Any:
        policy_cls, config_cls, PolicyFeature, FeatureType = self._resolve_policy_api()
        del policy_cls

        if self.policy_spec.policy_type == "smolvla":
            image_shape = (3, 256, 256)
            input_features = {
                "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=image_shape),
                "observation.images.camera2": PolicyFeature(type=FeatureType.VISUAL, shape=image_shape),
                "observation.images.camera3": PolicyFeature(type=FeatureType.VISUAL, shape=image_shape),
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            }
        else:
            input_features = {
                "observation.images.main": PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=self._feature_shape_hwc_to_chw(self.interface_spec.image_shape),
                ),
                "observation.state": PolicyFeature(
                    type=FeatureType.STATE,
                    shape=(int(self.interface_spec.proprio_dim),),
                ),
            }
        if self.interface_spec.uses_language:
            input_features["observation.task"] = PolicyFeature(
                type=FeatureType.LANGUAGE,
                shape=(48,),
            )
        output_features = {
            "action": PolicyFeature(
                type=FeatureType.ACTION,
                shape=(int(self.interface_spec.action_spec.action_dim),),
            )
        }
        config = config_cls(
            input_features=input_features,
            output_features=output_features,
            device=str(self._requested_device),
        )
        return config

    def _requested_torch_device(self) -> torch.device:
        requested = str(self._requested_device)
        if requested.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(requested)

    def _move_policy_to_requested_device(self) -> None:
        if self._policy_impl is None:
            return
        device = self._requested_torch_device()
        if hasattr(self._policy_impl, "to"):
            self._policy_impl.to(device)

    def _try_resolve_policy(self) -> None:
        try:
            importlib.import_module("lerobot")
        except Exception as exc:  # pragma: no cover - installation dependent
            extras = ""
            if self.policy_spec.extra_dependencies:
                extras = f" Required extras: {', '.join(self.policy_spec.extra_dependencies)}."
            self._availability_error = (
                "Official LeRobot policy package is not installed. "
                f"Install LeRobot and the required policy dependencies before using "
                f"the {self.policy_spec.policy_type} backbone.{extras} Original error: {exc}"
            )
            self._policy_impl = None
            return
        try:
            policy_cls, _config_cls, _PolicyFeature, _FeatureType = self._resolve_policy_api()
            if self._lazy_pretrained_load:
                self._policy_impl = None
                self._availability_error = None
                return
            config = self._build_policy_config()
            self._policy_impl = policy_cls(config)
            self._move_policy_to_requested_device()
            self._availability_error = None
        except Exception as exc:  # pragma: no cover - installation/runtime dependent
            extras = ""
            if self.policy_spec.extra_dependencies:
                extras = f" Required extras: {', '.join(self.policy_spec.extra_dependencies)}."
            self._availability_error = (
                f"Failed to initialize official policy '{self.policy_spec.policy_type}'."
                f"{extras} Original error: {exc}"
            )
            self._policy_impl = None

    def _ensure_policy_loaded(self) -> None:
        if self._policy_impl is not None:
            return
        if self._availability_error:
            raise RuntimeError(self._availability_error)
        if not self._lazy_pretrained_load:
            raise RuntimeError(f"Official policy '{self.policy_spec.policy_type}' is unavailable.")
        try:
            policy_cls, _config_cls, _PolicyFeature, _FeatureType = self._resolve_policy_api()
            self._policy_impl = policy_cls.from_pretrained(
                self.policy_spec.policy_path,
                cache_dir=os.environ.get("HF_HOME"),
            )
            self._move_policy_to_requested_device()
            self._availability_error = None
            self._lazy_pretrained_load = False
        except Exception as exc:  # pragma: no cover - installation/runtime dependent
            extras = ""
            if self.policy_spec.extra_dependencies:
                extras = f" Required extras: {', '.join(self.policy_spec.extra_dependencies)}."
            self._availability_error = (
                f"Failed to load official pretrained policy '{self.policy_spec.policy_type}'"
                f" from '{self.policy_spec.policy_path}'.{extras} Original error: {exc}"
            )
            self._policy_impl = None
            raise RuntimeError(self._availability_error) from exc

    def availability_error(self) -> str | None:
        return self._availability_error

    def _policy_device(self) -> torch.device:
        self._ensure_policy_loaded()
        if self._policy_impl is None:
            return torch.device("cpu")
        try:
            return next(self._policy_impl.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @staticmethod
    def _current_images(batch: ControlObservationBatch) -> torch.Tensor:
        images = batch.images
        if images.ndim == 5:
            return images[:, -1]
        if images.ndim == 4:
            return images
        raise ValueError(f"Expected images with 4 or 5 dims, got {images.shape}")

    @staticmethod
    def _current_proprio(batch: ControlObservationBatch) -> torch.Tensor:
        proprio = batch.proprio
        if proprio.ndim == 3:
            return proprio[:, -1]
        if proprio.ndim == 2:
            return proprio
        raise ValueError(f"Expected proprio with 2 or 3 dims, got {proprio.shape}")

    def _build_language_tokens(self, texts: list[str], device: torch.device) -> dict[str, torch.Tensor]:
        self._ensure_policy_loaded()
        if self._policy_impl is None or self.policy_spec.policy_type != "smolvla":
            return {}
        tokenizer = self._policy_impl.model.vlm_with_expert.processor.tokenizer
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=int(self._policy_impl.config.tokenizer_max_length),
            return_tensors="pt",
        )
        return {
            "observation.language.tokens": encoded["input_ids"].to(device),
            "observation.language.attention_mask": encoded["attention_mask"].to(device=device, dtype=torch.bool),
        }

    @staticmethod
    def _resize_images(images: torch.Tensor, height: int, width: int) -> torch.Tensor:
        if tuple(images.shape[-2:]) == (height, width):
            return images
        return F.interpolate(images, size=(height, width), mode="bilinear", align_corners=False)

    def _build_official_batch(self, batch: ControlObservationBatch) -> dict[str, torch.Tensor]:
        device = self._policy_device()
        images = self._current_images(batch).to(device)
        proprio = self._current_proprio(batch).to(device)
        if self.policy_spec.policy_type == "smolvla":
            images = self._resize_images(images, 256, 256)
            state = proprio[..., :6]
            official_batch: dict[str, torch.Tensor] = {
                "observation.images.camera1": images,
                "observation.images.camera2": images,
                "observation.images.camera3": images,
                "observation.state": state,
            }
        else:
            official_batch = {
                "observation.images.main": images,
                "observation.state": proprio,
            }
        if self.interface_spec.uses_language:
            batch_size = int(images.shape[0])
            task_text = batch.task_text or [""] * batch_size
            if len(task_text) == 1 and batch_size > 1:
                task_text = task_text * batch_size
            official_batch.update(self._build_language_tokens([str(v) for v in task_text], device))
        return official_batch

    @staticmethod
    def _ensure_batched_action(action: torch.Tensor) -> torch.Tensor:
        if action.ndim == 1:
            return action.unsqueeze(0)
        return action

    def _pending_action_queue(self) -> list[torch.Tensor]:
        self._ensure_policy_loaded()
        if self._policy_impl is None:
            return []
        if hasattr(self._policy_impl, "_action_queue"):
            return list(self._policy_impl._action_queue)
        if hasattr(self._policy_impl, "_queues") and self._policy_impl._queues is not None:
            queue = self._policy_impl._queues.get("action")
            if queue is not None:
                return list(queue)
        return []

    def _compose_plan(self, selected_action: torch.Tensor) -> torch.Tensor:
        selected_action = self._ensure_batched_action(selected_action)
        queue_actions = [self._ensure_batched_action(t) for t in self._pending_action_queue()]
        plan_steps = [selected_action] + queue_actions
        plan = torch.stack(plan_steps, dim=1)
        return plan

    def forward_policy(self, batch: ControlObservationBatch) -> ControlPolicyOutput:
        self._ensure_policy_loaded()
        if self._policy_impl is None:
            raise RuntimeError(
                self._availability_error
                or f"Official loader for {self.policy_spec.policy_type} is not available."
            )
        input_device = batch.images.device
        official_batch = self._build_official_batch(batch)
        with torch.no_grad():
            selected_action = self._policy_impl.select_action(official_batch)
            planned_actions = self._compose_plan(selected_action)
        selected_action = self._ensure_batched_action(selected_action).to(input_device)
        planned_actions = planned_actions.to(input_device)
        latent = planned_actions.flatten(start_dim=1)
        return ControlPolicyOutput(
            actions=selected_action.unsqueeze(1),
            latent=latent,
            aux={"planned_actions": planned_actions},
        )

    def latent_target_name(self) -> str:
        return self.policy_spec.latent_name

    def reset_policy_state(self) -> None:
        if self._policy_impl is not None and hasattr(self._policy_impl, "reset"):
            self._policy_impl.reset()

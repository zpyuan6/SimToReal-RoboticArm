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
        self._policy_preprocessor: Any | None = None
        self._policy_postprocessor: Any | None = None
        self._latent_hook_handle: Any | None = None
        self._last_policy_latent: torch.Tensor | None = None
        self._last_policy_adapted_latent: torch.Tensor | None = None
        self._latent_adapter: Any | None = None
        self._trajectory_adapter: Any | None = None
        self._diffusion_denoise_adapter: Any | None = None
        self._latent_adapter_context: dict[str, Any] = {}
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

    def _apply_runtime_config_overrides(self) -> None:
        if self._policy_impl is None:
            return
        overrides = self.official_cfg.get("config_overrides", {})
        config = getattr(self._policy_impl, "config", None)
        if config is not None and hasattr(config, "device"):
            config.device = str(self._requested_torch_device())
        for key, value in overrides.items():
            if config is not None and hasattr(config, key):
                setattr(config, key, value)
        if self.policy_spec.policy_type == "diffusion" and "num_inference_steps" in overrides:
            diffusion = getattr(self._policy_impl, "diffusion", None)
            if diffusion is not None:
                diffusion.num_inference_steps = int(overrides["num_inference_steps"])

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
            self._install_latent_capture()
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
            self._try_load_policy_processors()
            self._apply_runtime_config_overrides()
            self._move_policy_to_requested_device()
            self._install_latent_capture()
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

    def _try_load_policy_processors(self) -> None:
        self._policy_preprocessor = None
        self._policy_postprocessor = None
        if not self.policy_spec.policy_path:
            return
        try:
            from lerobot.processor import PolicyProcessorPipeline

            self._policy_preprocessor = PolicyProcessorPipeline.from_pretrained(
                self.policy_spec.policy_path,
                config_filename="policy_preprocessor.json",
            )
            self._policy_postprocessor = PolicyProcessorPipeline.from_pretrained(
                self.policy_spec.policy_path,
                config_filename="policy_postprocessor.json",
            )
        except Exception:
            self._policy_preprocessor = None
            self._policy_postprocessor = None

    def _find_policy_module(self, module_name: str) -> Any | None:
        if self._policy_impl is None:
            return None
        modules = dict(self._policy_impl.named_modules())
        return modules.get(module_name)

    def _install_latent_capture(self) -> None:
        self._last_policy_latent = None
        self._last_policy_adapted_latent = None
        if self._policy_impl is None:
            return
        if self.policy_spec.policy_type == "act":
            action_head = self._find_policy_module("model.action_head")
            if action_head is None:
                return
            if self._latent_hook_handle is not None:
                self._latent_hook_handle.remove()

            def _capture_or_adapt_action_head_input(_module, inputs):
                if not inputs:
                    return None
                latent = inputs[0]
                self._last_policy_latent = latent.detach()
                adapted = self._apply_runtime_latent_adapter(latent)
                if adapted is latent:
                    return None
                self._last_policy_adapted_latent = adapted.detach()
                return (adapted, *inputs[1:])

            self._latent_hook_handle = action_head.register_forward_pre_hook(_capture_or_adapt_action_head_input)
            return
        if self.policy_spec.policy_type == "diffusion":
            diffusion = getattr(self._policy_impl, "diffusion", None)
            if diffusion is None or not hasattr(diffusion, "_prepare_global_conditioning"):
                return
            if hasattr(diffusion, "_ttla_original_prepare_global_conditioning"):
                return
            original_prepare = diffusion._prepare_global_conditioning

            def _capture_global_conditioning(batch, _original_prepare=original_prepare):
                global_cond = _original_prepare(batch)
                self._last_policy_latent = global_cond.detach()
                adapted = self._apply_runtime_latent_adapter(global_cond)
                if adapted is not global_cond:
                    self._last_policy_adapted_latent = adapted.detach()
                return adapted

            diffusion._ttla_original_prepare_global_conditioning = original_prepare
            diffusion._prepare_global_conditioning = _capture_global_conditioning
            if hasattr(diffusion, "_ttla_original_conditional_sample"):
                return
            original_conditional_sample = diffusion.conditional_sample

            def _conditional_sample_with_adapter(
                batch_size: int,
                global_cond=None,
                generator=None,
                noise=None,
                _original_conditional_sample=original_conditional_sample,
                _diffusion=diffusion,
            ):
                if self._diffusion_denoise_adapter is None:
                    return _original_conditional_sample(
                        batch_size,
                        global_cond=global_cond,
                        generator=generator,
                        noise=noise,
                    )
                try:
                    param = next(_diffusion.parameters())
                    device = param.device
                    dtype = param.dtype
                except StopIteration:
                    device = torch.device("cpu")
                    dtype = torch.float32
                sample = (
                    noise
                    if noise is not None
                    else torch.randn(
                        size=(
                            int(batch_size),
                            int(_diffusion.config.horizon),
                            int(_diffusion.config.action_feature.shape[0]),
                        ),
                        dtype=dtype,
                        device=device,
                        generator=generator,
                    )
                )
                _diffusion.noise_scheduler.set_timesteps(_diffusion.num_inference_steps)
                for t in _diffusion.noise_scheduler.timesteps:
                    timestep = torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device)
                    model_output = _diffusion.unet(sample, timestep, global_cond=global_cond)
                    model_output = self._apply_runtime_diffusion_denoise_adapter(model_output, sample, timestep)
                    sample = _diffusion.noise_scheduler.step(model_output, t, sample, generator=generator).prev_sample
                return sample

            diffusion._ttla_original_conditional_sample = original_conditional_sample
            diffusion.conditional_sample = _conditional_sample_with_adapter

    def _captured_latent_or_plan(self, planned_actions: torch.Tensor, input_device: torch.device) -> torch.Tensor:
        latent = self._last_policy_latent
        if latent is None:
            return planned_actions.flatten(start_dim=1).to(input_device)
        if latent.ndim == 0:
            return planned_actions.flatten(start_dim=1).to(input_device)
        if latent.shape[0] != planned_actions.shape[0]:
            return planned_actions.flatten(start_dim=1).to(input_device)
        return latent.reshape(latent.shape[0], -1).to(input_device)

    def set_latent_adapter(self, adapter: Any | None) -> None:
        self._latent_adapter = adapter
        if adapter is not None and hasattr(adapter, "eval"):
            adapter.eval()

    def supports_runtime_latent_adapter(self) -> bool:
        return self.policy_spec.policy_type in {"act", "diffusion"}

    def set_trajectory_adapter(self, adapter: Any | None) -> None:
        self._trajectory_adapter = adapter
        if adapter is not None and hasattr(adapter, "eval"):
            adapter.eval()

    def supports_runtime_trajectory_adapter(self) -> bool:
        return self.policy_spec.policy_type in {"act", "diffusion"}

    def set_diffusion_denoise_adapter(self, adapter: Any | None) -> None:
        self._diffusion_denoise_adapter = adapter
        if adapter is not None and hasattr(adapter, "eval"):
            adapter.eval()

    def supports_runtime_diffusion_denoise_adapter(self) -> bool:
        return self.policy_spec.policy_type == "diffusion"

    def _current_task_id_tensor(self, batch: ControlObservationBatch, proprio: torch.Tensor) -> torch.Tensor | int | None:
        if batch.task_id is not None:
            return batch.task_id
        if proprio.shape[-1] >= 15:
            return torch.argmax(proprio[..., 12:15], dim=-1)
        return None

    def _set_latent_adapter_context(self, batch: ControlObservationBatch) -> None:
        proprio = self._current_proprio(batch).detach()
        self._latent_adapter_context = {
            "task_id": self._current_task_id_tensor(batch, proprio),
            "proprio": proprio,
        }

    def _apply_runtime_latent_adapter(self, latent: torch.Tensor) -> torch.Tensor:
        adapter = self._latent_adapter
        if adapter is None:
            return latent
        if hasattr(adapter, "to"):
            adapter.to(latent.device)
        context = self._latent_adapter_context
        task_id = context.get("task_id")
        proprio = context.get("proprio")
        if torch.is_tensor(task_id):
            task_id = task_id.to(device=latent.device)
        if torch.is_tensor(proprio):
            proprio = proprio.to(device=latent.device, dtype=latent.dtype)
        if hasattr(adapter, "adapt_tensor"):
            return adapter.adapt_tensor(latent, task_id=task_id, proprio=proprio)
        return adapter(latent)

    def _apply_runtime_trajectory_adapter(self, planned_actions: torch.Tensor) -> torch.Tensor:
        adapter = self._trajectory_adapter
        if adapter is None:
            return planned_actions
        if hasattr(adapter, "to"):
            adapter.to(planned_actions.device)
        context = self._latent_adapter_context
        task_id = context.get("task_id")
        proprio = context.get("proprio")
        if torch.is_tensor(task_id):
            task_id = task_id.to(device=planned_actions.device)
        if torch.is_tensor(proprio):
            proprio = proprio.to(device=planned_actions.device, dtype=planned_actions.dtype)
        if hasattr(adapter, "adapt_tensor"):
            return adapter.adapt_tensor(planned_actions, task_id=task_id, proprio=proprio)
        return adapter(planned_actions)

    def _apply_runtime_diffusion_denoise_adapter(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        adapter = self._diffusion_denoise_adapter
        if adapter is None:
            return model_output
        if hasattr(adapter, "to"):
            adapter.to(model_output.device)
        context = self._latent_adapter_context
        task_id = context.get("task_id")
        proprio = context.get("proprio")
        if torch.is_tensor(task_id):
            task_id = task_id.to(device=model_output.device)
        if torch.is_tensor(proprio):
            proprio = proprio.to(device=model_output.device, dtype=model_output.dtype)
        if hasattr(adapter, "adapt_denoise"):
            return adapter.adapt_denoise(model_output, sample, timestep, task_id=task_id, proprio=proprio)
        return adapter(model_output)

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
            task_text = [str(v) for v in task_text]
            if self.policy_spec.policy_type == "smolvla":
                official_batch["task"] = task_text
            else:
                official_batch.update(self._build_language_tokens(task_text, device))
        return official_batch

    def _preprocess_official_batch(self, official_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self._policy_preprocessor is None:
            return official_batch
        return self._policy_preprocessor(official_batch)

    def _move_official_batch_to_policy_device(self, value: Any) -> Any:
        device = self._policy_device()
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            return {key: self._move_official_batch_to_policy_device(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._move_official_batch_to_policy_device(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._move_official_batch_to_policy_device(item) for item in value)
        return value

    def _postprocess_action(self, action: torch.Tensor) -> torch.Tensor:
        if self._policy_postprocessor is None:
            return action
        return self._policy_postprocessor.process_action(action)

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
        selected_action = self._ensure_batched_action(self._postprocess_action(selected_action))
        queue_actions = [
            self._ensure_batched_action(self._postprocess_action(t)) for t in self._pending_action_queue()
        ]
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
        self._set_latent_adapter_context(batch)
        official_batch = self._build_official_batch(batch)
        official_batch = self._preprocess_official_batch(official_batch)
        official_batch = self._move_official_batch_to_policy_device(official_batch)
        with torch.no_grad():
            selected_action = self._policy_impl.select_action(official_batch)
            planned_actions = self._compose_plan(selected_action)
            planned_actions = self._apply_runtime_trajectory_adapter(planned_actions)
        selected_action = planned_actions[:, 0].to(input_device)
        planned_actions = planned_actions.to(input_device)
        latent = self._captured_latent_or_plan(planned_actions, input_device)
        return ControlPolicyOutput(
            actions=selected_action.unsqueeze(1),
            latent=latent,
            aux={"planned_actions": planned_actions},
        )

    def latent_target_name(self) -> str:
        return self.policy_spec.latent_name

    def reset_policy_state(self) -> None:
        self._last_policy_latent = None
        self._last_policy_adapted_latent = None
        if self._policy_impl is not None and hasattr(self._policy_impl, "reset"):
            self._policy_impl.reset()

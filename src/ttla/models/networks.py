from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _extract_task_ids(state: torch.Tensor, task_vocab_size: int) -> torch.Tensor:
    if state.ndim == 1:
        state = state.unsqueeze(0)
    if state.shape[-1] < 2:
        return torch.zeros(state.shape[0], dtype=torch.long, device=state.device)
    task_id = torch.round(state[..., -2]).long()
    return task_id.clamp(0, max(task_vocab_size - 1, 0))


def _resolve_task_ids(
    task_ids: torch.Tensor | None,
    state: torch.Tensor | None,
    task_vocab_size: int,
) -> torch.Tensor:
    if task_ids is not None:
        if task_ids.ndim > 1:
            task_ids = task_ids.squeeze(-1)
        return task_ids.long().clamp(0, max(task_vocab_size - 1, 0))
    if state is None:
        raise ValueError("Either task_ids or state must be provided.")
    return _extract_task_ids(state, task_vocab_size)


def _strip_task_feature(state: torch.Tensor) -> torch.Tensor:
    if state.shape[-1] < 2:
        return state
    return torch.cat([state[..., :-2], state[..., -1:]], dim=-1)


def _extract_progress_bins(state: torch.Tensor, num_bins: int) -> torch.Tensor:
    if state.ndim == 1:
        state = state.unsqueeze(0)
    if state.shape[-1] < 1:
        return torch.zeros(state.shape[0], dtype=torch.long, device=state.device)
    progress = state[..., -1].clamp(0.0, 1.0)
    scaled = torch.floor(progress * num_bins).long()
    return scaled.clamp(0, max(num_bins - 1, 0))


class ImageEncoder(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 11 * 11, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class BaseTTLAModel(nn.Module, ABC):
    uses_history = False

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
        hidden_dim: int,
        action_embed_dim: int = 16,
        task_vocab_size: int = 3,
        task_embed_dim: int = 16,
        transition_stage_bins: int = 4,
        adapter_hidden_dim: int = 64,
        adapter_scale: float = 0.1,
        **_unused: Any,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_embed_dim = action_embed_dim
        self.task_vocab_size = task_vocab_size
        self.task_embed_dim = task_embed_dim
        self.transition_stage_bins = transition_stage_bins
        self.adapter_scale = adapter_scale
        self.proprio_dim = max(state_dim - 1, 1)
        self.context_init = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.action_embedding = nn.Embedding(action_dim, action_embed_dim)
        self.task_embedding = nn.Embedding(task_vocab_size, task_embed_dim)
        self.transition_task_embedding = nn.Embedding(task_vocab_size, action_embed_dim)
        self.transition_stage_embedding = nn.Embedding(transition_stage_bins, action_embed_dim)
        self.transition = nn.Sequential(
            nn.Linear(latent_dim + action_embed_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.adapter = nn.Sequential(
            nn.Linear(latent_dim, adapter_hidden_dim),
            nn.ReLU(),
            nn.Linear(adapter_hidden_dim, latent_dim),
            nn.Tanh(),
        )
        nn.init.zeros_(self.adapter[2].weight)
        nn.init.zeros_(self.adapter[2].bias)

    @property
    def backbone_type(self) -> str:
        return getattr(self, "_backbone_type", "feedforward")

    def _task_embedding(self, task_ids: torch.Tensor | None, state: torch.Tensor | None) -> torch.Tensor:
        resolved_task_ids = _resolve_task_ids(task_ids, state, self.task_vocab_size)
        return self.task_embedding(resolved_task_ids)

    def _proprio_state(self, state: torch.Tensor) -> torch.Tensor:
        return _strip_task_feature(state)

    def predict_next(
        self,
        z: torch.Tensor,
        action_index: torch.Tensor,
        state: torch.Tensor | None = None,
        task_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        action_emb = self.action_embedding(action_index.long())
        if state is None:
            if task_ids is None:
                task_emb = torch.zeros_like(action_emb)
            else:
                resolved_task_ids = _resolve_task_ids(task_ids, None, self.task_vocab_size)
                task_emb = self.transition_task_embedding(resolved_task_ids)
            stage_emb = torch.zeros_like(action_emb)
        else:
            resolved_task_ids = _resolve_task_ids(task_ids, state, self.task_vocab_size)
            stage_ids = _extract_progress_bins(state, self.transition_stage_bins)
            task_emb = self.transition_task_embedding(resolved_task_ids)
            stage_emb = self.transition_stage_embedding(stage_ids)
        delta = self.transition(torch.cat([z, action_emb, task_emb, stage_emb], dim=-1))
        return z + delta

    def adapt(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.adapter(z) * self.adapter_scale

    def freeze_backbone(self) -> None:
        for name, param in self.named_parameters():
            if not name.startswith("adapter."):
                param.requires_grad_(False)

    def adapter_parameters(self):
        return self.adapter.parameters()

    def init_runtime_state(self, batch_size: int = 1, device: torch.device | None = None) -> Any:
        return None

    @abstractmethod
    def encode(self, image: torch.Tensor, state: torch.Tensor, task_ids: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def policy_logits(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def encode_step(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        runtime_state: Any = None,
        task_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Any]:
        return self.encode(image, state, task_ids=task_ids), runtime_state

    def compute_latents(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        task_ids = batch.get("task")
        z = self.encode(batch["image"], batch["state"], task_ids=task_ids)
        next_z = self.encode(batch["next_image"], batch["next_state"], task_ids=task_ids)
        return z, next_z

    def compute_policy_loss(self, batch: dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(self.policy_logits(z), batch["primitive_id"])

    def act(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        runtime_state: Any = None,
        use_adapter: bool = False,
        task_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Any, torch.Tensor]:
        z, next_runtime_state = self.encode_step(image, state, runtime_state, task_ids=task_ids)
        z_used = self.adapt(z) if use_adapter else z
        logits = self.policy_logits(z_used)
        primitive = logits.argmax(dim=-1)
        return primitive, next_runtime_state, z_used

    # Backward-compatible wrappers for older viewer/debug scripts.
    def adapted_latent(self, z: torch.Tensor, *_unused) -> torch.Tensor:
        return self.adapt(z)

    def predict_action(self, z: torch.Tensor, *_unused) -> torch.Tensor:
        return self.policy_logits(z)

    def predict_residual(self, z: torch.Tensor, *_unused) -> torch.Tensor:
        return torch.zeros(z.shape[0], 6, device=z.device, dtype=z.dtype)

    def action_features(self, action_index: torch.Tensor, action_residual: torch.Tensor | None = None) -> torch.Tensor:
        action_one_hot = torch.nn.functional.one_hot(action_index.long(), num_classes=self.action_dim).float()
        if action_residual is None:
            return action_one_hot
        return torch.cat([action_one_hot, action_residual.float()], dim=-1)


class FeedForwardTTLAModel(BaseTTLAModel):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._backbone_type = "feedforward"
        self.image_encoder = ImageEncoder(self.hidden_dim)
        self.state_encoder = nn.Sequential(
            nn.Linear(self.proprio_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + self.task_embed_dim, self.latent_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

    def encode(self, image: torch.Tensor, state: torch.Tensor, task_ids: torch.Tensor | None = None) -> torch.Tensor:
        visual = self.image_encoder(image)
        proprio = self.state_encoder(self._proprio_state(state))
        task_feat = self._task_embedding(task_ids, state)
        return self.fusion(torch.cat([visual, proprio, task_feat], dim=-1))

    def policy_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.policy_head(z)


class RecurrentTTLAModel(BaseTTLAModel):
    uses_history = True

    def __init__(self, history_len: int = 4, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._backbone_type = "recurrent"
        self.history_len = history_len
        self.image_encoder = ImageEncoder(self.hidden_dim)
        self.state_encoder = nn.Sequential(
            nn.Linear(self.proprio_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + self.task_embed_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(self.hidden_dim, self.latent_dim, batch_first=True)
        self.policy_head = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

    def _feature_step(self, image: torch.Tensor, state: torch.Tensor, task_ids: torch.Tensor | None = None) -> torch.Tensor:
        visual = self.image_encoder(image)
        proprio = self.state_encoder(self._proprio_state(state))
        task_feat = self._task_embedding(task_ids, state)
        return self.feature_fusion(torch.cat([visual, proprio, task_feat], dim=-1))

    def init_runtime_state(self, batch_size: int = 1, device: torch.device | None = None) -> torch.Tensor:
        target_device = device or next(self.parameters()).device
        return torch.zeros(1, batch_size, self.latent_dim, device=target_device)

    def encode_step(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        runtime_state: Any = None,
        task_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Any]:
        feat = self._feature_step(image, state, task_ids=task_ids).unsqueeze(1)
        if runtime_state is None:
            runtime_state = self.init_runtime_state(batch_size=image.shape[0], device=image.device)
        out, hidden = self.gru(feat, runtime_state)
        return out[:, -1], hidden

    def encode(self, image: torch.Tensor, state: torch.Tensor, task_ids: torch.Tensor | None = None) -> torch.Tensor:
        z, _ = self.encode_step(image, state, None, task_ids=task_ids)
        return z

    def encode_history(
        self,
        images: torch.Tensor,
        states: torch.Tensor,
        mask: torch.Tensor | None = None,
        task_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, history_len = images.shape[:2]
        flat_images = images.reshape(batch_size * history_len, *images.shape[2:])
        flat_states = states.reshape(batch_size * history_len, states.shape[-1])
        if task_ids is not None:
            expanded_task_ids = task_ids.unsqueeze(1).expand(-1, history_len).reshape(batch_size * history_len)
        else:
            expanded_task_ids = None
        fused = self._feature_step(flat_images, flat_states, task_ids=expanded_task_ids).reshape(batch_size, history_len, self.hidden_dim)
        output, _ = self.gru(fused)
        if mask is None:
            return output[:, -1]
        lengths = mask.long().sum(dim=1).clamp_min(1)
        gather_idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, output.shape[-1])
        return output.gather(1, gather_idx).squeeze(1)

    def compute_latents(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if "history_images" not in batch:
            return super().compute_latents(batch)
        task_ids = batch.get("task")
        z = self.encode_history(batch["history_images"], batch["history_states"], batch.get("history_mask"), task_ids=task_ids)
        next_z = self.encode_history(batch["next_history_images"], batch["next_history_states"], batch.get("next_history_mask"), task_ids=task_ids)
        return z, next_z

    def policy_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.policy_head(z)


class ChunkingTTLAModel(BaseTTLAModel):
    uses_history = True

    def __init__(self, chunk_size: int = 3, history_len: int = 4, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._backbone_type = "chunking"
        self.chunk_size = chunk_size
        self.history_len = history_len
        self.image_encoder = ImageEncoder(self.hidden_dim)
        self.state_encoder = nn.Sequential(
            nn.Linear(self.proprio_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + self.task_embed_dim, self.latent_dim),
            nn.ReLU(),
        )
        self.chunk_head = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim * self.chunk_size),
        )

    def encode(self, image: torch.Tensor, state: torch.Tensor, task_ids: torch.Tensor | None = None) -> torch.Tensor:
        visual = self.image_encoder(image)
        proprio = self.state_encoder(self._proprio_state(state))
        task_feat = self._task_embedding(task_ids, state)
        return self.fusion(torch.cat([visual, proprio, task_feat], dim=-1))

    def encode_history(
        self,
        images: torch.Tensor,
        states: torch.Tensor,
        mask: torch.Tensor | None = None,
        task_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is None:
            return self.encode(images[:, -1], states[:, -1], task_ids=task_ids)
        valid_index = (mask.long().sum(dim=1).clamp_min(1) - 1)
        batch_idx = torch.arange(images.shape[0], device=images.device)
        return self.encode(images[batch_idx, valid_index], states[batch_idx, valid_index], task_ids=task_ids)

    def compute_latents(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if "history_images" not in batch:
            return super().compute_latents(batch)
        task_ids = batch.get("task")
        z = self.encode_history(batch["history_images"], batch["history_states"], batch.get("history_mask"), task_ids=task_ids)
        next_z = self.encode_history(batch["next_history_images"], batch["next_history_states"], batch.get("next_history_mask"), task_ids=task_ids)
        return z, next_z

    def policy_chunk_logits(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.chunk_head(z)
        return logits.view(z.shape[0], self.chunk_size, self.action_dim)

    def policy_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.policy_chunk_logits(z)[:, 0]

    def compute_policy_loss(self, batch: dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        chunk_logits = self.policy_chunk_logits(z)
        if "chunk_primitive_ids" not in batch:
            return F.cross_entropy(chunk_logits[:, 0], batch["primitive_id"])
        targets = batch["chunk_primitive_ids"]
        mask = batch.get("chunk_mask")
        flat_logits = chunk_logits.reshape(-1, self.action_dim)
        flat_targets = targets.reshape(-1)
        losses = F.cross_entropy(flat_logits, flat_targets, reduction="none").view_as(targets)
        if mask is None:
            return losses.mean()
        denom = mask.sum().clamp_min(1.0)
        return (losses * mask).sum() / denom

    def init_runtime_state(self, batch_size: int = 1, device: torch.device | None = None) -> dict[str, list[int]]:
        return {"pending": []}

    def act(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        runtime_state: Any = None,
        use_adapter: bool = False,
        task_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Any, torch.Tensor]:
        runtime_state = runtime_state or self.init_runtime_state(batch_size=image.shape[0], device=image.device)
        z = self.encode(image, state, task_ids=task_ids)
        z_used = self.adapt(z) if use_adapter else z
        pending = list(runtime_state.get("pending", []))
        if pending:
            primitive = torch.tensor([pending[0]], device=image.device, dtype=torch.long)
            next_state = {"pending": pending[1:]}
            return primitive, next_state, z_used
        chunk_logits = self.policy_chunk_logits(z_used)
        chunk_actions = chunk_logits.argmax(dim=-1)
        next_state = {"pending": chunk_actions[0, 1:].tolist()}
        return chunk_actions[:, 0], next_state, z_used


class LanguageConditionedTTLAModel(BaseTTLAModel):
    def __init__(self, task_vocab_size: int = 3, language_dim: int = 32, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._backbone_type = "language"
        self.task_vocab_size = task_vocab_size
        self.language_dim = language_dim
        self.image_encoder = ImageEncoder(self.hidden_dim)
        self.state_encoder = nn.Sequential(
            nn.Linear(self.proprio_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.task_embedding = nn.Embedding(task_vocab_size, language_dim)
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + language_dim, self.latent_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

    def encode(self, image: torch.Tensor, state: torch.Tensor, task_ids: torch.Tensor | None = None) -> torch.Tensor:
        visual = self.image_encoder(image)
        proprio = self.state_encoder(self._proprio_state(state))
        language = self.task_embedding(_resolve_task_ids(task_ids, state, self.task_vocab_size))
        return self.fusion(torch.cat([visual, proprio, language], dim=-1))

    def policy_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.policy_head(z)


class DiffusionPrimitiveTTLAModel(BaseTTLAModel):
    def __init__(self, diffusion_steps: int = 4, diffusion_dim: int = 32, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._backbone_type = "diffusion"
        self.diffusion_steps = diffusion_steps
        self.diffusion_dim = diffusion_dim
        self.image_encoder = ImageEncoder(self.hidden_dim)
        self.state_encoder = nn.Sequential(
            nn.Linear(self.proprio_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + self.task_embed_dim, self.latent_dim),
            nn.ReLU(),
        )
        self.action_codebook = nn.Embedding(self.action_dim, diffusion_dim)
        self.time_embedding = nn.Embedding(diffusion_steps, diffusion_dim)
        self.denoiser = nn.Sequential(
            nn.Linear(self.latent_dim + diffusion_dim + diffusion_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, diffusion_dim),
        )

    def encode(self, image: torch.Tensor, state: torch.Tensor, task_ids: torch.Tensor | None = None) -> torch.Tensor:
        visual = self.image_encoder(image)
        proprio = self.state_encoder(self._proprio_state(state))
        task_feat = self._task_embedding(task_ids, state)
        return self.fusion(torch.cat([visual, proprio, task_feat], dim=-1))

    def _denoise(self, z: torch.Tensor, noisy_action: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embedding(timestep.long())
        return self.denoiser(torch.cat([z, noisy_action, t_emb], dim=-1))

    def _action_logits_from_embed(self, action_embed: torch.Tensor) -> torch.Tensor:
        return action_embed @ self.action_codebook.weight.t()

    def policy_logits(self, z: torch.Tensor) -> torch.Tensor:
        action_embed = torch.zeros(z.shape[0], self.diffusion_dim, device=z.device, dtype=z.dtype)
        for step in reversed(range(self.diffusion_steps)):
            timestep = torch.full((z.shape[0],), step, device=z.device, dtype=torch.long)
            action_embed = self._denoise(z, action_embed, timestep)
        return self._action_logits_from_embed(action_embed)

    def compute_policy_loss(self, batch: dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        primitive_id = batch["primitive_id"]
        target_embed = self.action_codebook(primitive_id)
        timestep = torch.randint(0, self.diffusion_steps, (z.shape[0],), device=z.device)
        alpha = 1.0 - timestep.float().unsqueeze(-1) / max(self.diffusion_steps, 1)
        noise = torch.randn_like(target_embed)
        noisy = alpha * target_embed + (1.0 - alpha) * noise
        pred_embed = self._denoise(z, noisy, timestep)
        loss_diffusion = F.mse_loss(pred_embed, target_embed)
        logits = self._action_logits_from_embed(pred_embed)
        loss_ce = F.cross_entropy(logits, primitive_id)
        return loss_ce + 0.5 * loss_diffusion


TTLAModel = FeedForwardTTLAModel


def build_backbone_model(backbone_type: str, **model_kwargs: Any) -> BaseTTLAModel:
    normalized = backbone_type.lower()
    if normalized in {"feedforward", "ff", "mlp"}:
        return FeedForwardTTLAModel(**model_kwargs)
    if normalized in {"recurrent", "gru", "rnn"}:
        return RecurrentTTLAModel(**model_kwargs)
    if normalized in {"chunking", "chunk", "act"}:
        return ChunkingTTLAModel(**model_kwargs)
    if normalized in {"language", "language_conditioned", "tiny_vla"}:
        return LanguageConditionedTTLAModel(**model_kwargs)
    if normalized in {"diffusion", "primitive_diffusion"}:
        return DiffusionPrimitiveTTLAModel(**model_kwargs)
    raise KeyError(f"Unsupported backbone_type: {backbone_type}")

from __future__ import annotations

import torch
from torch import nn


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


class TTLAModel(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, latent_dim: int, context_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.context_dim = context_dim
        self.image_encoder = ImageEncoder(hidden_dim)
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.fusion = nn.Sequential(nn.Linear(hidden_dim * 2, latent_dim), nn.ReLU())
        self.policy_head = nn.Sequential(
            nn.Linear(latent_dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.context_filter = nn.GRUCell(latent_dim + action_dim + latent_dim, context_dim)
        self.context_init = nn.Parameter(torch.zeros(context_dim))
        self.transition = nn.Sequential(
            nn.Linear(latent_dim + action_dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.adapter = nn.Sequential(nn.Linear(context_dim, latent_dim), nn.Tanh())

    def encode(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        visual = self.image_encoder(image)
        proprio = self.state_encoder(state)
        return self.fusion(torch.cat([visual, proprio], dim=-1))

    def predict_action(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.policy_head(torch.cat([z, context], dim=-1))

    def predict_next(self, z: torch.Tensor, action_one_hot: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.transition(torch.cat([z, action_one_hot, context], dim=-1))

    def update_context(
        self,
        prev_context: torch.Tensor,
        z: torch.Tensor,
        action_one_hot: torch.Tensor,
        next_z: torch.Tensor,
    ) -> torch.Tensor:
        return self.context_filter(torch.cat([z, action_one_hot, next_z], dim=-1), prev_context)

    def adapted_latent(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return z + self.adapter(context)

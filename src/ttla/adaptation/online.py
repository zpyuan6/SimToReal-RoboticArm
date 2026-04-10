from __future__ import annotations

import torch
import torch.nn.functional as F


class OnlineContextAdapter:
    def __init__(self, model, cfg: dict, device: torch.device) -> None:
        self.model = model
        self.cfg = cfg
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.context = self.model.context_init.detach().clone().unsqueeze(0).to(self.device)

    def adapt(self, z: torch.Tensor, action: int, next_z: torch.Tensor) -> torch.Tensor:
        z = z.detach()
        next_z = next_z.detach()
        action_one_hot = F.one_hot(
            torch.tensor([action], device=self.device), num_classes=self.model.action_dim
        ).float()
        updated = self.model.update_context(self.context.detach(), z, action_one_hot, next_z).detach()
        blended = 0.7 * updated + 0.3 * self.context.detach()
        self.context = blended
        return self.context

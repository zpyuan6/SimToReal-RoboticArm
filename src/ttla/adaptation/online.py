from __future__ import annotations

import torch


class OnlineContextAdapter:
    """
    Backward-compatible wrapper around the calibrated latent adapter.

    The current method no longer performs online context-state updates during
    deployment. This class remains only so older viewer/debug scripts can keep
    calling a small adapter object without importing the model directly.
    """

    def __init__(self, model, cfg: dict, device: torch.device) -> None:
        self.model = model
        self.cfg = cfg
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.context = None

    def adapt(self, z: torch.Tensor, *_unused) -> torch.Tensor:
        return self.model.adapt(z.detach())

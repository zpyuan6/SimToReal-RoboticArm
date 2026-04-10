from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data import TrajectoryDataset
from ..models import TTLAModel
from ..utils.io import ensure_dir


def build_model(cfg: dict) -> TTLAModel:
    model_cfg = cfg["model"]
    return TTLAModel(
        state_dim=model_cfg["state_dim"],
        action_dim=model_cfg["action_dim"],
        latent_dim=model_cfg["latent_dim"],
        context_dim=model_cfg["context_dim"],
        hidden_dim=model_cfg["hidden_dim"],
    )


def train_model(cfg: dict, train_path: str | Path, val_path: str | Path) -> Path:
    device = torch.device(cfg["train"]["device"])
    model = build_model(cfg).to(device)
    train_loader = DataLoader(TrajectoryDataset(train_path), batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(TrajectoryDataset(val_path), batch_size=cfg["train"]["batch_size"], shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    best_val = float("inf")
    checkpoint_dir = ensure_dir(cfg["paths"]["checkpoint_dir"])
    best_path = checkpoint_dir / "best_model.pt"
    for _ in range(cfg["train"]["epochs"]):
        model.train()
        for batch in tqdm(train_loader, desc="train", leave=False):
            optimizer.zero_grad(set_to_none=True)
            image = batch["image"].to(device)
            state = batch["state"].to(device)
            next_image = batch["next_image"].to(device)
            next_state = batch["next_state"].to(device)
            actions = batch["action"].to(device)
            context = batch["context"].to(device)
            z = model.encode(image, state)
            next_z = model.encode(next_image, next_state)
            adapted_z = model.adapted_latent(z, context)
            logits = model.predict_action(adapted_z, context)
            action_oh = F.one_hot(actions, num_classes=cfg["model"]["action_dim"]).float()
            pred_next = model.predict_next(adapted_z, action_oh, context)
            context_from_transition = model.update_context(context, z, action_oh, next_z)
            loss_policy = F.cross_entropy(logits, actions)
            loss_dyn = F.mse_loss(pred_next, next_z.detach())
            loss_temp = F.mse_loss(context_from_transition, context)
            loss_penalty = context.square().mean()
            loss = (
                cfg["train"]["policy_loss_weight"] * loss_policy
                + cfg["train"]["transition_loss_weight"] * loss_dyn
                + cfg["train"]["temporal_loss_weight"] * loss_temp
                + cfg["train"]["context_penalty_weight"] * loss_penalty
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        val_loss = _evaluate_loss(model, val_loader, cfg, device)
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(), "config": cfg}, best_path)
    return best_path


@torch.no_grad()
def _evaluate_loss(model: TTLAModel, loader: DataLoader, cfg: dict, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        image = batch["image"].to(device)
        state = batch["state"].to(device)
        next_image = batch["next_image"].to(device)
        next_state = batch["next_state"].to(device)
        actions = batch["action"].to(device)
        context = batch["context"].to(device)
        z = model.encode(image, state)
        next_z = model.encode(next_image, next_state)
        logits = model.predict_action(model.adapted_latent(z, context), context)
        action_oh = F.one_hot(actions, num_classes=cfg["model"]["action_dim"]).float()
        pred_next = model.predict_next(z, action_oh, context)
        loss = F.cross_entropy(logits, actions) + F.mse_loss(pred_next, next_z)
        total += float(loss.item()) * image.size(0)
        count += image.size(0)
    return total / max(count, 1)

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import load_config
from ..control import (
    ContinuousActionSpec,
    ControlInterfaceSpec,
    ControlObservationBatch,
    build_control_backbone,
)
from ..data import ContinuousTrajectoryDataset
from ..utils.io import ensure_dir


def _build_control_spec(cfg: dict) -> ControlInterfaceSpec:
    control_cfg = cfg["control"]
    action_cfg = control_cfg["action"]
    return ControlInterfaceSpec(
        image_shape=tuple(control_cfg.get("image_shape", (224, 224, 3))),
        proprio_dim=int(control_cfg["proprio_dim"]),
        uses_language=bool(control_cfg.get("uses_language", False)),
        action_spec=ContinuousActionSpec(
            action_dim=int(action_cfg["action_dim"]),
            horizon=int(action_cfg["horizon"]),
            control_mode=str(action_cfg.get("control_mode", "joint_delta")),
            clamp_low=tuple(action_cfg["clamp_low"]) if "clamp_low" in action_cfg else None,
            clamp_high=tuple(action_cfg["clamp_high"]) if "clamp_high" in action_cfg else None,
        ),
    )


def _make_batch(batch: dict, device: torch.device) -> tuple[ControlObservationBatch, torch.Tensor]:
    obs = ControlObservationBatch(
        images=batch["images"].to(device),
        proprio=batch["proprio"].to(device),
        task_text=batch.get("task_text"),
    )
    actions = batch["actions"].to(device)
    return obs, actions


def build_continuous_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    data_cfg = cfg["data_continuous"]
    train_ds = ContinuousTrajectoryDataset(
        data_cfg["train_path"],
        history_len=int(data_cfg.get("history_len", 1)),
        action_horizon=int(data_cfg.get("action_horizon", 1)),
    )
    val_ds = ContinuousTrajectoryDataset(
        data_cfg["val_path"],
        history_len=int(data_cfg.get("history_len", 1)),
        action_horizon=int(data_cfg.get("action_horizon", 1)),
    )
    train_loader = DataLoader(train_ds, batch_size=int(cfg["train_continuous"]["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["train_continuous"]["batch_size"]), shuffle=False)
    return train_loader, val_loader


def build_continuous_backbone(cfg: dict) -> nn.Module:
    spec = _build_control_spec(cfg)
    control_cfg = cfg["control"]
    return build_control_backbone(
        control_cfg["backbone_name"],
        spec,
        official_cfg=control_cfg.get("official"),
    )


def train_continuous_backbone(config_path: str | Path) -> dict[str, float]:
    cfg = load_config(config_path)
    official_cfg = cfg.get("control", {}).get("official", {})
    if bool(official_cfg.get("enforce_official_training", True)):
        raise RuntimeError(
            "Formal ACT / Diffusion Policy / SmolVLA baselines must be trained through the official "
            "LeRobot pipeline, not through the local prototype trainer. "
            "Use scripts/launch_official_lerobot_train.py after exporting a local LeRobot dataset with "
            "scripts/export_continuous_to_lerobot.py."
        )
    device = torch.device(cfg["train_continuous"].get("device", "cpu"))
    model = build_continuous_backbone(cfg).to(device)
    if hasattr(model, "availability_error") and callable(getattr(model, "availability_error")):
        message = model.availability_error()
        if message:
            raise RuntimeError(
                "Official-policy training cannot start yet because the requested backbone "
                f"is not available in the current environment. {message}"
            )
    train_loader, val_loader = build_continuous_dataloaders(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train_continuous"]["lr"]),
        weight_decay=float(cfg["train_continuous"].get("weight_decay", 0.0)),
    )
    epochs = int(cfg["train_continuous"]["epochs"])
    ckpt_root = ensure_dir(Path(cfg["paths"]["checkpoint_dir"]) / "continuous" / cfg["control"]["backbone_name"])
    best_val = float("inf")
    history = {"train_loss": float("nan"), "val_loss": float("nan")}
    for _epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_count = 0
        for batch in tqdm(train_loader, leave=False):
            obs, target = _make_batch(batch, device)
            output = model(obs)
            loss = F.mse_loss(output.actions, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * int(target.shape[0])
            train_count += int(target.shape[0])
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                obs, target = _make_batch(batch, device)
                output = model(obs)
                loss = F.mse_loss(output.actions, target)
                val_loss += float(loss.item()) * int(target.shape[0])
                val_count += int(target.shape[0])
        history["train_loss"] = train_loss / max(1, train_count)
        history["val_loss"] = val_loss / max(1, val_count)
        if history["val_loss"] < best_val:
            best_val = history["val_loss"]
            torch.save({"model": model.state_dict(), "config": cfg}, ckpt_root / "best.pt")
    return history

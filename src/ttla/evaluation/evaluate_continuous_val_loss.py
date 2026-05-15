from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..control import ControlObservationBatch, build_control_backbone
from ..data.continuous_dataset import ContinuousTrajectoryDataset
from .evaluate_continuous import _build_interface_spec, _merge_official_eval_cfg, resolve_official_policy_path


def evaluate_continuous_validation_loss(
    cfg: dict,
    policy_path: str | Path | None,
    output_dir: str | Path,
    *,
    batch_size: int | None = None,
    max_batches: int | None = None,
    policy_device: str | None = None,
) -> Path:
    interface_spec = _build_interface_spec(cfg)
    resolved_policy_path = resolve_official_policy_path(policy_path)
    official_cfg = _merge_official_eval_cfg(cfg, resolved_policy_path, policy_device)
    backbone = build_control_backbone(cfg["control"]["backbone_name"], interface_spec, official_cfg=official_cfg)
    backbone.eval()

    data_cfg = cfg.get("data_continuous", {})
    dataset = ContinuousTrajectoryDataset(
        data_cfg["val_path"],
        history_len=int(data_cfg.get("history_len", 1)),
        action_horizon=int(data_cfg.get("action_horizon", 1)),
    )
    eval_batch_size = int(batch_size or cfg["official_train"]["batch_size"])
    loader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False)

    total_loss = 0.0
    total_first_loss = 0.0
    total_batches = 0
    total_samples = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        backbone.reset_policy_state()
        task_text = batch.get("task_text")
        if task_text is not None and not isinstance(task_text, list):
            task_text = list(task_text)
        obs = ControlObservationBatch(
            images=batch["images"].float(),
            proprio=batch["proprio"].float(),
            task_text=task_text if cfg["control"].get("uses_language", False) else None,
        )
        with torch.no_grad():
            output = backbone.forward_policy(obs)
        pred_plan = output.aux.get("planned_actions", output.actions)
        target_actions = batch["actions"].to(pred_plan.device)
        horizon = min(int(pred_plan.shape[1]), int(target_actions.shape[1]))
        pred = pred_plan[:, :horizon]
        target = target_actions[:, :horizon]
        loss = torch.mean((pred - target) ** 2)
        first_loss = torch.mean((pred[:, 0] - target[:, 0]) ** 2)
        batch_size_actual = int(pred.shape[0])
        total_loss += float(loss.item()) * batch_size_actual
        total_first_loss += float(first_loss.item()) * batch_size_actual
        total_samples += batch_size_actual
        total_batches += 1

    if total_samples == 0:
        raise RuntimeError("Validation loss evaluation saw zero samples.")

    summary = pd.DataFrame(
        [
            {
                "val_action_mse": total_loss / total_samples,
                "val_first_action_mse": total_first_loss / total_samples,
                "batches": total_batches,
                "samples": total_samples,
            }
        ]
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "val_loss_summary.csv"
    summary.to_csv(summary_path, index=False)
    return summary_path

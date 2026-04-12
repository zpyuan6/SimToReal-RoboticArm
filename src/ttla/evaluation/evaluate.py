from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..models import BaseTTLAModel
from ..sim import RoArmSimEnv
from ..sim.task_defs import TASK_TO_ID
from ..training import build_model
from .baselines import baseline_overrides


@torch.no_grad()
def _select_primitive(
    model: BaseTTLAModel,
    obs: dict,
    runtime_state,
    use_adapter: bool,
    device: torch.device,
    task_id: int,
) -> tuple[int, object]:
    image = torch.from_numpy(obs["image"]).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    state = torch.from_numpy(obs["state"]).unsqueeze(0).float().to(device)
    task_ids = torch.tensor([task_id], device=device, dtype=torch.long)
    primitive, runtime_state, _ = model.act(image, state, runtime_state, use_adapter=use_adapter, task_ids=task_ids)
    return int(primitive.item()), runtime_state


def evaluate_checkpoint(cfg: dict, checkpoint_path: str | Path, baseline_name: str, output_path: str | Path) -> Path:
    device = torch.device(cfg["train"]["device"])
    payload = torch.load(checkpoint_path, map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    env = RoArmSimEnv(cfg["sim"], seed=cfg["seed"] + 101)
    baseline_cfg = baseline_overrides(baseline_name)
    records: list[dict] = []
    for task_name in cfg["sim"]["tasks"]:
        for ep in range(cfg["evaluation"]["episodes_per_task"]):
            obs = env.reset(task_name=task_name)
            runtime_state = model.init_runtime_state(batch_size=1, device=device)
            success = 0
            info = {
                "visibility": 0.0,
                "center_error": 0.0,
                "verified": 0,
                "grasped": 0,
                "lifted": 0,
                "placed": 0,
                "ee_target_distance": float("nan"),
                "dropzone_distance": float("nan"),
            }
            ever_verified = 0
            ever_grasped = 0
            ever_lifted = 0
            ever_placed = 0
            for step in range(cfg["sim"]["episode_horizon"]):
                if baseline_cfg.get("input_norm", False):
                    obs = {**obs, "image": _normalize_image(obs["image"])}
                primitive_id, runtime_state = _select_primitive(
                    model,
                    obs,
                    runtime_state,
                    baseline_cfg.get("use_adapter", False),
                    device,
                    TASK_TO_ID[task_name],
                )
                next_obs, _, done, info = env.step(primitive_id)
                obs = next_obs
                success = info["success"]
                ever_verified = max(ever_verified, int(info.get("verified", 0)))
                ever_grasped = max(ever_grasped, int(info.get("grasped", 0)))
                ever_lifted = max(ever_lifted, int(info.get("lifted", 0)))
                ever_placed = max(ever_placed, int(info.get("placed", 0)))
                if done:
                    break
            records.append(
                {
                    "baseline": baseline_name,
                    "task": task_name,
                    "episode": ep,
                    "success": success,
                    "steps": step + 1,
                    "visibility": info["visibility"],
                    "center_error": info["center_error"],
                    "verified": ever_verified,
                    "grasped": ever_grasped,
                    "lifted": ever_lifted,
                    "placed": ever_placed,
                    "final_ee_target_distance": info["ee_target_distance"],
                    "final_dropzone_distance": info["dropzone_distance"],
                }
            )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(output, index=False)
    env.close()
    return output


def _normalize_image(image: np.ndarray) -> np.ndarray:
    x = image.astype(np.float32)
    x = (x - x.mean()) / max(x.std(), 1.0)
    x = np.clip(x * 48 + 127, 0, 255)
    return x.astype(np.uint8)

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..adaptation import OnlineContextAdapter
from ..models import TTLAModel
from ..sim import RoArmSimEnv
from ..sim.context import context_vector, sample_context
from ..sim.expert import ScriptedExpert
from .baselines import baseline_overrides


@torch.no_grad()
def _select_action(model: TTLAModel, obs: dict, context: torch.Tensor, device: torch.device) -> tuple[int, torch.Tensor]:
    image = torch.from_numpy(obs["image"]).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    state = torch.from_numpy(obs["state"]).unsqueeze(0).float().to(device)
    z = model.encode(image, state)
    logits = model.predict_action(model.adapted_latent(z, context), context)
    return int(logits.argmax(dim=-1).item()), z


def evaluate_checkpoint(cfg: dict, checkpoint_path: str | Path, baseline_name: str, output_path: str | Path) -> Path:
    device = torch.device(cfg["train"]["device"])
    payload = torch.load(checkpoint_path, map_location=device)
    model = TTLAModel(
        state_dim=cfg["model"]["state_dim"],
        action_dim=cfg["model"]["action_dim"],
        latent_dim=cfg["model"]["latent_dim"],
        context_dim=cfg["model"]["context_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    env = RoArmSimEnv(cfg["sim"], seed=cfg["seed"] + 101)
    baseline_cfg = baseline_overrides(baseline_name)
    adapter = OnlineContextAdapter(model, cfg["adaptation"], device)
    records: list[dict] = []
    expert = ScriptedExpert()
    for task_name in cfg["sim"]["tasks"]:
        for ep in range(cfg["evaluation"]["episodes_per_task"]):
            context = sample_context(env.context_cfg, env.rng)
            context["fov_bias"] *= 1.6
            context["action_gain"] = float(np.clip(context["action_gain"] * 1.12, 0.7, 1.35))
            context["joint_bias"] *= 1.5
            context["noise_std"] *= 1.4
            context["blur_sigma"] *= 1.3
            obs = env.reset(task_name=task_name, context=context)
            adapter.reset()
            if baseline_cfg.get("adapt", False):
                adapter.context = torch.from_numpy(context_vector(env.context)).unsqueeze(0).float().to(device)
            success = 0
            info = {"visibility": 0.0}
            for step in range(cfg["sim"]["episode_horizon"]):
                if baseline_cfg.get("input_norm"):
                    obs["image"] = _normalize_image(obs["image"])
                action, z = _select_action(model, obs, adapter.context, device)
                if baseline_cfg.get("fewshot") and ep < 2:
                    action = expert.act(env)
                    image = torch.from_numpy(obs["image"]).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                    state = torch.from_numpy(obs["state"]).unsqueeze(0).float().to(device)
                    z = model.encode(image, state)
                next_obs, _, done, info = env.step(action)
                next_image_t = torch.from_numpy(next_obs["image"]).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                next_state_t = torch.from_numpy(next_obs["state"]).unsqueeze(0).float().to(device)
                next_z = model.encode(next_image_t, next_state_t)
                if baseline_cfg.get("adapt", False):
                    adapter.adapt(z, action, next_z)
                obs = next_obs
                success = info["success"]
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
                }
            )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(output, index=False)
    return output


def _normalize_image(image: np.ndarray) -> np.ndarray:
    x = image.astype(np.float32)
    x = (x - x.mean()) / max(x.std(), 1.0)
    x = np.clip(x * 48 + 127, 0, 255)
    return x.astype(np.uint8)

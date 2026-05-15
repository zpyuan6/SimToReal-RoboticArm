from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from ..control import ContinuousActionSpec, ControlInterfaceSpec, ControlObservationBatch, build_control_backbone
from ..sim.continuous_env import ContinuousRoArmSimEnv


def _action_clamps_from_cfg(action_cfg: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
    low = action_cfg.get("clamp_low")
    high = action_cfg.get("clamp_high")
    low_arr = None if low is None else np.asarray(low, dtype=np.float32)
    high_arr = None if high is None else np.asarray(high, dtype=np.float32)
    return low_arr, high_arr


def _build_interface_spec(cfg: dict) -> ControlInterfaceSpec:
    control_cfg = cfg["control"]
    action_cfg = control_cfg["action"]
    action_spec = ContinuousActionSpec(
        action_dim=int(action_cfg["action_dim"]),
        horizon=int(action_cfg["horizon"]),
        control_mode=str(action_cfg.get("control_mode", "joint_delta")),
        clamp_low=tuple(action_cfg["clamp_low"]) if action_cfg.get("clamp_low") is not None else None,
        clamp_high=tuple(action_cfg["clamp_high"]) if action_cfg.get("clamp_high") is not None else None,
    )
    return ControlInterfaceSpec(
        image_shape=tuple(int(v) for v in control_cfg["image_shape"]),
        proprio_dim=int(control_cfg["proprio_dim"]),
        action_spec=action_spec,
        uses_language=bool(control_cfg.get("uses_language", False)),
    )


def _build_env(cfg: dict, seed: int) -> ContinuousRoArmSimEnv:
    action_cfg = cfg["control"]["action"]
    low, high = _action_clamps_from_cfg(action_cfg)
    return ContinuousRoArmSimEnv(
        cfg["sim"],
        seed=seed,
        action_low=low,
        action_high=high,
        control_mode=str(action_cfg.get("control_mode", "joint_delta")),
    )


def _window_indices(length: int, history_len: int) -> list[int]:
    end_pos = max(0, length - 1)
    positions = []
    for offset in range(history_len - 1, -1, -1):
        pos = max(0, end_pos - offset)
        positions.append(pos)
    return positions


def _build_observation_batch(
    obs_history: list[dict[str, np.ndarray]],
    history_len: int,
    task_text: str | None,
    uses_language: bool,
) -> ControlObservationBatch:
    indices = _window_indices(len(obs_history), history_len)
    image_stack = np.stack([obs_history[idx]["image"] for idx in indices], axis=0)
    proprio_stack = np.stack([obs_history[idx]["state"] for idx in indices], axis=0)
    images = torch.from_numpy(image_stack).permute(0, 3, 1, 2).unsqueeze(0).float() / 255.0
    proprio = torch.from_numpy(proprio_stack).unsqueeze(0).float()
    texts = [str(task_text or "")]
    return ControlObservationBatch(
        images=images,
        proprio=proprio,
        task_text=texts if uses_language else None,
    )


def _merge_official_eval_cfg(cfg: dict, policy_path: str | None, policy_device: str | None) -> dict:
    official_cfg = dict(cfg["control"].get("official", {}))
    if policy_path:
        official_cfg["policy_path"] = str(policy_path)
    if policy_device:
        official_cfg["policy_device"] = str(policy_device)
    else:
        default_device = cfg.get("official_train", {}).get("policy_device")
        if default_device:
            official_cfg["policy_device"] = str(default_device)
    return official_cfg


def resolve_official_policy_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"Policy path does not exist: {candidate}")
    if candidate.is_file() and candidate.name == "model.safetensors":
        return str(candidate.parent)
    if candidate.is_dir():
        if (candidate / "model.safetensors").exists() and (candidate / "config.json").exists():
            return str(candidate)
        if (candidate / "pretrained_model" / "model.safetensors").exists():
            return str(candidate / "pretrained_model")
        if (candidate / "checkpoints" / "last" / "pretrained_model" / "model.safetensors").exists():
            return str(candidate / "checkpoints" / "last" / "pretrained_model")
        if (candidate / "last" / "pretrained_model" / "model.safetensors").exists():
            return str(candidate / "last" / "pretrained_model")
    raise FileNotFoundError(
        "Could not resolve an official policy directory from path: "
        f"{candidate}. Expected either a pretrained_model directory, a model.safetensors file, "
        "or a LeRobot training output root containing checkpoints/last/pretrained_model."
    )


def _summarize_records(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "success",
        "steps",
        "visibility",
        "center_error",
        "verified",
        "grasped",
        "lifted",
        "placed",
        "final_ee_target_distance",
        "final_grasp_gap",
        "final_dropzone_distance",
    ]
    task_summary = df.groupby("task", as_index=False)[numeric_cols].mean()
    task_summary.insert(0, "split", "task")
    overall = {
        "split": "overall",
        "task": "overall",
    }
    for col in numeric_cols:
        overall[col] = float(df[col].mean())
    return pd.concat([task_summary, pd.DataFrame([overall])], ignore_index=True)


def evaluate_continuous_backbone(
    cfg: dict,
    policy_path: str | Path | None,
    output_dir: str | Path,
    *,
    episodes_per_task: int | None = None,
    policy_device: str | None = None,
    tasks: Iterable[str] | None = None,
    seed: int | None = None,
) -> tuple[Path, Path]:
    interface_spec = _build_interface_spec(cfg)
    resolved_policy_path = resolve_official_policy_path(policy_path)
    official_cfg = _merge_official_eval_cfg(cfg, resolved_policy_path, policy_device)
    backbone = build_control_backbone(cfg["control"]["backbone_name"], interface_spec, official_cfg=official_cfg)
    backbone.eval()

    env_seed = int(cfg["seed"] if seed is None else seed)
    env = _build_env(cfg, seed=env_seed + 101)
    history_len = int(cfg.get("data_continuous", {}).get("history_len", 1))
    task_names = list(tasks) if tasks is not None else list(cfg["sim"]["tasks"])
    num_episodes = int(episodes_per_task or cfg["evaluation"]["episodes_per_task"])

    records: list[dict[str, float | int | str]] = []
    for task_name in task_names:
        for episode in range(num_episodes):
            obs = env.reset(task_name=task_name)
            obs_history = [obs]
            backbone.reset_policy_state()
            total_reward = 0.0
            info: dict = {
                "visibility": 0.0,
                "center_error": 0.0,
                "verified": 0,
                "grasped": 0,
                "lifted": 0,
                "placed": 0,
                "ee_target_distance": float("nan"),
                "grasp_gap": float("nan"),
                "dropzone_distance": float("nan"),
            }
            success = 0
            for step in range(int(cfg["sim"]["episode_horizon"])):
                batch = _build_observation_batch(
                    obs_history,
                    history_len=history_len,
                    task_text=env.task_text(),
                    uses_language=interface_spec.uses_language,
                )
                with torch.no_grad():
                    policy_output = backbone.forward_policy(batch)
                action = policy_output.actions[0, 0].detach().cpu().numpy().astype(np.float32)
                next_obs, reward, done, info = env.step_action(action)
                total_reward += float(reward)
                obs_history.append(next_obs)
                success = int(info["success"])
                if done:
                    break
            records.append(
                {
                    "backbone": cfg["control"]["backbone_name"],
                    "task": task_name,
                    "episode": episode,
                    "success": success,
                    "steps": step + 1,
                    "reward": total_reward,
                    "visibility": float(info.get("visibility", 0.0)),
                    "center_error": float(info.get("center_error", 0.0)),
                    "verified": int(info.get("verified", 0)),
                    "grasped": int(info.get("grasped", 0)),
                    "lifted": int(info.get("lifted", 0)),
                    "placed": int(info.get("placed", 0)),
                    "final_ee_target_distance": float(info.get("ee_target_distance", float("nan"))),
                    "final_grasp_gap": float(info.get("grasp_gap", float("nan"))),
                    "final_dropzone_distance": float(info.get("dropzone_distance", float("nan"))),
                }
            )

    env.close()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    episodes_path = output_root / "episodes.csv"
    summary_path = output_root / "summary.csv"
    df = pd.DataFrame.from_records(records)
    df.to_csv(episodes_path, index=False)
    _summarize_records(df).to_csv(summary_path, index=False)
    return episodes_path, summary_path

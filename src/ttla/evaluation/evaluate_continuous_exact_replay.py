from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np
import pandas as pd
import torch

from ..sim.context import context_from_full_vector
from ..sim.task_defs import ID_TO_TASK
from .evaluate_continuous import (
    _build_env,
    _build_interface_spec,
    _build_observation_batch,
    _merge_official_eval_cfg,
    _summarize_records,
    resolve_official_policy_path,
)
from ..control import build_control_backbone


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: int
    indices: np.ndarray
    task_id: int
    task_name: str
    task_text: str
    context_full: np.ndarray
    target_init_pos: np.ndarray
    drop_init_pos: np.ndarray
    episode_success: int


def _load_records(payload: np.lib.npyio.NpzFile, task_filter: set[str]) -> list[EpisodeRecord]:
    required = {
        "images",
        "actions",
        "tasks",
        "success",
        "episode_ids",
        "step_ids",
        "contexts_full",
        "target_init_pos",
        "drop_init_pos",
    }
    missing = sorted(required.difference(payload.files))
    if missing:
        raise KeyError(
            "Dataset does not contain replay metadata required for exact closed-loop replay. "
            f"Missing fields: {missing}"
        )
    episode_ids = payload["episode_ids"].astype(np.int64)
    step_ids = payload["step_ids"].astype(np.int64)
    tasks = payload["tasks"].astype(np.int64)
    success = payload["success"].astype(np.int64)
    task_text = payload.get("task_text")
    order = np.lexsort((step_ids, episode_ids))
    unique_episodes: list[int] = []
    seen: set[int] = set()
    for episode_id in episode_ids[order]:
        episode_id = int(episode_id)
        if episode_id not in seen:
            unique_episodes.append(episode_id)
            seen.add(episode_id)
    records: list[EpisodeRecord] = []
    for episode_id in unique_episodes:
        indices = np.flatnonzero(episode_ids == episode_id)
        indices = indices[np.argsort(step_ids[indices])]
        task_id = int(tasks[indices[0]])
        task_name = ID_TO_TASK[task_id].name
        if task_filter and task_name not in task_filter:
            continue
        records.append(
            EpisodeRecord(
                episode_id=episode_id,
                indices=indices,
                task_id=task_id,
                task_name=task_name,
                task_text=str(task_text[indices[0]]) if task_text is not None else task_name,
                context_full=np.asarray(payload["contexts_full"][indices[0]], dtype=np.float32),
                target_init_pos=np.asarray(payload["target_init_pos"][indices[0]], dtype=np.float32),
                drop_init_pos=np.asarray(payload["drop_init_pos"][indices[0]], dtype=np.float32),
                episode_success=int(np.max(success[indices])),
            )
        )
    if not records:
        raise RuntimeError("No episodes left after task filtering.")
    return records


def _sample_records_per_task(
    records: list[EpisodeRecord],
    *,
    num_per_task: int,
    rng: np.random.Generator,
) -> list[EpisodeRecord]:
    if num_per_task <= 0:
        return records
    by_task: dict[str, list[EpisodeRecord]] = {}
    for record in records:
        by_task.setdefault(record.task_name, []).append(record)
    sampled: list[EpisodeRecord] = []
    for task_name in sorted(by_task):
        task_records = by_task[task_name]
        if len(task_records) < num_per_task:
            raise RuntimeError(
                f"Requested {num_per_task} episodes for task {task_name}, but only {len(task_records)} are available."
            )
        chosen = rng.choice(len(task_records), size=num_per_task, replace=False)
        sampled.extend(task_records[int(index)] for index in chosen)
    order = rng.permutation(len(sampled))
    return [sampled[int(index)] for index in order]


def _restore_episode(env, record: EpisodeRecord) -> None:
    context = context_from_full_vector(record.context_full)
    env.reset(task_name=record.task_name, context=context)
    target_body = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "target")
    drop_body = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "distractor")
    env.model.body_pos[target_body] = np.asarray(record.target_init_pos, dtype=np.float64)
    env.model.body_pos[drop_body] = np.asarray(record.drop_init_pos, dtype=np.float64)
    mujoco.mj_forward(env.model, env.data)


def evaluate_continuous_exact_replay_backbone(
    cfg: dict,
    policy_path: str | Path | None,
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    policy_device: str | None = None,
    tasks: Iterable[str] | None = None,
    seed: int | None = None,
    num_per_task: int = 0,
) -> tuple[Path, Path]:
    interface_spec = _build_interface_spec(cfg)
    resolved_policy_path = resolve_official_policy_path(policy_path)
    official_cfg = _merge_official_eval_cfg(cfg, resolved_policy_path, policy_device)
    backbone = build_control_backbone(cfg["control"]["backbone_name"], interface_spec, official_cfg=official_cfg)
    backbone.eval()

    payload = np.load(Path(dataset_path), allow_pickle=True)
    task_filter = set(tasks or [])
    rng = np.random.default_rng(int(cfg["seed"] if seed is None else seed))
    records = _sample_records_per_task(
        _load_records(payload, task_filter),
        num_per_task=int(num_per_task),
        rng=rng,
    )

    env_seed = int(cfg["seed"] if seed is None else seed)
    env = _build_env(cfg, seed=env_seed + 303)
    history_len = int(cfg.get("data_continuous", {}).get("history_len", 1))
    sim_horizon = int(cfg["sim"]["episode_horizon"])

    rows: list[dict[str, float | int | str]] = []
    for record in records:
        _restore_episode(env, record)
        obs = env.observe()
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
            "ee_ear_center_distance": float("nan"),
            "ee_target_distance": float("nan"),
            "grasp_gap": float("nan"),
            "dropzone_distance": float("nan"),
        }
        success = 0
        max_steps = max(sim_horizon, len(record.indices))
        for step in range(max_steps):
            batch = _build_observation_batch(
                obs_history,
                history_len=history_len,
                task_text=record.task_text,
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

        rows.append(
            {
                "backbone": cfg["control"]["backbone_name"],
                "task": record.task_name,
                "episode": record.episode_id,
                "stored_episode_success": record.episode_success,
                "stored_steps": len(record.indices),
                "success": success,
                "steps": step + 1,
                "reward": total_reward,
                "visibility": float(info.get("visibility", 0.0)),
                "center_error": float(info.get("center_error", 0.0)),
                "verified": int(info.get("verified", 0)),
                "grasped": int(info.get("grasped", 0)),
                "lifted": int(info.get("lifted", 0)),
                "placed": int(info.get("placed", 0)),
                "final_ee_ear_center_distance": float(info.get("ee_ear_center_distance", float("nan"))),
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
    df = pd.DataFrame.from_records(rows)
    df.to_csv(episodes_path, index=False)
    _summarize_records(df).to_csv(summary_path, index=False)
    return episodes_path, summary_path

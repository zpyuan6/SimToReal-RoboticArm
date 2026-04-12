from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ttla.config import load_config
from ttla.sim import RoArmSimEnv, ScriptedExpert
from ttla.utils.io import ensure_dir, save_npz


def _prepare_level3_episode(env: RoArmSimEnv, episode: int, retries: int = 6) -> None:
    modes = ("default", "pregrasp", "grasped", "lifted", "near_dropzone")
    target_mode = modes[episode % len(modes)]
    for _ in range(retries):
        env.reset(task_name="level3_pick_place")
        if target_mode == "default":
            return
        env._execute_pregrasp_servo()
        if target_mode == "pregrasp":
            return
        env._execute_grasp()
        if not env.object_attached:
            continue
        if target_mode == "grasped":
            return
        env._execute_lift()
        if target_mode == "lifted":
            return
        env._execute_transport()
        return
    env.reset(task_name="level3_pick_place")


def _prepare_episode(env: RoArmSimEnv, task_name: str, episode: int) -> None:
    if task_name == "level3_pick_place":
        _prepare_level3_episode(env, episode)
        return
    env.reset(task_name=task_name)


def collect_split(env: RoArmSimEnv, expert: ScriptedExpert, episodes: int) -> dict[str, np.ndarray]:
    images = []
    states = []
    primitive_ids = []
    next_images = []
    next_states = []
    tasks = []
    success = []
    contexts = []
    episode_ids = []
    step_ids = []
    task_names = env.cfg["tasks"]
    for episode in range(episodes):
        task_name = task_names[episode % len(task_names)]
        _prepare_episode(env, task_name, episode)
        done = False
        step_idx = 0
        while not done:
            primitive_id = expert.act(env)
            _, _, done, info = env.step(primitive_id)
            transition = info["transition"]
            images.append(transition.image)
            states.append(transition.state)
            primitive_ids.append(transition.primitive_id)
            next_images.append(transition.next_image)
            next_states.append(transition.next_state)
            tasks.append(transition.task_id)
            success.append(transition.success)
            contexts.append(transition.context)
            episode_ids.append(episode)
            step_ids.append(step_idx)
            step_idx += 1
    return {
        "images": np.asarray(images, dtype=np.uint8),
        "states": np.asarray(states, dtype=np.float32),
        "primitive_ids": np.asarray(primitive_ids, dtype=np.int64),
        "next_images": np.asarray(next_images, dtype=np.uint8),
        "next_states": np.asarray(next_states, dtype=np.float32),
        "tasks": np.asarray(tasks, dtype=np.int64),
        "success": np.asarray(success, dtype=np.int64),
        "contexts": np.asarray(contexts, dtype=np.float32),
        "episode_ids": np.asarray(episode_ids, dtype=np.int64),
        "step_ids": np.asarray(step_ids, dtype=np.int64),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    np.random.seed(cfg["seed"])
    env = RoArmSimEnv(cfg["sim"], seed=cfg["seed"])
    expert = ScriptedExpert()
    data_root = ensure_dir(cfg["paths"]["data_root"])
    split_sizes = {
        "train": cfg["data"]["train_episodes"],
        "val": cfg["data"]["val_episodes"],
        "test": cfg["data"]["test_episodes"],
    }
    for split, count in split_sizes.items():
        payload = collect_split(env, expert, count)
        save_npz(Path(data_root) / f"{split}.npz", **payload)
    env.close()


if __name__ == "__main__":
    main()

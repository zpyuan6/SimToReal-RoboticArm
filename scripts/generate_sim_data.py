from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ttla.config import load_config
from ttla.sim import RoArmSimEnv, ScriptedExpert
from ttla.utils.io import ensure_dir, save_npz


def collect_split(env: RoArmSimEnv, expert: ScriptedExpert, episodes: int) -> dict[str, np.ndarray]:
    images = []
    states = []
    actions = []
    next_images = []
    next_states = []
    tasks = []
    success = []
    contexts = []
    task_names = env.cfg["tasks"]
    for episode in range(episodes):
        task_name = task_names[episode % len(task_names)]
        _ = env.reset(task_name=task_name)
        done = False
        while not done:
            action = expert.act(env)
            _, _, done, info = env.step(action)
            transition = info["transition"]
            images.append(transition.image)
            states.append(transition.state)
            actions.append(transition.action)
            next_images.append(transition.next_image)
            next_states.append(transition.next_state)
            tasks.append(transition.task_id)
            success.append(transition.success)
            contexts.append(transition.context)
    return {
        "images": np.asarray(images, dtype=np.uint8),
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "next_images": np.asarray(next_images, dtype=np.uint8),
        "next_states": np.asarray(next_states, dtype=np.float32),
        "tasks": np.asarray(tasks, dtype=np.int64),
        "success": np.asarray(success, dtype=np.int64),
        "contexts": np.asarray(contexts, dtype=np.float32),
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


if __name__ == "__main__":
    main()

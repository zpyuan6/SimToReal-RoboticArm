from __future__ import annotations

import argparse
from pathlib import Path

from ttla.config import load_config
from ttla.sim import ContinuousRoArmSimEnv, ContinuousWaypointExpert
from ttla.sim.context import neutral_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit continuous expert success rates in MuJoCo.")
    parser.add_argument("--config", default="configs/continuous_act_template.yaml")
    parser.add_argument("--episodes-per-task", type=int, default=8)
    parser.add_argument("--context-mode", choices=("neutral", "random"), default="neutral")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    env = ContinuousRoArmSimEnv(
        cfg["sim"],
        seed=int(cfg.get("seed", 7)),
        action_low=cfg["control"]["action"].get("clamp_low"),
        action_high=cfg["control"]["action"].get("clamp_high"),
        control_mode=cfg["control"]["action"].get("control_mode", "joint_delta"),
    )
    expert = ContinuousWaypointExpert()
    print("task,successes,episodes,success_rate,avg_steps")
    for task_name in cfg["sim"]["tasks"]:
        successes = 0
        total_steps = 0
        for _ in range(int(args.episodes_per_task)):
            context = neutral_context() if args.context_mode == "neutral" else None
            env.reset(task_name=task_name, context=context)
            expert.reset(task_name=task_name)
            done = False
            episode_success = 0
            while not done:
                action = expert.act(env)
                _, _, done, info = env.step_action(action)
                episode_success = max(episode_success, int(info["success"]))
            successes += episode_success
            total_steps += int(env.step_idx)
        print(
            f"{task_name},{successes},{args.episodes_per_task},"
            f"{successes / max(1, int(args.episodes_per_task)):.3f},"
            f"{total_steps / max(1, int(args.episodes_per_task)):.2f}"
        )
    env.close()


if __name__ == "__main__":
    main()

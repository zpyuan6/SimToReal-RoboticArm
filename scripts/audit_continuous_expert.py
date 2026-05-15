from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ttla.config import load_config
from ttla.sim import ContinuousRoArmSimEnv, ContinuousWaypointExpert
from ttla.sim.context import neutral_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit continuous expert success rates in MuJoCo.")
    parser.add_argument("--config", default="configs/continuous_act_template.yaml")
    parser.add_argument("--episodes-per-task", type=int, default=8)
    parser.add_argument("--context-mode", choices=("neutral", "random"), default="neutral")
    parser.add_argument("--seed", type=int, help="Override simulation seed.")
    parser.add_argument("--summary-csv", help="Optional path to write per-task summary CSV.")
    parser.add_argument("--episodes-csv", help="Optional path to write per-episode audit CSV.")
    parser.add_argument(
        "--require-success-rate",
        type=float,
        help="Fail with non-zero exit if any task falls below this success rate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 7))
    env = ContinuousRoArmSimEnv(
        cfg["sim"],
        seed=seed,
        action_low=cfg["control"]["action"].get("clamp_low"),
        action_high=cfg["control"]["action"].get("clamp_high"),
        control_mode=cfg["control"]["action"].get("control_mode", "joint_delta"),
    )
    expert = ContinuousWaypointExpert()
    task_rows: list[dict[str, float | int | str]] = []
    episode_rows: list[dict[str, float | int | str]] = []
    print("task,successes,episodes,success_rate,avg_steps")
    for task_name in cfg["sim"]["tasks"]:
        successes = 0
        total_steps = 0
        for episode_idx in range(int(args.episodes_per_task)):
            context = neutral_context() if args.context_mode == "neutral" else None
            env.reset(task_name=task_name, context=context)
            expert.reset(task_name=task_name)
            done = False
            episode_success = 0
            last_info = {}
            while not done:
                action = expert.act(env)
                _, _, done, info = env.step_action(action)
                last_info = info
                episode_success = max(episode_success, int(info["success"]))
            successes += episode_success
            total_steps += int(env.step_idx)
            episode_rows.append(
                {
                    "task": task_name,
                    "episode_idx": episode_idx,
                    "success": episode_success,
                    "steps": int(env.step_idx),
                    "visibility": float(last_info.get("visibility", 0.0)),
                    "center_error": float(last_info.get("center_error", 0.0)),
                    "verified": int(last_info.get("verified", 0)),
                    "grasped": int(last_info.get("grasped", 0)),
                    "lifted": int(last_info.get("lifted", 0)),
                    "placed": int(last_info.get("placed", 0)),
                    "ee_target_distance": float(last_info.get("ee_target_distance", 0.0)),
                    "grasp_gap": float(last_info.get("grasp_gap", 0.0)),
                    "dropzone_distance": float(last_info.get("dropzone_distance", 0.0)),
                    "gripper_intrusion_ratio": float(env.gripper_intrusion_ratio()),
                    "target_occlusion_ratio": float(env.target_occlusion_ratio()),
                }
            )
        success_rate = successes / max(1, int(args.episodes_per_task))
        avg_steps = total_steps / max(1, int(args.episodes_per_task))
        print(f"{task_name},{successes},{args.episodes_per_task},{success_rate:.3f},{avg_steps:.2f}")
        task_rows.append(
            {
                "task": task_name,
                "successes": successes,
                "episodes": int(args.episodes_per_task),
                "success_rate": success_rate,
                "avg_steps": avg_steps,
            }
        )
    env.close()
    if args.summary_csv:
        summary_path = Path(args.summary_csv)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["task", "successes", "episodes", "success_rate", "avg_steps"])
            writer.writeheader()
            writer.writerows(task_rows)
    if args.episodes_csv:
        episodes_path = Path(args.episodes_csv)
        episodes_path.parent.mkdir(parents=True, exist_ok=True)
        with episodes_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "task",
                    "episode_idx",
                    "success",
                    "steps",
                    "visibility",
                    "center_error",
                    "verified",
                    "grasped",
                    "lifted",
                    "placed",
                    "ee_target_distance",
                    "grasp_gap",
                    "dropzone_distance",
                    "gripper_intrusion_ratio",
                    "target_occlusion_ratio",
                ],
            )
            writer.writeheader()
            writer.writerows(episode_rows)
    if args.require_success_rate is not None:
        threshold = float(args.require_success_rate)
        failing = [row for row in task_rows if float(row["success_rate"]) < threshold]
        if failing:
            failures = ", ".join(f"{row['task']}={float(row['success_rate']):.3f}" for row in failing)
            raise SystemExit(f"continuous expert audit failed threshold {threshold:.3f}: {failures}")


if __name__ == "__main__":
    main()

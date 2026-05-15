from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from sim_demo_common import (
    ACCENT,
    SUCCESS,
    WARN,
    add_shared_args,
    panel_frame,
    save_contact_sheet,
    save_video,
    write_manifest,
)
from ttla.config import load_config
from ttla.sim import ContinuousRoArmSimEnv, ContinuousWaypointExpert
from ttla.sim.context import neutral_context
from ttla.sim.task_defs import TASK_SPECS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render continuous teacher rollouts for each task and save videos/contact sheets."
    )
    add_shared_args(parser, default_output_root="results/continuous_teacher_demos")
    parser.add_argument(
        "--tasks",
        default="level1_verify,level2_approach,level3_pick_place",
        help="Comma-separated task names.",
    )
    parser.add_argument("--context-mode", choices=("neutral", "random"), default="neutral")
    parser.add_argument("--max-attempts", type=int, default=20)
    return parser.parse_args()


def _extra_lines(action: np.ndarray, info: dict, attempt: int, step_idx: int) -> list[str]:
    return [
        f"attempt={attempt}",
        f"teacher_step={step_idx}",
        f"action=[{', '.join(f'{float(v):+.3f}' for v in action)}]",
        f"task_success={int(info['success'])}",
        f"grasp_gap={float(info['grasp_gap']):+.4f}",
        f"ee_target_distance={float(info['ee_target_distance']):.4f}",
        f"dropzone_distance={float(info['dropzone_distance']):.4f}",
        (
            f"flags verified={int(info['verified'])} grasped={int(info['grasped'])} "
            f"lifted={int(info['lifted'])} placed={int(info['placed'])}"
        ),
    ]


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    env = ContinuousRoArmSimEnv(
        cfg["sim"],
        seed=int(args.seed),
        action_low=np.asarray(cfg["control"]["action"].get("clamp_low", [-0.25] * 6), dtype=np.float32),
        action_high=np.asarray(cfg["control"]["action"].get("clamp_high", [0.25] * 6), dtype=np.float32),
        control_mode=cfg["control"]["action"].get("control_mode", "joint_delta"),
    )
    expert = ContinuousWaypointExpert()
    output_root = Path(args.output_root)
    tasks = [token.strip() for token in args.tasks.split(",") if token.strip()]
    manifest: dict[str, object] = {
        "config": args.config,
        "seed": args.seed,
        "fps": args.fps,
        "context_mode": args.context_mode,
        "tasks": [],
    }

    try:
        for task_name in tasks:
            if task_name not in TASK_SPECS:
                raise KeyError(f"Unknown task: {task_name}")
            task_dir = output_root / task_name
            task_dir.mkdir(parents=True, exist_ok=True)
            final_info = None
            final_steps: list[dict[str, object]] = []
            dashboard_frames: list[np.ndarray] = []
            forearm_frames: list[np.ndarray] = []
            overview_frames: list[np.ndarray] = []
            accepted = False
            for attempt in range(1, int(args.max_attempts) + 1):
                context = neutral_context() if args.context_mode == "neutral" else None
                obs = env.reset(task_name=task_name, context=context)
                expert.reset(task_name=task_name)
                dashboard_frames = []
                forearm_frames = [obs["image"][:, :, ::-1].copy()]
                overview_frames = [env.render_debug_view("overview_cam")[:, :, ::-1].copy()]
                final_steps = []
                done = False
                while not done:
                    action = expert.act(env)
                    next_obs, reward, done, info = env.step_action(action)
                    final_steps.append(
                        {
                            "step_idx": int(env.step_idx),
                            "action": [float(v) for v in action],
                            "reward": float(reward),
                            "done": bool(done),
                            "success": int(info["success"]),
                            "visibility": float(info["visibility"]),
                            "center_error": float(info["center_error"]),
                            "verified": int(info["verified"]),
                            "grasped": int(info["grasped"]),
                            "lifted": int(info["lifted"]),
                            "placed": int(info["placed"]),
                            "ee_target_distance": float(info["ee_target_distance"]),
                            "grasp_gap": float(info["grasp_gap"]),
                            "dropzone_distance": float(info["dropzone_distance"]),
                        }
                    )
                    dashboard_frames.append(
                        panel_frame(
                            env,
                            next_obs,
                            title=f"Continuous Teacher: {task_name}",
                            subtitle=f"task={task_name} | attempt={attempt}",
                            lines=_extra_lines(action, info, attempt, int(env.step_idx)),
                            status_label="SUCCESS" if info["success"] else "DONE" if done else "RUNNING",
                            status_color=SUCCESS if info["success"] else WARN if done else ACCENT,
                        )
                    )
                    forearm_frames.append(next_obs["image"][:, :, ::-1].copy())
                    overview_frames.append(env.render_debug_view("overview_cam")[:, :, ::-1].copy())
                    final_info = info
                if final_info is not None and int(final_info["success"]) == 1:
                    accepted = True
                    break
            if not accepted:
                raise RuntimeError(f"Teacher could not produce a successful {task_name} demo within {args.max_attempts} attempts.")

            save_video(task_dir / "dashboard.mp4", dashboard_frames, fps=args.fps)
            save_video(task_dir / "forearm_cam.mp4", forearm_frames, fps=args.fps)
            save_video(task_dir / "overview_cam.mp4", overview_frames, fps=args.fps)
            save_contact_sheet(task_dir / "dashboard_contact_sheet.png", dashboard_frames, labels=["start", "mid", "final"])
            save_contact_sheet(task_dir / "forearm_contact_sheet.png", forearm_frames, labels=["start", "mid", "final"])
            save_contact_sheet(task_dir / "overview_contact_sheet.png", overview_frames, labels=["start", "mid", "final"])
            write_manifest(
                task_dir / "meta.json",
                {
                    "task": task_name,
                    "task_public_label": TASK_SPECS[task_name].public_label,
                    "context_mode": args.context_mode,
                    "successful_attempt": attempt,
                    "final_success": int(final_info["success"]) if final_info is not None else 0,
                    "final_flags": {
                        "verified": int(final_info["verified"]) if final_info is not None else 0,
                        "grasped": int(final_info["grasped"]) if final_info is not None else 0,
                        "lifted": int(final_info["lifted"]) if final_info is not None else 0,
                        "placed": int(final_info["placed"]) if final_info is not None else 0,
                    },
                    "step_summaries": final_steps,
                },
            )
            manifest["tasks"].append(
                {
                    "task": task_name,
                    "successful_attempt": attempt,
                    "output_dir": str(task_dir),
                    "final_success": int(final_info["success"]) if final_info is not None else 0,
                }
            )
    finally:
        env.close()

    write_manifest(output_root / "manifest.json", manifest)
    print(f"saved={output_root}")


if __name__ == "__main__":
    main()

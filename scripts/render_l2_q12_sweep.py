from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mujoco
import numpy as np

from ttla.config import load_config
from ttla.sim import ContinuousRoArmSimEnv, ContinuousWaypointExpert
from ttla.sim.context import neutral_context
from ttla.sim.skills import PREGRASP_QPOS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render L2 q1/q2 pregrasp sweep reference sheet.")
    parser.add_argument("--config", default="configs/continuous_act_preview_jointtarget.yaml")
    parser.add_argument(
        "--center-mode",
        choices=("l1", "pregrasp"),
        default="pregrasp",
        help="Center the sweep around the completed L1 pose or the current L2 pregrasp solve.",
    )
    parser.add_argument(
        "--q1-offsets",
        default="-0.08,0.00,0.08",
        help="Comma-separated q1 offsets around the chosen center pose.",
    )
    parser.add_argument(
        "--q2-offsets",
        default="-0.12,-0.06,0.00,0.06,0.12",
        help="Comma-separated q2 offsets around the chosen center pose.",
    )
    parser.add_argument(
        "--output",
        default="results/l2_q12_sweep/l2_q12_sweep.png",
        help="Output image path.",
    )
    parser.add_argument("--width", type=int, default=280)
    parser.add_argument("--height", type=int, default=180)
    return parser.parse_args()


def draw_label(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (245, 245, 245), thickness=-1)
    cv2.putText(canvas, title, (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (45, 45, 45), 1, cv2.LINE_AA)
    return canvas


def render_pose(env: ContinuousRoArmSimEnv, qpos: np.ndarray, width: int, height: int) -> np.ndarray:
    env.data.qpos[:6] = np.asarray(qpos, dtype=np.float64)
    env.data.ctrl[:6] = np.asarray(qpos, dtype=np.float64)
    mujoco.mj_forward(env.model, env.data)
    overview = env.render_debug_view("overview_cam")
    forearm = env.render_debug_view("forearm_cam")
    overview = cv2.resize(overview, (width, height), interpolation=cv2.INTER_AREA)
    forearm = cv2.resize(forearm, (width, height), interpolation=cv2.INTER_AREA)
    return np.concatenate([overview, forearm], axis=0)


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
    env.reset(task_name="level2_approach", context=neutral_context())
    expert.reset(task_name="level2_approach")

    # First reproduce the current "complete L1, then approach" starting point.
    for _ in range(24):
        action = expert._level1_action(env)
        _, _, _, _ = env.step_action(action)
        if expert._level1_stage_complete(env):
            break

    if args.center_mode == "pregrasp":
        base_qpos = expert._solve_level2_target_qpos(env, PREGRASP_QPOS, near=True).astype(np.float64).copy()
    else:
        base_qpos = env.data.qpos[:6].astype(np.float64).copy()
        base_qpos[3] = -0.25
        base_qpos[4] = 0.0

    q1_offsets = [float(item) for item in str(args.q1_offsets).split(",") if item.strip()]
    q2_offsets = [float(item) for item in str(args.q2_offsets).split(",") if item.strip()]

    rows: list[np.ndarray] = []
    for q1_delta in q1_offsets:
        tiles: list[np.ndarray] = []
        for q2_delta in q2_offsets:
            qpos = base_qpos.copy()
            qpos[1] = float(np.clip(base_qpos[1] + q1_delta, float(env.target_low[1]), float(env.target_high[1])))
            qpos[2] = float(np.clip(base_qpos[2] + q2_delta, float(env.target_low[2]), float(env.target_high[2])))
            image = render_pose(env, qpos, args.width, args.height)
            ee = env._ee_position()
            ear = env._target_ear_center_position()
            dist = float(np.linalg.norm(ee - ear))
            labeled = draw_label(
                image,
                f"q1={qpos[1]:.3f}, q2={qpos[2]:.3f}",
                f"d={dist:.3f}",
            )
            tiles.append(labeled)
        rows.append(np.concatenate(tiles, axis=1))

    sheet = np.concatenate(rows, axis=0)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet[:, :, ::-1])
    print(output_path)


if __name__ == "__main__":
    main()

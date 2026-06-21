from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mujoco
import numpy as np

from ttla.config import load_config
from ttla.sim import ContinuousRoArmSimEnv
from ttla.sim.context import neutral_context
from ttla.sim.skills import APPROACH_QPOS, PREGRASP_QPOS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render L2 q3 sweep reference sheet.")
    parser.add_argument("--config", default="configs/continuous_act_preview_jointtarget.yaml")
    parser.add_argument(
        "--q3-values",
        default="-1.00,-0.85,-0.70,-0.55,-0.40,-0.25",
        help="Comma-separated wrist-pitch (q3) values to compare.",
    )
    parser.add_argument(
        "--output",
        default="results/l2_q3_sweep/l2_q3_sweep.png",
        help="Output image path.",
    )
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=220)
    return parser.parse_args()


def draw_label(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (245, 245, 245), thickness=-1)
    cv2.putText(canvas, title, (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (45, 45, 45), 1, cv2.LINE_AA)
    return canvas


def render_pose(env: ContinuousRoArmSimEnv, qpos: np.ndarray, width: int, height: int) -> np.ndarray:
    env.data.qpos[:6] = np.asarray(qpos, dtype=np.float64)
    env.data.ctrl[:6] = np.asarray(qpos, dtype=np.float64)
    mujoco.mj_forward(env.model, env.data)
    image = env.render_debug_view("overview_cam")
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


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

    env.reset(task_name="level2_approach", context=neutral_context())

    q3_values = [float(item) for item in str(args.q3_values).split(",") if item.strip()]
    rows: list[np.ndarray] = []
    pose_specs = [
        ("APPROACH", np.asarray(APPROACH_QPOS, dtype=np.float64).copy()),
        ("PREGRASP", np.asarray(PREGRASP_QPOS, dtype=np.float64).copy()),
    ]

    for row_name, base_qpos in pose_specs:
        tiles: list[np.ndarray] = []
        for q3 in q3_values:
            qpos = base_qpos.copy()
            qpos[3] = q3
            image = render_pose(env, qpos, args.width, args.height)
            labeled = draw_label(image, row_name, f"q3={q3:.2f}")
            tiles.append(labeled)
        rows.append(np.concatenate(tiles, axis=1))

    sheet = np.concatenate(rows, axis=0)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet[:, :, ::-1])
    print(output_path)


if __name__ == "__main__":
    main()

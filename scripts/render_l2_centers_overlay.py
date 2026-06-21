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
    parser = argparse.ArgumentParser(description="Render L2 center overlays for ee center and ear center.")
    parser.add_argument("--config", default="configs/continuous_act_preview_jointtarget.yaml")
    parser.add_argument(
        "--output",
        default="results/l2_centers_overlay/l2_centers_overlay.png",
        help="Output image path.",
    )
    parser.add_argument("--width", type=int, default=360)
    parser.add_argument("--height", type=int, default=240)
    return parser.parse_args()


def project_camera(
    env: ContinuousRoArmSimEnv,
    camera_name: str,
    position: np.ndarray,
) -> tuple[bool, int, int]:
    image_size = int(env.cfg["image_size"])
    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_name == "forearm_cam":
        cam_pos, cam_rot = env._camera_pose()
        fovy = np.deg2rad(float(env.model.cam_fovy[cam_id]) + env.context["fov_bias"])
    else:
        cam_pos = env.data.cam_xpos[cam_id].copy()
        cam_rot = env.data.cam_xmat[cam_id].reshape(3, 3).copy()
        fovy = np.deg2rad(float(env.model.cam_fovy[cam_id]))
    rel = position - cam_pos
    cam_rel = cam_rot.T @ rel
    depth = -cam_rel[2]
    if depth <= 1e-6:
        return False, 0, 0
    tan_half = max(np.tan(fovy / 2.0), 1e-6)
    u = 0.5 + cam_rel[0] / (2.0 * depth * tan_half)
    v = 0.5 - cam_rel[1] / (2.0 * depth * tan_half)
    px = int(np.clip(u * image_size, 0, image_size - 1))
    py = int(np.clip(v * image_size, 0, image_size - 1))
    visible = 0.02 < u < 0.98 and 0.02 < v < 0.98
    return visible, px, py


def draw_marker(image: np.ndarray, px: int, py: int, color: tuple[int, int, int], label: str) -> None:
    cv2.drawMarker(image, (px, py), color, markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2, line_type=cv2.LINE_AA)
    cv2.circle(image, (px, py), 6, color, thickness=1, lineType=cv2.LINE_AA)
    cv2.putText(image, label, (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def annotate_view(
    image: np.ndarray,
    env: ContinuousRoArmSimEnv,
    camera_name: str,
    title: str,
) -> np.ndarray:
    canvas = image.copy()
    ee = env._ee_position()
    ear = env._target_ear_center_position()
    ee_ok, ee_x, ee_y = project_camera(env, camera_name, ee)
    ear_ok, ear_x, ear_y = project_camera(env, camera_name, ear)
    if ee_ok:
        draw_marker(canvas, ee_x, ee_y, (220, 40, 40), "EE")
    if ear_ok:
        draw_marker(canvas, ear_x, ear_y, (30, 190, 60), "EAR")
    if ee_ok and ear_ok:
        cv2.line(canvas, (ee_x, ee_y), (ear_x, ear_y), (255, 210, 0), 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 26), (245, 245, 245), thickness=-1)
    cv2.putText(canvas, title, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def render_pose_views(
    env: ContinuousRoArmSimEnv,
    qpos: np.ndarray,
    width: int,
    height: int,
    title: str,
) -> np.ndarray:
    env.data.qpos[:6] = np.asarray(qpos, dtype=np.float64)
    env.data.ctrl[:6] = np.asarray(qpos, dtype=np.float64)
    mujoco.mj_forward(env.model, env.data)
    overview = env.render_debug_view("overview_cam")
    forearm = env.render_debug_view("forearm_cam")
    overview = annotate_view(overview, env, "overview_cam", f"{title} - overview")
    forearm = annotate_view(forearm, env, "forearm_cam", f"{title} - forearm")
    overview = cv2.resize(overview, (width, height), interpolation=cv2.INTER_AREA)
    forearm = cv2.resize(forearm, (width, height), interpolation=cv2.INTER_AREA)
    return np.concatenate([overview, forearm], axis=0)


def render_info_panel(env: ContinuousRoArmSimEnv, width: int, height: int, title: str) -> np.ndarray:
    panel = np.full((height, width, 3), 250, dtype=np.uint8)
    ee = env._ee_position()
    ear = env._target_ear_center_position()
    delta = ear - ee
    dist = float(np.linalg.norm(delta))
    lines = [
        title,
        f"qpos: [{', '.join(f'{float(v):.3f}' for v in env.data.qpos[:6])}]",
        f"EE center:  [{ee[0]:.4f}, {ee[1]:.4f}, {ee[2]:.4f}]",
        f"Ear center: [{ear[0]:.4f}, {ear[1]:.4f}, {ear[2]:.4f}]",
        f"Delta:      [{delta[0]:.4f}, {delta[1]:.4f}, {delta[2]:.4f}]",
        f"Distance:   {dist:.4f} m",
    ]
    y = 28
    for idx, line in enumerate(lines):
        scale = 0.56 if idx == 0 else 0.50
        color = (20, 20, 20) if idx == 0 else (45, 45, 45)
        cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += 28
    return panel


def complete_l1(env: ContinuousRoArmSimEnv, expert: ContinuousWaypointExpert) -> None:
    for _ in range(int(env.cfg["episode_horizon"])):
        action = expert._level1_action(env)
        _, _, done, _ = env.step_action(action)
        if expert._level1_stage_complete(env) or done:
            break


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
    complete_l1(env, expert)
    l1_qpos = env.data.qpos[:6].astype(np.float64).copy()
    l2_qpos = expert._solve_level2_target_qpos(env, PREGRASP_QPOS, near=True).astype(np.float64).copy()

    l1_views = render_pose_views(env, l1_qpos, args.width, args.height, "After L1")
    l1_info = render_info_panel(env, args.width, 170, "Current interpretation after full L1")
    l1_block = np.concatenate([l1_views, l1_info], axis=0)

    l2_views = render_pose_views(env, l2_qpos, args.width, args.height, "Solved L2 pregrasp")
    l2_info = render_info_panel(env, args.width, 170, "Current interpretation of final L2 target")
    l2_block = np.concatenate([l2_views, l2_info], axis=0)

    sheet = np.concatenate([l1_block, l2_block], axis=1)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet[:, :, ::-1])
    print(output_path)


if __name__ == "__main__":
    main()

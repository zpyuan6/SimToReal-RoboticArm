from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from ttla.config import load_config
from ttla.sim import ContinuousRoArmSimEnv, ContinuousWaypointExpert
from ttla.sim.context import neutral_context
from ttla.sim.skills import PREGRASP_QPOS

try:
    import msvcrt
except ImportError:  # pragma: no cover
    msvcrt = None


VIEW_MODES = ("forearm_cam", "overview_cam", "free")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive MuJoCo inspector for L2 ee-center and ear-center markers.")
    parser.add_argument("--config", default="configs/continuous_act_preview_jointtarget.yaml")
    parser.add_argument("--frame-sleep-s", type=float, default=0.03)
    return parser.parse_args()


def _set_view_mode(env: ContinuousRoArmSimEnv, viewer, view_mode: str) -> None:
    if view_mode == "free":
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        return
    camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, view_mode)
    viewer.cam.fixedcamid = int(camera_id)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED


def _complete_l1(env: ContinuousRoArmSimEnv, expert: ContinuousWaypointExpert) -> None:
    for _ in range(int(env.cfg["episode_horizon"])):
        action = expert._level1_action(env)
        _, _, done, _ = env.step_action(action)
        if expert._level1_stage_complete(env) or done:
            break


def _prepare_states(env: ContinuousRoArmSimEnv, expert: ContinuousWaypointExpert) -> tuple[np.ndarray, np.ndarray]:
    env.reset(task_name="level2_approach", context=neutral_context())
    expert.reset(task_name="level2_approach")
    _complete_l1(env, expert)
    after_l1 = env.data.qpos[:6].astype(np.float64).copy()
    solved_l2 = expert._solve_level2_target_qpos(env, PREGRASP_QPOS, near=True).astype(np.float64).copy()
    return after_l1, solved_l2


def _apply_pose(env: ContinuousRoArmSimEnv, qpos: np.ndarray) -> None:
    env.data.qpos[:6] = np.asarray(qpos, dtype=np.float64)
    env.data.ctrl[:6] = np.asarray(qpos, dtype=np.float64)
    mujoco.mj_forward(env.model, env.data)


def _identity_mat_flat() -> np.ndarray:
    return np.eye(3, dtype=np.float32).reshape(-1)


def _add_sphere(viewer, pos: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    idx = viewer.user_scn.ngeom
    if idx >= len(viewer.user_scn.geoms):
        return
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[idx],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.asarray([radius, radius, radius], dtype=np.float32),
        np.asarray(pos, dtype=np.float32),
        _identity_mat_flat(),
        np.asarray(rgba, dtype=np.float32),
    )
    viewer.user_scn.ngeom += 1


def _add_capsule(viewer, p1: np.ndarray, p2: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    idx = viewer.user_scn.ngeom
    if idx >= len(viewer.user_scn.geoms):
        return
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[idx],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.asarray([radius, radius, radius], dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        _identity_mat_flat(),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        viewer.user_scn.geoms[idx],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        np.asarray(p1, dtype=np.float64),
        np.asarray(p2, dtype=np.float64),
    )
    viewer.user_scn.ngeom += 1


def _update_markers(viewer, env: ContinuousRoArmSimEnv) -> None:
    viewer.user_scn.ngeom = 0
    ee = env._gripper_center_position()
    ear = env._target_ear_center_position()
    hinge = env._gripper_open_joint_position()
    _add_sphere(viewer, ee, 0.0085, np.asarray([0.92, 0.10, 0.10, 1.0], dtype=np.float32))
    _add_sphere(viewer, ear, 0.0075, np.asarray([0.10, 0.45, 1.00, 1.0], dtype=np.float32))
    _add_sphere(viewer, hinge, 0.0070, np.asarray([1.00, 0.85, 0.05, 1.0], dtype=np.float32))
    _add_capsule(viewer, ee, ear, 0.0018, np.asarray([1.0, 0.85, 0.10, 0.85], dtype=np.float32))


def _set_overlay(viewer, env: ContinuousRoArmSimEnv, mode_name: str, view_mode: str) -> None:
    ee = env._gripper_center_position()
    ear = env._target_ear_center_position()
    hinge = env._gripper_open_joint_position()
    delta = ear - ee
    texts = [
        (
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            "L2 Centers Inspector",
            (
                f"mode={mode_name} | view={view_mode}\n"
                f"GRIP(red)=[{ee[0]:+.4f}, {ee[1]:+.4f}, {ee[2]:+.4f}]\n"
                f"HINGE(yellow)=[{hinge[0]:+.4f}, {hinge[1]:+.4f}, {hinge[2]:+.4f}]\n"
                f"EAR(blue)=[{ear[0]:+.4f}, {ear[1]:+.4f}, {ear[2]:+.4f}]\n"
                f"delta=[{delta[0]:+.4f}, {delta[1]:+.4f}, {delta[2]:+.4f}] | dist={env.ee_ear_center_distance():.4f}m"
            ),
        ),
        (
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPRIGHT,
            "controls",
            "terminal keys: 1 after_l1 | 2 solved_l2 | r resample | v view | q quit",
        ),
    ]
    viewer.set_texts(texts)


def _poll_console_key() -> str:
    if msvcrt is None:
        return ""
    if not msvcrt.kbhit():
        return ""
    key = msvcrt.getwch()
    return key.lower()


def main() -> None:
    args = _parse_args()
    cfg = load_config(Path(args.config))
    env = ContinuousRoArmSimEnv(
        cfg["sim"],
        seed=int(cfg.get("seed", 7)),
        action_low=cfg["control"]["action"].get("clamp_low"),
        action_high=cfg["control"]["action"].get("clamp_high"),
        control_mode=cfg["control"]["action"].get("control_mode", "joint_delta"),
    )
    expert = ContinuousWaypointExpert()
    after_l1, solved_l2 = _prepare_states(env, expert)
    mode = "after_l1"
    viewer = mujoco.viewer.launch_passive(env.model, env.data, show_left_ui=True, show_right_ui=True)
    view_index = 2
    _set_view_mode(env, viewer, VIEW_MODES[view_index])
    print("keys: 1 after_l1 | 2 solved_l2 | r resample | v view | q quit")
    try:
        while viewer.is_running():
            qpos = after_l1 if mode == "after_l1" else solved_l2
            _apply_pose(env, qpos)
            _update_markers(viewer, env)
            _set_overlay(viewer, env, "After L1" if mode == "after_l1" else "Solved L2 pregrasp", VIEW_MODES[view_index])
            viewer.sync()
            key = _poll_console_key()
            if key == "":
                time.sleep(args.frame_sleep_s)
                continue
            if key == "q":
                break
            if key == "1":
                mode = "after_l1"
                continue
            if key == "2":
                mode = "solved_l2"
                continue
            if key == "r":
                after_l1, solved_l2 = _prepare_states(env, expert)
                mode = "after_l1"
                continue
            if key == "v":
                view_index = (view_index + 1) % len(VIEW_MODES)
                _set_view_mode(env, viewer, VIEW_MODES[view_index])
                continue
    finally:
        viewer.close()


if __name__ == "__main__":
    main()

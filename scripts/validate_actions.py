from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from ttla.config import load_config
from ttla.deployment import DeploymentRunner
from ttla.deployment.primitives import (
    REAL_CARRY_QPOS,
    REAL_DROPZONE_QPOS,
    REAL_GRIPPER_CLOSED_QPOS,
    REAL_GRIPPER_OPEN_QPOS,
    REAL_HOME_QPOS,
    REAL_OBS_CENTER_QPOS,
    REAL_OBS_LEFT_QPOS,
    REAL_OBS_RIGHT_QPOS,
    REAL_PREALIGN_QPOS,
)
from ttla.sim import RoArmSimEnv
from ttla.sim.skills import (
    ABORT_ID,
    APPROACH_COARSE_ID,
    APPROACH_FINE_ID,
    GRASP_EXECUTE_ID,
    HOME_QPOS as SIM_HOME_QPOS,
    LIFT_OBJECT_ID,
    OBS_CENTER_ID,
    OBS_CENTER_QPOS as SIM_OBS_CENTER_QPOS,
    OBS_LEFT_ID,
    OBS_LEFT_QPOS as SIM_OBS_LEFT_QPOS,
    OBS_RIGHT_ID,
    OBS_RIGHT_QPOS as SIM_OBS_RIGHT_QPOS,
    PLACE_OBJECT_ID,
    CARRY_QPOS as SIM_CARRY_QPOS,
    DROPZONE_QPOS as SIM_DROPZONE_QPOS,
    PREALIGN_BASE_QPOS as SIM_PREALIGN_BASE_QPOS,
    PREALIGN_GRASP_ID,
    PREGRASP_SERVO_ID,
    PRIMITIVE_NAMES,
    REOBSERVE_ID,
    RETREAT_ID,
    TRANSPORT_TO_DROPZONE_ID,
    primitive_name,
)
from ttla.utils.io import ensure_dir, write_json


REAL_WINDOW_NAME = "TTLA Real Action Comparison"
WINDOW_FLAGS = cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED
BG = (242, 244, 248)
CARD = (251, 252, 254)
BORDER = (218, 223, 232)
TEXT = (32, 37, 48)
SUBTLE = (98, 108, 125)
ACCENT = (54, 106, 255)
SUCCESS = (44, 142, 86)
WARN = (204, 129, 54)


# Per-joint validation mapping. For base / shoulder / elbow / wrist / roll we
# currently keep a conservative near-identity mapping because the real-arm
# validation already matched the expected simulator poses reasonably well.
# The gripper uses an explicit reversed linear map because the real RoArm hand
# opens as its angle decreases, while the MuJoCo gripper opens as qpos grows.
SIM_JOINT_LIMITS = np.asarray(
    [
        [-np.pi, np.pi],
        [-np.pi / 2.0, np.pi / 2.0],
        [-1.0, 2.95],
        [-np.pi / 2.0, np.pi / 2.0],
        [-np.pi, np.pi],
        [0.0, 2.3562],
    ],
    dtype=np.float32,
)

REAL_JOINT_LIMITS = np.asarray(
    [
        [-np.pi, np.pi],
        [-np.pi / 2.0, np.pi / 2.0],
        [0.0, np.pi],
        [-np.pi / 2.0, np.pi / 2.0],
        [-np.pi, np.pi],
        [float(REAL_GRIPPER_OPEN_QPOS), float(REAL_GRIPPER_CLOSED_QPOS)],
    ],
    dtype=np.float32,
)

ELBOW_REAL_REFS = np.asarray(
    [
        REAL_HOME_QPOS[2],
        REAL_OBS_CENTER_QPOS[2],
        REAL_OBS_LEFT_QPOS[2],
        REAL_OBS_RIGHT_QPOS[2],
        REAL_PREALIGN_QPOS[2],
        REAL_CARRY_QPOS[2],
        REAL_DROPZONE_QPOS[2],
    ],
    dtype=np.float32,
)
ELBOW_SIM_REFS = np.asarray(
    [
        SIM_HOME_QPOS[2],
        SIM_OBS_CENTER_QPOS[2],
        SIM_OBS_LEFT_QPOS[2],
        SIM_OBS_RIGHT_QPOS[2],
        SIM_PREALIGN_BASE_QPOS[2],
        SIM_CARRY_QPOS[2],
        SIM_DROPZONE_QPOS[2],
    ],
    dtype=np.float32,
)


def _fit_scalar_affine(real_refs: np.ndarray, sim_refs: np.ndarray) -> tuple[np.float32, np.float32]:
    x = np.asarray(real_refs, dtype=np.float64)
    y = np.asarray(sim_refs, dtype=np.float64)
    design = np.stack([x, np.ones_like(x)], axis=1)
    solution, *_ = np.linalg.lstsq(design, y, rcond=None)
    return np.float32(solution[0]), np.float32(solution[1])


ELBOW_TO_SIM_SLOPE, ELBOW_TO_SIM_INTERCEPT = _fit_scalar_affine(ELBOW_REAL_REFS, ELBOW_SIM_REFS)


def _map_joint_real_to_sim(joint_idx: int, value: float) -> np.float32:
    # Joint indices 0-4 already use compatible semantics in this validator, so
    # we retain them directly and only clamp to the MuJoCo actuator limits,
    # except elbow which benefits from an explicit affine fit.
    if joint_idx == 2:
        mapped = ELBOW_TO_SIM_SLOPE * np.float32(value) + ELBOW_TO_SIM_INTERCEPT
        lo, hi = SIM_JOINT_LIMITS[joint_idx]
        return np.float32(np.clip(mapped, lo, hi))

    if joint_idx < 5:
        lo, hi = SIM_JOINT_LIMITS[joint_idx]
        return np.float32(np.clip(value, lo, hi))

    real_open = float(REAL_JOINT_LIMITS[5, 0])
    real_closed = float(REAL_JOINT_LIMITS[5, 1])
    sim_open = float(SIM_JOINT_LIMITS[5, 1])
    sim_closed = 0.0
    slope = (sim_closed - sim_open) / (real_closed - real_open)
    intercept = sim_open - slope * real_open
    mapped = slope * float(value) + intercept
    lo, hi = SIM_JOINT_LIMITS[5]
    return np.float32(np.clip(mapped, lo, hi))


def _map_real_to_sim_qpos(real_q: np.ndarray) -> np.ndarray:
    real_q = np.asarray(real_q, dtype=np.float32)
    sim_q = real_q.copy()
    for joint_idx in range(sim_q.shape[0]):
        sim_q[joint_idx] = _map_joint_real_to_sim(joint_idx, sim_q[joint_idx])
    return sim_q


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--deploy-config", default="configs/deployment.yaml")
    parser.add_argument("--task", default="level1_verify", choices=["level1_verify", "level2_approach", "level3_pick_place"])
    parser.add_argument("--primitives", required=True, help="Comma-separated primitive ids, e.g. 2,0,1,2,3")
    parser.add_argument("--save-dir", default="data/raw/action_validation")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--step-delay", type=float, default=0.5, help="Pause after a real action before reading the after-frame.")
    parser.add_argument("--auto-advance", action="store_true", help="Advance automatically instead of waiting for a key between steps.")
    parser.add_argument("--hide-left-ui", action="store_true")
    parser.add_argument("--hide-right-ui", action="store_true")
    parser.add_argument("--disable-gui", action="store_true")
    return parser.parse_args()


def _put(canvas: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.58, color: tuple[int, int, int] = TEXT, thickness: int = 1) -> None:
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, lineType=cv2.LINE_AA)


def _card(canvas: np.ndarray, top_left: tuple[int, int], bottom_right: tuple[int, int], title: str, subtitle: str | None = None) -> None:
    x1, y1 = top_left
    x2, y2 = bottom_right
    cv2.rectangle(canvas, (x1, y1), (x2, y2), CARD, -1, lineType=cv2.LINE_AA)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), BORDER, 1, lineType=cv2.LINE_AA)
    _put(canvas, title, (x1 + 18, y1 + 28), 0.72, TEXT, 2)
    if subtitle:
        _put(canvas, subtitle, (x1 + 18, y1 + 52), 0.46, SUBTLE, 1)


def _badge(canvas: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = origin
    width = max(100, 18 + len(text) * 10)
    cv2.rectangle(canvas, (x, y), (x + width, y + 28), color, -1, lineType=cv2.LINE_AA)
    _put(canvas, text, (x + 10, y + 20), 0.48, (255, 255, 255), 1)


def _configure_camera(handle) -> None:
    with handle.lock():
        handle.cam.azimuth = 132.0
        handle.cam.elevation = -24.0
        handle.cam.distance = 0.95
        handle.cam.lookat[:] = np.asarray([0.26, 0.0, 0.12], dtype=np.float64)


def _update_overlay(handle, task_name: str, step_index: int, primitive_id: int, sim_info: dict) -> None:
    texts = [
        (
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            "Sim Action Validator",
            (
                f"task={task_name} | step={step_index} | primitive={primitive_name(primitive_id)}\n"
                f"success={int(sim_info.get('success', 0))} "
                f"vis={float(sim_info.get('visibility', 0.0)):.3f} "
                f"center={float(sim_info.get('center_error', 0.0)):.1f}px"
            ),
        ),
        (
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPRIGHT,
            "controls",
            "Free camera works here.\nPress N / Space / Enter in the OpenCV window to advance.\nPress Q / Esc in the OpenCV window to stop.",
        ),
    ]
    handle.set_texts(texts)


def _apply_expected_qpos(sim_env: RoArmSimEnv, qpos: np.ndarray) -> None:
    sim_env.data.qpos[:6] = np.asarray(qpos, dtype=np.float64)
    sim_env.data.ctrl[:6] = np.asarray(qpos, dtype=np.float64)
    mujoco.mj_forward(sim_env.model, sim_env.data)


def _expected_observe_pose(primitive_id: int) -> np.ndarray:
    if primitive_id == OBS_LEFT_ID:
        return REAL_OBS_LEFT_QPOS.copy()
    if primitive_id == OBS_RIGHT_ID:
        return REAL_OBS_RIGHT_QPOS.copy()
    return REAL_OBS_CENTER_QPOS.copy()


def _expected_next_real_qpos(current_q: np.ndarray, primitive_id: int) -> np.ndarray:
    current_q = np.asarray(current_q, dtype=np.float32).copy()
    if primitive_id in (OBS_LEFT_ID, OBS_RIGHT_ID, OBS_CENTER_ID):
        return _expected_observe_pose(primitive_id)
    if primitive_id == PREALIGN_GRASP_ID:
        return REAL_PREALIGN_QPOS.copy()
    if primitive_id == REOBSERVE_ID:
        return REAL_OBS_CENTER_QPOS.copy()
    if primitive_id == APPROACH_COARSE_ID:
        return current_q + np.asarray([0.0, -0.12, -0.16, 0.08, 0.0, 0.0], dtype=np.float32)
    if primitive_id == APPROACH_FINE_ID:
        return current_q + np.asarray([0.0, -0.05, -0.07, 0.04, 0.0, 0.0], dtype=np.float32)
    if primitive_id == RETREAT_ID:
        return current_q + np.asarray([0.0, 0.10, 0.14, -0.06, 0.0, 0.10], dtype=np.float32)
    if primitive_id == PREGRASP_SERVO_ID:
        next_q = REAL_PREALIGN_QPOS.copy() + np.asarray([0.0, -0.04, -0.04, 0.02, 0.0, -0.04], dtype=np.float32)
        return next_q
    if primitive_id == GRASP_EXECUTE_ID:
        next_q = current_q + np.asarray([0.0, -0.08, -0.10, 0.05, 0.0, 0.0], dtype=np.float32)
        next_q[5] = REAL_GRIPPER_CLOSED_QPOS
        return next_q
    if primitive_id == LIFT_OBJECT_ID:
        return REAL_CARRY_QPOS.copy()
    if primitive_id == TRANSPORT_TO_DROPZONE_ID:
        return REAL_DROPZONE_QPOS.copy()
    if primitive_id == PLACE_OBJECT_ID:
        next_q = current_q + np.asarray([0.0, 0.08, -0.05, 0.0, 0.0, 0.0], dtype=np.float32)
        next_q[5] = REAL_GRIPPER_OPEN_QPOS
        return next_q
    if primitive_id == ABORT_ID:
        return REAL_HOME_QPOS.copy()
    return current_q


def _compose_real_dashboard(
    task_name: str,
    step_index: int,
    primitive_id: int,
    real_before: np.ndarray,
    real_after: np.ndarray,
    sim_before: np.ndarray,
    sim_after: np.ndarray,
    real_info: dict,
) -> np.ndarray:
    canvas = np.full((980, 1280, 3), BG, dtype=np.uint8)
    _put(canvas, "TTLA Real Action Comparison", (28, 34), 0.92, TEXT, 2)
    _put(canvas, "Native MuJoCo viewer shows the simulated expected motion in a separate window.", (30, 60), 0.48, SUBTLE, 1)
    _badge(canvas, f"STEP {step_index}", (980, 22), ACCENT)
    _badge(canvas, primitive_name(primitive_id), (1090, 22), SUCCESS if primitive_id != ABORT_ID else WARN)

    _card(canvas, (24, 84), (620, 462), "Real Before", f"Task: {task_name}")
    _card(canvas, (660, 84), (1256, 462), "Real After", f"Primitive: {primitive_name(primitive_id)}")
    _card(canvas, (24, 500), (620, 878), "Sim Before Snapshot")
    _card(canvas, (660, 500), (1256, 878), "Sim After Snapshot")
    _card(canvas, (24, 900), (1256, 956), "Controls", None)

    real_before_view = cv2.resize(real_before, (560, 280), interpolation=cv2.INTER_AREA)
    real_after_view = cv2.resize(real_after, (560, 280), interpolation=cv2.INTER_AREA)
    sim_before_view = cv2.resize(sim_before, (560, 280), interpolation=cv2.INTER_AREA)
    sim_after_view = cv2.resize(sim_after, (560, 280), interpolation=cv2.INTER_AREA)
    canvas[150:430, 44:604] = real_before_view
    canvas[150:430, 680:1240] = real_after_view
    canvas[566:846, 44:604] = sim_before_view
    canvas[566:846, 680:1240] = sim_after_view

    controls_text = (
        "N / Space / Enter: next step    "
        "Q / Esc: quit    "
        "Use the MuJoCo viewer window to orbit / zoom / pan the simulated scene."
    )
    real_summary = real_info.get("primitive_name", primitive_name(primitive_id))
    _put(canvas, f"Real info: {real_summary}", (40, 930), 0.5, SUBTLE, 1)
    _put(canvas, controls_text, (420, 930), 0.46, SUBTLE, 1)
    return canvas


def _save_step_artifacts(
    step_dir: Path,
    primitive_id: int,
    sim_before: np.ndarray,
    sim_after: np.ndarray,
    real_before: np.ndarray,
    real_after: np.ndarray,
    sim_info: dict,
    real_info: dict,
) -> None:
    cv2.imwrite(str(step_dir / "sim_before.png"), sim_before)
    cv2.imwrite(str(step_dir / "sim_after.png"), sim_after)
    cv2.imwrite(str(step_dir / "real_before.png"), real_before)
    cv2.imwrite(str(step_dir / "real_after.png"), real_after)
    write_json(
        step_dir / "meta.json",
        {
            "primitive_id": primitive_id,
            "primitive_name": primitive_name(primitive_id),
            "sim_info": sim_info,
            "real_info": real_info,
        },
    )


def _await_next_step(auto_advance: bool, disable_gui: bool) -> bool:
    if disable_gui or auto_advance:
        return True
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            return False
        if key in (ord("n"), ord(" "), 13):
            return True


def main() -> None:
    args = _parse_args()
    primitive_ids = [int(item.strip()) for item in args.primitives.split(",") if item.strip()]
    if not primitive_ids:
        raise ValueError("No primitives were provided.")

    cfg = load_config(args.config)
    deploy_cfg = load_config(args.deploy_config)
    sim_env = RoArmSimEnv(cfg["sim"], seed=args.seed)
    runner = DeploymentRunner(deploy_cfg)
    session_dir = ensure_dir(Path(args.save_dir) / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    if not args.disable_gui:
        cv2.namedWindow(REAL_WINDOW_NAME, WINDOW_FLAGS)
        cv2.resizeWindow(REAL_WINDOW_NAME, 1320, 1020)

    obs = sim_env.reset(task_name=args.task)
    expected_real_q = REAL_HOME_QPOS.copy()
    _apply_expected_qpos(sim_env, _map_real_to_sim_qpos(expected_real_q))
    if deploy_cfg.get("safety", {}).get("reset_before_episode", True):
        runner.robot.reset_pose()
        time.sleep(1.5)

    viewer = None
    if not args.disable_gui:
        viewer = mujoco.viewer.launch_passive(
            sim_env.model,
            sim_env.data,
            show_left_ui=not args.hide_left_ui,
            show_right_ui=not args.hide_right_ui,
        )
        _configure_camera(viewer)

    try:
        for step_index, primitive_id in enumerate(primitive_ids, start=1):
            if viewer is not None and not viewer.is_running():
                break

            step_dir = ensure_dir(session_dir / f"step_{step_index:03d}_{primitive_name(primitive_id)}")

            sim_before = sim_env.render_debug_view("overview_cam")
            real_before = runner.camera.read()

            expected_next_real_q = _expected_next_real_qpos(expected_real_q, primitive_id)
            _apply_expected_qpos(sim_env, _map_real_to_sim_qpos(expected_next_real_q))
            sim_info = {
                "task": args.task,
                "success": 0,
                "visibility": float(sim_env.visibility_score()),
                "center_error": float(sim_env.center_error_px()),
                "primitive_name": primitive_name(primitive_id),
            }
            if viewer is not None:
                _update_overlay(viewer, args.task, step_index, primitive_id, sim_info)
                viewer.sync()

            real_result = runner.executor.run(primitive_id)
            time.sleep(max(args.step_delay, 0.0))
            real_after = runner.camera.read()
            sim_after = sim_env.render_debug_view("overview_cam")

            dashboard = _compose_real_dashboard(
                args.task,
                step_index,
                primitive_id,
                real_before,
                real_after,
                sim_before,
                sim_after,
                real_result.info,
            )
            if not args.disable_gui:
                cv2.imshow(REAL_WINDOW_NAME, dashboard)
                cv2.waitKey(1)

            _save_step_artifacts(
                step_dir,
                primitive_id,
                sim_before,
                sim_after,
                real_before,
                real_after,
                {
                    "task": args.task,
                    "success": int(sim_info.get("success", 0)),
                    "visibility": float(sim_info.get("visibility", 0.0)),
                    "center_error": float(sim_info.get("center_error", 0.0)),
                    "primitive_name": sim_info.get("primitive_name", primitive_name(primitive_id)),
                },
                real_result.info,
            )
            cv2.imwrite(str(step_dir / "comparison.png"), dashboard)

            expected_real_q = expected_next_real_q
            if not _await_next_step(args.auto_advance, args.disable_gui):
                break
    finally:
        write_json(
            session_dir / "session_meta.json",
            {
                "task": args.task,
                "primitives": primitive_ids,
                "primitive_names": [PRIMITIVE_NAMES[idx] for idx in primitive_ids],
            },
        )
        if viewer is not None:
            viewer.close()
        runner.close()
        sim_env.close()
        cv2.destroyAllWindows()

    print(f"saved_validation_session={session_dir}")


if __name__ == "__main__":
    main()

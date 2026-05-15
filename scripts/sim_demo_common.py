from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np

from ttla.config import load_config
from ttla.sim import RoArmSimEnv
from ttla.sim.skills import (
    ABORT_ID,
    APPROACH_ID,
    GRASP_EXECUTE_ID,
    LIFT_OBJECT_ID,
    OBS_CENTER_ID,
    OBS_LEFT_ID,
    OBS_RIGHT_ID,
    PLACE_OBJECT_ID,
    PREGRASP_SERVO_ID,
    RETREAT_ID,
    TRANSPORT_TO_DROPZONE_ID,
    primitive_count,
    primitive_id,
    primitive_name,
)
from ttla.sim.task_defs import TASK_SPECS
from ttla.utils.io import ensure_dir, write_json


CANVAS_BG = (242, 244, 248)
CARD_BG = (251, 252, 254)
CARD_BORDER = (218, 223, 232)
TEXT = (32, 37, 48)
SUBTLE = (98, 108, 125)
ACCENT = (54, 106, 255)
SUCCESS = (44, 142, 86)
WARN = (204, 129, 54)
DANGER = (186, 63, 63)


PRIMITIVE_DEMO_ORDER = (
    OBS_LEFT_ID,
    OBS_RIGHT_ID,
    OBS_CENTER_ID,
    APPROACH_ID,
    RETREAT_ID,
    PREGRASP_SERVO_ID,
    GRASP_EXECUTE_ID,
    LIFT_OBJECT_ID,
    TRANSPORT_TO_DROPZONE_ID,
    PLACE_OBJECT_ID,
    ABORT_ID,
)


PRIMITIVE_CANONICAL_TASK = {
    OBS_LEFT_ID: "level1_verify",
    OBS_RIGHT_ID: "level1_verify",
    OBS_CENTER_ID: "level1_verify",
    APPROACH_ID: "level2_approach",
    RETREAT_ID: "level2_approach",
    PREGRASP_SERVO_ID: "level3_pick_place",
    GRASP_EXECUTE_ID: "level3_pick_place",
    LIFT_OBJECT_ID: "level3_pick_place",
    TRANSPORT_TO_DROPZONE_ID: "level3_pick_place",
    PLACE_OBJECT_ID: "level3_pick_place",
    ABORT_ID: "level3_pick_place",
}


PRIMITIVE_SETUP_PREFIXES = {
    OBS_LEFT_ID: (),
    OBS_RIGHT_ID: (),
    OBS_CENTER_ID: (),
    APPROACH_ID: (OBS_CENTER_ID,),
    RETREAT_ID: (OBS_CENTER_ID, APPROACH_ID),
    PREGRASP_SERVO_ID: (OBS_CENTER_ID, APPROACH_ID),
    GRASP_EXECUTE_ID: (OBS_CENTER_ID, APPROACH_ID, PREGRASP_SERVO_ID),
    LIFT_OBJECT_ID: (OBS_CENTER_ID, APPROACH_ID, PREGRASP_SERVO_ID, GRASP_EXECUTE_ID),
    TRANSPORT_TO_DROPZONE_ID: (OBS_CENTER_ID, APPROACH_ID, PREGRASP_SERVO_ID, GRASP_EXECUTE_ID, LIFT_OBJECT_ID),
    PLACE_OBJECT_ID: (
        OBS_CENTER_ID,
        APPROACH_ID,
        PREGRASP_SERVO_ID,
        GRASP_EXECUTE_ID,
        LIFT_OBJECT_ID,
        TRANSPORT_TO_DROPZONE_ID,
    ),
    ABORT_ID: (OBS_CENTER_ID, APPROACH_ID),
}


TASK_PRESET_FLOWS = {
    "level1_verify": (OBS_CENTER_ID,),
    "level2_approach": (
        OBS_CENTER_ID,
        APPROACH_ID,
    ),
    "level3_pick_place": (
        OBS_CENTER_ID,
        APPROACH_ID,
        PREGRASP_SERVO_ID,
        GRASP_EXECUTE_ID,
        LIFT_OBJECT_ID,
        TRANSPORT_TO_DROPZONE_ID,
        PLACE_OBJECT_ID,
    ),
}


def add_shared_args(parser: argparse.ArgumentParser, default_output_root: str) -> None:
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output-root", default=default_output_root)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fps", type=float, default=6.0)


def load_env(config_path: str, seed: int) -> tuple[dict[str, Any], RoArmSimEnv]:
    cfg = load_config(config_path)
    env = RoArmSimEnv(cfg["sim"], seed=seed)
    return cfg, env


def canonical_context() -> dict[str, float]:
    return {
        "cam_x": 0.0,
        "cam_y": 0.0,
        "cam_z": 0.0,
        "cam_roll": 0.0,
        "cam_pitch": 0.0,
        "cam_yaw": 0.0,
        "fov_bias": 0.0,
        "light_gain": 1.0,
        "blur_sigma": 0.0,
        "noise_std": 0.0,
        "action_gain": 1.0,
        "action_delay": 0,
        "joint_bias": 0.0,
    }


def parse_primitive_tokens(tokens: str | None, primitive_vocabulary: str) -> list[int]:
    if not tokens:
        return list(PRIMITIVE_DEMO_ORDER)
    result: list[int] = []
    for raw in tokens.split(","):
        token = raw.strip()
        if not token:
            continue
        if token.isdigit():
            result.append(int(token))
            continue
        result.append(primitive_id(token, primitive_vocabulary=primitive_vocabulary))
    return result


def set_canonical_layout(env: RoArmSimEnv, task_name: str) -> None:
    if task_name == "level3_pick_place":
        target = np.asarray([0.285, 0.000, 0.040], dtype=np.float64)
        drop = np.asarray([0.245, -0.115, 0.045], dtype=np.float64)
    else:
        target = np.asarray([0.300, 0.000, 0.040], dtype=np.float64)
        drop = np.asarray([0.245, -0.115, 0.045], dtype=np.float64)
    target_body = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "target")
    drop_body = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "distractor")
    env.model.body_pos[target_body] = target
    env.model.body_pos[drop_body] = drop
    mujoco.mj_forward(env.model, env.data)


def panel_frame(
    env: RoArmSimEnv,
    obs: dict[str, np.ndarray],
    *,
    title: str,
    subtitle: str,
    lines: list[str],
    status_label: str,
    status_color: tuple[int, int, int],
) -> np.ndarray:
    canvas = np.full((860, 1280, 3), CANVAS_BG, dtype=np.uint8)
    _put(canvas, title, (28, 34), 0.92, TEXT, 2)
    _put(canvas, subtitle, (30, 62), 0.48, SUBTLE, 1)
    _badge(canvas, status_label, (1090, 20), status_color)

    _card(canvas, (24, 84), (620, 612), "Forearm Camera")
    _card(canvas, (660, 84), (1256, 612), "Overview Camera")
    _card(canvas, (24, 640), (1256, 834), "Primitive Diagnostics")

    forearm = cv2.resize(obs["image"][:, :, ::-1], (560, 440), interpolation=cv2.INTER_CUBIC)
    overview = cv2.resize(env.render_debug_view("overview_cam")[:, :, ::-1], (560, 440), interpolation=cv2.INTER_CUBIC)
    canvas[126:566, 44:604] = forearm
    canvas[126:566, 680:1240] = overview

    y = 676
    for line in lines:
        _put(canvas, line, (42, y), 0.54, TEXT if y == 676 else SUBTLE, 1)
        y += 26
    return canvas


def _put(
    canvas: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.58,
    color: tuple[int, int, int] = TEXT,
    thickness: int = 1,
) -> None:
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, lineType=cv2.LINE_AA)


def _badge(canvas: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = origin
    width = max(96, 18 + len(text) * 11)
    cv2.rectangle(canvas, (x, y), (x + width, y + 30), color, -1, lineType=cv2.LINE_AA)
    _put(canvas, text, (x + 10, y + 21), 0.5, (255, 255, 255), 1)


def _card(canvas: np.ndarray, top_left: tuple[int, int], bottom_right: tuple[int, int], title: str) -> None:
    x1, y1 = top_left
    x2, y2 = bottom_right
    cv2.rectangle(canvas, (x1, y1), (x2, y2), CARD_BG, -1, lineType=cv2.LINE_AA)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), CARD_BORDER, 1, lineType=cv2.LINE_AA)
    _put(canvas, title, (x1 + 18, y1 + 28), 0.72, TEXT, 2)


def lines_for_env(
    env: RoArmSimEnv,
    *,
    primitive_id_value: int,
    micro_step: int | None = None,
    phase: str | None = None,
    extra_lines: list[str] | None = None,
) -> list[str]:
    lines = [
        f"task={env.task_name}",
        f"primitive={primitive_name(primitive_id_value, primitive_vocabulary=env.primitive_vocabulary)}",
        f"step_idx={env.step_idx}",
        f"visibility={env.visibility_score():.3f}",
        f"center_error_px={env.center_error_px():.1f}",
        f"grasp_gap={env.grasp_gap():+.3f}",
        f"ee_target_distance={env.ee_target_distance():.3f}",
        f"dropzone_distance={env.dropzone_distance():.3f}",
        f"flags verified={int(env.verified)} grasped={int(env.object_attached)} lifted={int(env.lifted)} placed={int(env.placed)}",
    ]
    if micro_step is not None:
        lines.append(f"internal_control_step={micro_step}")
    if phase is not None:
        lines.append(f"phase={phase}")
    if extra_lines:
        lines.extend(extra_lines)
    return lines


def save_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def save_contact_sheet(path: Path, frames: list[np.ndarray], labels: list[str]) -> None:
    if not frames:
        return
    selected_indices = [0]
    if len(frames) > 2:
        selected_indices.append(len(frames) // 2)
    if len(frames) > 1:
        selected_indices.append(len(frames) - 1)
    tiles: list[np.ndarray] = []
    for idx, frame_idx in enumerate(selected_indices):
        tile = cv2.resize(frames[frame_idx], (420, 280), interpolation=cv2.INTER_AREA)
        label = labels[idx] if idx < len(labels) else f"frame_{frame_idx}"
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 34), (255, 255, 255), -1)
        _put(tile, label, (12, 24), 0.62, TEXT, 1)
        tiles.append(tile)
    sheet = np.hstack(tiles)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)


def run_trace(
    env: RoArmSimEnv,
    primitive_id_value: int,
    *,
    title: str,
    subtitle: str,
    extra_lines: list[str] | None = None,
) -> dict[str, Any]:
    frames: list[np.ndarray] = []
    forearm_frames: list[np.ndarray] = []
    overview_frames: list[np.ndarray] = []
    micro_step = 0
    obs_before = env.observe()
    forearm_frames.append(obs_before["image"][:, :, ::-1].copy())
    overview_frames.append(env.render_debug_view("overview_cam")[:, :, ::-1].copy())
    frames.append(
        panel_frame(
            env,
            obs_before,
            title=title,
            subtitle=subtitle,
            lines=lines_for_env(
                env,
                primitive_id_value=primitive_id_value,
                micro_step=None,
                phase="before",
                extra_lines=extra_lines,
            ),
            status_label="BEFORE",
            status_color=ACCENT,
        )
    )
    original_apply_target_pose = env._apply_target_pose

    def traced_apply_target_pose(target_qpos: np.ndarray, dwell: int = 1) -> None:
        nonlocal micro_step
        original_apply_target_pose(target_qpos, dwell=dwell)
        micro_step += 1
        traced_obs = env.observe()
        forearm_frames.append(traced_obs["image"][:, :, ::-1].copy())
        overview_frames.append(env.render_debug_view("overview_cam")[:, :, ::-1].copy())
        frames.append(
            panel_frame(
                env,
                traced_obs,
                title=title,
                subtitle=subtitle,
                lines=lines_for_env(
                    env,
                    primitive_id_value=primitive_id_value,
                    micro_step=micro_step,
                    phase="internal",
                    extra_lines=extra_lines,
                ),
                status_label=f"STEP {micro_step}",
                status_color=ACCENT,
            )
        )

    env._apply_target_pose = traced_apply_target_pose  # type: ignore[method-assign]
    try:
        next_obs, reward, done, info = env.step(primitive_id_value)
    finally:
        env._apply_target_pose = original_apply_target_pose  # type: ignore[method-assign]

    status_label = "SUCCESS" if info["success"] else "DONE" if done else "AFTER"
    status_color = SUCCESS if info["success"] else WARN if done else ACCENT
    forearm_frames.append(next_obs["image"][:, :, ::-1].copy())
    overview_frames.append(env.render_debug_view("overview_cam")[:, :, ::-1].copy())
    frames.append(
        panel_frame(
            env,
            next_obs,
            title=title,
            subtitle=subtitle,
            lines=lines_for_env(
                env,
                primitive_id_value=primitive_id_value,
                micro_step=micro_step,
                phase="after",
                extra_lines=extra_lines,
            )
            + [
                f"reward={reward:.3f}",
                f"done={int(done)} success={int(info['success'])}",
                f"executor={info['executor_primitive_name']}",
            ],
            status_label=status_label,
            status_color=status_color,
        )
    )
    return {
        "frames": frames,
        "forearm_frames": forearm_frames,
        "overview_frames": overview_frames,
        "reward": reward,
        "done": bool(done),
        "info": info,
        "micro_steps": micro_step,
    }


def primitive_output_dir(root: Path, primitive_id_value: int, primitive_vocabulary: str) -> Path:
    return ensure_dir(root / f"{primitive_id_value:02d}_{primitive_name(primitive_id_value, primitive_vocabulary=primitive_vocabulary)}")


def task_output_dir(root: Path, task_name: str) -> Path:
    return ensure_dir(root / task_name)


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def execute_with_animation(
    env: RoArmSimEnv,
    primitive_id_value: int,
    *,
    on_frame=None,
    frame_sleep_s: float = 0.03,
):
    micro_step = 0
    if on_frame is not None:
        on_frame("before", micro_step)
    original_apply_target_pose = env._apply_target_pose

    def animated_apply_target_pose(target_qpos: np.ndarray, dwell: int = 1) -> None:
        nonlocal micro_step
        current = env.data.ctrl[:6].copy()
        target = np.asarray(target_qpos, dtype=np.float64).copy()
        target[:5] += env.context["joint_bias"]
        desired = current + env.context["action_gain"] * (target - current)
        low = env.model.actuator_ctrlrange[:6, 0]
        high = env.model.actuator_ctrlrange[:6, 1]
        desired = np.clip(desired, low, high)
        env.action_delay_queue.append(desired.copy())
        if len(env.action_delay_queue) > env.context["action_delay"]:
            applied = env.action_delay_queue.popleft()
        else:
            applied = env.data.ctrl[:6].copy()
        for _ in range(max(1, dwell) * env.cfg["action_repeat"]):
            env.data.ctrl[:6] = applied
            mujoco.mj_step(env.model, env.data)
            if env._ear_grasp_contact_count() > 0:
                env.recent_ear_contact = min(env.recent_ear_contact + 1, 6)
            else:
                env.recent_ear_contact = max(env.recent_ear_contact - 1, 0)
            if env.object_attached:
                if not env._gripper_closed_enough():
                    env.release_counter += 1
                else:
                    env.release_counter = 0
                if env.release_counter >= 3:
                    env.object_attached = False
                    env.lifted = False
                    env.release_counter = 0
                else:
                    env._update_attached_object_pose()
            micro_step += 1
            if on_frame is not None:
                on_frame("internal", micro_step)
            if frame_sleep_s > 0.0:
                time.sleep(frame_sleep_s)
        mujoco.mj_forward(env.model, env.data)

    env._apply_target_pose = animated_apply_target_pose  # type: ignore[method-assign]
    try:
        result = env.step(primitive_id_value)
    finally:
        env._apply_target_pose = original_apply_target_pose  # type: ignore[method-assign]
    if on_frame is not None:
        on_frame("after", micro_step)
    return result, micro_step

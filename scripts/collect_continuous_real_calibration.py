from __future__ import annotations

import argparse
import copy
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ttla.config import load_config
from ttla.deployment.camera import USBCamera
from ttla.deployment.primitives import (
    REAL_APPROACH_QPOS,
    REAL_GRIPPER_CLOSED_QPOS,
    REAL_GRIPPER_OPEN_QPOS,
    REAL_HOME_QPOS,
    REAL_LIFT_QPOS,
    REAL_OBS_CENTER_QPOS,
    REAL_OBS_LEFT_QPOS,
    REAL_OBS_RIGHT_QPOS,
    REAL_PLACE_RELEASE_QPOS,
    REAL_PREGRASP_QPOS,
    REAL_TRANSPORT_QPOS,
)
from ttla.deployment.roarm_serial import RoArmSerialClient
from ttla.sim.task_defs import TASK_TO_ID
from ttla.task_runtime import build_runtime_state
from ttla.utils.io import ensure_dir, save_npz, write_json


LIVE_WINDOW_NAME = "Continuous Real Collection Camera"

TASK_TEXT_BY_ID = {
    0: "center the target object in the camera view",
    1: "move the gripper into a stable pre-grasp approach state",
    2: "pick up the object and place it in the blue drop zone",
}

STATIC_POSES = {
    "home": REAL_HOME_QPOS,
    "obs_center": REAL_OBS_CENTER_QPOS,
    "obs_left": REAL_OBS_LEFT_QPOS,
    "obs_right": REAL_OBS_RIGHT_QPOS,
    "approach": REAL_APPROACH_QPOS,
    "pregrasp": REAL_PREGRASP_QPOS,
    "lift": REAL_LIFT_QPOS,
    "transport": REAL_TRANSPORT_QPOS,
    "place_release": REAL_PLACE_RELEASE_QPOS,
}

RETREAT_DELTA = np.asarray([0.0, 0.10, 0.14, -0.06, 0.0, 0.10], dtype=np.float32)


@dataclass
class TransitionRecord:
    image: np.ndarray
    proprio: np.ndarray
    action_joint_target: np.ndarray
    action_joint_delta: np.ndarray
    next_image: np.ndarray
    next_proprio: np.ndarray
    task_id: int
    episode_id: int
    step_id: int
    waypoint_name: str
    q_before: np.ndarray
    q_after: np.ndarray
    frame_before_path: str
    frame_after_path: str


class DryRunCamera:
    def __init__(self, width: int, height: int) -> None:
        self.width = int(width)
        self.height = int(height)
        self.count = 0

    def read(self) -> np.ndarray:
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            f"dry-run frame {self.count}",
            (24, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        self.count += 1
        return frame

    def close(self) -> None:
        return None


class DryRunRobot:
    def __init__(self) -> None:
        self.last_q = REAL_HOME_QPOS.copy()

    def reset_pose(self) -> None:
        self.last_q = REAL_HOME_QPOS.copy()

    def move_joint_vector(self, joints: np.ndarray) -> None:
        self.last_q = np.asarray(joints, dtype=np.float32).copy()

    def close(self) -> None:
        return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect real continuous calibration trajectories for frozen ACT/Diffusion policies."
    )
    parser.add_argument("--plan", default="configs/continuous_real_collection_plan_v1.yaml")
    parser.add_argument("--session", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--deploy-config", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--operator", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--action-format", choices=["joint_target", "joint_delta", "both"], default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--max-joint-step-rad", type=float, default=None)
    parser.add_argument("--step-sleep-s", type=float, default=None)
    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument("--auto-accept", action="store_true")
    parser.add_argument("--save-preview", action="store_true")
    parser.add_argument("--live-preview", dest="live_preview", action="store_true", default=None)
    parser.add_argument("--no-live-preview", dest="live_preview", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _session_spec(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    plan = load_config(args.plan)
    sessions = plan.get("sessions", {})
    if args.session not in sessions:
        raise KeyError(f"Unknown session {args.session!r}. Available: {sorted(sessions)}")
    shared = copy.deepcopy(plan.get("shared", {}))
    spec = {**shared, **copy.deepcopy(sessions[args.session])}
    spec["sequences"] = copy.deepcopy(sessions[args.session].get("sequences", []))
    if args.config:
        spec["config"] = args.config
    if args.deploy_config:
        spec["deploy_config"] = args.deploy_config
    if args.output_root:
        spec["output_root"] = args.output_root
    if args.operator:
        spec["operator"] = args.operator
    if args.notes:
        spec["notes"] = args.notes
    if args.action_format:
        spec["action_format"] = args.action_format
    if args.repeats is not None:
        spec["repeats"] = int(args.repeats)
    if args.max_joint_step_rad is not None:
        spec["max_joint_step_rad"] = float(args.max_joint_step_rad)
    if args.step_sleep_s is not None:
        spec["step_sleep_s"] = float(args.step_sleep_s)
    if args.auto_start:
        spec["auto_start"] = True
    if args.auto_accept:
        spec["auto_accept"] = True
    if args.save_preview:
        spec["save_preview"] = True
    if args.live_preview is not None:
        spec["live_preview"] = bool(args.live_preview)
    if not spec.get("sequences"):
        raise ValueError(f"Session {args.session!r} does not define sequences.")
    return spec, args.session


def _action_formats(raw: str) -> list[str]:
    if raw == "both":
        return ["joint_target", "joint_delta"]
    return [str(raw)]


def _image_shape_from_config(cfg: dict) -> tuple[int, int]:
    shape = cfg.get("control", {}).get("image_shape", [224, 224, 3])
    return int(shape[0]), int(shape[1])


def _dataset_frame(frame: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    height, width = image_hw
    if tuple(frame.shape[:2]) == (height, width):
        return frame.astype(np.uint8)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA).astype(np.uint8)


def _pose_for_waypoint(name: str, current_q: np.ndarray) -> np.ndarray:
    key = str(name)
    if key in STATIC_POSES:
        return np.asarray(STATIC_POSES[key], dtype=np.float32).copy()
    if key == "grasp_close":
        q = np.asarray(current_q, dtype=np.float32).copy()
        q[5] = float(REAL_GRIPPER_CLOSED_QPOS)
        return q
    if key == "gripper_open":
        q = np.asarray(current_q, dtype=np.float32).copy()
        q[5] = float(REAL_GRIPPER_OPEN_QPOS)
        return q
    if key == "retreat":
        return np.asarray(current_q, dtype=np.float32) + RETREAT_DELTA
    raise KeyError(f"Unknown continuous real waypoint: {name!r}")


def _interpolate_targets(current_q: np.ndarray, target_q: np.ndarray, max_step: float) -> list[np.ndarray]:
    current = np.asarray(current_q, dtype=np.float32)
    target = np.asarray(target_q, dtype=np.float32)
    delta = target - current
    steps = max(1, int(np.ceil(float(np.max(np.abs(delta))) / max(float(max_step), 1.0e-6))))
    return [(current + delta * (idx / steps)).astype(np.float32) for idx in range(1, steps + 1)]


def _planned_targets(sequence: list[str], *, start_q: np.ndarray, max_step: float) -> list[tuple[str, np.ndarray]]:
    current_q = np.asarray(start_q, dtype=np.float32).copy()
    out: list[tuple[str, np.ndarray]] = []
    for waypoint in sequence:
        target_q = _pose_for_waypoint(waypoint, current_q)
        for sub_target in _interpolate_targets(current_q, target_q, max_step):
            out.append((str(waypoint), sub_target))
            current_q = sub_target.copy()
    return out


def _prompt_continue(prompt: str) -> bool:
    while True:
        answer = input(prompt).strip().lower()
        if answer in {"", "y", "yes"}:
            return True
        if answer in {"q", "quit", "n", "no"}:
            return False
        print("Enter/y continues; q/n stops.")


def _prompt_accept() -> str:
    while True:
        answer = input("Keep this episode [k], redo [r], or quit [q]? ").strip().lower()
        if answer in {"", "k", "keep"}:
            return "keep"
        if answer in {"r", "redo"}:
            return "redo"
        if answer in {"q", "quit"}:
            return "quit"
        print("Please enter k / r / q.")


def _show_live_frame(frame: np.ndarray, enabled: bool) -> None:
    if not enabled:
        return
    cv2.imshow(LIVE_WINDOW_NAME, frame)
    cv2.waitKey(1)


def _prompt_continue_with_live_preview(camera: Any, prompt: str, enabled: bool) -> bool:
    if not enabled:
        return _prompt_continue(prompt)

    cv2.namedWindow(LIVE_WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    decision_ready = threading.Event()
    decision: dict[str, bool] = {"continue": False}

    def _read_terminal_decision() -> None:
        try:
            decision["continue"] = _prompt_continue(prompt)
        finally:
            decision_ready.set()

    input_thread = threading.Thread(target=_read_terminal_decision, daemon=True)
    input_thread.start()
    while not decision_ready.is_set():
        _show_live_frame(camera.read(), enabled=True)
        time.sleep(0.03)
    input_thread.join(timeout=0.1)
    return bool(decision["continue"])


def _prompt_accept_with_live_preview(camera: Any, enabled: bool) -> str:
    if not enabled:
        return _prompt_accept()

    cv2.namedWindow(LIVE_WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    decision_ready = threading.Event()
    decision: dict[str, str] = {"value": "keep"}

    def _read_terminal_decision() -> None:
        try:
            decision["value"] = _prompt_accept()
        finally:
            decision_ready.set()

    input_thread = threading.Thread(target=_read_terminal_decision, daemon=True)
    input_thread.start()
    while not decision_ready.is_set():
        _show_live_frame(camera.read(), enabled=True)
        time.sleep(0.03)
    input_thread.join(timeout=0.1)
    return str(decision["value"])


def _sleep_with_live_preview(camera: Any, seconds: float, enabled: bool) -> None:
    if not enabled:
        time.sleep(seconds)
        return
    deadline = time.time() + max(0.0, seconds)
    while time.time() < deadline:
        _show_live_frame(camera.read(), enabled=True)
        time.sleep(min(0.03, max(0.0, deadline - time.time())))


def _write_preview(session_dir: Path, records: list[TransitionRecord], frame_size: tuple[int, int]) -> None:
    if not records:
        return
    writer = cv2.VideoWriter(
        str(session_dir / "preview.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        6.0,
        frame_size,
    )
    try:
        for record in records:
            before = cv2.imread(record.frame_before_path)
            after = cv2.imread(record.frame_after_path)
            if before is None or after is None:
                continue
            cv2.putText(before, f"before: {record.waypoint_name}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(after, f"after: {record.waypoint_name}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            writer.write(before)
            writer.write(after)
    finally:
        writer.release()


def _save_dataset(path: Path, records: list[TransitionRecord], *, action_format: str) -> None:
    if action_format == "joint_target":
        actions = [record.action_joint_target for record in records]
    elif action_format == "joint_delta":
        actions = [record.action_joint_delta for record in records]
    else:
        raise ValueError(f"Unsupported action format: {action_format}")
    save_npz(
        path,
        images=np.asarray([record.image for record in records], dtype=np.uint8),
        proprio=np.asarray([record.proprio for record in records], dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        tasks=np.asarray([record.task_id for record in records], dtype=np.int64),
        success=np.zeros((len(records),), dtype=np.int64),
        contexts=np.zeros((len(records), 8), dtype=np.float32),
        episode_ids=np.asarray([record.episode_id for record in records], dtype=np.int64),
        step_ids=np.asarray([record.step_id for record in records], dtype=np.int64),
        next_images=np.asarray([record.next_image for record in records], dtype=np.uint8),
        next_proprio=np.asarray([record.next_proprio for record in records], dtype=np.float32),
        q_before=np.asarray([record.q_before for record in records], dtype=np.float32),
        q_after=np.asarray([record.q_after for record in records], dtype=np.float32),
        action_joint_target=np.asarray([record.action_joint_target for record in records], dtype=np.float32),
        action_joint_delta=np.asarray([record.action_joint_delta for record in records], dtype=np.float32),
        task_text=np.asarray([TASK_TEXT_BY_ID[record.task_id] for record in records], dtype=object),
        waypoint_name=np.asarray([record.waypoint_name for record in records], dtype=object),
    )


def main() -> None:
    args = _parse_args()
    spec, session_key = _session_spec(args)
    cfg = load_config(spec.get("config", "configs/continuous_act_jointtarget_staged_frozen_best.yaml"))
    deploy_cfg = load_config(spec.get("deploy_config", "configs/deployment_l3.yaml"))
    image_hw = _image_shape_from_config(cfg)
    output_root = ensure_dir(spec.get("output_root", "data/real_continuous_v1/sessions"))
    session_dir = ensure_dir(output_root / f"{session_key}_{_timestamp()}")
    frames_dir = ensure_dir(session_dir / "frames")

    task_name = str(spec["task"])
    task_id = int(TASK_TO_ID[task_name])
    repeats = int(spec.get("repeats", 1))
    max_step = float(spec.get("max_joint_step_rad", 0.18))
    step_sleep_s = float(spec.get("step_sleep_s", 0.35))
    auto_start = bool(spec.get("auto_start", False))
    auto_accept = bool(spec.get("auto_accept", False))
    reset_between_episodes = bool(spec.get("reset_between_episodes", True))
    save_preview = bool(spec.get("save_preview", False))
    live_preview = bool(spec.get("live_preview", True))
    reset_settle_s = float(spec.get("reset_settle_s", 1.5 if not args.dry_run else 0.1))
    post_episode_settle_s = float(spec.get("post_episode_settle_s", 1.5 if not args.dry_run else 0.1))
    selected_formats = _action_formats(str(spec.get("action_format", "both")))

    if args.dry_run:
        width = int(deploy_cfg.get("camera", {}).get("width", 640))
        height = int(deploy_cfg.get("camera", {}).get("height", 480))
        camera = DryRunCamera(width=width, height=height)
        robot = DryRunRobot()
    else:
        camera = USBCamera(**deploy_cfg["camera"])
        robot = RoArmSerialClient(**deploy_cfg["serial"])

    all_records: list[TransitionRecord] = []
    episode_meta: list[dict[str, Any]] = []
    frame_size: tuple[int, int] | None = None
    next_episode_id = 0

    try:
        for sequence in spec["sequences"]:
            sequence_name = str(sequence["name"])
            waypoints = [str(value) for value in sequence["waypoints"]]
            sequence_repeats = int(sequence.get("repeats", repeats))
            repeat_idx = 0
            while repeat_idx < sequence_repeats:
                episode_name = f"{sequence_name}_r{repeat_idx:02d}"
                planned = _planned_targets(waypoints, start_q=REAL_HOME_QPOS, max_step=max_step)
                if reset_between_episodes:
                    robot.reset_pose()
                    _sleep_with_live_preview(camera, reset_settle_s, live_preview)
                    robot.move_joint_vector(REAL_HOME_QPOS)
                    _sleep_with_live_preview(camera, reset_settle_s, live_preview)
                if not auto_start:
                    prompt = (
                        f"[{episode_name}] task={task_name} waypoints={waypoints} "
                        f"substeps={len(planned)}. Press Enter to execute or q to stop: "
                    )
                    if not _prompt_continue_with_live_preview(camera, prompt, live_preview):
                        raise KeyboardInterrupt
                current_q = REAL_HOME_QPOS.copy()
                local_records: list[TransitionRecord] = []
                for local_step, (waypoint_name, target_q) in enumerate(planned):
                    before = camera.read()
                    _show_live_frame(before, live_preview)
                    if frame_size is None:
                        frame_size = (int(before.shape[1]), int(before.shape[0]))
                    before_dataset = _dataset_frame(before, image_hw)
                    state = build_runtime_state(
                        current_q=current_q,
                        task_id=task_id,
                        step_idx=local_step,
                        horizon=len(planned),
                    )
                    robot.move_joint_vector(target_q)
                    _sleep_with_live_preview(camera, step_sleep_s, live_preview)
                    after = camera.read()
                    _show_live_frame(after, live_preview)
                    after_dataset = _dataset_frame(after, image_hw)
                    next_state = build_runtime_state(
                        current_q=target_q,
                        task_id=task_id,
                        step_idx=local_step + 1,
                        horizon=len(planned),
                    )
                    before_path = frames_dir / f"episode_{next_episode_id:04d}_step_{local_step:03d}_before_{waypoint_name}.jpg"
                    after_path = frames_dir / f"episode_{next_episode_id:04d}_step_{local_step:03d}_after_{waypoint_name}.jpg"
                    cv2.imwrite(str(before_path), before)
                    cv2.imwrite(str(after_path), after)
                    record = TransitionRecord(
                        image=before_dataset,
                        proprio=state,
                        action_joint_target=target_q.astype(np.float32),
                        action_joint_delta=(target_q - current_q).astype(np.float32),
                        next_image=after_dataset,
                        next_proprio=next_state,
                        task_id=task_id,
                        episode_id=next_episode_id,
                        step_id=local_step,
                        waypoint_name=waypoint_name,
                        q_before=current_q.astype(np.float32),
                        q_after=target_q.astype(np.float32),
                        frame_before_path=str(before_path),
                        frame_after_path=str(after_path),
                    )
                    local_records.append(record)
                    current_q = target_q.copy()
                _sleep_with_live_preview(camera, post_episode_settle_s, live_preview)
                decision = "keep" if auto_accept else _prompt_accept_with_live_preview(camera, live_preview)
                if decision == "quit":
                    raise KeyboardInterrupt
                if decision == "redo":
                    print(f"redo_episode={episode_name}; retrying same planned episode", flush=True)
                    continue
                all_records.extend(local_records)
                episode_meta.append(
                    {
                        "episode_id": next_episode_id,
                        "episode_name": episode_name,
                        "sequence_name": sequence_name,
                        "repeat_idx": repeat_idx,
                        "waypoints": waypoints,
                        "substeps": len(local_records),
                    }
                )
                next_episode_id += 1
                repeat_idx += 1
    except KeyboardInterrupt:
        print("collection_stopped=true", flush=True)
    finally:
        camera.close()
        robot.close()
        if live_preview:
            cv2.destroyAllWindows()

    dataset_paths: dict[str, str] = {}
    for action_format in selected_formats:
        dataset_path = session_dir / f"session_dataset_{action_format}.npz"
        _save_dataset(dataset_path, all_records, action_format=action_format)
        dataset_paths[action_format] = str(dataset_path)
    if selected_formats:
        first_path = Path(dataset_paths[selected_formats[0]])
        alias_path = session_dir / "session_dataset.npz"
        alias_path.write_bytes(first_path.read_bytes())

    if save_preview and frame_size is not None:
        _write_preview(session_dir, all_records, frame_size)

    meta = {
        "created_at": time.time(),
        "session_key": session_key,
        "session_dir": str(session_dir),
        "session_dataset_paths": dataset_paths,
        "default_session_dataset_path": str(session_dir / "session_dataset.npz"),
        "plan_path": args.plan,
        "config_path": spec.get("config"),
        "deploy_config_path": spec.get("deploy_config"),
        "task": task_name,
        "task_id": task_id,
        "split_role": spec.get("split_role", ""),
        "layout_tag": spec.get("layout_tag", ""),
        "operator": spec.get("operator", args.operator),
        "notes": spec.get("notes", args.notes),
        "object_name": spec.get("object_name", ""),
        "dropzone_name": spec.get("dropzone_name", ""),
        "action_formats": selected_formats,
        "max_joint_step_rad": max_step,
        "step_sleep_s": step_sleep_s,
        "reset_settle_s": reset_settle_s,
        "post_episode_settle_s": post_episode_settle_s,
        "episodes_collected": len(episode_meta),
        "transitions_collected": len(all_records),
        "dry_run": bool(args.dry_run),
        "frame_size": None if frame_size is None else {"width": frame_size[0], "height": frame_size[1]},
        "image_shape": {"height": image_hw[0], "width": image_hw[1], "channels": 3},
        "scene_constraints": spec.get("scene_constraints", []),
        "placement_guide": spec.get("placement_guide", []),
        "collection_target": spec.get("collection_target", {}),
        "repeat_placement_policy": spec.get("repeat_placement_policy", []),
        "episode_records": episode_meta,
    }
    write_json(session_dir / "meta.json", meta)

    print(f"saved_continuous_real_session={session_dir}")
    for action_format, path in dataset_paths.items():
        print(f"{action_format}_dataset={path}")
    print(f"episodes_collected={len(episode_meta)}")
    print(f"transitions_collected={len(all_records)}")


if __name__ == "__main__":
    main()

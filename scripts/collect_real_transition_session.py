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
from ttla.deployment import DeploymentRunner
from ttla.deployment.primitives import REAL_HOME_QPOS
from ttla.sim.skills import primitive_id, primitive_name
from ttla.sim.task_defs import TASK_TO_ID, supervision_stage_id
from ttla.task_runtime import build_runtime_state
from ttla.utils.io import ensure_dir, save_npz, write_json

PREVIEW_WINDOW_NAME = "TTLA Real Collection Preview"


class CameraFrameBuffer:
    def __init__(self, camera) -> None:
        self.camera = camera
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._frame: np.ndarray | None = None
        self._error: BaseException | None = None
        self._thread.start()

    def _read_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self.camera.read()
            except BaseException as exc:  # Keep the main thread in charge of reporting failures.
                with self._lock:
                    self._error = exc
                time.sleep(0.05)
                continue
            with self._lock:
                self._frame = frame
                self._error = None

    def read(self, timeout_s: float = 2.0) -> np.ndarray:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self._lock:
                if self._frame is not None:
                    return self._frame.copy()
                error = self._error
            if error is not None:
                raise RuntimeError("Failed to read frame from USB camera.") from error
            time.sleep(0.02)
        raise RuntimeError("Timed out waiting for USB camera frame.")

    def close(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=1.0)


@dataclass
class RuntimeFlags:
    attached: bool = False
    lifted: bool = False
    placed: bool = False

    def apply(self, primitive_idx: int) -> None:
        label = primitive_name(int(primitive_idx))
        if label == "grasp_execute":
            self.attached = True
            self.placed = False
        elif label == "lift_object":
            if self.attached:
                self.lifted = True
        elif label == "place_object":
            self.placed = True
            self.attached = False
            self.lifted = False
        elif label == "abort":
            self.attached = False
            self.lifted = False
            self.placed = False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--deploy-config", default="configs/deployment.yaml")
    parser.add_argument("--plan", default=None, help="YAML collection plan.")
    parser.add_argument("--session", default=None, help="Session key inside the plan file.")
    parser.add_argument("--output-root", default=None, help="Override output root from plan.")
    parser.add_argument("--task", default=None, choices=sorted(TASK_TO_ID.keys()))
    parser.add_argument("--split-role", default=None, choices=["calibration", "heldout", "debug"])
    parser.add_argument("--layout-tag", default=None)
    parser.add_argument("--primitives", default=None, help="Comma-separated primitive names or ids for manual mode.")
    parser.add_argument("--sequence-name", default="manual_sequence")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--operator", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--session-tag", default="")
    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument("--auto-accept", action="store_true")
    parser.add_argument("--save-preview", action="store_true")
    parser.add_argument("--live-preview", action="store_true", help="Show the OpenCV live preview window during collection.")
    parser.add_argument("--record-episode-video", action="store_true", help="Record a continuous MP4 for each accepted episode.")
    parser.add_argument("--post-primitive-settle-s", type=float, default=None, help="Extra settle time after each primitive before capturing the after-frame.")
    return parser.parse_args()


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _parse_primitive_sequence(values: list[Any] | str) -> list[int]:
    if isinstance(values, str):
        raw_items = [item.strip() for item in values.split(",") if item.strip()]
    else:
        raw_items = list(values)
    sequence: list[int] = []
    for item in raw_items:
        if isinstance(item, int):
            sequence.append(int(item))
            continue
        text = str(item).strip()
        if text.isdigit():
            sequence.append(int(text))
        else:
            sequence.append(primitive_id(text))
    return sequence


def _load_plan_session(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if args.plan is None:
        if args.task is None or args.primitives is None or args.split_role is None:
            raise ValueError("Manual mode requires --task, --primitives, and --split-role.")
        session_key = args.session_tag or f"manual_{args.task}_{_timestamp()}"
        session_spec = {
            "task": args.task,
            "split_role": args.split_role,
            "layout_tag": args.layout_tag or "unspecified",
            "repeats": args.repeats or 1,
            "auto_start": args.auto_start,
            "save_preview": args.save_preview,
            "notes": args.notes,
            "operator": args.operator,
            "sequences": [
                {
                    "name": args.sequence_name,
                    "primitives": _parse_primitive_sequence(args.primitives),
                }
            ],
        }
        return session_spec, session_key

    plan = load_config(args.plan)
    sessions = plan.get("sessions", {})
    if not sessions:
        raise ValueError(f"No sessions defined in plan: {args.plan}")
    if args.session is None:
        raise ValueError("Plan mode requires --session.")
    if args.session not in sessions:
        raise KeyError(f"Unknown session key '{args.session}' in {args.plan}")
    shared = copy.deepcopy(plan.get("shared", {}))
    session_spec = copy.deepcopy(sessions[args.session])
    merged = {**shared, **session_spec}
    merged["sequences"] = copy.deepcopy(session_spec.get("sequences", shared.get("sequences", [])))
    if not merged.get("sequences"):
        raise ValueError(f"Session '{args.session}' does not define any sequences.")
    if args.output_root:
        merged["output_root"] = args.output_root
    if args.operator:
        merged["operator"] = args.operator
    if args.notes:
        merged["notes"] = args.notes
    if args.auto_start:
        merged["auto_start"] = True
    if args.auto_accept:
        merged["auto_accept"] = True
    if args.save_preview:
        merged["save_preview"] = True
    if args.live_preview:
        merged["live_preview"] = True
    if args.record_episode_video:
        merged["record_episode_video"] = True
    if args.post_primitive_settle_s is not None:
        merged["post_primitive_settle_s"] = float(args.post_primitive_settle_s)
    return merged, args.session


def _expand_episodes(session_spec: dict[str, Any]) -> list[dict[str, Any]]:
    default_repeats = int(session_spec.get("repeats", 1))
    expanded: list[dict[str, Any]] = []
    for sequence in session_spec.get("sequences", []):
        repeats = int(sequence.get("repeats", default_repeats))
        primitive_ids = _parse_primitive_sequence(sequence["primitives"])
        for repeat_idx in range(repeats):
            expanded.append(
                {
                    "name": sequence["name"],
                    "repeat_idx": repeat_idx,
                    "layout_tag": sequence.get("layout_tag", session_spec.get("layout_tag", "unspecified")),
                    "notes": sequence.get("notes", ""),
                    "primitive_ids": primitive_ids,
                }
            )
    return expanded


def _prompt_continue(prompt: str) -> bool:
    while True:
        answer = input(prompt).strip().lower()
        if answer in {"", "y", "yes"}:
            return True
        if answer in {"q", "quit", "n", "no"}:
            return False
        print("Please enter Enter/y to continue or q/n to stop.")


def _print_episode_prompt(
    episode_name: str,
    task_name: str,
    layout_tag: str,
    primitive_sequence: list[int],
    placement_guide: list[str],
) -> None:
    print()
    print(f"[{episode_name}] task={task_name} layout={layout_tag}")
    print("Sequence:", " -> ".join(primitive_name(pid) for pid in primitive_sequence))
    if placement_guide:
        print("Placement guide:")
        for guide in placement_guide:
            print(f"  - {guide}")


def _draw_live_preview(
    frame: np.ndarray,
    episode_name: str,
    task_name: str,
    layout_tag: str,
    primitive_sequence: list[int],
    placement_guide: list[str],
) -> np.ndarray:
    view = cv2.resize(frame, (800, 600))
    panel = np.full((600, 460, 3), 246, dtype=np.uint8)

    def _put(text: str, y: int, scale: float = 0.56, color: tuple[int, int, int] = (36, 40, 48), thickness: int = 1) -> int:
        cv2.putText(panel, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, lineType=cv2.LINE_AA)
        return y + 28

    y = 32
    y = _put("Real Collection Preview", y, 0.8, (24, 28, 36), 2)
    y += 8
    y = _put(f"Episode: {episode_name}", y)
    y = _put(f"Task: {task_name}", y)
    y = _put(f"Layout: {layout_tag}", y)
    y += 6
    y = _put("Sequence:", y, 0.62, (52, 64, 92), 1)
    sequence_text = " -> ".join(primitive_name(pid) for pid in primitive_sequence)
    for chunk_start in range(0, len(sequence_text), 32):
        y = _put(sequence_text[chunk_start : chunk_start + 32], y, 0.5, (70, 78, 92), 1)
    y += 6
    y = _put("Placement guide:", y, 0.62, (52, 64, 92), 1)
    for guide in placement_guide[:6]:
        line = f"- {guide}"
        for chunk_start in range(0, len(line), 44):
            y = _put(line[chunk_start : chunk_start + 44], y, 0.46, (76, 82, 94), 1)
    y = min(y + 10, 520)
    y = _put("Terminal controls this step.", y, 0.56, (52, 64, 92), 1)
    y = _put("Use the preview to judge placement.", y + 8, 0.48, (76, 82, 94), 1)

    return np.concatenate([view, panel], axis=1)


def _draw_execution_preview(
    frame: np.ndarray,
    episode_name: str,
    task_name: str,
    primitive_label: str,
    step_index: int,
    total_steps: int,
) -> np.ndarray:
    view = cv2.resize(frame, (800, 600))
    panel = np.full((600, 460, 3), 246, dtype=np.uint8)

    def _put(text: str, y: int, scale: float = 0.58, color: tuple[int, int, int] = (36, 40, 48), thickness: int = 1) -> int:
        cv2.putText(panel, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, lineType=cv2.LINE_AA)
        return y + 30

    y = 36
    y = _put("Real Collection Execution", y, 0.82, (24, 28, 36), 2)
    y += 10
    y = _put(f"Episode: {episode_name}", y)
    y = _put(f"Task: {task_name}", y)
    y = _put(f"Step: {step_index + 1}/{total_steps}", y)
    y = _put(f"Primitive: {primitive_label}", y, 0.62, (52, 64, 92), 1)
    y += 14
    y = _put("Live camera continues during motion.", y, 0.52, (52, 64, 92), 1)
    y = _put("Terminal stays in control for keep/redo.", y, 0.48, (76, 82, 94), 1)
    return np.concatenate([view, panel], axis=1)


def _preview_and_confirm_start(
    frame_buffer: CameraFrameBuffer,
    episode_name: str,
    task_name: str,
    layout_tag: str,
    primitive_sequence: list[int],
    placement_guide: list[str],
) -> bool:
    cv2.namedWindow(PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(PREVIEW_WINDOW_NAME, 1280, 680)
    decision_ready = threading.Event()
    decision: dict[str, bool] = {"continue": False}

    _print_episode_prompt(
        episode_name=episode_name,
        task_name=task_name,
        layout_tag=layout_tag,
        primitive_sequence=primitive_sequence,
        placement_guide=placement_guide,
    )

    def _read_terminal_decision() -> None:
        try:
            decision["continue"] = _prompt_continue("Check the preview window, then press Enter to execute or q to stop: ")
        finally:
            decision_ready.set()

    input_thread = threading.Thread(target=_read_terminal_decision, daemon=True)
    input_thread.start()
    try:
        while not decision_ready.is_set():
            frame = frame_buffer.read()
            dashboard = _draw_live_preview(
                frame,
                episode_name=episode_name,
                task_name=task_name,
                layout_tag=layout_tag,
                primitive_sequence=primitive_sequence,
                placement_guide=placement_guide,
            )
            cv2.imshow(PREVIEW_WINDOW_NAME, dashboard)
            cv2.waitKey(50)
        return bool(decision["continue"])
    finally:
        input_thread.join(timeout=0.1)


def _run_primitive_with_live_preview(
    runner: DeploymentRunner,
    frame_buffer: CameraFrameBuffer,
    primitive_idx: int,
    live_preview: bool,
    video_writer: cv2.VideoWriter | None,
    episode_name: str,
    task_name: str,
    step_index: int,
    total_steps: int,
) -> Any:
    result_box: dict[str, Any] = {}
    done_event = threading.Event()

    def _run() -> None:
        try:
            result_box["result"] = runner.executor.run(primitive_idx)
        finally:
            done_event.set()

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    primitive_label = primitive_name(primitive_idx)
    while not done_event.is_set():
        frame = frame_buffer.read()
        if video_writer is not None:
            video_writer.write(frame)
        if live_preview:
            dashboard = _draw_execution_preview(
                frame,
                episode_name=episode_name,
                task_name=task_name,
                primitive_label=primitive_label,
                step_index=step_index,
                total_steps=total_steps,
            )
            cv2.imshow(PREVIEW_WINDOW_NAME, dashboard)
            cv2.waitKey(50)
        else:
            time.sleep(0.05)

    worker.join(timeout=0.1)
    frame = frame_buffer.read()
    if video_writer is not None:
        video_writer.write(frame)
    if live_preview:
        dashboard = _draw_execution_preview(
            frame,
            episode_name=episode_name,
            task_name=task_name,
            primitive_label=primitive_label,
            step_index=step_index,
            total_steps=total_steps,
        )
        cv2.imshow(PREVIEW_WINDOW_NAME, dashboard)
        cv2.waitKey(1)
    return result_box["result"]


def _wait_after_primitive(
    frame_buffer: CameraFrameBuffer,
    settle_s: float,
    live_preview: bool,
    video_writer: cv2.VideoWriter | None,
    episode_name: str,
    task_name: str,
    primitive_label: str,
    step_index: int,
    total_steps: int,
) -> None:
    if settle_s <= 0:
        return
    deadline = time.time() + settle_s
    while time.time() < deadline:
        frame = frame_buffer.read()
        if video_writer is not None:
            video_writer.write(frame)
        if live_preview:
            dashboard = _draw_execution_preview(
                frame,
                episode_name=episode_name,
                task_name=task_name,
                primitive_label=f"{primitive_label} settling",
                step_index=step_index,
                total_steps=total_steps,
            )
            cv2.imshow(PREVIEW_WINDOW_NAME, dashboard)
            cv2.waitKey(50)
        else:
            time.sleep(min(0.05, max(0.0, deadline - time.time())))


def _open_episode_video(
    videos_dir: Path,
    episode_index: int,
    episode_name: str,
    frame: np.ndarray,
    fps: float,
) -> tuple[cv2.VideoWriter, Path]:
    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in episode_name)
    path = videos_dir / f"episode_{episode_index:04d}_{safe_name}.mp4"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (int(frame.shape[1]), int(frame.shape[0])),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open episode video writer: {path}")
    return writer, path


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


def _write_preview(session_dir: Path, entries: list[dict[str, Any]], frame_size: tuple[int, int]) -> None:
    if not entries:
        return
    preview_path = session_dir / "preview.mp4"
    writer = cv2.VideoWriter(
        str(preview_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        4.0,
        frame_size,
    )
    try:
        for entry in entries:
            before = cv2.imread(entry["before_path"])
            after = cv2.imread(entry["after_path"])
            if before is None or after is None:
                continue
            primitive_label = entry["primitive_name"]
            before_annotated = before.copy()
            after_annotated = after.copy()
            cv2.putText(before_annotated, f"before: {primitive_label}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(after_annotated, f"after: {primitive_label}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            writer.write(before_annotated)
            writer.write(after_annotated)
    finally:
        writer.release()


def main() -> None:
    args = _parse_args()
    session_spec, session_key = _load_plan_session(args)
    _cfg = load_config(session_spec.get("config", args.config))
    deploy_cfg = load_config(session_spec.get("deploy_config", args.deploy_config))
    deploy_cfg.setdefault("runtime", {})
    deploy_cfg.setdefault("serial", {})
    if "primitive_sleep_s" in session_spec:
        deploy_cfg["runtime"]["primitive_sleep_s"] = float(session_spec["primitive_sleep_s"])
    if "motion_spd" in session_spec:
        deploy_cfg["serial"]["spd"] = int(session_spec["motion_spd"])
    if "motion_acc" in session_spec:
        deploy_cfg["serial"]["acc"] = int(session_spec["motion_acc"])
    output_root = ensure_dir(session_spec.get("output_root", "data/real_v2/transitions"))
    session_dir = ensure_dir(output_root / f"{session_key}_{_timestamp()}")
    frames_dir = ensure_dir(session_dir / "frames")
    videos_dir = ensure_dir(session_dir / "videos")

    task_name = str(session_spec["task"])
    task_id = int(TASK_TO_ID[task_name])
    split_role = str(session_spec["split_role"])
    episodes = _expand_episodes(session_spec)
    auto_start = bool(session_spec.get("auto_start", False))
    auto_accept = bool(session_spec.get("auto_accept", False))
    save_preview = bool(session_spec.get("save_preview", False))
    live_preview = bool(session_spec.get("live_preview", False))
    record_episode_video = bool(session_spec.get("record_episode_video", False))
    episode_video_fps = float(session_spec.get("episode_video_fps", 10.0))
    post_primitive_settle_s = float(session_spec.get("post_primitive_settle_s", 1.5))
    reset_between_episodes = bool(session_spec.get("reset_between_episodes", deploy_cfg.get("safety", {}).get("reset_before_episode", True)))
    operator = str(session_spec.get("operator", ""))
    notes = str(session_spec.get("notes", ""))

    images: list[np.ndarray] = []
    states: list[np.ndarray] = []
    next_images: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    primitive_ids: list[int] = []
    tasks: list[int] = []
    episode_ids: list[int] = []
    step_ids: list[int] = []
    stage_ids: list[int] = []
    contexts: list[np.ndarray] = []
    success_flags: list[float] = []
    preview_entries: list[dict[str, Any]] = []
    episode_records: list[dict[str, Any]] = []
    frame_size: tuple[int, int] | None = None

    runner = DeploymentRunner(deploy_cfg)
    frame_buffer = CameraFrameBuffer(runner.camera)
    next_episode_id = 0

    try:
        for episode_index, episode_spec in enumerate(episodes):
            episode_name = f"{episode_spec['name']}_r{episode_spec['repeat_idx']:02d}"
            layout_tag = str(episode_spec.get("layout_tag", "unspecified"))
            primitive_sequence = list(episode_spec["primitive_ids"])

            if reset_between_episodes:
                print(f"[{episode_name}] resetting to standard pose before preview...")
                runner.robot.reset_pose()
                time.sleep(1.5)
                runner.executor.current_q = REAL_HOME_QPOS.copy()
                runner.robot.move_joint_vector(runner.executor.current_q)
                time.sleep(1.5)

            if not auto_start:
                if live_preview:
                    should_continue = _preview_and_confirm_start(
                        frame_buffer,
                        episode_name=episode_name,
                        task_name=task_name,
                        layout_tag=layout_tag,
                        primitive_sequence=primitive_sequence,
                        placement_guide=list(session_spec.get("placement_guide", [])),
                    )
                else:
                    _print_episode_prompt(
                        episode_name=episode_name,
                        task_name=task_name,
                        layout_tag=layout_tag,
                        primitive_sequence=primitive_sequence,
                        placement_guide=list(session_spec.get("placement_guide", [])),
                    )
                    should_continue = _prompt_continue("Check the external camera view, then press Enter to execute or q to stop: ")
                if not should_continue:
                    break

            current_q = runner.executor.current_q.copy()
            flags = RuntimeFlags()

            local_images: list[np.ndarray] = []
            local_states: list[np.ndarray] = []
            local_next_images: list[np.ndarray] = []
            local_next_states: list[np.ndarray] = []
            local_primitives: list[int] = []
            local_stage_ids: list[int] = []
            local_preview: list[dict[str, Any]] = []
            local_frame_paths: list[str] = []
            local_next_frame_paths: list[str] = []
            local_step_records: list[dict[str, Any]] = []
            local_video_path: str | None = None
            video_writer: cv2.VideoWriter | None = None

            for step_index, primitive_idx in enumerate(primitive_sequence):
                before = frame_buffer.read()
                if frame_size is None:
                    frame_size = (int(before.shape[1]), int(before.shape[0]))
                if record_episode_video and video_writer is None:
                    video_writer, video_path = _open_episode_video(
                        videos_dir,
                        episode_index=episode_index,
                        episode_name=episode_name,
                        frame=before,
                        fps=episode_video_fps,
                    )
                    local_video_path = str(video_path)
                if video_writer is not None:
                    video_writer.write(before)
                stage_before = supervision_stage_id(task_id, primitive_idx)
                state = build_runtime_state(
                    current_q=current_q,
                    task_id=task_id,
                    step_idx=step_index,
                    horizon=len(primitive_sequence),
                )
                if live_preview or video_writer is not None:
                    result = _run_primitive_with_live_preview(
                        runner,
                        frame_buffer,
                        primitive_idx,
                        live_preview=live_preview,
                        video_writer=video_writer,
                        episode_name=episode_name,
                        task_name=task_name,
                        step_index=step_index,
                        total_steps=len(primitive_sequence),
                    )
                else:
                    result = runner.executor.run(primitive_idx)
                current_q = runner.executor.current_q.copy()
                flags.apply(primitive_idx)
                primitive_label = primitive_name(primitive_idx)
                _wait_after_primitive(
                    frame_buffer,
                    settle_s=post_primitive_settle_s,
                    live_preview=live_preview,
                    video_writer=video_writer,
                    episode_name=episode_name,
                    task_name=task_name,
                    primitive_label=primitive_label,
                    step_index=step_index,
                    total_steps=len(primitive_sequence),
                )
                after = frame_buffer.read()
                if video_writer is not None:
                    video_writer.write(after)
                next_state = build_runtime_state(
                    current_q=current_q,
                    task_id=task_id,
                    step_idx=step_index + 1,
                    horizon=len(primitive_sequence),
                )
                before_path = frames_dir / f"episode_{episode_index:04d}_step_{step_index:03d}_before_{primitive_label}.jpg"
                after_path = frames_dir / f"episode_{episode_index:04d}_step_{step_index:03d}_after_{primitive_label}.jpg"
                cv2.imwrite(str(before_path), before)
                cv2.imwrite(str(after_path), after)
                local_images.append(before)
                local_states.append(state)
                local_next_images.append(after)
                local_next_states.append(next_state)
                local_primitives.append(int(primitive_idx))
                local_stage_ids.append(int(stage_before))
                local_frame_paths.append(str(before_path))
                local_next_frame_paths.append(str(after_path))
                local_step_records.append(
                    {
                        "step_index": step_index,
                        "primitive_id": int(primitive_idx),
                        "primitive_name": primitive_label,
                        "done": bool(result.done),
                        "result_info": result.info,
                    }
                )
                local_preview.append(
                    {
                        "before_path": str(before_path),
                        "after_path": str(after_path),
                        "primitive_name": primitive_label,
                    }
                )
                if result.done:
                    break

            if video_writer is not None:
                video_writer.release()

            decision = "keep" if auto_accept else _prompt_accept()
            if decision == "quit":
                break
            if decision == "redo":
                print(f"redo_episode={episode_name}")
                continue

            for step_index, primitive_idx in enumerate(local_primitives):
                images.append(local_images[step_index])
                states.append(local_states[step_index])
                next_images.append(local_next_images[step_index])
                next_states.append(local_next_states[step_index])
                primitive_ids.append(primitive_idx)
                tasks.append(task_id)
                episode_ids.append(next_episode_id)
                step_ids.append(step_index)
                stage_ids.append(local_stage_ids[step_index])
                contexts.append(np.zeros(8, dtype=np.float32))
                success_flags.append(0.0)
            preview_entries.extend(local_preview)
            episode_records.append(
                {
                    "episode_id": next_episode_id,
                    "episode_name": episode_name,
                    "layout_tag": layout_tag,
                    "sequence_notes": episode_spec.get("notes", ""),
                    "primitive_ids": local_primitives,
                    "primitive_names": [primitive_name(pid) for pid in local_primitives],
                    "frame_paths": local_frame_paths,
                    "next_frame_paths": local_next_frame_paths,
                    "steps_executed": len(local_primitives),
                    "task": task_name,
                    "video_path": local_video_path,
                }
            )
            next_episode_id += 1
    finally:
        frame_buffer.close()
        runner.close()
        if live_preview:
            cv2.destroyAllWindows()

    dataset_path = session_dir / "session_dataset.npz"
    camera_width = int(deploy_cfg.get("camera", {}).get("width", 640))
    camera_height = int(deploy_cfg.get("camera", {}).get("height", 480))
    if primitive_ids:
        save_npz(
            dataset_path,
            images=np.asarray(images, dtype=np.uint8),
            states=np.asarray(states, dtype=np.float32),
            primitive_ids=np.asarray(primitive_ids, dtype=np.int64),
            next_images=np.asarray(next_images, dtype=np.uint8),
            next_states=np.asarray(next_states, dtype=np.float32),
            tasks=np.asarray(tasks, dtype=np.int64),
            contexts=np.asarray(contexts, dtype=np.float32),
            success=np.asarray(success_flags, dtype=np.float32),
            episode_ids=np.asarray(episode_ids, dtype=np.int64),
            step_ids=np.asarray(step_ids, dtype=np.int64),
            stage_ids=np.asarray(stage_ids, dtype=np.int64),
        )
    else:
        save_npz(
            dataset_path,
            images=np.zeros((0, camera_height, camera_width, 3), dtype=np.uint8),
            states=np.zeros((0, 18), dtype=np.float32),
            primitive_ids=np.zeros((0,), dtype=np.int64),
            next_images=np.zeros((0, camera_height, camera_width, 3), dtype=np.uint8),
            next_states=np.zeros((0, 18), dtype=np.float32),
            tasks=np.zeros((0,), dtype=np.int64),
            contexts=np.zeros((0, 8), dtype=np.float32),
            success=np.zeros((0,), dtype=np.float32),
            episode_ids=np.zeros((0,), dtype=np.int64),
            step_ids=np.zeros((0,), dtype=np.int64),
            stage_ids=np.zeros((0,), dtype=np.int64),
        )

    meta = {
        "created_at": time.time(),
        "plan_path": args.plan,
        "config_path": session_spec.get("config", args.config),
        "deploy_config_path": session_spec.get("deploy_config", args.deploy_config),
        "primitive_sleep_s": float(deploy_cfg.get("runtime", {}).get("primitive_sleep_s", 0.8)),
        "post_primitive_settle_s": post_primitive_settle_s,
        "live_preview": live_preview,
        "record_episode_video": record_episode_video,
        "episode_video_fps": episode_video_fps,
        "session_key": session_key,
        "session_dir": str(session_dir),
        "session_dataset_path": str(dataset_path),
        "split_role": split_role,
        "task": task_name,
        "task_id": task_id,
        "layout_tag": session_spec.get("layout_tag", "unspecified"),
        "operator": operator,
        "notes": notes,
        "object_name": session_spec.get("object_name", ""),
        "dropzone_name": session_spec.get("dropzone_name", ""),
        "scene_constraints": session_spec.get("scene_constraints", []),
        "video_global_expectations": session_spec.get("video_global_expectations", []),
        "placement_guide": session_spec.get("placement_guide", []),
        "acceptance_criteria": session_spec.get("acceptance_criteria", []),
        "video_expectations": session_spec.get("video_expectations", []),
        "primitive_vocabulary": "legacy",
        "episodes_planned": len(episodes),
        "episodes_collected": len(episode_records),
        "transitions_collected": len(primitive_ids),
        "frame_size": None if frame_size is None else {"width": frame_size[0], "height": frame_size[1]},
        "episode_records": episode_records,
    }
    write_json(session_dir / "meta.json", meta)

    if save_preview and frame_size is not None:
        _write_preview(session_dir, preview_entries, frame_size)

    print(f"saved_transition_session={session_dir}")
    print(f"saved_transition_dataset={dataset_path}")
    print(f"episodes_collected={len(episode_records)}")
    print(f"transitions_collected={len(primitive_ids)}")


if __name__ == "__main__":
    main()

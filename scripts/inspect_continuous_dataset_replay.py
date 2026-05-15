from __future__ import annotations

import argparse
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from sim_demo_common import ACCENT, SUCCESS, WARN, add_shared_args
from ttla.config import load_config
from ttla.sim import ContinuousRoArmSimEnv
from ttla.sim.context import context_from_full_vector
from ttla.sim.task_defs import ID_TO_TASK


WINDOW_NAME = "TTLA Continuous Dataset Replay Inspector"
GUI_DISABLED_MESSAGE: Final[str] = (
    "OpenCV highgui backend unavailable; falling back to a Tk preview window while keeping MuJoCo viewer interactive."
)
VIEW_MODES: Final[tuple[str, ...]] = ("free", "forearm_cam", "overview_cam")


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: int
    indices: np.ndarray
    task_id: int
    task_name: str
    task_text: str
    context_full: np.ndarray
    target_init_pos: np.ndarray
    drop_init_pos: np.ndarray
    episode_success: int


class _TkPreviewWindow:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title(WINDOW_NAME)
        self.root.geometry("1600x980")
        self.label = tk.Label(self.root)
        self.label.pack(fill="both", expand=True)
        self._image = None
        self._queue: deque[str] = deque()
        self._closed = False
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        for key in ("space", "Return", "a", "n", "p", "j", "r", "v", "q", "Escape"):
            self.root.bind(f"<{key}>", self._on_key)

    def _on_key(self, event) -> None:
        mapping = {"space": " ", "Return": " ", "Escape": "q"}
        self._queue.append(mapping.get(event.keysym, str(event.keysym).lower()))

    def _on_close(self) -> None:
        self._closed = True
        self._queue.append("q")
        self.root.destroy()

    def show(self, frame: np.ndarray) -> None:
        if self._closed:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ok, encoded = cv2.imencode(".png", rgb)
        if not ok:
            return
        self._image = tk.PhotoImage(data=encoded.tobytes())
        self.label.configure(image=self._image)

    def poll_key(self, delay_ms: int) -> str:
        if self._closed:
            return "q"
        self.root.update_idletasks()
        self.root.update()
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)
        if self._queue:
            return self._queue.popleft()
        return ""

    def close(self) -> None:
        if not self._closed:
            self.root.destroy()
            self._closed = True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactively replay saved continuous dataset episodes in MuJoCo.")
    add_shared_args(parser, default_output_root="results/continuous_dataset_replay")
    parser.add_argument("--input", required=True, help="Path to a continuous NPZ file.")
    parser.add_argument(
        "--tasks",
        default="",
        help="Optional comma-separated task names to filter episodes before sampling.",
    )
    parser.add_argument("--frame-sleep-s", type=float, default=0.03)
    return parser.parse_args()


def _init_preview_window():
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
        cv2.resizeWindow(WINDOW_NAME, 1600, 980)
        return "cv2"
    except cv2.error:
        print(GUI_DISABLED_MESSAGE)
    try:
        return _TkPreviewWindow()
    except tk.TclError:
        return None


def _render_preview_frame(preview_backend, frame: np.ndarray) -> None:
    if preview_backend is None:
        return
    if preview_backend == "cv2":
        cv2.imshow(WINDOW_NAME, frame)
        return
    preview_backend.show(frame)


def _poll_preview_key(preview_backend, delay_ms: int) -> str:
    if preview_backend is None:
        return ""
    if preview_backend == "cv2":
        raw_key = cv2.waitKey(delay_ms)
        return "" if raw_key < 0 else chr(raw_key & 0xFF)
    return preview_backend.poll_key(delay_ms)


def _process_gui_delay(preview_backend, frame_sleep_s: float) -> None:
    if preview_backend == "cv2":
        cv2.waitKey(max(1, int(frame_sleep_s * 1000)))
        return
    if preview_backend is not None:
        preview_backend.poll_key(max(1, int(frame_sleep_s * 1000)))
        return
    if frame_sleep_s > 0:
        time.sleep(frame_sleep_s)


def _terminal_command() -> str:
    command = input(
        "[dataset replay] command: space(step) | a(auto) | n(next) | p(prev) | j(random) | r(reset) | v(view) | q(quit) > "
    ).strip().lower()
    if command in ("", "space", "step", "enter"):
        return " "
    if command in ("esc", "escape", "quit"):
        return "q"
    return command


def _set_view_mode(env: ContinuousRoArmSimEnv, viewer, view_mode: str) -> None:
    if view_mode == "free":
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        return
    camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, view_mode)
    viewer.cam.fixedcamid = int(camera_id)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED


def _put(canvas: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.56, color=(32, 37, 48), thickness: int = 1) -> None:
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, lineType=cv2.LINE_AA)


def _frame_tile(image_rgb: np.ndarray, label: str) -> np.ndarray:
    tile = np.full((480, 480, 3), 248, dtype=np.uint8)
    cv2.rectangle(tile, (0, 0), (479, 479), (218, 223, 232), 1, lineType=cv2.LINE_AA)
    resized = cv2.resize(image_rgb[:, :, ::-1], (440, 440), interpolation=cv2.INTER_CUBIC)
    tile[24:464, 20:460] = resized
    cv2.rectangle(tile, (0, 0), (479, 34), (255, 255, 255), -1)
    _put(tile, label, (14, 24), 0.64, (32, 37, 48), 1)
    return tile


def _preview(
    env: ContinuousRoArmSimEnv,
    record: EpisodeRecord,
    stored_images: np.ndarray,
    actions: np.ndarray,
    success_flags: np.ndarray,
    current_local_idx: int,
    status: str,
    status_color,
    view_mode: str,
) -> np.ndarray:
    live_image = env.observe()["image"]
    stored_image = stored_images[record.indices[current_local_idx]]
    diff = cv2.absdiff(live_image, stored_image)
    mae = float(np.mean(np.abs(live_image.astype(np.int16) - stored_image.astype(np.int16))))
    canvas = np.full((980, 1540, 3), 242, dtype=np.uint8)
    _put(canvas, "Continuous Dataset Replay Inspector", (28, 34), 0.88, (32, 37, 48), 2)
    _put(
        canvas,
        f"episode={record.episode_id} task={record.task_name} local_step={current_local_idx}/{len(record.indices) - 1} view={view_mode}",
        (30, 62),
        0.48,
        (98, 108, 125),
        1,
    )
    live_tile = _frame_tile(live_image, "replay current forearm_cam")
    stored_tile = _frame_tile(stored_image, "stored dataset image")
    diff_tile = _frame_tile(diff, f"abs diff (mae={mae:.2f})")
    canvas[92:572, 24:504] = live_tile
    canvas[92:572, 530:1010] = stored_tile
    canvas[92:572, 1036:1516] = diff_tile

    idx = record.indices[current_local_idx]
    y = 620
    lines = [
        f"status={status}",
        f"episode_success={record.episode_success}",
        f"stored_success_flag={int(success_flags[idx])}",
        f"visibility={env.visibility_score():.3f} center_error_px={env.center_error_px():.1f}",
        f"target_occlusion_ratio={env.target_occlusion_ratio():.3f}",
        f"gripper_intrusion_ratio={env.gripper_intrusion_ratio():.3f}",
        f"grasp_gap={env.grasp_gap():+.3f} ee_target_distance={env.ee_target_distance():.3f}",
        f"dropzone_distance={env.dropzone_distance():.3f} placed={int(env.placed)}",
        f"target_pixels={env.target_visible_pixels()} components={env.target_visible_component_count()} keypoint_ratio={env.target_keypoint_visibility_ratio():.2f}",
        f"action=[{', '.join(f'{float(v):+.3f}' for v in actions[idx])}]",
        f"qpos=[{', '.join(f'{float(v):+.3f}' for v in env.data.qpos[:6])}]",
        "controls: Space step | A autoplay | N/P next-prev episode | J random episode | R reset episode | V viewer view | Q quit",
    ]
    for line in lines:
        _put(canvas, line, (32, y), 0.54, status_color if line.startswith("status=") else (32, 37, 48), 1)
        y += 28
    return canvas


def _viewer_overlay(
    viewer,
    env: ContinuousRoArmSimEnv,
    record: EpisodeRecord,
    success_flags: np.ndarray,
    local_idx: int,
    view_mode: str,
) -> None:
    idx = record.indices[local_idx]
    texts = [
        (
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            "TTLA Dataset Replay Inspector",
            (
                f"episode={record.episode_id} task={record.task_name}\n"
                f"local_step={local_idx}/{len(record.indices) - 1} | view={view_mode}\n"
                f"vis={env.visibility_score():.3f} center={env.center_error_px():.1f}px "
                f"occ={env.target_occlusion_ratio():.3f} intrusion={env.gripper_intrusion_ratio():.3f} gap={env.grasp_gap():+.3f}m placed={int(env.placed)}\n"
                f"stored_success_flag={int(success_flags[idx])} episode_success={record.episode_success}"
            ),
        ),
        (
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPRIGHT,
            "controls",
            "Space/Enter one action\nA autoplay to end\nN/P next/prev episode\nJ random episode\nR reset episode\nV cycle view\nQ quit",
        ),
    ]
    viewer.set_texts(texts)


def _load_records(payload: np.lib.npyio.NpzFile, task_filter: set[str]) -> list[EpisodeRecord]:
    required = {"images", "actions", "tasks", "success", "episode_ids", "step_ids", "contexts_full", "target_init_pos", "drop_init_pos"}
    missing = sorted(required.difference(payload.files))
    if missing:
        raise KeyError(
            "Dataset does not contain replay metadata required for exact MuJoCo spot checks. "
            f"Missing fields: {missing}"
        )
    episode_ids = payload["episode_ids"].astype(np.int64)
    step_ids = payload["step_ids"].astype(np.int64)
    tasks = payload["tasks"].astype(np.int64)
    success = payload["success"].astype(np.int64)
    task_text = payload.get("task_text")
    order = np.lexsort((step_ids, episode_ids))
    unique_episodes: list[int] = []
    seen: set[int] = set()
    for episode_id in episode_ids[order]:
        episode_id = int(episode_id)
        if episode_id not in seen:
            unique_episodes.append(episode_id)
            seen.add(episode_id)
    records: list[EpisodeRecord] = []
    for episode_id in unique_episodes:
        indices = np.flatnonzero(episode_ids == episode_id)
        indices = indices[np.argsort(step_ids[indices])]
        task_id = int(tasks[indices[0]])
        task_name = ID_TO_TASK[task_id].name
        if task_filter and task_name not in task_filter:
            continue
        records.append(
            EpisodeRecord(
                episode_id=episode_id,
                indices=indices,
                task_id=task_id,
                task_name=task_name,
                task_text=str(task_text[indices[0]]) if task_text is not None else task_name,
                context_full=np.asarray(payload["contexts_full"][indices[0]], dtype=np.float32),
                target_init_pos=np.asarray(payload["target_init_pos"][indices[0]], dtype=np.float32),
                drop_init_pos=np.asarray(payload["drop_init_pos"][indices[0]], dtype=np.float32),
                episode_success=int(np.max(success[indices])),
            )
        )
    if not records:
        raise RuntimeError("No episodes left after task filtering.")
    return records


def _restore_episode(env: ContinuousRoArmSimEnv, record: EpisodeRecord) -> None:
    context = context_from_full_vector(record.context_full)
    env.reset(task_name=record.task_name, context=context)
    target_body = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "target")
    drop_body = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "distractor")
    env.model.body_pos[target_body] = np.asarray(record.target_init_pos, dtype=np.float64)
    env.model.body_pos[drop_body] = np.asarray(record.drop_init_pos, dtype=np.float64)
    mujoco.mj_forward(env.model, env.data)


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    payload = np.load(Path(args.input), allow_pickle=True)
    payload_images = payload["images"]
    payload_actions = payload["actions"].astype(np.float32)
    payload_success = payload["success"].astype(np.int64)
    task_filter = {token.strip() for token in args.tasks.split(",") if token.strip()}
    records = _load_records(payload, task_filter)
    rng = np.random.default_rng(int(args.seed))

    env = ContinuousRoArmSimEnv(
        cfg["sim"],
        seed=int(args.seed),
        action_low=np.asarray(cfg["control"]["action"].get("clamp_low", [-0.25] * 6), dtype=np.float32),
        action_high=np.asarray(cfg["control"]["action"].get("clamp_high", [0.25] * 6), dtype=np.float32),
        control_mode=cfg["control"]["action"].get("control_mode", "joint_delta"),
    )
    preview_backend = _init_preview_window()
    viewer = mujoco.viewer.launch_passive(env.model, env.data, show_left_ui=True, show_right_ui=True)

    record_index = int(rng.integers(0, len(records)))
    local_idx = 0
    status = "READY"
    status_color = ACCENT
    view_index = 0

    def set_record(index: int, *, random_pick: bool = False) -> EpisodeRecord:
        nonlocal record_index, local_idx, status, status_color
        record_index = int(index)
        local_idx = 0
        record = records[record_index]
        _restore_episode(env, record)
        status = "RANDOM READY" if random_pick else "READY"
        status_color = ACCENT if not random_pick else WARN
        return record

    record = set_record(record_index, random_pick=True)
    _set_view_mode(env, viewer, VIEW_MODES[view_index])

    try:
        while viewer.is_running():
            _viewer_overlay(viewer, env, record, payload_success, local_idx, VIEW_MODES[view_index])
            viewer.sync()
            _render_preview_frame(
                preview_backend,
                _preview(
                    env,
                    record,
                    payload_images,
                    payload_actions,
                    payload_success,
                    local_idx,
                    status,
                    status_color,
                    VIEW_MODES[view_index],
                ),
            )
            if preview_backend is not None:
                key = _poll_preview_key(preview_backend, 15)
            else:
                print(
                    f"[dataset replay] episode={record.episode_id} task={record.task_name} step={local_idx}/{len(record.indices)-1} "
                    f"vis={env.visibility_score():.3f} center={env.center_error_px():.1f} placed={int(env.placed)} status={status}"
                )
                key = _terminal_command()

            if key in ("q", "\x1b"):
                break
            if key == "n":
                record = set_record((record_index + 1) % len(records))
                continue
            if key == "p":
                record = set_record((record_index - 1) % len(records))
                continue
            if key == "j":
                record = set_record(int(rng.integers(0, len(records))), random_pick=True)
                continue
            if key == "r":
                record = set_record(record_index)
                status, status_color = "RESET", WARN
                continue
            if key == "v":
                view_index = (view_index + 1) % len(VIEW_MODES)
                _set_view_mode(env, viewer, VIEW_MODES[view_index])
                status, status_color = f"VIEW {VIEW_MODES[view_index]}", ACCENT
                continue

            if key in (" ", "\r"):
                idx = record.indices[local_idx]
                _, _, done, info = env.step_action(payload_actions[idx])
                if info["success"]:
                    status, status_color = "SUCCESS", SUCCESS
                elif done or local_idx >= len(record.indices) - 1:
                    status, status_color = "DONE", WARN
                else:
                    status, status_color = f"STEP {local_idx + 1}", ACCENT
                if local_idx < len(record.indices) - 1:
                    local_idx += 1
                continue

            if key == "a":
                while viewer.is_running() and local_idx < len(record.indices):
                    idx = record.indices[local_idx]
                    _, _, done, info = env.step_action(payload_actions[idx])
                    if info["success"]:
                        status, status_color = "SUCCESS", SUCCESS
                    elif done:
                        status, status_color = "DONE", WARN
                    else:
                        status, status_color = f"STEP {local_idx + 1}", ACCENT
                    if local_idx < len(record.indices) - 1:
                        local_idx += 1
                    _viewer_overlay(viewer, env, record, payload_success, local_idx, VIEW_MODES[view_index])
                    viewer.sync()
                    _render_preview_frame(
                        preview_backend,
                        _preview(
                            env,
                            record,
                            payload_images,
                            payload_actions,
                            payload_success,
                            local_idx,
                            status,
                            status_color,
                            VIEW_MODES[view_index],
                        ),
                    )
                    _process_gui_delay(preview_backend, float(args.frame_sleep_s))
                    if done or status == "SUCCESS":
                        break
                continue
    finally:
        viewer.close()
        env.close()
        if preview_backend == "cv2":
            cv2.destroyAllWindows()
        elif preview_backend is not None:
            preview_backend.close()


if __name__ == "__main__":
    main()

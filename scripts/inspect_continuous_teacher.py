from __future__ import annotations

import argparse
import time
import tkinter as tk
from collections import deque
from typing import Final

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from sim_demo_common import ACCENT, SUCCESS, WARN, add_shared_args, panel_frame
from ttla.config import load_config
from ttla.sim import ContinuousRoArmSimEnv, ContinuousWaypointExpert
from ttla.sim.context import neutral_context
from ttla.sim.task_defs import TASK_SPECS


WINDOW_NAME = "TTLA Continuous Teacher Inspector"
GUI_DISABLED_MESSAGE: Final[str] = (
    "OpenCV highgui backend unavailable; falling back to a Tk preview window while keeping MuJoCo viewer interactive."
)
VIEW_MODES: Final[tuple[str, ...]] = ("forearm_cam", "overview_cam", "free")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive MuJoCo inspector for continuous teacher rollouts."
    )
    add_shared_args(parser, default_output_root="results/continuous_teacher_demos")
    parser.add_argument(
        "--tasks",
        default="level1_verify,level2_approach,level3_pick_place",
        help="Comma-separated task names.",
    )
    parser.add_argument("--context-mode", choices=("neutral", "random"), default="neutral")
    parser.add_argument("--frame-sleep-s", type=float, default=0.03)
    return parser.parse_args()


def _viewer_overlay(
    handle,
    env,
    task_name: str,
    context_mode: str,
    view_mode: str,
    phase: str,
    micro_step: int,
    action: np.ndarray | None,
) -> None:
    action_text = "none" if action is None else "[" + ", ".join(f"{float(v):+.3f}" for v in action) + "]"
    texts = [
        (
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            "TTLA Continuous Teacher Inspector",
            (
                f"task={task_name} | context={context_mode} | step={env.step_idx}\n"
                f"view={view_mode} | phase={phase} | internal_step={micro_step}\n"
                f"vis={env.visibility_score():.3f} center={env.center_error_px():.1f}px "
                f"pixels={env.target_visible_pixels()} comps={env.target_visible_component_count()} "
                f"occ={env.target_occlusion_ratio():.3f} intrusion={env.gripper_intrusion_ratio():.3f} "
                f"gap={env.grasp_gap():+.3f}m dist={env.ee_target_distance():.3f}m\n"
                f"action={action_text}"
            ),
        ),
        (
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPRIGHT,
            "controls",
            "Space/Enter one teacher step\nA autoplay until done\nN next task\nP previous task\nR reset task\nC toggle neutral/random\nV cycle view\nQ quit",
        ),
    ]
    handle.set_texts(texts)


def _preview(env, task_name: str, context_mode: str, status: str, status_color, last_action: np.ndarray | None, last_reward: float | None) -> np.ndarray:
    obs = env.observe()
    action_text = "none" if last_action is None else "[" + ", ".join(f"{float(v):+.3f}" for v in last_action) + "]"
    lines = [
        f"task={task_name}",
        f"context_mode={context_mode}",
        f"step_idx={env.step_idx}",
        f"visibility={env.visibility_score():.3f}",
        f"center_error_px={env.center_error_px():.1f}",
        f"target_pixels={env.target_visible_pixels()} components={env.target_visible_component_count()} keypoint_ratio={env.target_keypoint_visibility_ratio():.2f}",
        f"target_occlusion_ratio={env.target_occlusion_ratio():.3f}",
        f"gripper_intrusion_ratio={env.gripper_intrusion_ratio():.3f}",
        f"grasp_gap={env.grasp_gap():+.3f}",
        f"ee_target_distance={env.ee_target_distance():.3f}",
        f"dropzone_distance={env.dropzone_distance():.3f}",
        f"flags verified={int(env.verified)} grasped={int(env.object_attached)} lifted={int(env.lifted)} placed={int(env.placed)}",
        f"last_action={action_text}",
    ]
    if last_reward is not None:
        lines.append(f"last_reward={last_reward:+.3f}")
    lines.append("controls: Space step | A autoplay | N/P task | R reset | C toggle context | Q quit")
    return panel_frame(
        env,
        obs,
        title=f"Continuous Teacher Inspector: {task_name}",
        subtitle=f"context={context_mode}",
        lines=lines,
        status_label=status,
        status_color=status_color,
    )


class _TkPreviewWindow:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title(WINDOW_NAME)
        self.root.geometry("1320x900")
        self.label = tk.Label(self.root)
        self.label.pack(fill="both", expand=True)
        self._image = None
        self._queue: deque[str] = deque()
        self._closed = False
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        for key in ("space", "Return", "a", "n", "p", "r", "c", "v", "q", "Escape"):
            self.root.bind(f"<{key}>", self._on_key)

    def _on_key(self, event) -> None:
        key = event.keysym
        mapping = {
            "space": " ",
            "Return": " ",
            "Escape": "q",
        }
        self._queue.append(mapping.get(key, str(key).lower()))

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


def _init_preview_window():
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
        cv2.resizeWindow(WINDOW_NAME, 1320, 900)
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
        cv2.waitKey(1)
        if frame_sleep_s > 0:
            cv2.waitKey(max(1, int(frame_sleep_s * 1000)))
        return
    if preview_backend is not None:
        preview_backend.poll_key(max(1, int(frame_sleep_s * 1000)))
        return
    if frame_sleep_s > 0:
        time.sleep(frame_sleep_s)


def _terminal_command() -> str:
    command = input(
        "[teacher inspector] command: space(step) | a(auto) | n(next) | p(prev) | r(reset) | c(context) | v(view) | q(quit) > "
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


def _teacher_step_with_animation(
    env: ContinuousRoArmSimEnv,
    expert: ContinuousWaypointExpert,
    viewer,
    task_name: str,
    context_mode: str,
    view_mode: str,
    frame_sleep_s: float,
    preview_backend,
) -> tuple[np.ndarray, float, bool, dict]:
    action = expert.act(env)
    micro_step = 0
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
            mujoco.mj_forward(env.model, env.data)
            micro_step += 1
            _viewer_overlay(viewer, env, task_name, context_mode, view_mode, "internal", micro_step, action)
            viewer.sync()
            _render_preview_frame(
                preview_backend,
                _preview(env, task_name, context_mode, "RUNNING", ACCENT, action, None),
            )
            _process_gui_delay(preview_backend, frame_sleep_s)

    env._apply_target_pose = animated_apply_target_pose  # type: ignore[method-assign]
    try:
        _, reward, done, info = env.step_action(action)
    finally:
        env._apply_target_pose = original_apply_target_pose  # type: ignore[method-assign]
    return action, reward, bool(done), info


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
    tasks = [token.strip() for token in args.tasks.split(",") if token.strip()]
    for task_name in tasks:
        if task_name not in TASK_SPECS:
            raise KeyError(f"Unknown task: {task_name}")

    preview_backend = _init_preview_window()
    viewer = mujoco.viewer.launch_passive(env.model, env.data, show_left_ui=True, show_right_ui=True)

    task_index = 0
    context_mode = str(args.context_mode)
    status = "READY"
    status_color = ACCENT
    last_action: np.ndarray | None = None
    last_reward: float | None = None
    last_done = False
    view_index = 2 if preview_backend is not None else 0

    def reset_current() -> str:
        nonlocal last_action, last_reward, last_done
        task_name = tasks[task_index]
        context = neutral_context() if context_mode == "neutral" else None
        env.reset(task_name=task_name, context=context)
        expert.reset(task_name=task_name)
        last_action = None
        last_reward = None
        last_done = False
        return task_name

    task_name = reset_current()
    _set_view_mode(env, viewer, VIEW_MODES[view_index])

    try:
        while viewer.is_running():
            _viewer_overlay(viewer, env, task_name, context_mode, VIEW_MODES[view_index], "idle", 0, last_action)
            viewer.sync()
            _render_preview_frame(
                preview_backend,
                _preview(env, task_name, context_mode, status, status_color, last_action, last_reward),
            )
            if preview_backend is not None:
                key = _poll_preview_key(preview_backend, 15)
            else:
                print(
                    f"[teacher inspector] task={task_name} context={context_mode} view={VIEW_MODES[view_index]} step={env.step_idx} "
                    f"vis={env.visibility_score():.3f} center={env.center_error_px():.1f} "
                    f"occ={env.target_occlusion_ratio():.3f} intrusion={env.gripper_intrusion_ratio():.3f} gap={env.grasp_gap():+.3f} placed={int(env.placed)} status={status}"
                )
                key = _terminal_command()
            if key in ("q", "\x1b"):
                break
            if key == "n":
                task_index = (task_index + 1) % len(tasks)
                task_name = reset_current()
                status, status_color = "READY", ACCENT
                continue
            if key == "p":
                task_index = (task_index - 1) % len(tasks)
                task_name = reset_current()
                status, status_color = "READY", ACCENT
                continue
            if key == "c":
                context_mode = "random" if context_mode == "neutral" else "neutral"
                task_name = reset_current()
                status, status_color = context_mode.upper(), WARN
                continue
            if key == "v":
                view_index = (view_index + 1) % len(VIEW_MODES)
                _set_view_mode(env, viewer, VIEW_MODES[view_index])
                status, status_color = f"VIEW {VIEW_MODES[view_index]}", ACCENT
                continue
            if key == "r":
                task_name = reset_current()
                status, status_color = "RESET", WARN
                continue

            if key in (" ", "\r"):
                action, reward, done, info = _teacher_step_with_animation(
                    env,
                    expert,
                    viewer,
                    task_name,
                    context_mode,
                    VIEW_MODES[view_index],
                    float(args.frame_sleep_s),
                    preview_backend,
                )
                last_action = action
                last_reward = float(reward)
                last_done = bool(done)
                if info["success"]:
                    status, status_color = "SUCCESS", SUCCESS
                elif done:
                    status, status_color = "DONE", WARN
                else:
                    status, status_color = f"STEP {env.step_idx}", ACCENT
                continue

            if key == "a":
                while viewer.is_running() and not last_done:
                    action, reward, done, info = _teacher_step_with_animation(
                        env,
                        expert,
                        viewer,
                        task_name,
                        context_mode,
                        VIEW_MODES[view_index],
                        float(args.frame_sleep_s),
                        preview_backend,
                    )
                    last_action = action
                    last_reward = float(reward)
                    last_done = bool(done)
                    if info["success"]:
                        status, status_color = "SUCCESS", SUCCESS
                        break
                    if done:
                        status, status_color = "DONE", WARN
                        break
                    status, status_color = f"STEP {env.step_idx}", ACCENT
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

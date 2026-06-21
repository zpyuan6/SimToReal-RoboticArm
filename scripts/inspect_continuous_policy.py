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
import torch

from sim_demo_common import ACCENT, SUCCESS, WARN, add_shared_args, panel_frame
from ttla.config import load_config
from ttla.evaluation.evaluate_continuous import (
    _build_env,
    _build_interface_spec,
    _build_observation_batch,
    _merge_official_eval_cfg,
    resolve_official_policy_path,
)
from ttla.control import build_control_backbone
from ttla.sim.task_defs import TASK_SPECS


WINDOW_NAME = "TTLA Continuous Policy Inspector"
GUI_DISABLED_MESSAGE: Final[str] = (
    "OpenCV highgui backend unavailable; falling back to a Tk preview window while keeping MuJoCo viewer interactive."
)
VIEW_MODES: Final[tuple[str, ...]] = ("forearm_cam", "overview_cam", "free")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive MuJoCo inspector for continuous policy rollouts."
    )
    add_shared_args(parser, default_output_root="results/continuous_policy_demos")
    parser.add_argument(
        "--policy-path",
        required=True,
        help="Official LeRobot policy path or training output root.",
    )
    parser.add_argument(
        "--tasks",
        default="level1_verify,level2_approach,level3_pick_place",
        help="Comma-separated task names.",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Target episode index within the selected task.",
    )
    parser.add_argument(
        "--eval-replay",
        action="store_true",
        help="Advance RNG in the same task order as evaluate_continuous_backbone so episode-index matches episodes.csv.",
    )
    parser.add_argument(
        "--eval-episodes-per-task",
        type=int,
        default=12,
        help="Episodes per task used during rollout evaluation when --eval-replay is enabled.",
    )
    parser.add_argument("--frame-sleep-s", type=float, default=0.03)
    return parser.parse_args()


def _viewer_overlay(
    handle,
    env,
    task_name: str,
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
            "TTLA Continuous Policy Inspector",
            (
                f"task={task_name} | step={env.step_idx}\n"
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
            "Space/Enter one policy step\nA autoplay until done\nN next task\nP previous task\nR reset task\nV cycle view\nQ quit",
        ),
    ]
    handle.set_texts(texts)


def _preview(env, task_name: str, status: str, status_color, last_action: np.ndarray | None, last_reward: float | None) -> np.ndarray:
    obs = env.observe()
    action_text = "none" if last_action is None else "[" + ", ".join(f"{float(v):+.3f}" for v in last_action) + "]"
    lines = [
        f"task={task_name}",
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
    lines.append("controls: Space step | A autoplay | N/P task | R reset | Q quit")
    return panel_frame(
        env,
        obs,
        title=f"Continuous Policy Inspector: {task_name}",
        subtitle="policy rollout",
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
        for key in ("space", "Return", "a", "n", "p", "r", "v", "q", "Escape"):
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
        "[policy inspector] command: space(step) | a(auto) | n(next) | p(prev) | r(reset) | v(view) | q(quit) > "
    ).strip().lower()
    if command in ("", "space", "step", "enter"):
        return " "
    if command in ("esc", "escape", "quit"):
        return "q"
    return command


def _set_view_mode(env, viewer, view_mode: str) -> None:
    if view_mode == "free":
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        return
    camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, view_mode)
    viewer.cam.fixedcamid = int(camera_id)
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED


def _policy_action(backbone, batch) -> np.ndarray:
    with torch.no_grad():
        policy_output = backbone.forward_policy(batch)
    return policy_output.actions[0, 0].detach().cpu().numpy().astype(np.float32)


def _policy_step_with_animation(
    env,
    backbone,
    history_len: int,
    uses_language: bool,
    viewer,
    task_name: str,
    view_mode: str,
    frame_sleep_s: float,
    preview_backend,
    obs_history: list[dict[str, np.ndarray]],
) -> tuple[np.ndarray, float, bool, dict, dict[str, np.ndarray]]:
    batch = _build_observation_batch(
        obs_history,
        history_len=history_len,
        task_text=env.task_text(),
        uses_language=uses_language,
    )
    action = _policy_action(backbone, batch)
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
            _viewer_overlay(viewer, env, task_name, view_mode, "internal", micro_step, action)
            viewer.sync()
            _render_preview_frame(
                preview_backend,
                _preview(env, task_name, "RUNNING", ACCENT, action, None),
            )
            _process_gui_delay(preview_backend, frame_sleep_s)

    env._apply_target_pose = animated_apply_target_pose  # type: ignore[method-assign]
    try:
        next_obs, reward, done, info = env.step_action(action)
    finally:
        env._apply_target_pose = original_apply_target_pose  # type: ignore[method-assign]
    return action, reward, bool(done), info, next_obs


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    interface_spec = _build_interface_spec(cfg)
    official_cfg = _merge_official_eval_cfg(
        cfg,
        resolve_official_policy_path(args.policy_path),
        policy_device=None,
    )
    backbone = build_control_backbone(cfg["control"]["backbone_name"], interface_spec, official_cfg=official_cfg)
    backbone.eval()

    env_seed = int(args.seed if args.seed is not None else cfg["seed"])
    env = _build_env(cfg, seed=env_seed + 101)
    history_len = int(cfg.get("data_continuous", {}).get("history_len", 1))
    uses_language = bool(interface_spec.uses_language)

    tasks = [token.strip() for token in args.tasks.split(",") if token.strip()]
    for task_name in tasks:
        if task_name not in TASK_SPECS:
            raise KeyError(f"Unknown task: {task_name}")

    preview_backend = _init_preview_window()
    viewer = mujoco.viewer.launch_passive(env.model, env.data, show_left_ui=True, show_right_ui=True)

    task_index = 0
    status = "READY"
    status_color = ACCENT
    last_action: np.ndarray | None = None
    last_reward: float | None = None
    last_done = False
    view_index = 2 if preview_backend is not None else 0
    obs_history: list[dict[str, np.ndarray]] = []

    def reset_current() -> str:
        nonlocal last_action, last_reward, last_done, obs_history
        task_name = tasks[task_index]
        if args.eval_replay:
            task_order = list(cfg["sim"]["tasks"])
            if task_name not in task_order:
                raise KeyError(f"Task {task_name} not found in config sim.tasks for eval replay.")
            for warm_task in task_order:
                warm_repeats = max(0, int(args.episode_index)) + 1 if warm_task == task_name else int(args.eval_episodes_per_task)
                for _ in range(warm_repeats):
                    obs = env.reset(task_name=warm_task)
                    backbone.reset_policy_state()
                    obs_history = [obs]
                if warm_task == task_name:
                    break
        else:
            repeats = max(0, int(args.episode_index)) + 1
            for _ in range(repeats):
                obs = env.reset(task_name=task_name)
                backbone.reset_policy_state()
                obs_history = [obs]
        last_action = None
        last_reward = None
        last_done = False
        return task_name

    task_name = reset_current()
    _set_view_mode(env, viewer, VIEW_MODES[view_index])

    try:
        while viewer.is_running():
            _viewer_overlay(viewer, env, task_name, VIEW_MODES[view_index], "idle", 0, last_action)
            viewer.sync()
            _render_preview_frame(
                preview_backend,
                _preview(env, task_name, status, status_color, last_action, last_reward),
            )
            if preview_backend is not None:
                key = _poll_preview_key(preview_backend, 15)
            else:
                print(
                    f"[policy inspector] task={task_name} view={VIEW_MODES[view_index]} step={env.step_idx} "
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
                action, reward, done, info, next_obs = _policy_step_with_animation(
                    env,
                    backbone,
                    history_len,
                    uses_language,
                    viewer,
                    task_name,
                    VIEW_MODES[view_index],
                    float(args.frame_sleep_s),
                    preview_backend,
                    obs_history,
                )
                obs_history.append(next_obs)
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
                    action, reward, done, info, next_obs = _policy_step_with_animation(
                        env,
                        backbone,
                        history_len,
                        uses_language,
                        viewer,
                        task_name,
                        VIEW_MODES[view_index],
                        float(args.frame_sleep_s),
                        preview_backend,
                        obs_history,
                    )
                    obs_history.append(next_obs)
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

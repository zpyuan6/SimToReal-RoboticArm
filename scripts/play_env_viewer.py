from __future__ import annotations

import argparse
import queue
import time
from datetime import datetime
from pathlib import Path

import cv2
import glfw
import mujoco
import mujoco.viewer
import numpy as np
import torch

from ttla.config import load_config
from ttla.evaluation.baselines import baseline_overrides
from ttla.training import build_model
from ttla.sim import RoArmSimEnv, ScriptedExpert
from ttla.sim.skills import (
    ABORT_ID,
    APPROACH_COARSE_ID,
    APPROACH_FINE_ID,
    GRASP_EXECUTE_ID,
    LIFT_OBJECT_ID,
    OBS_CENTER_ID,
    OBS_LEFT_ID,
    OBS_RIGHT_ID,
    PLACE_OBJECT_ID,
    PREALIGN_GRASP_ID,
    PREGRASP_SERVO_ID,
    PRIMITIVE_NAMES,
    REOBSERVE_ID,
    RETREAT_ID,
    TRANSPORT_TO_DROPZONE_ID,
    VERIFY_TARGET_ID,
    primitive_name,
)
from ttla.sim.task_defs import TASK_TO_ID
from ttla.utils.episode import EpisodeBuffer
from ttla.utils.io import ensure_dir


MANUAL_KEYMAP = {
    glfw.KEY_A: OBS_LEFT_ID,
    glfw.KEY_D: OBS_RIGHT_ID,
    glfw.KEY_C: OBS_CENTER_ID,
    glfw.KEY_V: VERIFY_TARGET_ID,
    glfw.KEY_P: PREALIGN_GRASP_ID,
    glfw.KEY_E: APPROACH_COARSE_ID,
    glfw.KEY_R: APPROACH_FINE_ID,
    glfw.KEY_T: RETREAT_ID,
    glfw.KEY_O: REOBSERVE_ID,
    glfw.KEY_G: PREGRASP_SERVO_ID,
    glfw.KEY_F: GRASP_EXECUTE_ID,
    glfw.KEY_L: LIFT_OBJECT_ID,
    glfw.KEY_M: TRANSPORT_TO_DROPZONE_ID,
    glfw.KEY_Y: PLACE_OBJECT_ID,
    glfw.KEY_X: ABORT_ID,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", default="level1_verify", choices=["level1_verify", "level2_approach", "level3_pick_place"])
    parser.add_argument("--mode", default="manual", choices=["manual", "expert", "policy"])
    parser.add_argument("--baseline", default="ours", choices=["no_adaptation", "domain_randomization_only", "input_normalization", "few_shot_finetuning", "ours"])
    parser.add_argument("--checkpoint")
    parser.add_argument("--save-dir")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--hide-left-ui", action="store_true")
    parser.add_argument("--hide-right-ui", action="store_true")
    return parser.parse_args()


def _load_model(cfg: dict, checkpoint_path: str | Path, device: torch.device):
    payload = torch.load(checkpoint_path, map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def _normalize_image(image: np.ndarray) -> np.ndarray:
    x = image.astype(np.float32)
    x = (x - x.mean()) / max(float(x.std()), 1.0)
    x = np.clip(x * 48.0 + 127.0, 0, 255)
    return x.astype(np.uint8)


@torch.no_grad()
def _select_primitive(
    model,
    obs: dict[str, np.ndarray],
    runtime_state,
    use_adapter: bool,
    device: torch.device,
    task_id: int,
) -> tuple[int, object]:
    image_t = torch.from_numpy(obs["image"]).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    state_t = torch.from_numpy(obs["state"]).unsqueeze(0).float().to(device)
    task_ids = torch.tensor([task_id], device=device, dtype=torch.long)
    primitive, runtime_state, _ = model.act(image_t, state_t, runtime_state, use_adapter=use_adapter, task_ids=task_ids)
    return int(primitive.item()), runtime_state


def _episode_root(base_dir: str | Path) -> Path:
    root = ensure_dir(base_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / f"episode_{stamp}"


def _save_episode(episode: EpisodeBuffer, root: Path | None, args: argparse.Namespace, env: RoArmSimEnv, success: int, run_index: int) -> Path | None:
    if root is None or not episode.actions:
        return None
    saved = episode.save(
        root / f"run_{run_index:03d}",
        metadata={
            "task": env.task_name,
            "mode": args.mode,
            "baseline": args.baseline,
            "seed": args.seed,
            "steps": len(episode.actions),
            "success": success,
            "action_names": PRIMITIVE_NAMES,
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        },
    )
    return saved


def _preview_frame(env: RoArmSimEnv, obs: dict[str, np.ndarray], primitive_id: int, success: int, step: int) -> np.ndarray:
    left = cv2.resize(obs["image"], (320, 320), interpolation=cv2.INTER_CUBIC)
    right = cv2.resize(env.render_debug_view("overview_cam"), (320, 320), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((380, 640, 3), 245, dtype=np.uint8)
    canvas[18:338, 14:334] = left
    canvas[18:338, 306:626] = right
    cv2.putText(canvas, "Forearm Camera", (18, 356), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (35, 35, 35), 1, lineType=cv2.LINE_AA)
    cv2.putText(canvas, "Overview", (312, 356), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (35, 35, 35), 1, lineType=cv2.LINE_AA)
    cv2.putText(canvas, f"step {step} | primitive {primitive_name(primitive_id)} | success {success}", (18, 372), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (55, 55, 55), 1, lineType=cv2.LINE_AA)
    return canvas


def _configure_camera(handle) -> None:
    with handle.lock():
        handle.cam.azimuth = 132.0
        handle.cam.elevation = -24.0
        handle.cam.distance = 0.95
        handle.cam.lookat[:] = np.asarray([0.26, 0.0, 0.12], dtype=np.float64)


def _status_label(success: int, episode_done: bool, last_action: int) -> str:
    if success:
        return "SUCCESS"
    if episode_done and last_action == ABORT_ID:
        return "ABORTED"
    if episode_done:
        return "DONE"
    return "RUNNING"


def _controls_text() -> str:
    return (
        "a/d/c observe L/R/C\n"
        "v verify | p prealign\n"
        "e/r approach coarse/fine\n"
        "t retreat | o reobserve\n"
        "g servo | f grasp\n"
        "l lift | m move | y place\n"
        "x abort | n reset | q quit"
    )


def _update_overlay(
    handle,
    env: RoArmSimEnv,
    obs: dict[str, np.ndarray],
    args: argparse.Namespace,
    step: int,
    last_action: int,
    last_reward: float,
    success: int,
    episode_done: bool,
    action_history: list[str],
) -> None:
    status = _status_label(success, episode_done, last_action)
    history = ", ".join(action_history[-5:]) if action_history else "none"
    task_metrics = (
        f"vis={env.visibility_score():.3f} "
        f"center={env.center_error_px():.1f}px "
        f"ee->obj={env.ee_target_distance():.3f}m "
        f"ee->drop={env.dropzone_distance():.3f}m"
    )
    flags = f"verified={int(env.verified)} grasped={int(env.object_attached)} placed={int(env.placed)}"
    texts = [
        (
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            "TTLA Primitive Viewer",
            f"{status} | task={env.task_name} | mode={args.mode}\n{task_metrics}\n{flags}",
        ),
        (
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
            "step / reward / primitive",
            f"{step} / {last_reward:+.2f} / {primitive_name(last_action)}",
        ),
        (
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
            "history",
            history,
        ),
        (
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPRIGHT,
            "controls",
            _controls_text(),
        ),
    ]
    handle.set_texts(texts)
    viewport = handle.viewport
    if viewport is None:
        return
    overlay_size = min(260, max(180, viewport.width // 5))
    rect = mujoco.MjrRect(viewport.width - overlay_size - 24, viewport.height - overlay_size - 86, overlay_size, overlay_size)
    image = cv2.resize(obs["image"], (overlay_size, overlay_size), interpolation=cv2.INTER_CUBIC)
    handle.set_images((rect, image))


def main() -> None:
    args = _parse_args()
    if args.mode == "policy" and not args.checkpoint:
        raise ValueError("--checkpoint is required when --mode policy.")

    cfg = load_config(args.config)
    env = RoArmSimEnv(cfg["sim"], seed=args.seed)
    expert = ScriptedExpert()
    baseline_cfg = baseline_overrides(args.baseline)
    device = torch.device(cfg["train"]["device"])
    model = _load_model(cfg, args.checkpoint, device) if args.mode == "policy" else None

    obs = env.reset(task_name=args.task)
    episode = EpisodeBuffer()
    run_index = 0
    save_root = _episode_root(args.save_dir) if args.save_dir else None
    last_action = OBS_CENTER_ID
    last_reward = 0.0
    success = 0
    episode_done = False
    action_history: list[str] = []
    runtime_state = model.init_runtime_state(batch_size=1, device=device) if model is not None else None
    event_queue: queue.SimpleQueue[int] = queue.SimpleQueue()

    def on_key(keycode: int) -> None:
        event_queue.put(keycode)

    viewer = mujoco.viewer.launch_passive(
        env.model,
        env.data,
        key_callback=on_key,
        show_left_ui=not args.hide_left_ui,
        show_right_ui=not args.hide_right_ui,
    )
    _configure_camera(viewer)
    last_step_time = 0.0
    idle_step_time = 0.0

    try:
        step = 0
        while viewer.is_running():
            _update_overlay(viewer, env, obs, args, step, last_action, last_reward, success, episode_done, action_history)
            viewer.sync()

            requested_action = None
            reset_requested = False
            quit_requested = False
            while True:
                try:
                    key = event_queue.get_nowait()
                except queue.Empty:
                    break
                if key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
                    quit_requested = True
                elif key == glfw.KEY_N:
                    reset_requested = True
                elif key in MANUAL_KEYMAP and not episode_done:
                    requested_action = MANUAL_KEYMAP[key]

            if quit_requested:
                break

            if reset_requested:
                saved = _save_episode(episode, save_root, args, env, success, run_index)
                if saved is not None:
                    print(f"saved_episode={saved}")
                    run_index += 1
                obs = env.reset(task_name=args.task)
                episode = EpisodeBuffer()
                last_action = OBS_CENTER_ID
                last_reward = 0.0
                success = 0
                episode_done = False
                action_history = []
                step = 0
                runtime_state = model.init_runtime_state(batch_size=1, device=device) if model is not None else None
                continue

            if args.mode == "manual":
                if requested_action is None:
                    now = time.perf_counter()
                    if now - idle_step_time >= 1.0 / 60.0:
                        idle_step_time = now
                        obs = env.idle_step()
                    else:
                        time.sleep(0.005)
                    continue
                primitive_id = requested_action
            else:
                now = time.perf_counter()
                if episode_done or step >= min(args.max_steps, cfg["sim"]["episode_horizon"]):
                    break
                if now - last_step_time < 1.0 / max(args.fps, 1e-3):
                    time.sleep(0.005)
                    continue
                last_step_time = now
                if args.mode == "expert":
                    primitive_id = expert.act(env)
                else:
                    model_input = dict(obs)
                    if baseline_cfg.get("input_norm"):
                        model_input["image"] = _normalize_image(model_input["image"])
                    primitive_id, runtime_state = _select_primitive(
                        model,
                        model_input,
                        runtime_state,
                        baseline_cfg.get("use_adapter", False),
                        device,
                        TASK_TO_ID[env.task_name],
                    )

            next_obs, reward, done, info = env.step(primitive_id)
            episode.add(
                frame=_preview_frame(env, obs, primitive_id, success, step),
                state=obs["state"],
                action=primitive_id,
                context=info["context"],
                reward=reward,
                info={"task": env.task_name, "success": int(info["success"]), "visibility": float(info["visibility"])},
            )
            obs = next_obs
            last_action = primitive_id
            last_reward = reward
            success = int(info["success"])
            episode_done = done
            action_history.append(primitive_name(primitive_id))
            step += 1

            if args.mode != "manual" and episode_done:
                break

        saved = _save_episode(episode, save_root, args, env, success, run_index)
        if saved is not None:
            print(f"saved_episode={saved}")
    finally:
        viewer.close()
        env.close()


if __name__ == "__main__":
    main()

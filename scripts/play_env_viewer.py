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

from ttla.adaptation import OnlineContextAdapter
from ttla.config import load_config
from ttla.evaluation.baselines import baseline_overrides
from ttla.models import TTLAModel
from ttla.sim import RoArmSimEnv, ScriptedExpert
from ttla.sim.skills import SKILL_NAMES, skill_name
from ttla.utils.episode import EpisodeBuffer
from ttla.utils.io import ensure_dir


MANUAL_KEYMAP = {
    glfw.KEY_A: 0,
    glfw.KEY_D: 1,
    glfw.KEY_W: 2,
    glfw.KEY_S: 3,
    glfw.KEY_E: 4,
    glfw.KEY_H: 5,
    glfw.KEY_X: 6,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", default="approach", choices=["approach", "verify", "observe_then_act"])
    parser.add_argument("--mode", default="manual", choices=["manual", "expert", "policy"])
    parser.add_argument("--baseline", default="ours", choices=["no_adaptation", "domain_randomization_only", "input_normalization", "few_shot_finetuning", "ours"])
    parser.add_argument("--checkpoint")
    parser.add_argument("--save-dir")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--hide-left-ui", action="store_true")
    parser.add_argument("--hide-right-ui", action="store_true")
    return parser.parse_args()


def _load_model(cfg: dict, checkpoint_path: str | Path, device: torch.device) -> TTLAModel:
    payload = torch.load(checkpoint_path, map_location=device)
    model = TTLAModel(
        state_dim=cfg["model"]["state_dim"],
        action_dim=cfg["model"]["action_dim"],
        latent_dim=cfg["model"]["latent_dim"],
        context_dim=cfg["model"]["context_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def _normalize_image(image: np.ndarray) -> np.ndarray:
    x = image.astype(np.float32)
    x = (x - x.mean()) / max(float(x.std()), 1.0)
    x = np.clip(x * 48.0 + 127.0, 0, 255)
    return x.astype(np.uint8)


def _select_action(
    model: TTLAModel,
    obs: dict[str, np.ndarray],
    context: torch.Tensor,
    device: torch.device,
) -> tuple[int, torch.Tensor]:
    image_t = torch.from_numpy(obs["image"]).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    state_t = torch.from_numpy(obs["state"]).unsqueeze(0).float().to(device)
    z = model.encode(image_t, state_t)
    logits = model.predict_action(model.adapted_latent(z, context), context)
    return int(logits.argmax(dim=-1).item()), z


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
            "action_names": SKILL_NAMES,
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        },
    )
    return saved


def _preview_frame(env: RoArmSimEnv, obs: dict[str, np.ndarray], action_name: str, success: int, step: int) -> np.ndarray:
    left = cv2.resize(obs["image"], (320, 320), interpolation=cv2.INTER_CUBIC)
    right = cv2.resize(env.render_debug_view("overview_cam"), (320, 320), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((380, 640, 3), 245, dtype=np.uint8)
    canvas[18:338, 14:334] = left
    canvas[18:338, 306:626] = right
    cv2.putText(canvas, "Forearm Camera", (18, 356), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (35, 35, 35), 1, lineType=cv2.LINE_AA)
    cv2.putText(canvas, "Overview", (312, 356), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (35, 35, 35), 1, lineType=cv2.LINE_AA)
    cv2.putText(canvas, f"step {step} | action {action_name} | success {success}", (18, 372), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (55, 55, 55), 1, lineType=cv2.LINE_AA)
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
    if episode_done and last_action == 6:
        return "STOPPED"
    if episode_done:
        return "DONE"
    return "RUNNING"


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
    context: np.ndarray,
) -> None:
    status = _status_label(success, episode_done, last_action)
    distance = float(np.linalg.norm(env._ee_position() - env._target_position()))
    history = ", ".join(action_history[-5:]) if action_history else "none"
    context_text = " ".join(f"c{i}:{context[i]:+.2f}" for i in range(min(4, len(context))))
    mode_hint = (
        "manual: sliders live, a/d/w/s/e/h/x skills, n reset, q quit"
        if args.mode == "manual"
        else f"{args.mode}: autoplay at {args.fps:.1f} FPS"
    )
    if args.mode == "manual" and episode_done:
        mode_hint = "manual complete: n save+reset, q save+quit"

    texts = [
        (
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            "TTLA Viewer",
            f"{status} | task={env.task_name} | mode={args.mode}\ncontext {context_text}",
        ),
        (
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
            "step / reward / visibility / distance",
            f"{step} / {last_reward:+.2f} / {env.visibility_score():.3f} / {distance:.3f}m",
        ),
        (
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
            "action / history",
            f"{skill_name(last_action)} | {history}",
        ),
        (
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPRIGHT,
            "controls",
            mode_hint,
        ),
    ]
    handle.set_texts(texts)

    viewport = handle.viewport
    if viewport is None:
        return
    overlay_size = min(260, max(180, viewport.width // 5))
    rect = mujoco.MjrRect(
        viewport.width - overlay_size - 24,
        viewport.height - overlay_size - 86,
        overlay_size,
        overlay_size,
    )
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
    model = None
    adapter = None
    if args.mode == "policy":
        model = _load_model(cfg, args.checkpoint, device)
        adapter = OnlineContextAdapter(model, cfg["adaptation"], device)

    obs = env.reset(task_name=args.task)
    episode = EpisodeBuffer()
    run_index = 0
    save_root = _episode_root(args.save_dir) if args.save_dir else None
    last_action = 5
    last_reward = 0.0
    success = 0
    episode_done = False
    action_history: list[str] = []
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
            current_context = (
                adapter.context.squeeze(0).detach().cpu().numpy()
                if adapter is not None
                else np.zeros(cfg["model"]["context_dim"], dtype=np.float32)
            )
            _update_overlay(viewer, env, obs, args, step, last_action, last_reward, success, episode_done, action_history, current_context)
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
                if adapter is not None:
                    adapter.reset()
                episode = EpisodeBuffer()
                last_action = 5
                last_reward = 0.0
                success = 0
                episode_done = False
                action_history = []
                step = 0
                continue

            if args.mode == "manual":
                if requested_action is None:
                    now = time.perf_counter()
                    if now - idle_step_time >= 1.0 / 60.0:
                        idle_step_time = now
                        obs = env.idle_step()
                        success = env.task_success()
                        episode_done = bool(success)
                    else:
                        time.sleep(0.005)
                    continue
                action = requested_action
            else:
                now = time.perf_counter()
                if episode_done or step >= min(args.max_steps, cfg["sim"]["episode_horizon"]):
                    break
                if now - last_step_time < 1.0 / max(args.fps, 1e-3):
                    time.sleep(0.005)
                    continue
                last_step_time = now
                if args.mode == "expert":
                    action = expert.act(env)
                else:
                    model_input = dict(obs)
                    if baseline_cfg.get("input_norm"):
                        model_input["image"] = _normalize_image(model_input["image"])
                    action, z = _select_action(model, model_input, adapter.context, device)
                    if baseline_cfg.get("fewshot") and step < 2:
                        action = expert.act(env)

            next_obs, reward, done, info = env.step(action)
            success = int(info["success"])
            next_context = (
                adapter.context.squeeze(0).detach().cpu().numpy()
                if adapter is not None
                else np.zeros(cfg["model"]["context_dim"], dtype=np.float32)
            )
            episode.add(
                frame=_preview_frame(env, obs, skill_name(action), success, step),
                state=obs["state"],
                action=action,
                context=next_context,
                reward=reward,
                info={"task": env.task_name, "success": success, "visibility": float(info["visibility"])},
            )
            if args.mode == "policy" and baseline_cfg.get("adapt", False):
                next_image_t = torch.from_numpy(next_obs["image"]).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                next_state_t = torch.from_numpy(next_obs["state"]).unsqueeze(0).float().to(device)
                next_z = model.encode(next_image_t, next_state_t)
                adapter.adapt(z, action, next_z)

            obs = next_obs
            last_action = action
            last_reward = reward
            episode_done = done
            action_history.append(skill_name(action))
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

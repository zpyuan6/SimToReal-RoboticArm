from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
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


WINDOW_NAME = "TTLA Env Player"
WINDOW_FLAGS = cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED
BG = (242, 244, 248)
CARD = (251, 252, 254)
BORDER = (218, 223, 232)
TEXT = (32, 37, 48)
SUBTLE = (98, 108, 125)
ACCENT = (54, 106, 255)
SUCCESS = (44, 142, 86)
WARN = (204, 129, 54)
DANGER = (186, 63, 63)
MANUAL_KEYS = {
    ord("a"): 0,
    ord("d"): 1,
    ord("w"): 2,
    ord("s"): 3,
    ord("e"): 4,
    ord("h"): 5,
    ord("x"): 6,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", default="approach", choices=["approach", "verify", "observe_then_act"])
    parser.add_argument("--mode", default="manual", choices=["manual", "expert", "policy"])
    parser.add_argument("--baseline", default="ours", choices=list(_baseline_names()))
    parser.add_argument("--checkpoint")
    parser.add_argument("--save-dir")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--disable-gui", action="store_true")
    return parser.parse_args()


def _baseline_names() -> tuple[str, ...]:
    return (
        "no_adaptation",
        "domain_randomization_only",
        "input_normalization",
        "few_shot_finetuning",
        "ours",
    )


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
    width = max(96, 18 + len(text) * 10)
    cv2.rectangle(canvas, (x, y), (x + width, y + 28), color, -1, lineType=cv2.LINE_AA)
    _put(canvas, text, (x + 10, y + 20), 0.48, (255, 255, 255), 1)


def _keycap(canvas: np.ndarray, text: str, origin: tuple[int, int], active: bool = False) -> None:
    x, y = origin
    fill = (233, 239, 255) if active else (245, 247, 250)
    edge = ACCENT if active else BORDER
    cv2.rectangle(canvas, (x, y), (x + 42, y + 32), fill, -1, lineType=cv2.LINE_AA)
    cv2.rectangle(canvas, (x, y), (x + 42, y + 32), edge, 1, lineType=cv2.LINE_AA)
    _put(canvas, text, (x + 12, y + 21), 0.56, TEXT, 1)


def _metric_line(canvas: np.ndarray, label: str, value: str, y: int) -> None:
    _put(canvas, label, (18, y), 0.48, SUBTLE, 1)
    _put(canvas, value, (180, y), 0.54, TEXT, 1)


def _draw_context_strip(canvas: np.ndarray, context: np.ndarray, origin: tuple[int, int], width: int) -> None:
    x, y = origin
    height = 72
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (248, 249, 252), -1, lineType=cv2.LINE_AA)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), BORDER, 1, lineType=cv2.LINE_AA)
    _put(canvas, "Context State", (x + 12, y + 22), 0.48, SUBTLE, 1)
    dims = min(8, len(context))
    if dims <= 0:
        return
    bar_zone_y = y + 34
    bar_zone_h = 26
    spacing = 10
    bar_w = max(18, (width - 24 - (dims - 1) * spacing) // dims)
    for idx in range(dims):
        bx = x + 12 + idx * (bar_w + spacing)
        value = float(np.clip(context[idx], -1.5, 1.5))
        zero_y = bar_zone_y + bar_zone_h // 2
        top_y = int(zero_y - value / 1.5 * (bar_zone_h // 2))
        color = SUCCESS if value >= 0 else ACCENT
        cv2.rectangle(canvas, (bx, min(zero_y, top_y)), (bx + bar_w, max(zero_y, top_y)), color, -1, lineType=cv2.LINE_AA)
        _put(canvas, f"c{idx}", (bx + 2, y + 66), 0.36, SUBTLE, 1)


def _draw_controls(canvas: np.ndarray, origin: tuple[int, int], episode_done: bool) -> None:
    x, y = origin
    _put(canvas, "Controls", (x, y), 0.58, TEXT, 1)
    rows = [
        [("A", "scan L"), ("D", "scan R")],
        [("W", "lift"), ("S", "dip"), ("E", "approach")],
        [("H", "hold"), ("X", "stop"), ("N", "reset"), ("Q", "quit")],
    ]
    row_y = y + 20
    for row in rows:
        cursor_x = x
        for key, label in row:
            _keycap(canvas, key, (cursor_x, row_y), active=episode_done and key in {"N", "Q"})
            _put(canvas, label, (cursor_x + 50, row_y + 22), 0.44, SUBTLE, 1)
            cursor_x += 128
        row_y += 42


def _status_text(success: int, episode_done: bool, action: int) -> tuple[str, tuple[int, int, int]]:
    if success:
        return "SUCCESS", SUCCESS
    if episode_done and action == 6:
        return "STOPPED", WARN
    if episode_done:
        return "DONE", DANGER
    return "RUNNING", ACCENT


def _save_episode(episode: EpisodeBuffer, args: argparse.Namespace, env: RoArmSimEnv, success: int, episode_index: int) -> Path | None:
    if not args.save_dir or not episode.actions:
        return None
    episode_dir = _episode_dir(args.save_dir) / f"run_{episode_index:03d}"
    saved = episode.save(
        episode_dir,
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


def _compose_dashboard(
    env: RoArmSimEnv,
    obs: dict[str, np.ndarray],
    mode: str,
    baseline: str,
    step: int,
    action: int,
    success: int,
    reward: float,
    context: np.ndarray,
    episode_done: bool,
    action_history: list[str],
) -> np.ndarray:
    dashboard = np.full((860, 1280, 3), BG, dtype=np.uint8)
    _put(dashboard, "TTLA MuJoCo Visualizer", (28, 34), 0.92, TEXT, 2)
    _put(dashboard, "True 3D forearm-camera rendering with skill-level interaction", (30, 60), 0.48, SUBTLE, 1)
    status_text, status_color = _status_text(success, episode_done, action)
    _badge(dashboard, status_text, (1040, 22), status_color)
    _badge(dashboard, mode.upper(), (910, 22), (90, 98, 115))

    _card(dashboard, (24, 84), (620, 612), "Forearm Camera", f"Task: {env.task_name}")
    _card(dashboard, (660, 84), (1256, 612), "MuJoCo Overview", f"Baseline: {baseline}")
    _card(dashboard, (24, 640), (520, 834), "Episode State")
    _card(dashboard, (544, 640), (860, 834), "Action History")
    _card(dashboard, (884, 640), (1256, 834), "Controls")

    camera = cv2.resize(obs["image"], (560, 440), interpolation=cv2.INTER_CUBIC)
    overview = cv2.resize(env.render_debug_view("overview_cam"), (560, 440), interpolation=cv2.INTER_CUBIC)
    dashboard[126:566, 44:604] = camera
    dashboard[126:566, 680:1240] = overview

    distance = float(np.linalg.norm(env._ee_position() - env._target_position()))
    _metric_line(dashboard[664:820, 40:500], "Mode", mode, 30)
    _metric_line(dashboard[664:820, 40:500], "Task", env.task_name, 58)
    _metric_line(dashboard[664:820, 40:500], "Action", skill_name(action), 86)
    _metric_line(dashboard[664:820, 40:500], "Step", f"{step}", 114)
    _metric_line(dashboard[664:820, 40:500], "Visibility", f"{env.visibility_score():.3f}", 142)
    _metric_line(dashboard[664:820, 40:500], "Target Dist", f"{distance:.3f} m", 170)
    _metric_line(dashboard[664:820, 40:500], "Reward", f"{reward:.3f}", 198)

    qpos = env.data.qpos[:6]
    joint_text = f"b {qpos[0]:+.2f} | s {qpos[1]:+.2f} | e {qpos[2]:+.2f} | wp {qpos[3]:+.2f} | wr {qpos[4]:+.2f} | g {qpos[5]:+.2f}"
    _put(dashboard, joint_text, (42, 814), 0.46, SUBTLE, 1)
    _draw_context_strip(dashboard, context, (248, 730), 246)

    history_y = 676
    visible_history = action_history[-7:] if action_history else ["none"]
    for idx, item in enumerate(reversed(visible_history)):
        label = f"{len(visible_history) - idx:02d}. {item}"
        color = TEXT if idx == 0 else SUBTLE
        _put(dashboard, label, (564, history_y + idx * 22), 0.52 if idx == 0 else 0.46, color, 1)

    _draw_controls(dashboard, (904, 676), episode_done)

    if episode_done:
        banner_color = SUCCESS if success else WARN if action == 6 else DANGER
        cv2.rectangle(dashboard, (350, 350), (930, 450), banner_color, -1, lineType=cv2.LINE_AA)
        headline = "SUCCESS" if success else "EPISODE COMPLETE"
        subline = "Press N to save and reset, or Q to exit." if mode == "manual" else "Autoplay episode finished."
        _put(dashboard, headline, (520, 390), 1.18, (255, 255, 255), 2)
        _put(dashboard, subline, (432, 424), 0.64, (255, 255, 255), 1)
    return dashboard


def _episode_dir(base_dir: str | Path) -> Path:
    root = ensure_dir(base_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / f"episode_{stamp}"


def _manual_action(
    env: RoArmSimEnv,
    obs: dict[str, np.ndarray],
    baseline: str,
    step: int,
    last_action: int,
    success: int,
    reward: float,
    context: np.ndarray,
    episode_done: bool,
    action_history: list[str],
) -> int | None:
    while True:
        frame = _compose_dashboard(env, obs, "manual", baseline, step, last_action, success, reward, context, episode_done, action_history)
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(33) & 0xFF
        if key in MANUAL_KEYS and not episode_done:
            return MANUAL_KEYS[key]
        if key == ord("q"):
            return None
        if key == ord("n"):
            return -1


def main() -> None:
    args = _parse_args()
    if args.disable_gui and args.mode == "manual":
        raise ValueError("--disable-gui cannot be used with --mode manual.")
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
    episode_index = 0
    last_action = 5
    last_reward = 0.0
    success = 0
    episode_done = False
    action_history: list[str] = []

    try:
        if not args.disable_gui:
            cv2.namedWindow(WINDOW_NAME, WINDOW_FLAGS)
            cv2.resizeWindow(WINDOW_NAME, 1320, 900)
        step = 0
        while True:
            if args.mode != "manual" and step >= min(args.max_steps, cfg["sim"]["episode_horizon"]):
                break
            current_context = (
                adapter.context.squeeze(0).detach().cpu().numpy()
                if adapter is not None
                else np.zeros(cfg["model"]["context_dim"], dtype=np.float32)
            )
            frame = _compose_dashboard(env, obs, args.mode, args.baseline, step, last_action, success, last_reward, current_context, episode_done, action_history)
            if not args.disable_gui and args.mode != "manual":
                cv2.imshow(WINDOW_NAME, frame)
                cv2.waitKey(1)

            if args.mode == "manual":
                action = _manual_action(env, obs, args.baseline, step, last_action, success, last_reward, current_context, episode_done, action_history)
                if action is None:
                    break
                if action == -1:
                    saved = _save_episode(episode, args, env, success, episode_index)
                    if saved is not None:
                        print(f"saved_episode={saved}")
                        episode_index += 1
                    obs = env.reset(task_name=args.task)
                    if adapter is not None:
                        adapter.reset()
                    episode = EpisodeBuffer()
                    step = 0
                    last_action = 5
                    last_reward = 0.0
                    success = 0
                    episode_done = False
                    action_history = []
                    continue
            elif args.mode == "expert":
                action = expert.act(env)
                if not args.disable_gui:
                    time.sleep(1.0 / max(args.fps, 1e-3))
            else:
                model_input = dict(obs)
                if baseline_cfg.get("input_norm"):
                    model_input["image"] = _normalize_image(model_input["image"])
                action, z = _select_action(model, model_input, adapter.context, device)
                if baseline_cfg.get("fewshot") and step < 2:
                    action = expert.act(env)
                if not args.disable_gui:
                    time.sleep(1.0 / max(args.fps, 1e-3))

            next_obs, reward, done, info = env.step(action)
            success = int(info["success"])
            next_context = (
                adapter.context.squeeze(0).detach().cpu().numpy()
                if adapter is not None
                else np.zeros(cfg["model"]["context_dim"], dtype=np.float32)
            )
            episode.add(
                frame=frame,
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
            action_history.append(skill_name(action))
            last_action = action
            last_reward = reward
            step += 1
            episode_done = done
            if done and args.mode != "manual":
                break

        saved = _save_episode(episode, args, env, success, episode_index)
        if saved is not None:
            print(f"saved_episode={saved}")
    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

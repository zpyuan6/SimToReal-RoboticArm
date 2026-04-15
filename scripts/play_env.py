from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from ttla.config import load_config
from ttla.evaluation.baselines import baseline_overrides
from ttla.models import load_model_state
from ttla.training import build_model
from ttla.sim import RoArmSimEnv, ScriptedExpert
from ttla.sim.skills import (
    ABORT_ID,
    APPROACH_COARSE_ID,
    APPROACH_FINE_ID,
    GRASP_EXECUTE_ID,
    HOLD_POSITION_ID,
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


WINDOW_NAME = "TTLA Primitive Player"
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
    ord("a"): OBS_LEFT_ID,
    ord("d"): OBS_RIGHT_ID,
    ord("c"): OBS_CENTER_ID,
    ord("v"): VERIFY_TARGET_ID,
    ord("p"): PREALIGN_GRASP_ID,
    ord("e"): APPROACH_COARSE_ID,
    ord("r"): APPROACH_FINE_ID,
    ord("t"): RETREAT_ID,
    ord("o"): REOBSERVE_ID,
    ord("g"): PREGRASP_SERVO_ID,
    ord("f"): GRASP_EXECUTE_ID,
    ord("l"): LIFT_OBJECT_ID,
    ord("m"): TRANSPORT_TO_DROPZONE_ID,
    ord("y"): PLACE_OBJECT_ID,
    ord("h"): HOLD_POSITION_ID,
    ord("x"): ABORT_ID,
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
    parser.add_argument("--disable-gui", action="store_true")
    return parser.parse_args()


def _load_model(cfg: dict, checkpoint_path: str | Path, device: torch.device):
    payload = torch.load(checkpoint_path, map_location=device)
    model = build_model(cfg).to(device)
    load_model_state(model, payload["model_state"])
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


def _metric_line(canvas: np.ndarray, label: str, value: str, y: int) -> None:
    _put(canvas, label, (18, y), 0.48, SUBTLE, 1)
    _put(canvas, value, (200, y), 0.54, TEXT, 1)


def _primitive_control_lines() -> list[str]:
    return [
        "a/d/c: observe left/right/center",
        "v: verify  p: prealign",
        "e/r: coarse/fine approach",
        "t: retreat  o: reobserve",
        "g: pregrasp servo  f: grasp",
        "l: lift  m: move to dropzone",
        "y: place  x: abort",
        "n: save+reset  q: quit",
    ]


def _status_text(success: int, episode_done: bool, action_id: int) -> tuple[str, tuple[int, int, int]]:
    if success:
        return "SUCCESS", SUCCESS
    if episode_done and action_id == ABORT_ID:
        return "ABORTED", WARN
    if episode_done:
        return "DONE", DANGER
    return "RUNNING", ACCENT


def _save_episode(episode: EpisodeBuffer, args: argparse.Namespace, env: RoArmSimEnv, success: int, episode_index: int) -> Path | None:
    if not args.save_dir or not episode.actions:
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = ensure_dir(Path(args.save_dir) / f"episode_{stamp}")
    saved = episode.save(
        root / f"run_{episode_index:03d}",
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


def _compose_dashboard(
    env: RoArmSimEnv,
    obs: dict[str, np.ndarray],
    mode: str,
    baseline: str,
    step: int,
    action_id: int,
    success: int,
    reward: float,
    episode_done: bool,
    action_history: list[str],
) -> np.ndarray:
    dashboard = np.full((860, 1280, 3), BG, dtype=np.uint8)
    _put(dashboard, "TTLA Primitive Task Viewer", (28, 34), 0.92, TEXT, 2)
    _put(dashboard, "Discrete primitive policy with task-level execution scripts", (30, 60), 0.48, SUBTLE, 1)
    status_text, status_color = _status_text(success, episode_done, action_id)
    _badge(dashboard, status_text, (1040, 22), status_color)
    _badge(dashboard, mode.upper(), (910, 22), (90, 98, 115))

    _card(dashboard, (24, 84), (620, 612), "Forearm Camera", f"Task: {env.task_name}")
    _card(dashboard, (660, 84), (1256, 612), "MuJoCo Overview", f"Baseline: {baseline}")
    _card(dashboard, (24, 640), (460, 834), "Episode State")
    _card(dashboard, (484, 640), (860, 834), "Action History")
    _card(dashboard, (884, 640), (1256, 834), "Primitive Controls")

    camera = cv2.resize(obs["image"], (560, 440), interpolation=cv2.INTER_CUBIC)
    overview = cv2.resize(env.render_debug_view("overview_cam"), (560, 440), interpolation=cv2.INTER_CUBIC)
    dashboard[126:566, 44:604] = camera
    dashboard[126:566, 680:1240] = overview

    distance = env.ee_target_distance()
    drop_distance = env.dropzone_distance()
    _metric_line(dashboard[664:820, 40:440], "Task", env.task_name, 30)
    _metric_line(dashboard[664:820, 40:440], "Mode", mode, 58)
    _metric_line(dashboard[664:820, 40:440], "Primitive", primitive_name(action_id), 86)
    _metric_line(dashboard[664:820, 40:440], "Step", f"{step}", 114)
    _metric_line(dashboard[664:820, 40:440], "Visibility", f"{env.visibility_score():.3f}", 142)
    _metric_line(dashboard[664:820, 40:440], "Center Error", f"{env.center_error_px():.1f}px", 170)
    _metric_line(dashboard[664:820, 40:440], "EE->Target", f"{distance:.3f} m", 198)
    _metric_line(dashboard[664:820, 40:440], "EE->Dropzone", f"{drop_distance:.3f} m", 226)
    _metric_line(dashboard[664:820, 40:440], "Flags", f"verified={int(env.verified)} grasped={int(env.object_attached)} placed={int(env.placed)}", 254)

    history_y = 676
    visible_history = action_history[-7:] if action_history else ["none"]
    for idx, item in enumerate(reversed(visible_history)):
        label = f"{len(visible_history) - idx:02d}. {item}"
        color = TEXT if idx == 0 else SUBTLE
        _put(dashboard, label, (506, history_y + idx * 22), 0.52 if idx == 0 else 0.46, color, 1)

    controls_y = 676
    for line in _primitive_control_lines():
        _put(dashboard, line, (904, controls_y), 0.46, SUBTLE, 1)
        controls_y += 22

    if episode_done:
        banner_color = SUCCESS if success else WARN if action_id == ABORT_ID else DANGER
        cv2.rectangle(dashboard, (340, 350), (940, 450), banner_color, -1, lineType=cv2.LINE_AA)
        headline = "SUCCESS" if success else "EPISODE COMPLETE"
        subline = "Press N to save and reset, or Q to exit." if mode == "manual" else "Autoplay episode finished."
        _put(dashboard, headline, (515, 390), 1.18, (255, 255, 255), 2)
        _put(dashboard, subline, (420, 424), 0.64, (255, 255, 255), 1)
    return dashboard


def _manual_action(
    env: RoArmSimEnv,
    obs: dict[str, np.ndarray],
    baseline: str,
    step: int,
    last_action: int,
    success: int,
    reward: float,
    episode_done: bool,
    action_history: list[str],
) -> int | None:
    while True:
        frame = _compose_dashboard(env, obs, "manual", baseline, step, last_action, success, reward, episode_done, action_history)
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
    model = _load_model(cfg, args.checkpoint, device) if args.mode == "policy" else None

    obs = env.reset(task_name=args.task)
    episode = EpisodeBuffer()
    episode_index = 0
    last_action = OBS_CENTER_ID
    last_reward = 0.0
    success = 0
    episode_done = False
    action_history: list[str] = []
    runtime_state = model.init_runtime_state(batch_size=1, device=device) if model is not None else None

    try:
        if not args.disable_gui:
            cv2.namedWindow(WINDOW_NAME, WINDOW_FLAGS)
            cv2.resizeWindow(WINDOW_NAME, 1320, 900)
        step = 0
        while True:
            if args.mode != "manual" and step >= min(args.max_steps, cfg["sim"]["episode_horizon"]):
                break

            if args.mode == "manual":
                primitive_id = _manual_action(env, obs, args.baseline, step, last_action, success, last_reward, episode_done, action_history)
                if primitive_id is None:
                    break
                if primitive_id == -1:
                    saved = _save_episode(episode, args, env, success, episode_index)
                    if saved is not None:
                        print(f"saved_episode={saved}")
                        episode_index += 1
                    obs = env.reset(task_name=args.task)
                    episode = EpisodeBuffer()
                    step = 0
                    last_action = OBS_CENTER_ID
                    last_reward = 0.0
                    success = 0
                    episode_done = False
                    action_history = []
                    runtime_state = model.init_runtime_state(batch_size=1, device=device) if model is not None else None
                    continue
            elif args.mode == "expert":
                primitive_id = expert.act(env)
                if not args.disable_gui:
                    time.sleep(1.0 / max(args.fps, 1e-3))
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
                if not args.disable_gui:
                    time.sleep(1.0 / max(args.fps, 1e-3))

            next_obs, reward, done, info = env.step(primitive_id)
            episode.add(
                frame=_compose_dashboard(env, obs, args.mode, args.baseline, step, primitive_id, success, reward, done, action_history),
                state=obs["state"],
                action=primitive_id,
                context=info["context"],
                reward=reward,
                info={"task": env.task_name, "success": int(info["success"]), "visibility": float(info["visibility"])},
            )
            obs = next_obs
            success = int(info["success"])
            last_action = primitive_id
            last_reward = reward
            action_history.append(primitive_name(primitive_id))
            step += 1
            episode_done = done

            if args.mode != "manual":
                frame = _compose_dashboard(env, obs, args.mode, args.baseline, step, last_action, success, last_reward, episode_done, action_history)
                if not args.disable_gui:
                    cv2.imshow(WINDOW_NAME, frame)
                    cv2.waitKey(1)
                if done:
                    break

        saved = _save_episode(episode, args, env, success, episode_index)
        if saved is not None:
            print(f"saved_episode={saved}")
    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

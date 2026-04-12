from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


WINDOW_NAME = "TTLA Episode Replay"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-dir", required=True)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--disable-gui", action="store_true")
    return parser.parse_args()


def _load_episode(episode_dir: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    root = Path(episode_dir)
    arrays = np.load(root / "episode.npz")
    with (root / "meta.json").open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    return {key: arrays[key] for key in arrays.files}, meta


def _overlay(frame: np.ndarray, arrays: dict[str, np.ndarray], meta: dict, idx: int) -> np.ndarray:
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    panel = np.full((height, 320, 3), 248, dtype=np.uint8)
    actions = meta.get("action_names", [])
    action_id = int(arrays["actions"][idx]) if idx < len(arrays["actions"]) else -1
    action_name = actions[action_id] if 0 <= action_id < len(actions) else str(action_id)
    reward = float(arrays["rewards"][idx]) if idx < len(arrays["rewards"]) else 0.0
    context = arrays["contexts"][idx] if idx < len(arrays["contexts"]) else np.zeros(8, dtype=np.float32)
    state = arrays["states"][idx] if idx < len(arrays["states"]) else np.zeros(16, dtype=np.float32)
    info_raw = None
    if "infos" in arrays and idx < len(arrays["infos"]):
        info_candidate = arrays["infos"][idx]
        if isinstance(info_candidate, np.ndarray) and info_candidate.shape == ():
            info_raw = info_candidate.item()
        else:
            info_raw = info_candidate
    info_dict = info_raw if isinstance(info_raw, dict) else {}
    visibility = float(info_dict.get("visibility", 0.0))
    success = int(info_dict.get("success", meta.get("success", 0)))
    verified = float(state[-4]) if len(state) >= 4 else 0.0
    grasped = float(state[-3]) if len(state) >= 3 else 0.0
    task_id = int(round(float(state[-2]))) if len(state) >= 2 else -1
    progress = float(state[-1]) if len(state) >= 1 else 0.0
    lines = [
        f"Task: {meta.get('task', 'unknown')}",
        f"Mode: {meta.get('mode', 'unknown')}",
        f"Baseline: {meta.get('baseline', 'unknown')}",
        f"Step: {idx + 1}/{len(arrays['frames'])}",
        f"Primitive: {action_name}",
        f"Reward: {reward:.3f}",
        f"Visibility: {visibility:.3f}",
        f"Flags: verified={int(verified)} grasped={int(grasped)}",
        f"Task Id: {task_id}  Progress: {progress:.2f}",
        f"Success: {success}",
        "Controls: space pause, a prev, d next, q quit",
    ]
    y = 28
    for line in lines:
        cv2.putText(panel, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (35, 35, 35), 1)
        y += 24

    cv2.putText(panel, "Context", (12, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (35, 35, 35), 1)
    graph_x = 14
    graph_y = 248
    graph_h = 72
    graph_w = 290
    cv2.rectangle(panel, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (215, 215, 215), 1)
    dims = min(8, len(context))
    for j in range(dims):
        bar_w = 24
        x = graph_x + 8 + j * 34
        value = float(np.clip(context[j], -1.5, 1.5))
        zero_y = graph_y + graph_h // 2
        top_y = int(zero_y - value / 1.5 * (graph_h // 2 - 4))
        color = (70, 170, 70) if value >= 0 else (60, 90, 210)
        cv2.rectangle(panel, (x, min(zero_y, top_y)), (x + bar_w, max(zero_y, top_y)), color, -1)
        cv2.putText(panel, f"c{j}", (x, graph_y + graph_h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)

    cv2.putText(panel, "State Preview", (12, 354), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (35, 35, 35), 1)
    preview = ", ".join(f"{value:.2f}" for value in state[:6])
    cv2.putText(panel, preview, (12, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (45, 45, 45), 1)
    return np.concatenate([canvas, panel], axis=1)


def main() -> None:
    args = _parse_args()
    arrays, meta = _load_episode(args.episode_dir)
    frame_count = len(arrays["frames"])
    print(
        f"episode_dir={Path(args.episode_dir).resolve()} "
        f"task={meta.get('task')} mode={meta.get('mode')} baseline={meta.get('baseline')} "
        f"steps={frame_count} success={meta.get('success')}"
    )
    if args.disable_gui:
        return

    idx = 0
    playing = True
    delay = max(1, int(1000 / max(args.fps, 1e-3)))
    while True:
        frame = _overlay(arrays["frames"][idx], arrays, meta, idx)
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(delay if playing else 0) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            playing = not playing
            continue
        if key == ord("a"):
            idx = max(0, idx - 1)
            continue
        if key == ord("d"):
            idx = min(frame_count - 1, idx + 1)
            continue
        if playing:
            idx += 1
            if idx >= frame_count:
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

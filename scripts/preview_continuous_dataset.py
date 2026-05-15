from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from ttla.sim.task_defs import ID_TO_TASK
from ttla.utils.io import ensure_dir, write_json


TEXT = (32, 37, 48)
SUBTLE = (98, 108, 125)
BG = (242, 244, 248)
CARD = (251, 252, 254)
BORDER = (218, 223, 232)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview stored continuous teacher dataset episodes.")
    parser.add_argument("--input", required=True, help="Path to a continuous NPZ file.")
    parser.add_argument("--output-root", default="results/continuous_dataset_preview")
    parser.add_argument("--max-episodes", type=int, default=6)
    parser.add_argument("--fps", type=float, default=6.0)
    return parser.parse_args()


def _put(canvas: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.54, color=TEXT, thickness: int = 1) -> None:
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, lineType=cv2.LINE_AA)


def _render_frame(image_rgb: np.ndarray, lines: list[str], title: str, subtitle: str) -> np.ndarray:
    canvas = np.full((760, 1220, 3), BG, dtype=np.uint8)
    _put(canvas, title, (28, 34), 0.88, TEXT, 2)
    _put(canvas, subtitle, (30, 62), 0.46, SUBTLE, 1)
    cv2.rectangle(canvas, (24, 84), (700, 700), CARD, -1, lineType=cv2.LINE_AA)
    cv2.rectangle(canvas, (24, 84), (700, 700), BORDER, 1, lineType=cv2.LINE_AA)
    cv2.rectangle(canvas, (724, 84), (1196, 700), CARD, -1, lineType=cv2.LINE_AA)
    cv2.rectangle(canvas, (724, 84), (1196, 700), BORDER, 1, lineType=cv2.LINE_AA)
    resized = cv2.resize(image_rgb[:, :, ::-1], (636, 476), interpolation=cv2.INTER_CUBIC)
    canvas[154:630, 44:680] = resized
    y = 126
    for line in lines:
        _put(canvas, line, (744, y), 0.54, TEXT if y == 126 else SUBTLE, 1)
        y += 26
    return canvas


def _save_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def _save_contact_sheet(path: Path, frames: list[np.ndarray]) -> None:
    if not frames:
        return
    indices = [0]
    if len(frames) > 2:
        indices.append(len(frames) // 2)
    if len(frames) > 1:
        indices.append(len(frames) - 1)
    tiles = []
    for idx, frame_idx in enumerate(indices):
        tile = cv2.resize(frames[frame_idx], (380, 236), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 34), (255, 255, 255), -1)
        _put(tile, ["start", "mid", "final"][idx], (12, 24), 0.62, TEXT, 1)
        tiles.append(tile)
    sheet = np.hstack(tiles)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)


def main() -> None:
    args = _parse_args()
    payload = np.load(Path(args.input), allow_pickle=True)
    images = payload["images"]
    proprio = payload["proprio"]
    actions = payload["actions"]
    tasks = payload["tasks"].astype(np.int64)
    success = payload["success"].astype(np.int64)
    episode_ids = payload["episode_ids"].astype(np.int64)
    step_ids = payload["step_ids"].astype(np.int64)
    task_text = payload.get("task_text")
    output_root = ensure_dir(args.output_root)

    unique_episodes = []
    seen = set()
    for episode_id in episode_ids:
        episode_id = int(episode_id)
        if episode_id in seen:
            continue
        seen.add(episode_id)
        unique_episodes.append(episode_id)
    selected_episodes = unique_episodes[: int(args.max_episodes)]

    manifest: dict[str, object] = {
        "input": str(args.input),
        "episodes": [],
    }
    for episode_id in selected_episodes:
        indices = np.flatnonzero(episode_ids == episode_id)
        indices = indices[np.argsort(step_ids[indices])]
        task_id = int(tasks[indices[0]])
        task_name = ID_TO_TASK[task_id].name
        episode_success = int(np.max(success[indices]))
        episode_dir = ensure_dir(output_root / f"episode_{episode_id:04d}_{task_name}")
        frames: list[np.ndarray] = []
        for idx in indices:
            line_block = [
                f"task={task_name}",
                f"task_id={task_id}",
                f"episode_id={episode_id}",
                f"step_id={int(step_ids[idx])}",
                f"success_flag={int(success[idx])}",
                f"action=[{', '.join(f'{float(v):+.3f}' for v in actions[idx])}]",
                f"qpos=[{', '.join(f'{float(v):+.3f}' for v in proprio[idx][:6])}]",
                f"qvel=[{', '.join(f'{float(v):+.3f}' for v in proprio[idx][6:12])}]",
                f"progress={float(proprio[idx][-1]):.3f}",
            ]
            if task_text is not None:
                line_block.append(f"task_text={str(task_text[idx])}")
            frames.append(
                _render_frame(
                    images[idx],
                    line_block,
                    title=f"Continuous Dataset Preview: {task_name}",
                    subtitle=f"episode={episode_id} | final_episode_success={episode_success}",
                )
            )
        _save_video(episode_dir / "preview.mp4", frames, fps=args.fps)
        _save_contact_sheet(episode_dir / "contact_sheet.png", frames)
        write_json(
            episode_dir / "meta.json",
            {
                "task": task_name,
                "task_id": task_id,
                "episode_id": episode_id,
                "episode_success": episode_success,
                "num_steps": int(len(indices)),
            },
        )
        manifest["episodes"].append(
            {
                "episode_id": episode_id,
                "task": task_name,
                "episode_success": episode_success,
                "output_dir": str(episode_dir),
            }
        )

    write_json(output_root / "manifest.json", manifest)
    print(f"saved={output_root}")


if __name__ == "__main__":
    main()

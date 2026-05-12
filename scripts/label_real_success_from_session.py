from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ttla.utils.io import ensure_dir, save_npz, write_json


EPISODE_PATTERN = re.compile(r"(?:repeat|attempt)_(\d+)_step_(\d+)")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return _repo_root() / path


def _load_session(session_dir: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], Path]:
    meta_path = session_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json in {session_dir}")
    with meta_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    output_npz = metadata.get("output_npz")
    if not output_npz:
        raise ValueError(f"meta.json in {session_dir} does not contain output_npz")
    npz_path = _resolve_path(str(output_npz))
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing NPZ data file: {npz_path}")
    arrays = dict(np.load(npz_path))
    return metadata, arrays, npz_path


def _episode_key(frame_path: str, default_idx: int, episode_len: int) -> tuple[int, int]:
    match = EPISODE_PATTERN.search(frame_path)
    if match:
        return int(match.group(1)), int(match.group(2))
    return default_idx // episode_len, default_idx % episode_len


def _build_episodes(metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    frame_paths = list(metadata.get("frame_paths", []))
    next_frame_paths = list(metadata.get("next_frame_paths", []))
    primitive_names = list(metadata.get("primitive_names", []))
    primitive_ids = arrays["primitive_ids"].tolist()
    tasks = arrays["tasks"].tolist()
    total = len(primitive_ids)
    if not (len(frame_paths) == len(next_frame_paths) == len(primitive_names) == total):
        raise ValueError("Session metadata and NPZ arrays have inconsistent lengths")

    episode_len = max(1, len(metadata.get("primitives", [])))
    grouped: dict[int, list[tuple[int, int]]] = {}
    for idx, frame_path in enumerate(frame_paths):
        episode_idx, step_idx = _episode_key(frame_path, idx, episode_len)
        grouped.setdefault(episode_idx, []).append((step_idx, idx))

    episodes: list[dict[str, Any]] = []
    for episode_idx in sorted(grouped):
        ordered = [idx for step, idx in sorted(grouped[episode_idx])]
        episodes.append(
            {
                "episode_idx": episode_idx,
                "indices": ordered,
                "primitive_ids": [primitive_ids[i] for i in ordered],
                "primitive_names": [primitive_names[i] for i in ordered],
                "frame_paths": [frame_paths[i] for i in ordered],
                "next_frame_paths": [next_frame_paths[i] for i in ordered],
                "task_id": int(tasks[ordered[0]]) if ordered else int(metadata.get("task_id", 0)),
            }
        )
    return episodes


def _annotate_pair(
    before: np.ndarray,
    after: np.ndarray,
    episode_idx: int,
    step_idx: int,
    primitive_name: str,
) -> np.ndarray:
    before_annotated = before.copy()
    after_annotated = after.copy()
    cv2.putText(before_annotated, f"episode {episode_idx:03d} | step {step_idx:02d} | before", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(after_annotated, f"episode {episode_idx:03d} | step {step_idx:02d} | after", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(before_annotated, primitive_name, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(after_annotated, primitive_name, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return np.concatenate([before_annotated, after_annotated], axis=1)


def _render_episode_video(
    episode: dict[str, Any],
    video_dir: Path,
    fps: float,
    hold_frames: int,
    overwrite: bool,
) -> Path:
    video_path = video_dir / f"episode_{episode['episode_idx']:03d}.mp4"
    if video_path.exists() and not overwrite:
        return video_path

    first_frame = cv2.imread(str(_resolve_path(episode["frame_paths"][0])))
    if first_frame is None:
        raise FileNotFoundError(f"Could not read frame {episode['frame_paths'][0]}")
    height, width = first_frame.shape[:2]
    canvas_size = (width * 2, height)
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, canvas_size)
    try:
        for step_idx, (before_rel, after_rel, primitive_name) in enumerate(
            zip(episode["frame_paths"], episode["next_frame_paths"], episode["primitive_names"])
        ):
            before = cv2.imread(str(_resolve_path(before_rel)))
            after = cv2.imread(str(_resolve_path(after_rel)))
            if before is None or after is None:
                raise FileNotFoundError(f"Could not read episode frames {before_rel} / {after_rel}")
            composed = _annotate_pair(before, after, int(episode["episode_idx"]), step_idx, str(primitive_name))
            for _ in range(max(1, hold_frames)):
                writer.write(composed)
    finally:
        writer.release()
    return video_path


def _play_video(video_path: Path, window_name: str) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(40) & 0xFF
            if key in {27, ord("q")}:
                break
    finally:
        capture.release()
        cv2.destroyWindow(window_name)


def _prompt_label(episode_idx: int, video_path: Path, previous: str | None) -> str:
    prior = f" [{previous}]" if previous else ""
    while True:
        answer = input(
            f"[episode {episode_idx:03d}] {video_path.name}{prior} -> success (y), failure (n), skip (s), redo (r), quit (q): "
        ).strip().lower()
        if answer in {"y", "n", "s", "r", "q"}:
            return answer
        print("Please enter one of: y / n / s / r / q")


def _load_existing_labels(labels_path: Path) -> dict[str, Any]:
    if not labels_path.exists():
        return {"episodes": []}
    with labels_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_label_manifest(labels_path: Path, metadata: dict[str, Any], episode_labels: list[dict[str, Any]]) -> None:
    payload = {
        "created_at": time.time(),
        "source_session_dir": metadata.get("session_dir"),
        "source_output_npz": metadata.get("output_npz"),
        "task_id": metadata.get("task_id"),
        "episodes": episode_labels,
    }
    write_json(labels_path, payload)


def _write_label_csv(labels_csv: Path, episode_labels: list[dict[str, Any]]) -> None:
    labels_csv.parent.mkdir(parents=True, exist_ok=True)
    with labels_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "episode_idx",
                "label",
                "transition_count",
                "task_id",
                "video_path",
                "primitive_names",
            ],
        )
        writer.writeheader()
        for item in episode_labels:
            writer.writerow(item)


def _append_episode(container: dict[str, list[Any]], episode_record: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    indices = episode_record["indices"]
    for idx in indices:
        container["images"].append(arrays["images"][idx])
        container["states"].append(arrays["states"][idx])
        container["next_images"].append(arrays["next_images"][idx])
        container["next_states"].append(arrays["next_states"][idx])
        container["primitive_ids"].append(arrays["primitive_ids"][idx])
        container["tasks"].append(arrays["tasks"][idx])
    container["episodes"].append(
        {
            "episode_idx": episode_record["episode_idx"],
            "transition_count": len(indices),
            "primitive_names": episode_record["primitive_names"],
            "primitive_ids": episode_record["primitive_ids"],
            "frame_paths": episode_record["frame_paths"],
            "next_frame_paths": episode_record["next_frame_paths"],
            "task_id": episode_record["task_id"],
            "label": episode_record["label"],
            "video_path": episode_record["video_path"],
        }
    )


def _save_split_dataset(
    output_path: Path,
    session_dir: Path,
    dataset: dict[str, list[Any]],
    metadata: dict[str, Any],
) -> None:
    save_npz(
        output_path,
        images=np.asarray(dataset["images"], dtype=np.uint8),
        states=np.asarray(dataset["states"], dtype=np.float32),
        primitive_ids=np.asarray(dataset["primitive_ids"], dtype=np.int64),
        next_images=np.asarray(dataset["next_images"], dtype=np.uint8),
        next_states=np.asarray(dataset["next_states"], dtype=np.float32),
        tasks=np.asarray(dataset["tasks"], dtype=np.int64),
    )
    write_json(session_dir / "meta.json", metadata)


def _write_labeled_splits(
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
    session_dir: Path,
    episode_labels: list[dict[str, Any]],
) -> None:
    success_dir = ensure_dir(session_dir / "success")
    failure_dir = ensure_dir(session_dir / "failure")

    success_dataset = {key: [] for key in ("images", "states", "next_images", "next_states", "primitive_ids", "tasks", "episodes")}
    failure_dataset = {key: [] for key in ("images", "states", "next_images", "next_states", "primitive_ids", "tasks", "episodes")}

    for episode in episode_labels:
        if episode["label"] == "success":
            _append_episode(success_dataset, episode, arrays)
        elif episode["label"] == "failure":
            _append_episode(failure_dataset, episode, arrays)

    output_npz = Path(str(metadata["output_npz"]))
    stem = output_npz.stem
    success_output = output_npz.with_name(f"{stem}_success.npz")
    failure_output = output_npz.with_name(f"{stem}_failure.npz")

    common = {
        "created_at": time.time(),
        "output_prefix": str(output_npz.with_suffix("")),
        "session_root": str(session_dir),
        "task_id": int(metadata.get("task_id", 0)),
        "primitives": metadata.get("primitives", []),
        "attempts_recorded": len([ep for ep in episode_labels if ep["label"] in {"success", "failure"}]),
        "frame_size": metadata.get("frame_size"),
    }

    _save_split_dataset(
        _resolve_path(str(success_output)),
        success_dir,
        success_dataset,
        {
            **common,
            "label": "success",
            "success_count": len(success_dataset["episodes"]),
            "failure_count": len(failure_dataset["episodes"]),
            "episodes": success_dataset["episodes"],
        },
    )
    _save_split_dataset(
        _resolve_path(str(failure_output)),
        failure_dir,
        failure_dataset,
        {
            **common,
            "label": "failure",
            "success_count": len(success_dataset["episodes"]),
            "failure_count": len(failure_dataset["episodes"]),
            "episodes": failure_dataset["episodes"],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Label success/failure for existing real-data sessions.")
    parser.add_argument("--session-dir", required=True, help="Path to a *_session directory under data/real.")
    parser.add_argument("--fps", type=float, default=2.5, help="Episode video FPS.")
    parser.add_argument("--hold-frames", type=int, default=3, help="Number of repeated frames per step in rendered episode videos.")
    parser.add_argument("--render-only", action="store_true", help="Only render per-episode videos; do not prompt for labels.")
    parser.add_argument("--play-inline", action="store_true", help="Play each rendered episode video in an OpenCV window before prompting.")
    parser.add_argument("--overwrite-videos", action="store_true", help="Re-render episode videos even if they already exist.")
    parser.add_argument("--overwrite-labels", action="store_true", help="Ignore existing labels and relabel from scratch.")
    args = parser.parse_args()

    session_dir = Path(args.session_dir)
    if not session_dir.is_absolute():
        session_dir = (_repo_root() / session_dir).resolve()
    metadata, arrays, _ = _load_session(session_dir)
    episodes = _build_episodes(metadata, arrays)

    video_dir = ensure_dir(session_dir / "episode_videos")
    labels_path = session_dir / "episode_labels.json"
    labels_csv = session_dir / "episode_labels.csv"

    existing = {} if args.overwrite_labels else {
        int(item["episode_idx"]): item for item in _load_existing_labels(labels_path).get("episodes", [])
    }
    episode_labels: list[dict[str, Any]] = []

    for episode in episodes:
        video_path = _render_episode_video(
            episode=episode,
            video_dir=video_dir,
            fps=args.fps,
            hold_frames=args.hold_frames,
            overwrite=args.overwrite_videos,
        )
        if args.render_only:
            continue

        previous = existing.get(int(episode["episode_idx"]))
        if previous is not None:
            episode["label"] = previous["label"]
            episode["video_path"] = previous.get("video_path", str(video_path))
            episode_labels.append(episode.copy())
            print(f"[episode {episode['episode_idx']:03d}] reusing existing label: {episode['label']}")
            continue

        while True:
            if args.play_inline:
                _play_video(video_path, f"episode-{episode['episode_idx']:03d}")
            else:
                print(f"Review video: {video_path}")
            answer = _prompt_label(int(episode["episode_idx"]), video_path, None)
            if answer == "q":
                _save_label_manifest(labels_path, metadata, episode_labels)
                _write_label_csv(labels_csv, episode_labels)
                print("Stopped before finishing all episodes.")
                return
            if answer == "r":
                continue
            if answer == "s":
                break
            episode["label"] = "success" if answer == "y" else "failure"
            episode["video_path"] = str(video_path)
            episode_labels.append(episode.copy())
            _save_label_manifest(labels_path, metadata, episode_labels)
            _write_label_csv(labels_csv, episode_labels)
            break

    if args.render_only:
        print(f"rendered_videos={video_dir}")
        print(f"episode_count={len(episodes)}")
        return

    labeled = [ep for ep in episode_labels if ep.get("label") in {"success", "failure"}]
    _write_labeled_splits(metadata, arrays, session_dir, labeled)

    success_count = sum(1 for ep in labeled if ep["label"] == "success")
    failure_count = sum(1 for ep in labeled if ep["label"] == "failure")
    success_rate = (success_count / len(labeled)) if labeled else 0.0
    print(f"labels_json={labels_path}")
    print(f"labels_csv={labels_csv}")
    print(f"labeled_episodes={len(labeled)}")
    print(f"success_count={success_count}")
    print(f"failure_count={failure_count}")
    print(f"success_rate={success_rate:.4f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

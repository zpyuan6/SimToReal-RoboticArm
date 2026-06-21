from __future__ import annotations

import argparse
import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from ttla.sim.task_defs import ID_TO_TASK
except Exception:  # pragma: no cover - keeps the audit tool usable without sim imports.
    ID_TO_TASK = {}


TEXT = (32, 37, 48)
SUBTLE = (98, 108, 125)
BG = (242, 244, 248)
CARD = (251, 252, 254)
BORDER = (218, 223, 232)
WARN = (36, 108, 245)
OK = (76, 146, 92)


@dataclass(frozen=True)
class Episode:
    episode_id: int
    indices: np.ndarray
    task_id: int
    task_name: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit continuous real collection sessions without opening mp4 files."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Path to a session or merged continuous real NPZ file.")
    source.add_argument("--session-dir", help="Path to one collected session directory.")
    parser.add_argument(
        "--action-format",
        choices=("joint_target", "joint_delta"),
        default="joint_target",
        help="Dataset file to load when --session-dir is used.",
    )
    parser.add_argument("--output-dir", default="results/continuous_real_dataset_audit")
    parser.add_argument(
        "--tasks",
        default="",
        help="Optional comma-separated task names or task ids to inspect.",
    )
    parser.add_argument(
        "--episode-ids",
        default="",
        help="Optional comma-separated episode ids. Overrides --num sampling.",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=0,
        help="If > 0, randomly sample this many episodes per task.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps-per-page", type=int, default=5)
    parser.add_argument("--image-format", choices=("bgr", "rgb"), default="bgr")
    parser.add_argument("--warn-delta-rad", type=float, default=0.35)
    parser.add_argument("--interactive", action="store_true")
    return parser.parse_args()


def _dataset_path(args: argparse.Namespace) -> Path:
    if args.input:
        return Path(args.input)
    session_dir = Path(args.session_dir)
    preferred = session_dir / f"session_dataset_{args.action_format}.npz"
    if preferred.exists():
        return preferred
    fallback = session_dir / "session_dataset.npz"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Could not find {preferred.name} or session_dataset.npz in {session_dir}"
    )


def _task_name(task_id: int) -> str:
    spec = ID_TO_TASK.get(int(task_id)) if isinstance(ID_TO_TASK, dict) else None
    if spec is not None and hasattr(spec, "name"):
        return str(spec.name)
    return f"task_{int(task_id)}"


def _as_text_array(payload: np.lib.npyio.NpzFile, key: str, length: int) -> list[str]:
    if key not in payload:
        return [""] * length
    raw = payload[key]
    out: list[str] = []
    for value in raw.tolist():
        if isinstance(value, bytes):
            out.append(value.decode("utf-8", errors="replace"))
        else:
            out.append(str(value))
    return out


def _load_episodes(payload: np.lib.npyio.NpzFile) -> list[Episode]:
    episode_ids = payload["episode_ids"].astype(np.int64)
    step_ids = payload["step_ids"].astype(np.int64)
    tasks = payload["tasks"].astype(np.int64)
    ordered: list[int] = []
    seen: set[int] = set()
    for value in episode_ids.tolist():
        episode_id = int(value)
        if episode_id in seen:
            continue
        seen.add(episode_id)
        ordered.append(episode_id)

    episodes: list[Episode] = []
    for episode_id in ordered:
        indices = np.flatnonzero(episode_ids == episode_id)
        indices = indices[np.argsort(step_ids[indices])]
        task_id = int(tasks[indices[0]]) if indices.size else -1
        episodes.append(
            Episode(
                episode_id=episode_id,
                indices=indices,
                task_id=task_id,
                task_name=_task_name(task_id),
            )
        )
    return episodes


def _parse_task_filter(raw: str) -> set[str]:
    return {token.strip() for token in raw.split(",") if token.strip()}


def _select_episodes(args: argparse.Namespace, episodes: list[Episode]) -> list[Episode]:
    task_filter = _parse_task_filter(args.tasks)
    if task_filter:
        episodes = [
            episode
            for episode in episodes
            if episode.task_name in task_filter or str(episode.task_id) in task_filter
        ]
    if args.episode_ids:
        requested = {int(token.strip()) for token in args.episode_ids.split(",") if token.strip()}
        return [episode for episode in episodes if episode.episode_id in requested]
    if int(args.num) <= 0:
        return episodes

    rng = np.random.default_rng(int(args.seed))
    by_task: dict[int, list[Episode]] = {}
    for episode in episodes:
        by_task.setdefault(episode.task_id, []).append(episode)
    selected: list[Episode] = []
    for task_id in sorted(by_task):
        group = by_task[task_id]
        if len(group) <= int(args.num):
            selected.extend(group)
            continue
        choices = rng.choice(np.arange(len(group)), size=int(args.num), replace=False)
        selected.extend(group[int(idx)] for idx in sorted(choices.tolist()))
    return selected


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _put(
    canvas: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.52,
    color: tuple[int, int, int] = TEXT,
    thickness: int = 1,
) -> None:
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, lineType=cv2.LINE_AA)


def _fmt_vec(values: np.ndarray, precision: int = 3) -> str:
    return "[" + ", ".join(f"{float(value):+.{precision}f}" for value in values.tolist()) + "]"


def _to_bgr(image: np.ndarray, image_format: str) -> np.ndarray:
    image = image.astype(np.uint8, copy=False)
    if image_format == "rgb":
        return image[:, :, ::-1].copy()
    return image.copy()


def _image_tile(image: np.ndarray, label: str, image_format: str) -> np.ndarray:
    tile = np.full((230, 260, 3), CARD, dtype=np.uint8)
    cv2.rectangle(tile, (0, 0), (259, 229), BORDER, 1, lineType=cv2.LINE_AA)
    cv2.rectangle(tile, (0, 0), (259, 30), (255, 255, 255), -1)
    _put(tile, label, (10, 21), 0.52, TEXT, 1)
    bgr = _to_bgr(image, image_format)
    resized = cv2.resize(bgr, (220, 176), interpolation=cv2.INTER_AREA)
    tile[42:218, 20:240] = resized
    return tile


def _value(payload: np.lib.npyio.NpzFile, key: str, idx: int, width: int = 6) -> np.ndarray:
    if key not in payload:
        return np.zeros((width,), dtype=np.float32)
    return np.asarray(payload[key][idx], dtype=np.float32)


def _render_step_tile(
    payload: np.lib.npyio.NpzFile,
    episode: Episode,
    idx: int,
    local_step: int,
    waypoint_names: list[str],
    task_text: list[str],
    image_format: str,
    warn_delta_rad: float,
) -> np.ndarray:
    tile = np.full((275, 1500, 3), CARD, dtype=np.uint8)
    cv2.rectangle(tile, (0, 0), (1499, 274), BORDER, 1, lineType=cv2.LINE_AA)

    before = _image_tile(payload["images"][idx], "before", image_format)
    after = _image_tile(payload["next_images"][idx], "after", image_format)
    tile[28:258, 20:280] = before
    tile[28:258, 296:556] = after

    q_before = _value(payload, "q_before", idx)
    q_after = _value(payload, "q_after", idx)
    action = _value(payload, "actions", idx)
    action_delta = _value(payload, "action_joint_delta", idx)
    max_delta = float(np.max(np.abs(action_delta))) if action_delta.size else 0.0
    progress = float(payload["proprio"][idx][-1]) if "proprio" in payload else 0.0
    color = WARN if max_delta > float(warn_delta_rad) else OK

    x = 585
    y = 34
    lines = [
        f"episode={episode.episode_id}  local_step={local_step}/{len(episode.indices) - 1}  npz_index={idx}",
        f"task={episode.task_name}  task_id={episode.task_id}  waypoint={waypoint_names[idx]}",
        f"task_text={task_text[idx]}",
        f"q_before={_fmt_vec(q_before)}",
        f"q_after ={_fmt_vec(q_after)}",
        f"action  ={_fmt_vec(action)}",
        f"delta   ={_fmt_vec(action_delta)}",
        f"progress={progress:.3f}  max_abs_delta={max_delta:.4f}",
    ]
    for line_idx, line in enumerate(lines):
        _put(tile, line[:118], (x, y), 0.52, color if line_idx == len(lines) - 1 else TEXT, 1)
        y += 29
    return tile


def _render_episode_page(
    payload: np.lib.npyio.NpzFile,
    episode: Episode,
    page_indices: np.ndarray,
    page_num: int,
    num_pages: int,
    waypoint_names: list[str],
    task_text: list[str],
    image_format: str,
    warn_delta_rad: float,
) -> np.ndarray:
    tile_h = 275
    header_h = 86
    height = header_h + tile_h * len(page_indices) + 24
    canvas = np.full((height, 1540, 3), BG, dtype=np.uint8)
    _put(canvas, "Continuous Real Dataset Audit", (28, 34), 0.86, TEXT, 2)
    _put(
        canvas,
        f"episode={episode.episode_id} task={episode.task_name} page={page_num + 1}/{num_pages} steps={len(episode.indices)}",
        (30, 64),
        0.52,
        SUBTLE,
        1,
    )
    for row, idx in enumerate(page_indices.tolist()):
        local_step_matches = np.flatnonzero(episode.indices == idx)
        local_step = int(local_step_matches[0]) if local_step_matches.size else row
        tile = _render_step_tile(
            payload,
            episode,
            int(idx),
            local_step,
            waypoint_names,
            task_text,
            image_format,
            warn_delta_rad,
        )
        top = header_h + row * tile_h
        canvas[top : top + tile_h, 20:1520] = tile
    return canvas


def _episode_summary(payload: np.lib.npyio.NpzFile, episode: Episode, waypoint_names: list[str]) -> dict[str, Any]:
    indices = episode.indices
    deltas = payload["action_joint_delta"][indices] if "action_joint_delta" in payload else np.zeros((len(indices), 6))
    q_before = payload["q_before"][indices[0]] if "q_before" in payload and len(indices) else np.zeros((6,))
    q_after = payload["q_after"][indices[-1]] if "q_after" in payload and len(indices) else np.zeros((6,))
    return {
        "episode_id": episode.episode_id,
        "task_id": episode.task_id,
        "task_name": episode.task_name,
        "steps": int(len(indices)),
        "first_waypoint": waypoint_names[int(indices[0])] if len(indices) else "",
        "last_waypoint": waypoint_names[int(indices[-1])] if len(indices) else "",
        "max_abs_delta": float(np.max(np.abs(deltas))) if deltas.size else 0.0,
        "q_start": [float(v) for v in np.asarray(q_before).tolist()],
        "q_end": [float(v) for v in np.asarray(q_after).tolist()],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_html(output_dir: Path, manifest: dict[str, Any]) -> None:
    rows: list[str] = []
    for episode in manifest["episodes"]:
        page_links = " ".join(
            f'<a href="{html.escape(page)}">page {idx + 1}</a>'
            for idx, page in enumerate(episode["pages"])
        )
        rows.append(
            "<tr>"
            f"<td>{episode['episode_id']}</td>"
            f"<td>{html.escape(str(episode['task_name']))}</td>"
            f"<td>{episode['steps']}</td>"
            f"<td>{episode['max_abs_delta']:.4f}</td>"
            f"<td>{html.escape(str(episode['first_waypoint']))}</td>"
            f"<td>{html.escape(str(episode['last_waypoint']))}</td>"
            f"<td>{page_links}</td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Continuous Real Dataset Audit</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #202530; background: #f5f7fb; }}
    table {{ border-collapse: collapse; width: 100%; background: white; }}
    th, td {{ border: 1px solid #d8dee9; padding: 8px; text-align: left; font-size: 14px; }}
    th {{ background: #eef2f7; }}
    a {{ color: #1559b7; }}
  </style>
</head>
<body>
  <h1>Continuous Real Dataset Audit</h1>
  <p>source: {html.escape(str(manifest["source"]))}</p>
  <p>episodes: {len(manifest["episodes"])}</p>
  <table>
    <thead>
      <tr><th>episode</th><th>task</th><th>steps</th><th>max_abs_delta</th><th>first waypoint</th><th>last waypoint</th><th>pages</th></tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def _export_pages(
    payload: np.lib.npyio.NpzFile,
    episodes: list[Episode],
    output_dir: Path,
    args: argparse.Namespace,
    waypoint_names: list[str],
    task_text: list[str],
) -> dict[str, Any]:
    output_dir = _ensure_dir(output_dir)
    manifest: dict[str, Any] = {
        "source": str(_dataset_path(args)),
        "image_format": args.image_format,
        "steps_per_page": int(args.steps_per_page),
        "episodes": [],
    }
    summaries: list[dict[str, Any]] = []
    for episode in episodes:
        episode_dir = _ensure_dir(output_dir / f"episode_{episode.episode_id:04d}_{episode.task_name}")
        pages: list[str] = []
        chunks = [
            episode.indices[start : start + int(args.steps_per_page)]
            for start in range(0, len(episode.indices), int(args.steps_per_page))
        ]
        for page_num, chunk in enumerate(chunks):
            page = _render_episode_page(
                payload,
                episode,
                chunk,
                page_num,
                len(chunks),
                waypoint_names,
                task_text,
                args.image_format,
                float(args.warn_delta_rad),
            )
            page_name = f"page_{page_num:02d}.png"
            cv2.imwrite(str(episode_dir / page_name), page)
            pages.append(f"{episode_dir.name}/{page_name}")
        summary = _episode_summary(payload, episode, waypoint_names)
        summary["pages"] = pages
        manifest["episodes"].append(summary)
        summaries.append({key: value for key, value in summary.items() if key not in {"q_start", "q_end", "pages"}})

    _write_json(output_dir / "manifest.json", manifest)
    _write_csv(output_dir / "summary.csv", summaries)
    _write_html(output_dir, manifest)
    return manifest


def _interactive(
    payload: np.lib.npyio.NpzFile,
    episodes: list[Episode],
    args: argparse.Namespace,
    waypoint_names: list[str],
    task_text: list[str],
) -> None:
    if not episodes:
        print("no_episodes_to_show=true")
        return
    try:
        cv2.namedWindow("Continuous Real Dataset Audit", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Continuous Real Dataset Audit", 1540, 900)
    except cv2.error as exc:
        print(f"interactive_unavailable={exc}")
        return

    episode_pos = 0
    step_pos = 0
    while True:
        episode = episodes[episode_pos]
        idx = int(episode.indices[step_pos])
        page = _render_episode_page(
            payload,
            episode,
            np.asarray([idx], dtype=np.int64),
            step_pos,
            len(episode.indices),
            waypoint_names,
            task_text,
            args.image_format,
            float(args.warn_delta_rad),
        )
        _put(
            page,
            "keys: n/p step | e/d episode | q quit",
            (870, 64),
            0.52,
            SUBTLE,
            1,
        )
        cv2.imshow("Continuous Real Dataset Audit", page)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("n"):
            step_pos = min(step_pos + 1, len(episode.indices) - 1)
        elif key == ord("p"):
            step_pos = max(step_pos - 1, 0)
        elif key == ord("e"):
            episode_pos = min(episode_pos + 1, len(episodes) - 1)
            step_pos = 0
        elif key == ord("d"):
            episode_pos = max(episode_pos - 1, 0)
            step_pos = 0
    cv2.destroyWindow("Continuous Real Dataset Audit")


def main() -> None:
    args = _parse_args()
    path = _dataset_path(args)
    payload = np.load(path, allow_pickle=True)
    required = {"images", "next_images", "proprio", "actions", "tasks", "episode_ids", "step_ids"}
    missing = sorted(required.difference(payload.files))
    if missing:
        raise KeyError(f"{path} is missing required fields: {missing}")

    length = int(payload["images"].shape[0])
    waypoint_names = _as_text_array(payload, "waypoint_name", length)
    task_text = _as_text_array(payload, "task_text", length)
    episodes = _select_episodes(args, _load_episodes(payload))
    manifest = _export_pages(payload, episodes, Path(args.output_dir), args, waypoint_names, task_text)
    if args.interactive:
        _interactive(payload, episodes, args, waypoint_names, task_text)
    print(f"audit_output={Path(args.output_dir)}")
    print(f"index={Path(args.output_dir) / 'index.html'}")
    print(f"episodes={len(manifest['episodes'])}")


if __name__ == "__main__":
    main()

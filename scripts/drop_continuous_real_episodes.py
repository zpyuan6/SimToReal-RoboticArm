from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drop bad episodes from a continuous real collection session."
    )
    parser.add_argument("--session-dir", required=True, help="Collected session directory containing meta.json.")
    parser.add_argument(
        "--drop-episode-names",
        default="",
        help="Comma-separated episode names from meta.json, e.g. center_pick_place_r10,left_pick_place_r11.",
    )
    parser.add_argument(
        "--drop-episode-ids",
        default="",
        help="Comma-separated numeric episode ids. Can be combined with --drop-episode-names.",
    )
    parser.add_argument(
        "--action-formats",
        nargs="*",
        default=["joint_target", "joint_delta"],
        help="Dataset action formats to rewrite.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite the session in place after creating a timestamped backup.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional cleaned output session dir. If omitted, --in-place is required.",
    )
    return parser.parse_args()


def _load_meta(session_dir: Path) -> dict[str, Any]:
    meta_path = session_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json in {session_dir}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _parse_csv_ints(raw: str) -> set[int]:
    out: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if token:
            out.add(int(token))
    return out


def _parse_csv_strings(raw: str) -> set[str]:
    return {token.strip() for token in raw.split(",") if token.strip()}


def _episode_ids_from_names(meta: dict[str, Any], names: set[str]) -> set[int]:
    if not names:
        return set()
    records = meta.get("episode_records", [])
    name_to_id: dict[str, int] = {}
    for record in records:
        if "episode_name" in record and "episode_id" in record:
            name_to_id[str(record["episode_name"])] = int(record["episode_id"])
    missing = sorted(name for name in names if name not in name_to_id)
    if missing:
        available = ", ".join(sorted(name_to_id)[:20])
        raise KeyError(f"Unknown episode names {missing}. First available names: {available}")
    return {name_to_id[name] for name in names}


def _dataset_path(meta: dict[str, Any], session_dir: Path, action_format: str) -> Path:
    paths = meta.get("session_dataset_paths", {})
    if isinstance(paths, dict) and action_format in paths:
        path = Path(str(paths[action_format]))
        if path.is_absolute() or path.exists():
            return path
        return session_dir / path
    fallback = session_dir / f"session_dataset_{action_format}.npz"
    if fallback.exists():
        return fallback
    if action_format == "joint_target":
        return session_dir / "session_dataset.npz"
    return fallback


def _backup_files(session_dir: Path, paths: list[Path]) -> Path:
    backup_dir = session_dir / f"backup_before_drop_{time.strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    for path in paths:
        if path.exists():
            shutil.copy2(path, backup_dir / path.name)
    return backup_dir


def _filter_dataset(input_path: Path, output_path: Path, drop_ids: set[int]) -> tuple[int, int, int]:
    with np.load(input_path, allow_pickle=True) as payload_npz:
        payload = {key: payload_npz[key] for key in payload_npz.files}
    if "episode_ids" not in payload:
        raise KeyError(f"{input_path} does not contain episode_ids.")

    episode_ids = payload["episode_ids"].astype(np.int64)
    keep_mask = ~np.isin(episode_ids, np.asarray(sorted(drop_ids), dtype=np.int64))
    before_transitions = int(len(episode_ids))
    after_transitions = int(np.count_nonzero(keep_mask))
    if before_transitions == after_transitions:
        raise ValueError(f"No matching transitions were dropped from {input_path}. Drop ids: {sorted(drop_ids)}")

    output_payload: dict[str, np.ndarray] = {}
    for key, values in payload.items():
        if hasattr(values, "shape") and values.shape[:1] == episode_ids.shape[:1]:
            output_payload[key] = values[keep_mask]
        else:
            output_payload[key] = values

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if input_path.resolve() == output_path.resolve():
        tmp_path = output_path.with_name(f"{output_path.name}.tmp")
        with tmp_path.open("wb") as handle:
            np.savez(handle, **output_payload)
        tmp_path.replace(output_path)
    else:
        np.savez(output_path, **output_payload)
    dropped_transitions = before_transitions - after_transitions
    return before_transitions, after_transitions, dropped_transitions


def _rewrite_meta(meta: dict[str, Any], drop_ids: set[int], output_dir: Path, dataset_paths: dict[str, str]) -> dict[str, Any]:
    records = list(meta.get("episode_records", []))
    kept_records = [record for record in records if int(record.get("episode_id", -1)) not in drop_ids]
    transitions_collected = int(sum(int(record.get("substeps", 0)) for record in kept_records))
    out = dict(meta)
    out["session_dir"] = str(output_dir)
    out["session_dataset_paths"] = dataset_paths
    out["default_session_dataset_path"] = str(output_dir / "session_dataset.npz")
    out["episode_records"] = kept_records
    out["episodes_collected"] = len(kept_records)
    out["transitions_collected"] = transitions_collected
    out["dropped_episode_ids"] = sorted(drop_ids)
    out["cleaned_at"] = time.time()
    return out


def main() -> None:
    args = _parse_args()
    session_dir = Path(args.session_dir)
    meta = _load_meta(session_dir)
    drop_names = _parse_csv_strings(args.drop_episode_names)
    drop_ids = _parse_csv_ints(args.drop_episode_ids)
    drop_ids.update(_episode_ids_from_names(meta, drop_names))
    if not drop_ids:
        raise ValueError("No episodes selected. Use --drop-episode-names or --drop-episode-ids.")
    if not args.in_place and not args.output_dir:
        raise ValueError("Use --in-place or provide --output-dir.")

    output_dir = session_dir if args.in_place else Path(args.output_dir)
    assert output_dir is not None

    dataset_inputs = [_dataset_path(meta, session_dir, fmt) for fmt in args.action_formats]
    existing_inputs = [path for path in dataset_inputs if path.exists()]
    if not existing_inputs:
        raise FileNotFoundError(f"No dataset files found for action formats: {args.action_formats}")
    backup_dir = _backup_files(session_dir, [session_dir / "meta.json", *existing_inputs]) if args.in_place else None

    dataset_paths: dict[str, str] = {}
    summaries: list[tuple[str, int, int, int]] = []
    for action_format, input_path in zip(args.action_formats, dataset_inputs):
        if not input_path.exists():
            continue
        output_path = output_dir / f"session_dataset_{action_format}.npz"
        before, after, dropped = _filter_dataset(input_path, output_path, drop_ids)
        dataset_paths[action_format] = str(output_path)
        summaries.append((action_format, before, after, dropped))

    if "joint_target" in dataset_paths:
        alias_path = output_dir / "session_dataset.npz"
        alias_path.write_bytes(Path(dataset_paths["joint_target"]).read_bytes())

    cleaned_meta = _rewrite_meta(meta, drop_ids, output_dir, dataset_paths)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "meta.json").write_text(json.dumps(cleaned_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"cleaned_session={output_dir}")
    if backup_dir is not None:
        print(f"backup_dir={backup_dir}")
    print(f"dropped_episode_ids={sorted(drop_ids)}")
    for action_format, before, after, dropped in summaries:
        print(f"{action_format}: transitions {before}->{after} dropped={dropped}")


if __name__ == "__main__":
    main()

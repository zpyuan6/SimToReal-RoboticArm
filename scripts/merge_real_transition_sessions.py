from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ttla.config import load_config
from ttla.sim.task_defs import supervision_stage_id
from ttla.utils.io import ensure_dir, save_npz, write_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Directory containing collected transition session subdirectories.")
    parser.add_argument("--output-dir", default="data/real_v2/merged")
    parser.add_argument("--plan", default=None, help="Optional YAML plan used to report missing collection episodes.")
    parser.add_argument("--roles", nargs="*", default=["calibration", "heldout"])
    return parser.parse_args()


def _find_session_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_dir() and (path / "meta.json").exists())


def _load_session_payload(session_dir: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with (session_dir / "meta.json").open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    dataset_path = Path(meta.get("session_dataset_path", session_dir / "session_dataset.npz"))
    payload_npz = np.load(dataset_path, allow_pickle=False)
    payload = {key: payload_npz[key] for key in payload_npz.files}
    return meta, payload


def _ensure_fields(payload: dict[str, np.ndarray], task_id: int) -> dict[str, np.ndarray]:
    out = dict(payload)
    num_samples = int(len(out.get("primitive_ids", np.zeros(0, dtype=np.int64))))
    if "tasks" not in out:
        out["tasks"] = np.full(num_samples, int(task_id), dtype=np.int64)
    if "contexts" not in out:
        out["contexts"] = np.zeros((num_samples, 8), dtype=np.float32)
    if "success" not in out:
        out["success"] = np.zeros(num_samples, dtype=np.float32)
    if "episode_ids" not in out:
        out["episode_ids"] = np.arange(num_samples, dtype=np.int64)
    if "step_ids" not in out:
        out["step_ids"] = np.zeros(num_samples, dtype=np.int64)
    if "stage_ids" not in out:
        out["stage_ids"] = np.asarray(
            [supervision_stage_id(int(task), int(pid)) for task, pid in zip(out["tasks"], out["primitive_ids"])],
            dtype=np.int64,
        )
    return out


def _concat_payloads(payloads: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not payloads:
        raise ValueError("No payloads to merge.")
    merged: dict[str, list[np.ndarray]] = {}
    episode_offset = 0
    for payload in payloads:
        current = dict(payload)
        current_episode_ids = current["episode_ids"].astype(np.int64) + episode_offset
        current["episode_ids"] = current_episode_ids
        if len(current_episode_ids):
            episode_offset = int(current_episode_ids.max()) + 1
        for key, value in current.items():
            merged.setdefault(key, []).append(value)
    return {key: _concat_field(key, values) for key, values in merged.items()}


def _concat_field(key: str, values: list[np.ndarray]) -> np.ndarray:
    try:
        return np.concatenate(values, axis=0)
    except ValueError:
        if key not in {"states", "next_states", "contexts"}:
            raise
        padded = _pad_feature_arrays(values)
        return np.concatenate(padded, axis=0)


def _pad_feature_arrays(values: list[np.ndarray]) -> list[np.ndarray]:
    ranks = {value.ndim for value in values}
    if len(ranks) != 1:
        raise ValueError(f"Cannot merge feature arrays with different ranks: {sorted(ranks)}")
    ndim = values[0].ndim
    if ndim <= 1:
        return values
    target_shape = [max(value.shape[dim] for value in values) for dim in range(1, ndim)]
    padded: list[np.ndarray] = []
    for value in values:
        pad_width = [(0, 0)]
        needs_padding = False
        for dim, target in enumerate(target_shape, start=1):
            delta = target - value.shape[dim]
            if delta < 0:
                raise ValueError("Internal error while padding feature arrays.")
            needs_padding = needs_padding or delta > 0
            pad_width.append((0, delta))
        padded.append(np.pad(value, pad_width, mode="constant") if needs_padding else value)
    return padded


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame.from_records(rows).to_csv(path, index=False)


def _format_count(value: Any) -> str:
    if value in {"", None}:
        return "?"
    return str(int(value))


def _print_collection_status(status_by_session: list[dict[str, Any]], status_by_sequence: list[dict[str, Any]]) -> None:
    if not status_by_session:
        return
    print("collection_status_by_session:")
    for row in status_by_session:
        missing = int(row.get("missing_episodes", 0) or 0)
        extra = int(row.get("extra_episodes", 0) or 0)
        suffix_parts = []
        if missing:
            suffix_parts.append(f"missing={missing}")
        if extra:
            suffix_parts.append(f"extra={extra}")
        suffix = "" if not suffix_parts else " " + " ".join(suffix_parts)
        print(
            f"  {row['session_key']}: "
            f"{_format_count(row.get('collected_episodes'))}/{_format_count(row.get('expected_episodes'))}"
            f"{suffix}"
        )

    if not status_by_sequence:
        return
    print("collection_status_by_sequence:")
    for row in status_by_sequence:
        missing = int(row.get("missing_episodes", 0) or 0)
        extra = int(row.get("extra_episodes", 0) or 0)
        suffix_parts = []
        if missing:
            suffix_parts.append(f"missing={missing}")
        if extra:
            suffix_parts.append(f"extra={extra}")
        suffix = "" if not suffix_parts else " " + " ".join(suffix_parts)
        print(
            f"  {row['session_key']} / {row['sequence_name']}: "
            f"{_format_count(row.get('collected_episodes'))}/{_format_count(row.get('expected_episodes'))}"
            f"{suffix}"
        )


def _sequence_name_from_episode(episode_name: str) -> str:
    if "_r" not in episode_name:
        return episode_name
    return episode_name.rsplit("_r", 1)[0]


def _expected_rows_from_plan(plan_path: str | None) -> list[dict[str, Any]]:
    if plan_path is None:
        return []
    plan = load_config(plan_path)
    shared = plan.get("shared", {})
    rows: list[dict[str, Any]] = []
    for session_key, session_spec in plan.get("sessions", {}).items():
        merged = {**shared, **session_spec}
        default_repeats = int(merged.get("repeats", 1))
        for sequence in merged.get("sequences", []):
            sequence_name = str(sequence.get("name", ""))
            repeats = int(sequence.get("repeats", default_repeats))
            rows.append(
                {
                    "session_key": str(session_key),
                    "split_role": str(merged.get("split_role", "")),
                    "task": str(merged.get("task", "")),
                    "layout_tag": str(merged.get("layout_tag", "")),
                    "sequence_name": sequence_name,
                    "expected_episodes": repeats,
                }
            )
    return rows


def _summarize_collection_status(
    expected_rows: list[dict[str, Any]],
    session_metas: list[tuple[Path, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sequence_counts: dict[tuple[str, str], dict[str, Any]] = {}
    session_counts: dict[str, dict[str, Any]] = {}

    for session_dir, meta in session_metas:
        session_key = str(meta.get("session_key", session_dir.name))
        session_entry = session_counts.setdefault(
            session_key,
            {
                "session_key": session_key,
                "split_role": str(meta.get("split_role", "")),
                "task": str(meta.get("task", "")),
                "layout_tag": str(meta.get("layout_tag", "")),
                "collected_episodes": 0,
                "transitions_collected": 0,
                "session_dirs": [],
            },
        )
        session_entry["collected_episodes"] += int(meta.get("episodes_collected", 0))
        session_entry["transitions_collected"] += int(meta.get("transitions_collected", 0))
        session_entry["session_dirs"].append(str(session_dir))
        for record in meta.get("episode_records", []):
            sequence_name = _sequence_name_from_episode(str(record.get("episode_name", "")))
            key = (session_key, sequence_name)
            entry = sequence_counts.setdefault(
                key,
                {
                    "session_key": session_key,
                    "split_role": str(meta.get("split_role", "")),
                    "task": str(meta.get("task", "")),
                    "layout_tag": str(meta.get("layout_tag", "")),
                    "sequence_name": sequence_name,
                    "collected_episodes": 0,
                    "session_dirs": [],
                },
            )
            entry["collected_episodes"] += 1
            if str(session_dir) not in entry["session_dirs"]:
                entry["session_dirs"].append(str(session_dir))

    sequence_rows: list[dict[str, Any]] = []
    if expected_rows:
        seen_session_keys = {row["session_key"] for row in expected_rows}
        for expected in expected_rows:
            key = (expected["session_key"], expected["sequence_name"])
            observed = sequence_counts.get(key, {})
            collected = int(observed.get("collected_episodes", 0))
            expected_count = int(expected["expected_episodes"])
            row = {
                **expected,
                "collected_episodes": collected,
                "missing_episodes": max(expected_count - collected, 0),
                "extra_episodes": max(collected - expected_count, 0),
                "session_dirs": ";".join(observed.get("session_dirs", [])),
            }
            sequence_rows.append(row)
        for (session_key, sequence_name), observed in sequence_counts.items():
            if session_key in seen_session_keys:
                continue
            collected = int(observed.get("collected_episodes", 0))
            sequence_rows.append(
                {
                    "session_key": session_key,
                    "split_role": observed.get("split_role", ""),
                    "task": observed.get("task", ""),
                    "layout_tag": observed.get("layout_tag", ""),
                    "sequence_name": sequence_name,
                    "expected_episodes": 0,
                    "collected_episodes": collected,
                    "missing_episodes": 0,
                    "extra_episodes": collected,
                    "session_dirs": ";".join(observed.get("session_dirs", [])),
                }
            )
    else:
        for observed in sequence_counts.values():
            sequence_rows.append(
                {
                    **observed,
                    "expected_episodes": "",
                    "missing_episodes": "",
                    "extra_episodes": "",
                    "session_dirs": ";".join(observed.get("session_dirs", [])),
                }
            )

    session_expected: dict[str, dict[str, Any]] = {}
    for row in expected_rows:
        entry = session_expected.setdefault(
            row["session_key"],
            {
                "session_key": row["session_key"],
                "split_role": row["split_role"],
                "task": row["task"],
                "layout_tag": row["layout_tag"],
                "expected_episodes": 0,
            },
        )
        entry["expected_episodes"] += int(row["expected_episodes"])

    session_rows: list[dict[str, Any]] = []
    all_session_keys = set(session_expected) | set(session_counts)
    for session_key in sorted(all_session_keys):
        expected = session_expected.get(session_key, {})
        observed = session_counts.get(session_key, {})
        expected_count = int(expected.get("expected_episodes", 0))
        collected = int(observed.get("collected_episodes", 0))
        session_rows.append(
            {
                "session_key": session_key,
                "split_role": expected.get("split_role", observed.get("split_role", "")),
                "task": expected.get("task", observed.get("task", "")),
                "layout_tag": expected.get("layout_tag", observed.get("layout_tag", "")),
                "expected_episodes": expected_count,
                "collected_episodes": collected,
                "missing_episodes": max(expected_count - collected, 0),
                "extra_episodes": max(collected - expected_count, 0),
                "transitions_collected": int(observed.get("transitions_collected", 0)),
                "session_dirs": ";".join(observed.get("session_dirs", [])),
            }
        )
    return session_rows, sequence_rows


def main() -> None:
    args = _parse_args()
    root = Path(args.root)
    output_dir = ensure_dir(args.output_dir)
    requested_roles = set(args.roles)

    session_dirs = _find_session_dirs(root)
    if not session_dirs:
        raise FileNotFoundError(f"No session directories with meta.json found under {root}")

    sessions_by_role: dict[str, list[dict[str, np.ndarray]]] = {role: [] for role in requested_roles}
    manifests: dict[str, list[dict[str, Any]]] = {role: [] for role in requested_roles}
    session_metas: list[tuple[Path, dict[str, Any]]] = []

    for session_dir in session_dirs:
        meta, payload = _load_session_payload(session_dir)
        session_metas.append((session_dir, meta))
        split_role = str(meta.get("split_role", ""))
        if split_role not in requested_roles:
            continue
        task_id = int(meta.get("task_id", 0))
        normalized = _ensure_fields(payload, task_id=task_id)
        sessions_by_role[split_role].append(normalized)
        manifests[split_role].append(
            {
                "session_dir": str(session_dir),
                "dataset_path": str(meta.get("session_dataset_path", session_dir / "session_dataset.npz")),
                "split_role": split_role,
                "task": meta.get("task", ""),
                "task_id": task_id,
                "layout_tag": meta.get("layout_tag", ""),
                "operator": meta.get("operator", ""),
                "episodes_collected": int(meta.get("episodes_collected", 0)),
                "transitions_collected": int(meta.get("transitions_collected", len(normalized["primitive_ids"]))),
            }
        )

    written: dict[str, str] = {}
    expected_rows = _expected_rows_from_plan(args.plan)
    status_by_session, status_by_sequence = _summarize_collection_status(expected_rows, session_metas)
    if status_by_session:
        _write_manifest(output_dir / "collection_status_by_session.csv", status_by_session)
    if status_by_sequence:
        _write_manifest(output_dir / "collection_status_by_sequence.csv", status_by_sequence)
    write_json(
        output_dir / "collection_status.json",
        {
            "plan_path": args.plan,
            "source_root": str(root),
            "sessions": status_by_session,
            "sequences": status_by_sequence,
        },
    )

    for role in sorted(requested_roles):
        payloads = sessions_by_role.get(role, [])
        if not payloads:
            continue
        merged = _concat_payloads(payloads)
        output_path = output_dir / f"{role}_merged.npz"
        save_npz(output_path, **merged)
        _write_manifest(output_dir / f"{role}_sessions.csv", manifests[role])
        write_json(
            output_dir / f"{role}_meta.json",
            {
                "split_role": role,
                "output_path": str(output_path),
                "num_sessions": len(payloads),
                "num_transitions": int(len(merged["primitive_ids"])),
                "num_episodes": int(len(np.unique(merged["episode_ids"]))),
                "source_root": str(root),
                "merge_mode": "rebuild_from_session_directories",
            },
        )
        written[role] = str(output_path)

    if not written:
        raise RuntimeError(f"No sessions matched requested roles: {sorted(requested_roles)}")

    for role, path in written.items():
        print(f"{role}_merged={path}")
    _print_collection_status(status_by_session, status_by_sequence)
    missing_rows = [row for row in status_by_session if int(row.get("missing_episodes", 0) or 0) > 0]
    if missing_rows:
        print("missing_collection:")
        for row in missing_rows:
            print(
                f"  {row['session_key']}: missing={row['missing_episodes']} "
                f"collected={row['collected_episodes']}/{row['expected_episodes']}"
            )
        print(f"collection_status={output_dir / 'collection_status_by_session.csv'}")
    elif status_by_session:
        print("missing_collection: none")
        print(f"collection_status={output_dir / 'collection_status_by_session.csv'}")


if __name__ == "__main__":
    main()

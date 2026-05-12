from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ttla.sim.task_defs import supervision_stage_id
from ttla.utils.io import ensure_dir, save_npz, write_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Directory containing collected transition session subdirectories.")
    parser.add_argument("--output-dir", default="data/real_v2/merged")
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
    return {key: np.concatenate(values, axis=0) for key, values in merged.items()}


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame.from_records(rows).to_csv(path, index=False)


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

    for session_dir in session_dirs:
        meta, payload = _load_session_payload(session_dir)
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
            },
        )
        written[role] = str(output_path)

    if not written:
        raise RuntimeError(f"No sessions matched requested roles: {sorted(requested_roles)}")

    for role, path in written.items():
        print(f"{role}_merged={path}")


if __name__ == "__main__":
    main()

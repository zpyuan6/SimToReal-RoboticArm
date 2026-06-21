from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ttla.utils.io import ensure_dir, save_npz, write_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge continuous real calibration sessions by split role.")
    parser.add_argument("--root", default="data/real_continuous_v1/sessions")
    parser.add_argument("--output-dir", default="data/real_continuous_v1/merged")
    parser.add_argument("--roles", nargs="*", default=["calibration", "heldout"])
    parser.add_argument("--action-formats", nargs="*", default=["joint_target", "joint_delta"])
    return parser.parse_args()


def _session_meta_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("meta.json") if path.is_file())


def _load_meta(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _dataset_path(meta: dict[str, Any], action_format: str) -> Path | None:
    paths = meta.get("session_dataset_paths", {})
    if isinstance(paths, dict) and action_format in paths:
        return Path(str(paths[action_format]))
    if action_format == "joint_target" and meta.get("default_session_dataset_path"):
        return Path(str(meta["default_session_dataset_path"]))
    return None


def _load_payload(path: Path) -> dict[str, np.ndarray]:
    npz = np.load(path, allow_pickle=True)
    return {key: npz[key] for key in npz.files}


def _concat_payloads(payloads: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not payloads:
        raise ValueError("No payloads to concatenate.")
    common_keys = set(payloads[0])
    for payload in payloads[1:]:
        common_keys.intersection_update(payload)
    ordered_keys = [key for key in payloads[0] if key in common_keys]
    merged_lists: dict[str, list[np.ndarray]] = {key: [] for key in ordered_keys}
    episode_offset = 0
    for payload in payloads:
        current = {key: payload[key] for key in ordered_keys}
        if "episode_ids" in current:
            current["episode_ids"] = current["episode_ids"].astype(np.int64) + int(episode_offset)
            if len(current["episode_ids"]):
                episode_offset = int(current["episode_ids"].max()) + 1
        for key in ordered_keys:
            merged_lists[key].append(current[key])
    return {key: np.concatenate(values, axis=0) for key, values in merged_lists.items()}


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(rows).to_csv(path, index=False)


def main() -> None:
    args = _parse_args()
    root = Path(args.root)
    output_dir = ensure_dir(args.output_dir)
    requested_roles = {str(role) for role in args.roles}
    action_formats = [str(value) for value in args.action_formats]

    meta_paths = _session_meta_paths(root)
    if not meta_paths:
        raise FileNotFoundError(f"No continuous real session meta.json files found under {root}")

    written: dict[str, str] = {}
    for action_format in action_formats:
        payloads_by_role: dict[str, list[dict[str, np.ndarray]]] = {role: [] for role in requested_roles}
        manifest_by_role: dict[str, list[dict[str, Any]]] = {role: [] for role in requested_roles}
        for meta_path in meta_paths:
            meta = _load_meta(meta_path)
            role = str(meta.get("split_role", ""))
            if role not in requested_roles:
                continue
            dataset_path = _dataset_path(meta, action_format)
            if dataset_path is None or not dataset_path.exists():
                continue
            payload = _load_payload(dataset_path)
            if len(payload.get("actions", [])) == 0:
                continue
            payloads_by_role[role].append(payload)
            manifest_by_role[role].append(
                {
                    "session_dir": meta.get("session_dir", str(meta_path.parent)),
                    "dataset_path": str(dataset_path),
                    "split_role": role,
                    "action_format": action_format,
                    "task": meta.get("task", ""),
                    "task_id": meta.get("task_id", ""),
                    "layout_tag": meta.get("layout_tag", ""),
                    "operator": meta.get("operator", ""),
                    "episodes_collected": int(meta.get("episodes_collected", 0)),
                    "transitions_collected": int(meta.get("transitions_collected", 0)),
                    "dry_run": bool(meta.get("dry_run", False)),
                }
            )
        for role, payloads in payloads_by_role.items():
            if not payloads:
                continue
            merged = _concat_payloads(payloads)
            output_path = output_dir / f"{role}_{action_format}.npz"
            save_npz(output_path, **merged)
            _write_manifest(output_dir / f"{role}_{action_format}_sessions.csv", manifest_by_role[role])
            write_json(
                output_dir / f"{role}_{action_format}_meta.json",
                {
                    "split_role": role,
                    "action_format": action_format,
                    "output_path": str(output_path),
                    "num_sessions": len(payloads),
                    "num_transitions": int(len(merged["actions"])),
                    "num_episodes": int(len(np.unique(merged["episode_ids"]))) if "episode_ids" in merged else 0,
                    "source_root": str(root),
                },
            )
            written[f"{role}_{action_format}"] = str(output_path)

    if not written:
        raise RuntimeError(
            f"No sessions matched roles={sorted(requested_roles)} and action_formats={action_formats} under {root}"
        )
    for key, path in written.items():
        print(f"{key}={path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_repeat_specs(values: list[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid repeat spec '{value}'. Expected format task_name=repeat_count.")
        task, repeat_text = value.split("=", 1)
        task = task.strip()
        repeat_count = int(repeat_text)
        if repeat_count < 1:
            raise ValueError(f"Repeat count for '{task}' must be >= 1, got {repeat_count}.")
        mapping[task] = repeat_count
    return mapping


def ordered_episode_ids(episode_ids: np.ndarray) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for episode_id in episode_ids.tolist():
        episode_id = int(episode_id)
        if episode_id not in seen:
            seen.add(episode_id)
            ordered.append(episode_id)
    return ordered


def concatenate_values(parts: list[np.ndarray], sample: np.ndarray) -> np.ndarray:
    if sample.ndim == 0:
        return np.asarray(parts, dtype=sample.dtype)
    return np.concatenate(parts, axis=0)


def build_boosted_npz(input_path: Path, output_path: Path, repeat_map: dict[str, int]) -> None:
    bundle = np.load(input_path, allow_pickle=True)
    keys = list(bundle.keys())
    episode_ids = bundle["episode_ids"]
    tasks = bundle["tasks"]
    ordered_ids = ordered_episode_ids(episode_ids)

    collected: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    next_episode_id = 0

    for original_episode_id in ordered_ids:
        indices = np.nonzero(episode_ids == original_episode_id)[0]
        if indices.size == 0:
            continue
        indices = indices[np.argsort(bundle["step_ids"][indices])]
        task_name = str(tasks[indices[0]])
        repeat_count = repeat_map.get(task_name, 1)

        for _ in range(repeat_count):
            for key in keys:
                values = bundle[key][indices]
                if key == "episode_ids":
                    values = np.full(indices.shape[0], next_episode_id, dtype=episode_ids.dtype)
                collected[key].append(values)
            next_episode_id += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for key in keys:
        sample = bundle[key]
        payload[key] = concatenate_values(collected[key], sample)
    np.savez(output_path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a task-boosted continuous NPZ dataset.")
    parser.add_argument("--input", required=True, help="Input continuous NPZ path.")
    parser.add_argument("--output", required=True, help="Output boosted NPZ path.")
    parser.add_argument(
        "--repeat-task",
        action="append",
        default=[],
        help="Repeat spec in the form task_name=repeat_count. Repeat count includes the original copy.",
    )
    args = parser.parse_args()

    repeat_map = parse_repeat_specs(args.repeat_task)
    build_boosted_npz(Path(args.input), Path(args.output), repeat_map)


if __name__ == "__main__":
    main()

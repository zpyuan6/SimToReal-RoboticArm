from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np

from ttla.sim.task_defs import TASK_TO_ID


TASK_ID_TO_NAME = {int(value): str(key) for key, value in TASK_TO_ID.items()}


def _parse_task_int_specs(values: list[str], *, default: int) -> dict[int, int]:
    parsed: dict[int, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid task spec '{value}'. Expected task=value.")
        raw_task, raw_count = value.split("=", 1)
        task_name = raw_task.strip()
        if task_name not in TASK_TO_ID:
            known = ", ".join(sorted(TASK_TO_ID))
            raise KeyError(f"Unknown task '{task_name}'. Known tasks: {known}")
        count = int(raw_count)
        if count < 0:
            raise ValueError(f"Task value must be >= 0 for '{task_name}', got {count}.")
        parsed[int(TASK_TO_ID[task_name])] = count
    if parsed:
        return parsed
    return {int(task_id): int(default) for task_id in TASK_ID_TO_NAME}


def _episode_order(episode_ids: np.ndarray) -> list[int]:
    ordered: OrderedDict[int, None] = OrderedDict()
    for episode_id in episode_ids.tolist():
        ordered[int(episode_id)] = None
    return list(ordered.keys())


def _append_episode(
    additions: dict[str, list[np.ndarray]],
    bundle: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    episode_id: int,
) -> None:
    length = int(indices.size)
    for key, value in bundle.items():
        if key == "episode_ids":
            additions[key].append(np.full(length, int(episode_id), dtype=value.dtype))
        elif key == "step_ids":
            additions[key].append(np.arange(length, dtype=value.dtype))
        else:
            additions[key].append(value[indices].copy())


def build_terminal_focus_npz(
    input_path: str | Path,
    output_path: str | Path,
    *,
    repeats: dict[int, int],
    suffix_lens: dict[int, int],
    compression: str,
) -> None:
    source = np.load(input_path, allow_pickle=True)
    bundle = {key: source[key] for key in source.files}
    required = {"episode_ids", "step_ids", "tasks"}
    missing = sorted(required.difference(bundle))
    if missing:
        raise KeyError(f"Input NPZ is missing required arrays: {missing}")

    episode_ids = bundle["episode_ids"]
    step_ids = bundle["step_ids"]
    tasks = bundle["tasks"]
    additions: dict[str, list[np.ndarray]] = {key: [] for key in bundle}
    next_episode_id = int(np.max(episode_ids)) + 1 if episode_ids.size else 0
    added_counts = {task_name: 0 for task_name in TASK_TO_ID}
    added_frames = {task_name: 0 for task_name in TASK_TO_ID}

    for episode_id in _episode_order(episode_ids):
        indices = np.flatnonzero(episode_ids == episode_id)
        if indices.size == 0:
            continue
        indices = indices[np.argsort(step_ids[indices])]
        task_id = int(tasks[indices[0]])
        repeat_count = int(repeats.get(task_id, 0))
        if repeat_count <= 0:
            continue
        suffix_len = int(suffix_lens.get(task_id, indices.size))
        if suffix_len <= 0:
            continue
        suffix_indices = indices[-min(int(indices.size), suffix_len) :]
        task_name = TASK_ID_TO_NAME[task_id]
        for _ in range(repeat_count):
            _append_episode(additions, bundle, suffix_indices, episode_id=next_episode_id)
            next_episode_id += 1
            added_counts[task_name] += 1
            added_frames[task_name] += int(suffix_indices.size)

    merged = {}
    for key, value in bundle.items():
        if additions[key]:
            merged[key] = np.concatenate([value, *additions[key]], axis=0)
        else:
            merged[key] = value.copy()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if compression == "compressed":
        np.savez_compressed(output_path, **merged)
    elif compression == "raw":
        np.savez(output_path, **merged)
    else:
        raise KeyError(f"Unsupported compression mode: {compression}")

    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"source_frames={int(next(iter(bundle.values())).shape[0])}")
    print(f"output_frames={int(next(iter(merged.values())).shape[0])}")
    print(f"source_episodes={len(_episode_order(episode_ids))}")
    print(f"output_episodes={len(_episode_order(merged['episode_ids']))}")
    for task_name in sorted(TASK_TO_ID):
        print(f"{task_name}: added_episodes={added_counts[task_name]} added_frames={added_frames[task_name]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append terminal suffix episodes to a continuous NPZ so short-horizon policies see more success-critical frames."
    )
    parser.add_argument("--input", required=True, help="Source continuous NPZ.")
    parser.add_argument("--output", required=True, help="Output augmented continuous NPZ.")
    parser.add_argument(
        "--repeat",
        action="append",
        default=[],
        help="Task repeat spec, e.g. level2_approach=3. If omitted, all tasks use --default-repeat.",
    )
    parser.add_argument(
        "--suffix-len",
        action="append",
        default=[],
        help="Task suffix length spec, e.g. level2_approach=10. If omitted, all tasks use --default-suffix-len.",
    )
    parser.add_argument("--default-repeat", type=int, default=0)
    parser.add_argument("--default-suffix-len", type=int, default=8)
    parser.add_argument("--compression", choices=("compressed", "raw"), default="compressed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repeats = _parse_task_int_specs(args.repeat, default=int(args.default_repeat))
    suffix_lens = _parse_task_int_specs(args.suffix_len, default=int(args.default_suffix_len))
    build_terminal_focus_npz(
        args.input,
        args.output,
        repeats=repeats,
        suffix_lens=suffix_lens,
        compression=str(args.compression),
    )


if __name__ == "__main__":
    main()

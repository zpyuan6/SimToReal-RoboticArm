from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ttla.config import load_config


def _ordered_episode_ids(episode_ids: np.ndarray) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for episode_id in episode_ids.tolist():
        episode_id = int(episode_id)
        if episode_id not in seen:
            seen.add(episode_id)
            ordered.append(episode_id)
    return ordered


def _normalize_tasks(raw: str) -> list[str]:
    tasks = [part.strip() for part in raw.split(",") if part.strip()]
    if not tasks:
        raise ValueError("At least one task must be provided via --tasks.")
    return tasks


def _task_name_lookup(bundle: dict[str, np.ndarray], task_names: list[str] | None) -> dict[int, str]:
    raw_tasks = bundle["tasks"]
    if np.issubdtype(raw_tasks.dtype, np.integer):
        if task_names is None:
            raise ValueError("Integer task ids require a config-backed task name list.")
        lookup = {idx: str(name) for idx, name in enumerate(task_names)}
        observed = {int(v) for v in np.unique(raw_tasks).tolist()}
        missing = sorted(v for v in observed if v not in lookup)
        if missing:
            raise ValueError(f"Task ids {missing} are missing from the provided task name list.")
        return lookup
    return {}


def _episode_task_map(bundle: dict[str, np.ndarray], task_names: list[str] | None) -> dict[int, str]:
    episode_ids = bundle["episode_ids"]
    tasks = bundle["tasks"]
    lookup = _task_name_lookup(bundle, task_names)
    mapping: dict[int, str] = {}
    for episode_id in _ordered_episode_ids(episode_ids):
        indices = np.nonzero(episode_ids == episode_id)[0]
        if indices.size == 0:
            continue
        raw_task = tasks[int(indices[0])]
        if lookup:
            mapping[int(episode_id)] = lookup[int(raw_task)]
        else:
            mapping[int(episode_id)] = str(raw_task)
    return mapping


def _select_episode_ids(
    bundle: dict[str, np.ndarray],
    tasks: list[str],
    max_episodes_per_task: int | None,
    seed: int,
    task_names: list[str] | None,
) -> list[int]:
    episode_task = _episode_task_map(bundle, task_names)
    ordered_ids = _ordered_episode_ids(bundle["episode_ids"])
    rng = np.random.default_rng(seed)
    selected: set[int] = set()

    for task_name in tasks:
        task_episode_ids = [episode_id for episode_id in ordered_ids if episode_task.get(episode_id) == task_name]
        if not task_episode_ids:
            raise ValueError(f"Task '{task_name}' does not exist in the input NPZ.")
        if max_episodes_per_task is None:
            chosen = task_episode_ids
        else:
            if len(task_episode_ids) < max_episodes_per_task:
                raise ValueError(
                    f"Task '{task_name}' only has {len(task_episode_ids)} episodes, "
                    f"fewer than requested {max_episodes_per_task}."
                )
            picked = rng.choice(np.asarray(task_episode_ids, dtype=np.int64), size=max_episodes_per_task, replace=False)
            chosen = picked.tolist()
        selected.update(int(v) for v in chosen)

    return [episode_id for episode_id in ordered_ids if episode_id in selected]


def _concatenate_parts(parts: list[np.ndarray], sample: np.ndarray) -> np.ndarray:
    if sample.ndim == 0:
        return np.asarray(parts, dtype=sample.dtype)
    return np.concatenate(parts, axis=0)


def filter_continuous_npz(
    input_path: Path,
    output_path: Path,
    *,
    tasks: list[str],
    max_episodes_per_task: int | None,
    seed: int,
    task_names: list[str] | None,
) -> dict[str, int]:
    bundle_npz = np.load(input_path, allow_pickle=True)
    bundle = {key: bundle_npz[key] for key in bundle_npz.files}
    keys = list(bundle.keys())
    ordered_episode_ids = _select_episode_ids(bundle, tasks, max_episodes_per_task, seed, task_names)
    task_lookup = _task_name_lookup(bundle, task_names)

    collected: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    next_episode_id = 0
    accepted_counts = {task_name: 0 for task_name in tasks}

    for original_episode_id in ordered_episode_ids:
        indices = np.nonzero(bundle["episode_ids"] == original_episode_id)[0]
        if indices.size == 0:
            continue
        indices = indices[np.argsort(bundle["step_ids"][indices])]
        raw_task = bundle["tasks"][int(indices[0])]
        task_name = task_lookup[int(raw_task)] if task_lookup else str(raw_task)
        for key in keys:
            values = bundle[key][indices]
            if key == "episode_ids":
                values = np.full(indices.shape[0], next_episode_id, dtype=bundle["episode_ids"].dtype)
            collected[key].append(values)
        accepted_counts[task_name] = accepted_counts.get(task_name, 0) + 1
        next_episode_id += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for key in keys:
        payload[key] = _concatenate_parts(collected[key], bundle[key])
    np.savez(output_path, **payload)

    return accepted_counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter a continuous NPZ into task-specific episode subsets.")
    parser.add_argument("--input", required=True, help="Input continuous NPZ path.")
    parser.add_argument("--output", required=True, help="Output filtered NPZ path.")
    parser.add_argument("--tasks", required=True, help="Comma-separated task names to keep.")
    parser.add_argument(
        "--config",
        default="configs/continuous_act_template.yaml",
        help="Config used to map integer task ids to sim.tasks names.",
    )
    parser.add_argument(
        "--max-episodes-per-task",
        type=int,
        default=None,
        help="Optional maximum number of episodes to keep per task. Sampling is random without replacement.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for episode sampling.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sim_tasks = [str(v) for v in cfg.get("sim", {}).get("tasks", [])]
    counts = filter_continuous_npz(
        Path(args.input),
        Path(args.output),
        tasks=_normalize_tasks(args.tasks),
        max_episodes_per_task=args.max_episodes_per_task,
        seed=args.seed,
        task_names=sim_tasks or None,
    )
    print(f"saved={args.output}")
    print(f"accepted_counts={counts}")


if __name__ == "__main__":
    main()

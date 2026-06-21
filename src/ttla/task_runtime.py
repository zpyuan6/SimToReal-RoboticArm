from __future__ import annotations

import numpy as np


def build_runtime_state(
    current_q: np.ndarray,
    task_id: int,
    step_idx: int,
    horizon: int,
) -> np.ndarray:
    q = np.asarray(current_q, dtype=np.float32).reshape(-1)
    if q.size != 6:
        raise ValueError("Expected 6 joint values when building runtime state.")
    qvel = np.zeros(6, dtype=np.float32)
    task_one_hot = np.zeros(3, dtype=np.float32)
    clipped_task_id = int(np.clip(task_id, 0, 2))
    task_one_hot[clipped_task_id] = 1.0
    context = np.asarray(
        [
            float(step_idx / max(1, horizon)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([q, qvel, task_one_hot, context], dtype=np.float32)

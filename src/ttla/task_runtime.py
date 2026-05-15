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
    context = np.asarray(
        [
            float(task_id),
            float(step_idx / max(1, horizon)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([q, qvel, context], dtype=np.float32)

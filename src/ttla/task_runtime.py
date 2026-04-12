from __future__ import annotations

import numpy as np


def build_runtime_state(
    current_q: np.ndarray,
    attached: bool,
    verified: bool,
    task_id: int,
    step_idx: int,
    horizon: int,
    lifted: bool = False,
    placed: bool = False,
) -> np.ndarray:
    q = np.asarray(current_q, dtype=np.float32).reshape(-1)
    if q.size != 6:
        raise ValueError("Expected 6 joint values when building runtime state.")
    qvel = np.zeros(6, dtype=np.float32)
    flags = np.asarray(
        [
            float(attached),
            float(verified),
            float(lifted),
            float(placed),
            float(task_id),
            float(step_idx / max(1, horizon)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([q, qvel, flags], dtype=np.float32)

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SKILL_NAMES = [
    "scan_left",
    "scan_right",
    "lift_view",
    "dip_view",
    "approach_small",
    "hold",
    "stop",
]


@dataclass(frozen=True)
class SkillCommand:
    joint_delta: np.ndarray
    dwell: int = 1


def skill_name(skill_id: int) -> str:
    return SKILL_NAMES[skill_id]


def skill_count() -> int:
    return len(SKILL_NAMES)


def skill_to_joint_delta(skill_id: int) -> SkillCommand:
    mapping = {
        0: SkillCommand(np.asarray([0.12, 0.00, 0.00, 0.00, 0.18, 0.00], dtype=np.float32)),
        1: SkillCommand(np.asarray([-0.12, 0.00, 0.00, 0.00, -0.18, 0.00], dtype=np.float32)),
        2: SkillCommand(np.asarray([0.00, -0.08, 0.06, 0.10, 0.10, -0.05], dtype=np.float32)),
        3: SkillCommand(np.asarray([0.00, 0.08, -0.06, -0.10, -0.10, 0.05], dtype=np.float32)),
        4: SkillCommand(np.asarray([0.00, -0.06, -0.08, 0.05, 0.04, -0.08], dtype=np.float32)),
        5: SkillCommand(np.asarray([0.00, 0.00, 0.00, -0.03, 0.00, 0.04], dtype=np.float32), dwell=2),
        6: SkillCommand(np.asarray([0.00, 0.00, 0.00, 0.00, 0.00, 0.00], dtype=np.float32), dwell=1),
    }
    return mapping[skill_id]

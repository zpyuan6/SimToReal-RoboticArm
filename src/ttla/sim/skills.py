from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PRIMITIVE_NAMES = [
    "obs_left",
    "obs_right",
    "obs_center",
    "verify_target",
    "prealign_grasp",
    "approach_coarse",
    "approach_fine",
    "retreat",
    "reobserve",
    "pregrasp_servo",
    "grasp_execute",
    "lift_object",
    "transport_to_dropzone",
    "place_object",
    "abort",
]

OBS_LEFT_ID = 0
OBS_RIGHT_ID = 1
OBS_CENTER_ID = 2
VERIFY_TARGET_ID = 3
PREALIGN_GRASP_ID = 4
APPROACH_COARSE_ID = 5
APPROACH_FINE_ID = 6
RETREAT_ID = 7
REOBSERVE_ID = 8
PREGRASP_SERVO_ID = 9
GRASP_EXECUTE_ID = 10
LIFT_OBJECT_ID = 11
TRANSPORT_TO_DROPZONE_ID = 12
PLACE_OBJECT_ID = 13
ABORT_ID = 14

HOME_QPOS = np.asarray([0.0, 0.0, 2.618, -1.0472, 0.0, 0.85], dtype=np.float32)
OBS_CENTER_QPOS = np.asarray([0.0, -0.10, 2.55, -1.18, 0.0, 0.95], dtype=np.float32)
OBS_LEFT_QPOS = np.asarray([0.35, -0.10, 2.55, -1.18, 0.0, 0.95], dtype=np.float32)
OBS_RIGHT_QPOS = np.asarray([-0.35, -0.10, 2.55, -1.18, 0.0, 0.95], dtype=np.float32)
PREALIGN_BASE_QPOS = np.asarray([0.0, -0.28, 2.30, -1.20, 0.0, 1.05], dtype=np.float32)
CARRY_QPOS = np.asarray([0.0, -0.42, 2.08, -0.86, 0.0, 0.22], dtype=np.float32)
DROPZONE_QPOS = np.asarray([-0.55, -0.38, 2.00, -0.78, 0.0, 0.24], dtype=np.float32)


LEVEL1_PRIMITIVES = (
    OBS_LEFT_ID,
    OBS_RIGHT_ID,
    OBS_CENTER_ID,
    VERIFY_TARGET_ID,
    ABORT_ID,
)

LEVEL2_PRIMITIVES = (
    OBS_LEFT_ID,
    OBS_RIGHT_ID,
    OBS_CENTER_ID,
    VERIFY_TARGET_ID,
    PREALIGN_GRASP_ID,
    APPROACH_COARSE_ID,
    APPROACH_FINE_ID,
    RETREAT_ID,
    ABORT_ID,
)

LEVEL3_PRIMITIVES = (
    REOBSERVE_ID,
    PREGRASP_SERVO_ID,
    GRASP_EXECUTE_ID,
    LIFT_OBJECT_ID,
    TRANSPORT_TO_DROPZONE_ID,
    PLACE_OBJECT_ID,
    RETREAT_ID,
    ABORT_ID,
)


@dataclass(frozen=True)
class PrimitiveStep:
    joint_target: np.ndarray
    dwell: int = 1


@dataclass(frozen=True)
class HybridAction:
    skill_id: int
    residual: np.ndarray


def primitive_name(primitive_id: int) -> str:
    return PRIMITIVE_NAMES[int(primitive_id)]


def primitive_count() -> int:
    return len(PRIMITIVE_NAMES)


def primitive_id(name: str) -> int:
    return PRIMITIVE_NAMES.index(name)


def primitive_action(action: int | dict[str, int] | str) -> int:
    if isinstance(action, HybridAction):
        return int(action.skill_id)
    if isinstance(action, int):
        return int(action)
    if isinstance(action, str):
        return primitive_id(action)
    if isinstance(action, dict):
        if "primitive_id" in action:
            return int(action["primitive_id"])
        if "primitive" in action:
            return primitive_id(str(action["primitive"]))
    raise TypeError(f"Unsupported primitive action type: {type(action)!r}")


def allowed_primitives(task_level: int) -> tuple[int, ...]:
    if task_level == 1:
        return LEVEL1_PRIMITIVES
    if task_level == 2:
        return LEVEL2_PRIMITIVES
    if task_level == 3:
        return LEVEL3_PRIMITIVES
    raise KeyError(task_level)


def observe_pose(primitive_id_value: int) -> np.ndarray:
    if primitive_id_value == OBS_LEFT_ID:
        return OBS_LEFT_QPOS.copy()
    if primitive_id_value == OBS_RIGHT_ID:
        return OBS_RIGHT_QPOS.copy()
    return OBS_CENTER_QPOS.copy()


# Compatibility aliases for older scripts that still import skill_* names.
SKILL_NAMES = PRIMITIVE_NAMES
STOP_SKILL_ID = ABORT_ID


def skill_name(skill_id: int) -> str:
    return primitive_name(skill_id)


def skill_count() -> int:
    return primitive_count()


def zero_residual(residual_dim: int = 6) -> np.ndarray:
    return np.zeros(residual_dim, dtype=np.float32)

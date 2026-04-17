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
    "hold_position",
    "abort",
]

PRIMITIVE_DESCRIPTIONS = [
    "rotate to the left observation pose and look for the target",
    "rotate to the right observation pose and look for the target",
    "move to the centered observation pose and inspect the target",
    "hold view and confirm the target is visible and centered",
    "move into a coarse pre-grasp alignment in front of the target",
    "take a larger approach step toward the target",
    "take a smaller precise approach step toward the target",
    "back away from the target to recover from a bad approach",
    "return to a stable re-observation pose before trying again",
    "servo the gripper into the pre-grasp window using visual feedback",
    "execute the grasp by descending, closing the gripper, and checking attachment",
    "lift the grasped object upward while keeping the gripper closed",
    "carry the grasped object toward the blue drop zone",
    "lower over the blue drop zone, open the gripper, and release the object",
    "keep the current pose still when the task is already complete",
    "abort the task and return to a safe home pose",
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
HOLD_POSITION_ID = 14
ABORT_ID = 15

# These simulator pose presets are intentionally aligned to the real-arm
# primitive validator mapping rather than the older MuJoCo-authored defaults.
# The goal is to keep primitive semantics consistent across real and simulated
# execution without retraining the policy backbone.
HOME_QPOS = np.asarray([0.0, 0.0, 1.5708, 0.0, 0.0, 0.1964], dtype=np.float32)
OBS_CENTER_QPOS = np.asarray([0.0, 0.2094, 1.8850, 0.3142, 0.0, 0.4320], dtype=np.float32)
OBS_LEFT_QPOS = np.asarray([0.5236, 0.2094, 1.8850, 0.3142, 0.0, 0.4320], dtype=np.float32)
OBS_RIGHT_QPOS = np.asarray([-0.5236, 0.2094, 1.8850, 0.3142, 0.0, 0.4320], dtype=np.float32)
PREALIGN_BASE_QPOS = np.asarray([0.0, 0.3142, 2.0595, 0.4538, 0.0, 0.5105], dtype=np.float32)
CARRY_QPOS = np.asarray([0.0, -0.1396, 1.6755, -0.1396, 0.0, 0.0], dtype=np.float32)
DROPZONE_QPOS = np.asarray([-0.5833, -0.1745, 1.7104, -0.1745, 0.0, 0.0], dtype=np.float32)


LEVEL1_PRIMITIVES = (
    OBS_LEFT_ID,
    OBS_RIGHT_ID,
    OBS_CENTER_ID,
    VERIFY_TARGET_ID,
    HOLD_POSITION_ID,
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
    HOLD_POSITION_ID,
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
    HOLD_POSITION_ID,
    ABORT_ID,
)

ALL_PRIMITIVES = tuple(range(len(PRIMITIVE_NAMES)))


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


def primitive_description(primitive_id: int) -> str:
    return PRIMITIVE_DESCRIPTIONS[int(primitive_id)]


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
    if task_level in (1, 2, 3):
        return ALL_PRIMITIVES
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

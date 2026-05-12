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
    "briefly hold view to confirm the target is visible and centered",
    "move into a coarse pre-grasp alignment in front of the target",
    "take a larger approach step toward the target",
    "take a smaller precise approach step toward the target",
    "back away from the target to recover from a bad approach",
    "return to a stable centered observation pose before trying again",
    "servo the gripper into the near-target pre-grasp window using visual feedback",
    "execute the grasp by descending, closing the gripper, and checking attachment",
    "lift the grasped object upward while keeping the gripper closed",
    "carry the grasped object toward the blue drop zone",
    "lower over the blue drop zone, open the gripper, and release the object",
    "keep the current pose still when the task is already complete",
    "abort the task and return to a safe home pose",
]

PRIMITIVE_VOCAB_LEGACY = "legacy"
PRIMITIVE_VOCAB_COMPACT = "compact"

PRIMITIVE_FAMILY_NAMES = [
    "observe",
    "confirm",
    "approach",
    "recover",
    "grasp",
    "lift",
    "transport",
    "place",
    "hold",
    "abort",
]

COMPACT_PRIMITIVE_NAMES = [
    "observe_left",
    "observe_center",
    "observe_right",
    "confirm",
    "approach",
    "pregrasp",
    "retreat",
    "grasp",
    "lift",
    "transport",
    "place",
    "hold",
    "abort",
]

COMPACT_PRIMITIVE_DESCRIPTIONS = [
    "observe the scene from the left view",
    "observe the scene from the centered view",
    "observe the scene from the right view",
    "confirm that the target is well observed",
    "move toward a task-appropriate coarse approach state",
    "servo into the near-target pre-grasp window",
    "back away to recover from a poor alignment",
    "close the grasp on the target",
    "lift the grasped object upward",
    "carry the object toward the destination",
    "lower and release the object at the destination",
    "hold the current stable pose",
    "abort and return to a safe pose",
]

OBSERVE_FAMILY_ID = 0
CONFIRM_FAMILY_ID = 1
APPROACH_FAMILY_ID = 2
RECOVER_FAMILY_ID = 3
GRASP_FAMILY_ID = 4
LIFT_FAMILY_ID = 5
TRANSPORT_FAMILY_ID = 6
PLACE_FAMILY_ID = 7
HOLD_FAMILY_ID = 8
ABORT_FAMILY_ID = 9

COMPACT_OBS_LEFT_ID = 0
COMPACT_OBS_CENTER_ID = 1
COMPACT_OBS_RIGHT_ID = 2
COMPACT_CONFIRM_ID = 3
COMPACT_APPROACH_ID = 4
COMPACT_PREGRASP_ID = 5
COMPACT_RETREAT_ID = 6
COMPACT_GRASP_ID = 7
COMPACT_LIFT_ID = 8
COMPACT_TRANSPORT_ID = 9
COMPACT_PLACE_ID = 10
COMPACT_HOLD_ID = 11
COMPACT_ABORT_ID = 12

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

PRIMITIVE_TO_FAMILY = {
    OBS_LEFT_ID: OBSERVE_FAMILY_ID,
    OBS_RIGHT_ID: OBSERVE_FAMILY_ID,
    OBS_CENTER_ID: OBSERVE_FAMILY_ID,
    VERIFY_TARGET_ID: CONFIRM_FAMILY_ID,
    PREALIGN_GRASP_ID: APPROACH_FAMILY_ID,
    APPROACH_COARSE_ID: APPROACH_FAMILY_ID,
    APPROACH_FINE_ID: APPROACH_FAMILY_ID,
    RETREAT_ID: RECOVER_FAMILY_ID,
    REOBSERVE_ID: OBSERVE_FAMILY_ID,
    PREGRASP_SERVO_ID: APPROACH_FAMILY_ID,
    GRASP_EXECUTE_ID: GRASP_FAMILY_ID,
    LIFT_OBJECT_ID: LIFT_FAMILY_ID,
    TRANSPORT_TO_DROPZONE_ID: TRANSPORT_FAMILY_ID,
    PLACE_OBJECT_ID: PLACE_FAMILY_ID,
    HOLD_POSITION_ID: HOLD_FAMILY_ID,
    ABORT_ID: ABORT_FAMILY_ID,
}

FAMILY_TO_PRIMITIVES = {
    family_id: tuple(primitive_id for primitive_id, mapped in PRIMITIVE_TO_FAMILY.items() if mapped == family_id)
    for family_id in range(len(PRIMITIVE_FAMILY_NAMES))
}

LEGACY_TO_COMPACT = {
    OBS_LEFT_ID: COMPACT_OBS_LEFT_ID,
    OBS_CENTER_ID: COMPACT_OBS_CENTER_ID,
    OBS_RIGHT_ID: COMPACT_OBS_RIGHT_ID,
    VERIFY_TARGET_ID: COMPACT_CONFIRM_ID,
    PREALIGN_GRASP_ID: COMPACT_APPROACH_ID,
    APPROACH_COARSE_ID: COMPACT_APPROACH_ID,
    APPROACH_FINE_ID: COMPACT_APPROACH_ID,
    PREGRASP_SERVO_ID: COMPACT_PREGRASP_ID,
    RETREAT_ID: COMPACT_RETREAT_ID,
    REOBSERVE_ID: COMPACT_OBS_CENTER_ID,
    GRASP_EXECUTE_ID: COMPACT_GRASP_ID,
    LIFT_OBJECT_ID: COMPACT_LIFT_ID,
    TRANSPORT_TO_DROPZONE_ID: COMPACT_TRANSPORT_ID,
    PLACE_OBJECT_ID: COMPACT_PLACE_ID,
    HOLD_POSITION_ID: COMPACT_HOLD_ID,
    ABORT_ID: COMPACT_ABORT_ID,
}


@dataclass(frozen=True)
class PrimitiveStep:
    joint_target: np.ndarray
    dwell: int = 1


@dataclass(frozen=True)
class HybridAction:
    skill_id: int
    residual: np.ndarray


def primitive_name(primitive_id: int, primitive_vocabulary: str = PRIMITIVE_VOCAB_LEGACY) -> str:
    if primitive_vocabulary == PRIMITIVE_VOCAB_COMPACT:
        return COMPACT_PRIMITIVE_NAMES[int(primitive_id)]
    return PRIMITIVE_NAMES[int(primitive_id)]


def primitive_description(primitive_id: int, primitive_vocabulary: str = PRIMITIVE_VOCAB_LEGACY) -> str:
    if primitive_vocabulary == PRIMITIVE_VOCAB_COMPACT:
        return COMPACT_PRIMITIVE_DESCRIPTIONS[int(primitive_id)]
    return PRIMITIVE_DESCRIPTIONS[int(primitive_id)]


def primitive_family_id(
    primitive_id: int,
    primitive_vocabulary: str = PRIMITIVE_VOCAB_LEGACY,
) -> int:
    primitive_id = int(primitive_id)
    if primitive_vocabulary == PRIMITIVE_VOCAB_COMPACT:
        compact_to_family = {
            COMPACT_OBS_LEFT_ID: OBSERVE_FAMILY_ID,
            COMPACT_OBS_CENTER_ID: OBSERVE_FAMILY_ID,
            COMPACT_OBS_RIGHT_ID: OBSERVE_FAMILY_ID,
            COMPACT_CONFIRM_ID: CONFIRM_FAMILY_ID,
            COMPACT_APPROACH_ID: APPROACH_FAMILY_ID,
            COMPACT_PREGRASP_ID: APPROACH_FAMILY_ID,
            COMPACT_RETREAT_ID: RECOVER_FAMILY_ID,
            COMPACT_GRASP_ID: GRASP_FAMILY_ID,
            COMPACT_LIFT_ID: LIFT_FAMILY_ID,
            COMPACT_TRANSPORT_ID: TRANSPORT_FAMILY_ID,
            COMPACT_PLACE_ID: PLACE_FAMILY_ID,
            COMPACT_HOLD_ID: HOLD_FAMILY_ID,
            COMPACT_ABORT_ID: ABORT_FAMILY_ID,
        }
        return int(compact_to_family[primitive_id])
    return int(PRIMITIVE_TO_FAMILY[primitive_id])


def primitive_family_name(
    primitive_id_or_family_id: int,
    *,
    primitive: bool = True,
    primitive_vocabulary: str = PRIMITIVE_VOCAB_LEGACY,
) -> str:
    family_id = (
        primitive_family_id(int(primitive_id_or_family_id), primitive_vocabulary=primitive_vocabulary)
        if primitive
        else int(primitive_id_or_family_id)
    )
    return PRIMITIVE_FAMILY_NAMES[family_id]


def family_primitives(family_id: int) -> tuple[int, ...]:
    return FAMILY_TO_PRIMITIVES[int(family_id)]


def project_primitive_ids(
    primitive_ids: tuple[int, ...] | list[int] | set[int],
    primitive_vocabulary: str = PRIMITIVE_VOCAB_LEGACY,
) -> tuple[int, ...]:
    if primitive_vocabulary == PRIMITIVE_VOCAB_LEGACY:
        return tuple(int(primitive_id) for primitive_id in primitive_ids)
    projected: list[int] = []
    for primitive_id in primitive_ids:
        mapped = remap_primitive_id(int(primitive_id), primitive_vocabulary)
        if mapped not in projected:
            projected.append(mapped)
    return tuple(projected)


def family_projected_primitives(
    family_ids: tuple[int, ...] | list[int] | set[int],
    primitive_vocabulary: str = PRIMITIVE_VOCAB_LEGACY,
) -> tuple[int, ...]:
    primitive_ids: list[int] = []
    for family_id in family_ids:
        primitive_ids.extend(family_primitives(int(family_id)))
    return project_primitive_ids(tuple(primitive_ids), primitive_vocabulary=primitive_vocabulary)


def primitive_count(primitive_vocabulary: str = PRIMITIVE_VOCAB_LEGACY) -> int:
    if primitive_vocabulary == PRIMITIVE_VOCAB_COMPACT:
        return len(COMPACT_PRIMITIVE_NAMES)
    return len(PRIMITIVE_NAMES)


def primitive_id(name: str, primitive_vocabulary: str = PRIMITIVE_VOCAB_LEGACY) -> int:
    if primitive_vocabulary == PRIMITIVE_VOCAB_COMPACT:
        return COMPACT_PRIMITIVE_NAMES.index(name)
    return PRIMITIVE_NAMES.index(name)


def remap_primitive_id(primitive_id_value: int, primitive_vocabulary: str) -> int:
    if primitive_vocabulary == PRIMITIVE_VOCAB_COMPACT:
        return int(LEGACY_TO_COMPACT[int(primitive_id_value)])
    return int(primitive_id_value)


def primitive_action(
    action: int | dict[str, int] | str,
    primitive_vocabulary: str = PRIMITIVE_VOCAB_LEGACY,
) -> int:
    if isinstance(action, HybridAction):
        return int(action.skill_id)
    if isinstance(action, int):
        return int(action)
    if isinstance(action, str):
        return primitive_id(action, primitive_vocabulary=primitive_vocabulary)
    if isinstance(action, dict):
        if "primitive_id" in action:
            return int(action["primitive_id"])
        if "primitive" in action:
            return primitive_id(str(action["primitive"]), primitive_vocabulary=primitive_vocabulary)
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

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PRIMITIVE_NAMES = [
    "obs_left",
    "obs_right",
    "obs_center",
    "approach",
    "retreat",
    "pregrasp_servo",
    "grasp_execute",
    "lift_object",
    "transport_to_dropzone",
    "place_object",
    "abort",
]

PRIMITIVE_DESCRIPTIONS = [
    "rotate to the left observation pose and look for the target",
    "rotate to the right observation pose and look for the target",
    "move to the centered observation pose and inspect the target",
    "execute a fixed approach script from the current observation state toward the target region",
    "back away from the target to recover from a bad approach",
    "execute a fixed pre-grasp settling script before closing the gripper",
    "close the gripper and check whether the object becomes attached",
    "lift the grasped object upward while keeping the gripper closed",
    "move the grasped object along a fixed transport script toward the blue drop zone",
    "move to a fixed release pose over the blue drop zone and open the gripper",
    "abort the task and return to a safe home pose",
]

PRIMITIVE_VOCAB_LEGACY = "legacy"

PRIMITIVE_FAMILY_NAMES = [
    "observe",
    "approach",
    "recover",
    "grasp",
    "lift",
    "transport",
    "place",
    "abort",
]

OBSERVE_FAMILY_ID = 0
APPROACH_FAMILY_ID = 1
RECOVER_FAMILY_ID = 2
GRASP_FAMILY_ID = 3
LIFT_FAMILY_ID = 4
TRANSPORT_FAMILY_ID = 5
PLACE_FAMILY_ID = 6
ABORT_FAMILY_ID = 7

OBS_LEFT_ID = 0
OBS_RIGHT_ID = 1
OBS_CENTER_ID = 2
APPROACH_ID = 3
RETREAT_ID = 4
PREGRASP_SERVO_ID = 5
GRASP_EXECUTE_ID = 6
LIFT_OBJECT_ID = 7
TRANSPORT_TO_DROPZONE_ID = 8
PLACE_OBJECT_ID = 9
ABORT_ID = 10

# These simulator pose presets are intentionally aligned to the real-arm
# primitive validator mapping rather than the older MuJoCo-authored defaults.
# The goal is to keep primitive semantics consistent across real and simulated
# execution without retraining the policy backbone.
HOME_QPOS = np.asarray([0.0, 0.0, 1.5708, 0.0, 0.0, 0.1964], dtype=np.float32)
# Observation poses retuned to keep the forearm camera pitched down while
# lifting the gripper clear of the tabletop. The prior retune fixed visibility
# but left the tool point slightly below table height, which made the
# observation primitives skim the table surface.
OBS_CENTER_QPOS = np.asarray([0.0, 0.1, 2.6, -0.4189, 0.0, 0.4363], dtype=np.float32)
OBS_LEFT_QPOS = np.asarray([0.3142, 0.1, 2.6, -0.4189, 0.0, 0.4363], dtype=np.float32)
OBS_RIGHT_QPOS = np.asarray([-0.2443, 0.1, 2.6, -0.4189, 0.0, 0.4363], dtype=np.float32)
# The remaining manipulation primitives are fixed scripted joint targets,
# derived from the last visually validated MuJoCo task flow. This keeps sim and
# real execution on the same side of the abstraction boundary: the policy picks
# among fixed motor scripts, rather than relying on hidden sim-only closed-loop
# logic inside the primitive itself.
APPROACH_QPOS = np.asarray([0.0, 0.1478, 2.5743, -0.4197, 0.0, 0.9981], dtype=np.float32)
PREGRASP_QPOS = np.asarray([0.0, 0.2550, 2.4100, -0.3000, 0.0, 0.8989], dtype=np.float32)
LIFT_QPOS = np.asarray([-0.0643, 0.1944, 2.3820, -0.3147, 0.0, 0.0], dtype=np.float32)
TRANSPORT_QPOS = np.asarray([-0.4587, 0.2082, 2.3622, -0.1081, 0.0, 0.0], dtype=np.float32)
PLACE_RELEASE_QPOS = np.asarray([-0.4435, 0.2311, 2.3562, -0.1074, 0.0, 0.0], dtype=np.float32)

# Compatibility aliases retained for older scripts/imports.
PREALIGN_BASE_QPOS = APPROACH_QPOS.copy()
CARRY_QPOS = TRANSPORT_QPOS.copy()
DROPZONE_QPOS = PLACE_RELEASE_QPOS.copy()


LEVEL1_PRIMITIVES = (
    OBS_LEFT_ID,
    OBS_RIGHT_ID,
    OBS_CENTER_ID,
    ABORT_ID,
)

LEVEL2_PRIMITIVES = (
    OBS_LEFT_ID,
    OBS_RIGHT_ID,
    OBS_CENTER_ID,
    APPROACH_ID,
    RETREAT_ID,
    ABORT_ID,
)

LEVEL3_PRIMITIVES = (
    OBS_CENTER_ID,
    APPROACH_ID,
    PREGRASP_SERVO_ID,
    GRASP_EXECUTE_ID,
    LIFT_OBJECT_ID,
    TRANSPORT_TO_DROPZONE_ID,
    PLACE_OBJECT_ID,
    RETREAT_ID,
    ABORT_ID,
)

ALL_PRIMITIVES = tuple(range(len(PRIMITIVE_NAMES)))

PRIMITIVE_TO_FAMILY = {
    OBS_LEFT_ID: OBSERVE_FAMILY_ID,
    OBS_RIGHT_ID: OBSERVE_FAMILY_ID,
    OBS_CENTER_ID: OBSERVE_FAMILY_ID,
    APPROACH_ID: APPROACH_FAMILY_ID,
    RETREAT_ID: RECOVER_FAMILY_ID,
    PREGRASP_SERVO_ID: APPROACH_FAMILY_ID,
    GRASP_EXECUTE_ID: GRASP_FAMILY_ID,
    LIFT_OBJECT_ID: LIFT_FAMILY_ID,
    TRANSPORT_TO_DROPZONE_ID: TRANSPORT_FAMILY_ID,
    PLACE_OBJECT_ID: PLACE_FAMILY_ID,
    ABORT_ID: ABORT_FAMILY_ID,
}

FAMILY_TO_PRIMITIVES = {
    family_id: tuple(primitive_id for primitive_id, mapped in PRIMITIVE_TO_FAMILY.items() if mapped == family_id)
    for family_id in range(len(PRIMITIVE_FAMILY_NAMES))
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
    del primitive_vocabulary
    return PRIMITIVE_NAMES[int(primitive_id)]


def primitive_description(primitive_id: int, primitive_vocabulary: str = PRIMITIVE_VOCAB_LEGACY) -> str:
    del primitive_vocabulary
    return PRIMITIVE_DESCRIPTIONS[int(primitive_id)]


def primitive_family_id(
    primitive_id: int,
    primitive_vocabulary: str = PRIMITIVE_VOCAB_LEGACY,
) -> int:
    del primitive_vocabulary
    primitive_id = int(primitive_id)
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
    del primitive_vocabulary
    return tuple(int(primitive_id) for primitive_id in primitive_ids)


def family_projected_primitives(
    family_ids: tuple[int, ...] | list[int] | set[int],
    primitive_vocabulary: str = PRIMITIVE_VOCAB_LEGACY,
) -> tuple[int, ...]:
    primitive_ids: list[int] = []
    for family_id in family_ids:
        primitive_ids.extend(family_primitives(int(family_id)))
    return project_primitive_ids(tuple(primitive_ids), primitive_vocabulary=primitive_vocabulary)


def primitive_count(primitive_vocabulary: str = PRIMITIVE_VOCAB_LEGACY) -> int:
    del primitive_vocabulary
    return len(PRIMITIVE_NAMES)


def primitive_id(name: str, primitive_vocabulary: str = PRIMITIVE_VOCAB_LEGACY) -> int:
    del primitive_vocabulary
    return PRIMITIVE_NAMES.index(name)


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

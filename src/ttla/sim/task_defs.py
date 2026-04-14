from __future__ import annotations

from dataclasses import dataclass

from .skills import (
    ABORT_ID,
    ALL_PRIMITIVES,
    APPROACH_COARSE_ID,
    APPROACH_FINE_ID,
    GRASP_EXECUTE_ID,
    HOLD_POSITION_ID,
    LIFT_OBJECT_ID,
    OBS_CENTER_ID,
    OBS_LEFT_ID,
    OBS_RIGHT_ID,
    PLACE_OBJECT_ID,
    PREALIGN_GRASP_ID,
    PREGRASP_SERVO_ID,
    REOBSERVE_ID,
    RETREAT_ID,
    TRANSPORT_TO_DROPZONE_ID,
    VERIFY_TARGET_ID,
    primitive_description,
    primitive_name,
)

STAGE_OBSERVE_ID = 0
STAGE_VERIFY_ID = 1
STAGE_APPROACH_ID = 2
STAGE_PREGRASP_ID = 3
STAGE_GRASP_ID = 4
STAGE_LIFT_ID = 5
STAGE_TRANSPORT_ID = 6
STAGE_TERMINAL_ID = 7
NUM_SUPERVISION_STAGES = 8


@dataclass(frozen=True)
class TaskSpec:
    name: str
    task_id: int
    level: int
    description: str
    allowed_primitives: tuple[int, ...]
    primary_primitives: tuple[int, ...]


TASK_SPECS = {
    "level1_verify": TaskSpec(
        name="level1_verify",
        task_id=0,
        level=1,
        description="Observe and verify that the target is visible and centered.",
        allowed_primitives=ALL_PRIMITIVES,
        primary_primitives=(
            OBS_LEFT_ID,
            OBS_RIGHT_ID,
            OBS_CENTER_ID,
            VERIFY_TARGET_ID,
            HOLD_POSITION_ID,
            ABORT_ID,
        ),
    ),
    "level2_approach": TaskSpec(
        name="level2_approach",
        task_id=1,
        level=2,
        description="Observe, pre-align, and approach the target to a stable pre-grasp state.",
        allowed_primitives=ALL_PRIMITIVES,
        primary_primitives=(
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
        ),
    ),
    "level3_pick_place": TaskSpec(
        name="level3_pick_place",
        task_id=2,
        level=3,
        description="Re-observe, grasp the wire-ear cork target, lift it, transport it, and place it in the blue drop zone.",
        allowed_primitives=ALL_PRIMITIVES,
        primary_primitives=(
            REOBSERVE_ID,
            PREGRASP_SERVO_ID,
            GRASP_EXECUTE_ID,
            LIFT_OBJECT_ID,
            TRANSPORT_TO_DROPZONE_ID,
            PLACE_OBJECT_ID,
            RETREAT_ID,
            HOLD_POSITION_ID,
            ABORT_ID,
        ),
    ),
}

TASK_TO_ID = {name: spec.task_id for name, spec in TASK_SPECS.items()}
ID_TO_TASK = {spec.task_id: spec for spec in TASK_SPECS.values()}


def task_spec(task_name: str) -> TaskSpec:
    return TASK_SPECS[task_name]


def task_instruction(task_name_or_id: str | int) -> str:
    if isinstance(task_name_or_id, int):
        spec = ID_TO_TASK[int(task_name_or_id)]
    else:
        spec = TASK_SPECS[task_name_or_id]
    preferred = ", ".join(primitive_name(pid) for pid in spec.primary_primitives)
    if spec.task_id == 0:
        goal = (
            "Finish as soon as the target is visible and centered, then choose hold_position instead of extra motion."
        )
    elif spec.task_id == 1:
        goal = (
            "Finish when the target reaches a stable pre-grasp pose, then choose hold_position instead of extra motion."
        )
    else:
        goal = (
            "Use the blue drop zone as the placement target. Grip the ear handle, then choose hold_position after a successful place."
        )
    return (
        f"Task: {spec.description} "
        f"Useful primitives for this task: {preferred}. "
        f"{goal}"
    )


def task_action_hint(task_name_or_id: str | int) -> str:
    if isinstance(task_name_or_id, int):
        spec = ID_TO_TASK[int(task_name_or_id)]
    else:
        spec = TASK_SPECS[task_name_or_id]
    preferred = ", ".join(primitive_name(pid) for pid in spec.primary_primitives)
    if spec.task_id == 0:
        suffix = "prefer verify_target then hold_position once the target is centered"
    elif spec.task_id == 1:
        suffix = "prefer prealign_grasp then approach_coarse or approach_fine, then hold_position"
    else:
        suffix = "prefer pregrasp_servo, grasp_execute, lift_object, transport_to_dropzone, and place_object to move the wire-ear cork target, then hold_position"
    return f"Preferred primitives: {preferred}. For this task, {suffix}."


def primitive_instruction(primitive_id: int) -> str:
    return f"{primitive_name(primitive_id)}: {primitive_description(primitive_id)}."


def supervision_stage_id(task_id: int, primitive_id: int) -> int:
    if task_id == 0:
        if primitive_id in (OBS_LEFT_ID, OBS_RIGHT_ID, OBS_CENTER_ID):
            return STAGE_OBSERVE_ID
        if primitive_id == VERIFY_TARGET_ID:
            return STAGE_VERIFY_ID
        return STAGE_TERMINAL_ID
    if task_id == 1:
        if primitive_id in (OBS_LEFT_ID, OBS_RIGHT_ID, OBS_CENTER_ID, VERIFY_TARGET_ID):
            return STAGE_OBSERVE_ID
        if primitive_id in (PREALIGN_GRASP_ID, APPROACH_COARSE_ID, APPROACH_FINE_ID, RETREAT_ID):
            return STAGE_APPROACH_ID
        return STAGE_TERMINAL_ID
    if task_id == 2:
        if primitive_id == REOBSERVE_ID:
            return STAGE_OBSERVE_ID
        if primitive_id in (PREGRASP_SERVO_ID, RETREAT_ID):
            return STAGE_PREGRASP_ID
        if primitive_id == GRASP_EXECUTE_ID:
            return STAGE_GRASP_ID
        if primitive_id == LIFT_OBJECT_ID:
            return STAGE_LIFT_ID
        if primitive_id == TRANSPORT_TO_DROPZONE_ID:
            return STAGE_TRANSPORT_ID
        return STAGE_TERMINAL_ID
    return STAGE_OBSERVE_ID

from __future__ import annotations

from dataclasses import dataclass

from .skills import (
    ABORT_ID,
    APPROACH_COARSE_ID,
    APPROACH_FAMILY_ID,
    APPROACH_FINE_ID,
    CONFIRM_FAMILY_ID,
    GRASP_FAMILY_ID,
    GRASP_EXECUTE_ID,
    HOLD_FAMILY_ID,
    HOLD_POSITION_ID,
    LIFT_FAMILY_ID,
    LIFT_OBJECT_ID,
    OBS_CENTER_ID,
    OBSERVE_FAMILY_ID,
    OBS_LEFT_ID,
    OBS_RIGHT_ID,
    PLACE_FAMILY_ID,
    PLACE_OBJECT_ID,
    PREALIGN_GRASP_ID,
    PREGRASP_SERVO_ID,
    RECOVER_FAMILY_ID,
    REOBSERVE_ID,
    RETREAT_ID,
    TRANSPORT_FAMILY_ID,
    TRANSPORT_TO_DROPZONE_ID,
    VERIFY_TARGET_ID,
    PRIMITIVE_FAMILY_NAMES,
    PRIMITIVE_VOCAB_COMPACT,
    PRIMITIVE_VOCAB_LEGACY,
    family_primitives,
    primitive_description,
    primitive_name,
    remap_primitive_id,
)

STAGE_OBSERVE_ID = 0
STAGE_CONFIRM_ID = 1
STAGE_APPROACH_ID = 2
STAGE_GRASP_ID = 3
STAGE_LIFT_ID = 4
STAGE_TRANSPORT_ID = 5
STAGE_PLACE_ID = 6
STAGE_TERMINAL_ID = 7
NUM_SUPERVISION_STAGES = 8


@dataclass(frozen=True)
class TaskSpec:
    name: str
    task_id: int
    level: int
    public_label: str
    description: str
    allowed_primitives: tuple[int, ...]
    primary_primitives: tuple[int, ...]
    primary_families: tuple[int, ...]
    compact_allowed_primitives: tuple[int, ...] | None = None
    compact_primary_primitives: tuple[int, ...] | None = None


def _expand_allowed_primitives(
    primary_primitives: tuple[int, ...],
    primary_families: tuple[int, ...],
) -> tuple[int, ...]:
    allowed: list[int] = []
    for family_id in primary_families:
        for primitive_id in family_primitives(family_id):
            if primitive_id not in allowed:
                allowed.append(primitive_id)
    for primitive_id in primary_primitives:
        if primitive_id not in allowed:
            allowed.append(primitive_id)
    return tuple(allowed)


TASK_SPECS = {
    "level1_verify": TaskSpec(
        name="level1_verify",
        task_id=0,
        level=1,
        public_label="observe_and_center",
        description="Use reusable observation primitives to bring the target into a stable centered view, then confirm completion.",
        primary_primitives=(
            OBS_LEFT_ID,
            OBS_RIGHT_ID,
            OBS_CENTER_ID,
            VERIFY_TARGET_ID,
            HOLD_POSITION_ID,
            ABORT_ID,
        ),
        primary_families=(OBSERVE_FAMILY_ID, CONFIRM_FAMILY_ID, HOLD_FAMILY_ID),
        allowed_primitives=_expand_allowed_primitives(
            (
                OBS_LEFT_ID,
                OBS_RIGHT_ID,
                OBS_CENTER_ID,
                VERIFY_TARGET_ID,
                HOLD_POSITION_ID,
                ABORT_ID,
            ),
            (OBSERVE_FAMILY_ID, CONFIRM_FAMILY_ID, HOLD_FAMILY_ID),
        ),
    ),
    "level2_approach": TaskSpec(
        name="level2_approach",
        task_id=1,
        level=2,
        public_label="approach_to_pregrasp",
        description="Reuse observation and approach primitives to reach a stable pre-grasp state near the target.",
        primary_primitives=(
            OBS_LEFT_ID,
            OBS_RIGHT_ID,
            OBS_CENTER_ID,
            PREALIGN_GRASP_ID,
            APPROACH_COARSE_ID,
            APPROACH_FINE_ID,
            RETREAT_ID,
            HOLD_POSITION_ID,
            ABORT_ID,
        ),
        primary_families=(OBSERVE_FAMILY_ID, APPROACH_FAMILY_ID, RECOVER_FAMILY_ID, HOLD_FAMILY_ID),
        allowed_primitives=_expand_allowed_primitives(
            (
                OBS_LEFT_ID,
                OBS_RIGHT_ID,
                OBS_CENTER_ID,
                PREALIGN_GRASP_ID,
                APPROACH_COARSE_ID,
                APPROACH_FINE_ID,
                RETREAT_ID,
                HOLD_POSITION_ID,
                ABORT_ID,
            ),
            (OBSERVE_FAMILY_ID, APPROACH_FAMILY_ID, RECOVER_FAMILY_ID, HOLD_FAMILY_ID),
        ),
    ),
    "level3_pick_place": TaskSpec(
        name="level3_pick_place",
        task_id=2,
        level=3,
        public_label="pick_and_place",
        description="Reuse observation, approach, grasp, lift, transport, and place primitives to move the wire-ear cork target into the blue drop zone.",
        primary_primitives=(
            OBS_CENTER_ID,
            PREGRASP_SERVO_ID,
            GRASP_EXECUTE_ID,
            LIFT_OBJECT_ID,
            TRANSPORT_TO_DROPZONE_ID,
            PLACE_OBJECT_ID,
            RETREAT_ID,
            HOLD_POSITION_ID,
            ABORT_ID,
        ),
        primary_families=(
            OBSERVE_FAMILY_ID,
            APPROACH_FAMILY_ID,
            RECOVER_FAMILY_ID,
            GRASP_FAMILY_ID,
            LIFT_FAMILY_ID,
            TRANSPORT_FAMILY_ID,
            PLACE_FAMILY_ID,
            HOLD_FAMILY_ID,
        ),
        allowed_primitives=_expand_allowed_primitives(
            (
                OBS_CENTER_ID,
                PREGRASP_SERVO_ID,
                GRASP_EXECUTE_ID,
                LIFT_OBJECT_ID,
                TRANSPORT_TO_DROPZONE_ID,
                PLACE_OBJECT_ID,
                RETREAT_ID,
                HOLD_POSITION_ID,
                ABORT_ID,
            ),
            (
                OBSERVE_FAMILY_ID,
                APPROACH_FAMILY_ID,
                RECOVER_FAMILY_ID,
                GRASP_FAMILY_ID,
                LIFT_FAMILY_ID,
                TRANSPORT_FAMILY_ID,
                PLACE_FAMILY_ID,
                HOLD_FAMILY_ID,
            ),
        ),
        compact_primary_primitives=(
            OBS_LEFT_ID,
            OBS_RIGHT_ID,
            OBS_CENTER_ID,
            PREGRASP_SERVO_ID,
            RETREAT_ID,
            GRASP_EXECUTE_ID,
            LIFT_OBJECT_ID,
            TRANSPORT_TO_DROPZONE_ID,
            PLACE_OBJECT_ID,
            HOLD_POSITION_ID,
            ABORT_ID,
        ),
        compact_allowed_primitives=(
            OBS_LEFT_ID,
            OBS_RIGHT_ID,
            OBS_CENTER_ID,
            PREGRASP_SERVO_ID,
            RETREAT_ID,
            GRASP_EXECUTE_ID,
            LIFT_OBJECT_ID,
            TRANSPORT_TO_DROPZONE_ID,
            PLACE_OBJECT_ID,
            HOLD_POSITION_ID,
            ABORT_ID,
        ),
    ),
}

TASK_TO_ID = {name: spec.task_id for name, spec in TASK_SPECS.items()}
ID_TO_TASK = {spec.task_id: spec for spec in TASK_SPECS.values()}


def task_spec(task_name: str) -> TaskSpec:
    return TASK_SPECS[task_name]


def _project_primitives(
    primitive_ids: tuple[int, ...],
    primitive_vocabulary: str = PRIMITIVE_VOCAB_LEGACY,
) -> tuple[int, ...]:
    if primitive_vocabulary == PRIMITIVE_VOCAB_LEGACY:
        return tuple(primitive_ids)
    projected: list[int] = []
    for primitive_id in primitive_ids:
        mapped = remap_primitive_id(int(primitive_id), primitive_vocabulary)
        if mapped not in projected:
            projected.append(mapped)
    return tuple(projected)


def task_allowed_primitives(
    task_name_or_id: str | int,
    primitive_vocabulary: str = PRIMITIVE_VOCAB_LEGACY,
) -> tuple[int, ...]:
    if isinstance(task_name_or_id, int):
        spec = ID_TO_TASK[int(task_name_or_id)]
    else:
        spec = TASK_SPECS[task_name_or_id]
    primitive_ids = (
        spec.compact_allowed_primitives
        if primitive_vocabulary == PRIMITIVE_VOCAB_COMPACT and spec.compact_allowed_primitives is not None
        else spec.allowed_primitives
    )
    return _project_primitives(primitive_ids, primitive_vocabulary=primitive_vocabulary)


def task_primary_primitives(
    task_name_or_id: str | int,
    primitive_vocabulary: str = PRIMITIVE_VOCAB_LEGACY,
) -> tuple[int, ...]:
    if isinstance(task_name_or_id, int):
        spec = ID_TO_TASK[int(task_name_or_id)]
    else:
        spec = TASK_SPECS[task_name_or_id]
    primitive_ids = (
        spec.compact_primary_primitives
        if primitive_vocabulary == PRIMITIVE_VOCAB_COMPACT and spec.compact_primary_primitives is not None
        else spec.primary_primitives
    )
    return _project_primitives(primitive_ids, primitive_vocabulary=primitive_vocabulary)


def _family_phrase(spec: TaskSpec) -> str:
    return ", ".join(PRIMITIVE_FAMILY_NAMES[family_id] for family_id in spec.primary_families)


def task_instruction(
    task_name_or_id: str | int,
    primitive_vocabulary: str = PRIMITIVE_VOCAB_LEGACY,
) -> str:
    if isinstance(task_name_or_id, int):
        spec = ID_TO_TASK[int(task_name_or_id)]
    else:
        spec = TASK_SPECS[task_name_or_id]
    preferred_ids = _project_primitives(spec.primary_primitives, primitive_vocabulary=primitive_vocabulary)
    preferred = ", ".join(primitive_name(pid, primitive_vocabulary=primitive_vocabulary) for pid in preferred_ids)
    family_names = _family_phrase(spec)
    if spec.task_id == 0:
        goal = "Finish as soon as the target is visible and centered, then choose hold_position instead of extra motion."
    elif spec.task_id == 1:
        goal = "Finish when the target reaches a stable pre-grasp pose, then choose hold_position instead of extra motion."
    else:
        goal = "Use the blue drop zone as the placement target. Grip the ear handle, then choose hold_position after a successful place."
    return (
        f"Task: {spec.description} "
        f"Primary reusable primitive families: {family_names}. "
        f"Typical executable primitives for this task: {preferred}. "
        f"{goal}"
    )


def task_action_hint(
    task_name_or_id: str | int,
    primitive_vocabulary: str = PRIMITIVE_VOCAB_LEGACY,
) -> str:
    if isinstance(task_name_or_id, int):
        spec = ID_TO_TASK[int(task_name_or_id)]
    else:
        spec = TASK_SPECS[task_name_or_id]
    preferred_ids = _project_primitives(spec.primary_primitives, primitive_vocabulary=primitive_vocabulary)
    preferred = ", ".join(primitive_name(pid, primitive_vocabulary=primitive_vocabulary) for pid in preferred_ids)
    family_names = _family_phrase(spec)
    if spec.task_id == 0:
        suffix = "prefer observation primitives until centered, then confirm and hold"
    elif spec.task_id == 1:
        suffix = "prefer reusable approach primitives to reach pre-grasp, then hold_position"
    else:
        suffix = "prefer observe/approach/grasp/lift/transport/place reuse rather than task-specific detours, then hold_position"
    return f"Primary families: {family_names}. Typical primitives: {preferred}. For this task, {suffix}."


def primitive_instruction(
    primitive_id: int,
    primitive_vocabulary: str = PRIMITIVE_VOCAB_LEGACY,
) -> str:
    return (
        f"{primitive_name(primitive_id, primitive_vocabulary=primitive_vocabulary)}: "
        f"{primitive_description(primitive_id, primitive_vocabulary=primitive_vocabulary)}."
    )


def supervision_stage_id(
    task_id: int,
    primitive_id: int,
    primitive_vocabulary: str = PRIMITIVE_VOCAB_LEGACY,
) -> int:
    del task_id
    if primitive_vocabulary == PRIMITIVE_VOCAB_COMPACT:
        if primitive_id in (0, 1, 2):
            return STAGE_OBSERVE_ID
        if primitive_id == 3:
            return STAGE_CONFIRM_ID
        if primitive_id in (4, 5):
            return STAGE_APPROACH_ID
        if primitive_id == 6:
            return STAGE_GRASP_ID
        if primitive_id == 7:
            return STAGE_LIFT_ID
        if primitive_id == 8:
            return STAGE_TRANSPORT_ID
        if primitive_id == 9:
            return STAGE_PLACE_ID
        return STAGE_TERMINAL_ID
    if primitive_id in (OBS_LEFT_ID, OBS_RIGHT_ID, OBS_CENTER_ID, REOBSERVE_ID):
        return STAGE_OBSERVE_ID
    if primitive_id == VERIFY_TARGET_ID:
        return STAGE_CONFIRM_ID
    if primitive_id in (PREALIGN_GRASP_ID, APPROACH_COARSE_ID, APPROACH_FINE_ID, PREGRASP_SERVO_ID, RETREAT_ID):
        return STAGE_APPROACH_ID
    if primitive_id == GRASP_EXECUTE_ID:
        return STAGE_GRASP_ID
    if primitive_id == LIFT_OBJECT_ID:
        return STAGE_LIFT_ID
    if primitive_id == TRANSPORT_TO_DROPZONE_ID:
        return STAGE_TRANSPORT_ID
    if primitive_id == PLACE_OBJECT_ID:
        return STAGE_PLACE_ID
    return STAGE_TERMINAL_ID

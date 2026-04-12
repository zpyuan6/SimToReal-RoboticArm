from __future__ import annotations

from dataclasses import dataclass

from .skills import LEVEL1_PRIMITIVES, LEVEL2_PRIMITIVES, LEVEL3_PRIMITIVES


@dataclass(frozen=True)
class TaskSpec:
    name: str
    task_id: int
    level: int
    description: str
    allowed_primitives: tuple[int, ...]


TASK_SPECS = {
    "level1_verify": TaskSpec(
        name="level1_verify",
        task_id=0,
        level=1,
        description="Observe and verify that the target is visible and centered.",
        allowed_primitives=LEVEL1_PRIMITIVES,
    ),
    "level2_approach": TaskSpec(
        name="level2_approach",
        task_id=1,
        level=2,
        description="Observe, pre-align, and approach the target to a stable pre-grasp state.",
        allowed_primitives=LEVEL2_PRIMITIVES,
    ),
    "level3_pick_place": TaskSpec(
        name="level3_pick_place",
        task_id=2,
        level=3,
        description="Re-observe, servo into a grasp, lift, transport, and place the object.",
        allowed_primitives=LEVEL3_PRIMITIVES,
    ),
}

TASK_TO_ID = {name: spec.task_id for name, spec in TASK_SPECS.items()}
ID_TO_TASK = {spec.task_id: spec for spec in TASK_SPECS.values()}


def task_spec(task_name: str) -> TaskSpec:
    return TASK_SPECS[task_name]

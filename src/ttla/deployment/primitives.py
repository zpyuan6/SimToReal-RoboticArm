from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..sim.skills import (
    ABORT_ID,
    APPROACH_ID,
    GRASP_EXECUTE_ID,
    LIFT_OBJECT_ID,
    OBS_CENTER_ID,
    OBS_LEFT_ID,
    OBS_RIGHT_ID,
    PLACE_OBJECT_ID,
    PREGRASP_SERVO_ID,
    RETREAT_ID,
    TRANSPORT_TO_DROPZONE_ID,
    primitive_action,
    primitive_name,
)


# Real-robot reference poses derived from the official RoArm-M3 joint semantics.
# These are intentionally separate from the simulator constants because the sim
# poses were tuned for MuJoCo task logic, while the real arm follows the vendor's
# documented zero positions and joint directions.
REAL_HOME_QPOS = np.deg2rad(np.asarray([0.0, 0.0, 90.0, 0.0, 0.0, 170.0], dtype=np.float32))
REAL_OBS_CENTER_QPOS = np.deg2rad(np.asarray([0.0, 12.0, 108.0, 18.0, 0.0, 158.0], dtype=np.float32))
REAL_OBS_LEFT_QPOS = np.deg2rad(np.asarray([30.0, 12.0, 108.0, 18.0, 0.0, 158.0], dtype=np.float32))
REAL_OBS_RIGHT_QPOS = np.deg2rad(np.asarray([-30.0, 12.0, 108.0, 18.0, 0.0, 158.0], dtype=np.float32))
REAL_APPROACH_QPOS = np.deg2rad(np.asarray([0.0, 13.4, 111.7, 29.4, 0.0, 154.0], dtype=np.float32))
REAL_PREGRASP_QPOS = np.deg2rad(np.asarray([0.0, 21.0, 114.0, -1.0, 0.0, 156.0], dtype=np.float32))
REAL_LIFT_QPOS = np.deg2rad(np.asarray([0.0, -8.0, 96.0, -8.0, 0.0, 180.0], dtype=np.float32))
REAL_TRANSPORT_QPOS = np.deg2rad(np.asarray([-20.0, 6.0, 139.0, -12.0, 0.0, 180.0], dtype=np.float32))
REAL_PLACE_RELEASE_QPOS = np.deg2rad(np.asarray([-20.0, 10.0, 148.0, -8.0, 0.0, 180.0], dtype=np.float32))
REAL_GRIPPER_HOME_QPOS = np.deg2rad(np.float32(170.0))
REAL_GRIPPER_OPEN_QPOS = np.deg2rad(np.float32(60.0))
REAL_GRIPPER_CLOSED_QPOS = np.deg2rad(np.float32(180.0))
REAL_GRIPPER_MIN_QPOS = np.deg2rad(np.float32(45.0))
REAL_GRIPPER_MAX_QPOS = np.deg2rad(np.float32(180.0))


@dataclass
class PrimitiveResult:
    success: bool
    done: bool
    timeout: bool
    info: dict


class PrimitiveExecutor:
    """Maps high-level primitive IDs to fixed RoArm joint scripts."""

    def __init__(self, robot_interface, runtime_cfg: dict | None = None) -> None:
        self.robot = robot_interface
        self.runtime_cfg = runtime_cfg or {}
        self.sleep_s = float(self.runtime_cfg.get("primitive_sleep_s", 0.8))
        self.current_q = REAL_HOME_QPOS.copy()

    def run(self, primitive_id: int | str | dict) -> PrimitiveResult:
        primitive_id_value = primitive_action(primitive_id)
        name = primitive_name(primitive_id_value)
        if primitive_id_value in (OBS_LEFT_ID, OBS_RIGHT_ID, OBS_CENTER_ID):
            self._goto(self._observe_pose(primitive_id_value))
            return PrimitiveResult(True, False, False, {"primitive_name": name})
        if primitive_id_value == APPROACH_ID:
            self._goto(REAL_APPROACH_QPOS)
            return PrimitiveResult(True, False, False, {"primitive_name": name})
        if primitive_id_value == RETREAT_ID:
            self._delta(np.asarray([0.0, 0.10, 0.14, -0.06, 0.0, 0.10], dtype=np.float32))
            return PrimitiveResult(True, False, False, {"primitive_name": name})
        if primitive_id_value == PREGRASP_SERVO_ID:
            self._goto(REAL_PREGRASP_QPOS)
            return PrimitiveResult(True, False, False, {"primitive_name": name})
        if primitive_id_value == GRASP_EXECUTE_ID:
            # Grasp execute is now a pure close-gripper primitive. Any final
            # target-relative positioning should already have happened during
            # approach/pregrasp, not here.
            self._set_gripper(REAL_GRIPPER_CLOSED_QPOS)
            return PrimitiveResult(True, False, False, {"primitive_name": name})
        if primitive_id_value == LIFT_OBJECT_ID:
            self._goto(REAL_LIFT_QPOS)
            return PrimitiveResult(True, False, False, {"primitive_name": name})
        if primitive_id_value == TRANSPORT_TO_DROPZONE_ID:
            self._goto(REAL_TRANSPORT_QPOS)
            return PrimitiveResult(True, False, False, {"primitive_name": name})
        if primitive_id_value == PLACE_OBJECT_ID:
            # Explicitly lower into the release pose instead of opening from the
            # hover posture, which previously caused releases to happen too high.
            self._goto(REAL_PLACE_RELEASE_QPOS)
            self._set_gripper(REAL_GRIPPER_OPEN_QPOS)
            return PrimitiveResult(True, False, False, {"primitive_name": name})
        if primitive_id_value == ABORT_ID:
            self._goto(REAL_HOME_QPOS)
            return PrimitiveResult(True, True, False, {"primitive_name": name})
        raise KeyError(primitive_id_value)

    def _observe_pose(self, primitive_id_value: int) -> np.ndarray:
        if primitive_id_value == OBS_LEFT_ID:
            return REAL_OBS_LEFT_QPOS.copy()
        if primitive_id_value == OBS_RIGHT_ID:
            return REAL_OBS_RIGHT_QPOS.copy()
        return REAL_OBS_CENTER_QPOS.copy()

    def _goto(self, q_target: np.ndarray) -> None:
        self.current_q = np.asarray(q_target, dtype=np.float32).copy()
        self.robot.move_joint_vector(self.current_q)
        time.sleep(self.sleep_s)

    def _delta(self, joint_delta: np.ndarray) -> None:
        self._goto(self.current_q + np.asarray(joint_delta, dtype=np.float32))

    def _set_gripper(self, value: float) -> None:
        q_target = self.current_q.copy()
        q_target[5] = np.clip(value, REAL_GRIPPER_MIN_QPOS, REAL_GRIPPER_MAX_QPOS)
        self._goto(q_target)

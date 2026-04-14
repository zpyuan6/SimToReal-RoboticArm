from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..sim.skills import (
    ABORT_ID,
    APPROACH_COARSE_ID,
    APPROACH_FINE_ID,
    CARRY_QPOS,
    DROPZONE_QPOS,
    GRASP_EXECUTE_ID,
    HOME_QPOS,
    HOLD_POSITION_ID,
    LIFT_OBJECT_ID,
    OBS_CENTER_ID,
    OBS_LEFT_ID,
    OBS_RIGHT_ID,
    OBS_CENTER_QPOS,
    PLACE_OBJECT_ID,
    PREALIGN_BASE_QPOS,
    PREALIGN_GRASP_ID,
    PREGRASP_SERVO_ID,
    REOBSERVE_ID,
    RETREAT_ID,
    TRANSPORT_TO_DROPZONE_ID,
    VERIFY_TARGET_ID,
    observe_pose,
    primitive_action,
    primitive_name,
)


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
        self.current_q = HOME_QPOS.copy()

    def run(self, primitive_id: int | str | dict) -> PrimitiveResult:
        primitive_id_value = primitive_action(primitive_id)
        name = primitive_name(primitive_id_value)
        if primitive_id_value in (OBS_LEFT_ID, OBS_RIGHT_ID, OBS_CENTER_ID):
            self._goto(observe_pose(primitive_id_value))
            return PrimitiveResult(True, False, False, {"primitive_name": name})
        if primitive_id_value == VERIFY_TARGET_ID:
            time.sleep(self.sleep_s * 0.5)
            return PrimitiveResult(True, False, False, {"primitive_name": name})
        if primitive_id_value == PREALIGN_GRASP_ID:
            self._goto(PREALIGN_BASE_QPOS)
            return PrimitiveResult(True, False, False, {"primitive_name": name})
        if primitive_id_value == APPROACH_COARSE_ID:
            self._delta(np.asarray([0.0, -0.12, -0.16, 0.08, 0.0, 0.0], dtype=np.float32))
            return PrimitiveResult(True, False, False, {"primitive_name": name})
        if primitive_id_value == APPROACH_FINE_ID:
            self._delta(np.asarray([0.0, -0.05, -0.07, 0.04, 0.0, 0.0], dtype=np.float32))
            return PrimitiveResult(True, False, False, {"primitive_name": name})
        if primitive_id_value == RETREAT_ID:
            self._delta(np.asarray([0.0, 0.10, 0.14, -0.06, 0.0, 0.10], dtype=np.float32))
            return PrimitiveResult(True, False, False, {"primitive_name": name})
        if primitive_id_value == REOBSERVE_ID:
            self._goto(OBS_CENTER_QPOS)
            return PrimitiveResult(True, False, False, {"primitive_name": name})
        if primitive_id_value == PREGRASP_SERVO_ID:
            # On real hardware this remains a short closed-loop primitive placeholder.
            self._goto(PREALIGN_BASE_QPOS)
            self._delta(np.asarray([0.0, -0.04, -0.04, 0.02, 0.0, -0.04], dtype=np.float32))
            return PrimitiveResult(True, False, False, {"primitive_name": name, "mode": "servo_stub"})
        if primitive_id_value == GRASP_EXECUTE_ID:
            self._delta(np.asarray([0.0, -0.08, -0.10, 0.05, 0.0, 0.0], dtype=np.float32))
            self._set_gripper(0.18)
            return PrimitiveResult(True, False, False, {"primitive_name": name})
        if primitive_id_value == LIFT_OBJECT_ID:
            self._goto(CARRY_QPOS)
            return PrimitiveResult(True, False, False, {"primitive_name": name})
        if primitive_id_value == TRANSPORT_TO_DROPZONE_ID:
            self._goto(DROPZONE_QPOS)
            return PrimitiveResult(True, False, False, {"primitive_name": name})
        if primitive_id_value == PLACE_OBJECT_ID:
            self._delta(np.asarray([0.0, 0.08, -0.05, 0.0, 0.0, 0.0], dtype=np.float32))
            self._set_gripper(1.05)
            return PrimitiveResult(True, False, False, {"primitive_name": name})
        if primitive_id_value == HOLD_POSITION_ID:
            self.robot.move_joint_vector(self.current_q)
            time.sleep(self.sleep_s * 0.5)
            return PrimitiveResult(True, False, False, {"primitive_name": name})
        if primitive_id_value == ABORT_ID:
            self._goto(HOME_QPOS)
            return PrimitiveResult(True, True, False, {"primitive_name": name})
        raise KeyError(primitive_id_value)

    def _goto(self, q_target: np.ndarray) -> None:
        self.current_q = np.asarray(q_target, dtype=np.float32).copy()
        self.robot.move_joint_vector(self.current_q)
        time.sleep(self.sleep_s)

    def _delta(self, joint_delta: np.ndarray) -> None:
        self._goto(self.current_q + np.asarray(joint_delta, dtype=np.float32))

    def _set_gripper(self, value: float) -> None:
        q_target = self.current_q.copy()
        q_target[5] = value
        self._goto(q_target)

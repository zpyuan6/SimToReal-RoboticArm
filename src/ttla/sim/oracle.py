from __future__ import annotations

import numpy as np

from .skills import (
    ABORT_ID,
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
)


class PrimitiveOracle:
    """Rule-based oracle that emits discrete primitive labels for sim training."""

    def act(self, env) -> int:
        task_name = env.task_name
        if task_name == "level1_verify":
            return self._act_level1(env)
        if task_name == "level2_approach":
            return self._act_level2(env)
        if task_name == "level3_pick_place":
            return self._act_level3(env)
        raise KeyError(task_name)

    def _observe_action(self, env) -> int:
        yaw_error = env.target_yaw_error()
        if yaw_error > 0.10:
            return OBS_LEFT_ID
        if yaw_error < -0.10:
            return OBS_RIGHT_ID
        return OBS_CENTER_ID

    def _act_level1(self, env) -> int:
        if env.visibility_score() < 0.10:
            return self._observe_action(env)
        if env.center_error_px() > 15.0:
            return self._observe_action(env)
        if env.verified:
            return HOLD_POSITION_ID
        return VERIFY_TARGET_ID

    def _act_level2(self, env) -> int:
        if env.verified and env.approach_success_ready():
            return HOLD_POSITION_ID
        if env.visibility_score() < 0.10:
            return self._observe_action(env)
        if env.center_error_px() > 14.0:
            return PREALIGN_GRASP_ID
        if env.pregrasp_ready():
            if env.visibility_score() > 0.26:
                return VERIFY_TARGET_ID
            return APPROACH_FINE_ID
        dist = env.ee_target_distance()
        if dist > 0.18:
            return APPROACH_COARSE_ID
        if dist > 0.11:
            return APPROACH_FINE_ID
        if env.center_error_px() > 16.0:
            return RETREAT_ID
        return VERIFY_TARGET_ID

    def _act_level3(self, env) -> int:
        if env.placed:
            return HOLD_POSITION_ID
        if env.object_attached:
            if not env.lifted:
                return LIFT_OBJECT_ID
            if env._dropzone_xy_distance() > 0.085:
                return TRANSPORT_TO_DROPZONE_ID
            return PLACE_OBJECT_ID
        if env.visibility_score() < 0.08:
            return REOBSERVE_ID
        if env._ear_grasp_contact_count() > 0:
            return GRASP_EXECUTE_ID
        if env.visibility_score() > 0.20 and env.center_error_px() < 14.0 and env.ee_target_distance() < 0.075:
            return GRASP_EXECUTE_ID
        if not env.pregrasp_ready():
            return PREGRASP_SERVO_ID
        if env.ee_target_distance() > 0.07:
            return PREGRASP_SERVO_ID
        return GRASP_EXECUTE_ID


# Compatibility export for code that still imports ScriptedExpert.
ScriptedExpert = PrimitiveOracle

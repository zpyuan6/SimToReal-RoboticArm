from __future__ import annotations

import numpy as np

from .skills import (
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
)


class PrimitiveOracle:
    """Rule-based oracle that emits discrete primitive labels for sim training."""

    @staticmethod
    def _emit(env, primitive_id: int) -> int:
        return int(primitive_id)

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

    @staticmethod
    def _scaled_px(env, reference_px_at_84: float) -> float:
        return float(reference_px_at_84 * (float(env.cfg["image_size"]) / 84.0))

    def _act_level1(self, env) -> int:
        if env.visibility_score() < 0.08:
            return self._emit(env, self._observe_action(env))
        if env.center_error_px() > self._scaled_px(env, 36.0):
            return self._emit(env, self._observe_action(env))
        return self._emit(env, OBS_CENTER_ID)

    def _act_level2(self, env) -> int:
        if env.visibility_score() < 0.08:
            return self._emit(env, self._observe_action(env))
        if env.center_error_px() > self._scaled_px(env, 18.0):
            return self._emit(env, self._observe_action(env))
        dist = env.ee_target_distance()
        if dist > 0.028 or env.center_error_px() > self._scaled_px(env, 10.0):
            return self._emit(env, APPROACH_ID)
        return self._emit(env, RETREAT_ID)

    def _act_level3(self, env) -> int:
        if env.object_attached:
            if not env.lifted:
                return self._emit(env, LIFT_OBJECT_ID)
            if env._dropzone_xy_distance() > 0.085:
                return self._emit(env, TRANSPORT_TO_DROPZONE_ID)
            return self._emit(env, PLACE_OBJECT_ID)
        if env.visibility_score() < 0.08:
            return self._emit(env, self._observe_action(env))
        if env._ear_grasp_contact_count() > 0:
            return self._emit(env, GRASP_EXECUTE_ID)
        if env.visibility_score() > 0.20 and env.center_error_px() < self._scaled_px(env, 6.0) and env.ee_target_distance() < 0.030:
            return self._emit(env, GRASP_EXECUTE_ID)
        if env.center_error_px() > self._scaled_px(env, 18.0):
            return self._emit(env, self._observe_action(env))
        dist = env.ee_target_distance()
        if not env.approach_success_ready():
            return self._emit(env, APPROACH_ID)
        if dist > 0.035 or env.center_error_px() > self._scaled_px(env, 10.0):
            return self._emit(env, APPROACH_ID)
        if not env.pregrasp_ready():
            return self._emit(env, PREGRASP_SERVO_ID)
        return self._emit(env, GRASP_EXECUTE_ID)


# Compatibility export for code that still imports ScriptedExpert.
ScriptedExpert = PrimitiveOracle

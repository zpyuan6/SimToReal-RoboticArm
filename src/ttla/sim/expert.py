from __future__ import annotations

import numpy as np


class ScriptedExpert:
    """Short-horizon expert tuned for the simplified skill space."""

    def act(self, env) -> int:
        target = env._target_position()
        ee = env._ee_position()
        rel = target - ee
        visibility = env.visibility_score()
        yaw_target = np.arctan2(target[1], max(target[0], 1e-6))
        base = float(env.data.qpos[0])
        wrist_pitch = float(env.data.qpos[3])
        wrist_roll = float(env.data.qpos[4])
        gripper = float(env.data.qpos[5])
        if env.task_name == "verify":
            if visibility < 0.16:
                return 0 if yaw_target > base else 1
            if visibility < 0.24 or wrist_pitch > -0.18:
                return 2
            if abs(wrist_roll) > 0.28:
                return 1 if wrist_roll > 0 else 0
            return 5
        if env.task_name == "observe_then_act":
            if env.step_idx < 2 and visibility < 0.15:
                return 0 if yaw_target > base else 1
            if np.linalg.norm(rel) > 0.11:
                return 4
            if visibility < 0.17 or gripper > 0.55:
                return 2
            return 5
        if abs(yaw_target - base) > 0.08:
            return 0 if yaw_target > base else 1
        if rel[0] > 0.08:
            return 4
        if visibility < 0.12 or wrist_pitch > -0.12:
            return 2
        return 5

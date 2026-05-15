from __future__ import annotations

import mujoco
import numpy as np

from .skills import (
    HOME_QPOS,
    OBS_CENTER_ID,
    OBS_LEFT_ID,
    OBS_RIGHT_ID,
    PREGRASP_QPOS,
    APPROACH_QPOS,
    observe_pose,
)


class ContinuousWaypointExpert:
    """Task-conditioned continuous teacher for source-data generation.

    Unlike the older primitive-imitation expert, this controller is expected to
    generate successful continuous trajectories. It therefore uses validated
    observation poses together with task-conditioned Jacobian servo actions for
    approach, pre-grasp, lift, transport, and place.
    """

    def __init__(
        self,
        max_arm_delta: float = 0.16,
        max_gripper_delta: float = 0.25,
        servo_gain: float = 2.0,
        damping: float = 1e-3,
    ) -> None:
        self.max_arm_delta = float(max_arm_delta)
        self.max_gripper_delta = float(max_gripper_delta)
        self.servo_gain = float(servo_gain)
        self.damping = float(damping)
        self.current_task: str | None = None
        self.phase: str = "observe"
        self.phase_steps: int = 0

    def reset(self, task_name: str | None = None) -> None:
        self.current_task = task_name
        self.phase = "observe"
        self.phase_steps = 0

    @staticmethod
    def _scaled_px(env, reference_px_at_84: float) -> float:
        return float(reference_px_at_84 * (float(env.cfg["image_size"]) / 84.0))

    def _observe_action(self, env) -> np.ndarray:
        yaw_error = env.target_yaw_error()
        if yaw_error > 0.10:
            pose = observe_pose(OBS_LEFT_ID)
        elif yaw_error < -0.10:
            pose = observe_pose(OBS_RIGHT_ID)
        else:
            pose = observe_pose(OBS_CENTER_ID)
        return self._joint_target_delta(env, pose)

    def _joint_target_delta(self, env, q_target: np.ndarray) -> np.ndarray:
        current = env.data.qpos[:6].astype(np.float32).copy()
        delta = q_target.astype(np.float32) - current
        delta[:5] = np.clip(delta[:5], -self.max_arm_delta, self.max_arm_delta)
        delta[5] = np.clip(delta[5], -self.max_gripper_delta, self.max_gripper_delta)
        return delta.astype(np.float32)

    def _position_servo_delta(
        self,
        env,
        target_pos: np.ndarray,
        *,
        gripper_target: float | None = None,
        posture_bias: np.ndarray | None = None,
    ) -> np.ndarray:
        site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        jacp = np.zeros((3, env.model.nv), dtype=np.float64)
        jacr = np.zeros((3, env.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(env.model, env.data, jacp, jacr, site_id)
        j = jacp[:, :5]
        pos_error = np.asarray(target_pos, dtype=np.float64) - env._ee_position()
        task_vel = self.servo_gain * pos_error
        jj_t = j @ j.T
        solve = np.linalg.solve(jj_t + self.damping * np.eye(3, dtype=np.float64), task_vel)
        dq = j.T @ solve
        if posture_bias is not None:
            dq += 0.05 * (np.asarray(posture_bias, dtype=np.float64)[:5] - env.data.qpos[:5].copy())
        delta = np.zeros(6, dtype=np.float32)
        delta[:5] = np.clip(dq.astype(np.float32), -self.max_arm_delta, self.max_arm_delta)
        if gripper_target is None:
            delta[5] = 0.0
        else:
            delta[5] = float(
                np.clip(
                    float(gripper_target) - float(env.data.qpos[5]),
                    -self.max_gripper_delta,
                    self.max_gripper_delta,
                )
            )
        return delta

    def _level1_action(self, env) -> np.ndarray:
        if env.verify_ready():
            return self._joint_target_delta(env, observe_pose(OBS_CENTER_ID))
        return self._observe_action(env)

    def _level2_action(self, env) -> np.ndarray:
        if env.visibility_score() < 0.12 or env.center_error_px() > self._scaled_px(env, 40.0):
            return self._observe_action(env)
        hover = np.asarray([0.0, 0.0, 0.030], dtype=np.float64)
        if env.ee_target_distance() > 0.080:
            target = env._target_grasp_position() + hover
            return self._position_servo_delta(env, target, gripper_target=1.10, posture_bias=APPROACH_QPOS)
        target = env._target_grasp_position() + np.asarray([0.0, 0.0, 0.018], dtype=np.float64)
        return self._position_servo_delta(env, target, gripper_target=1.10, posture_bias=None)

    def _level3_action(self, env) -> np.ndarray:
        if env.object_attached:
            if not env.lifted:
                target = env._ee_position() + np.asarray([0.0, 0.0, 0.060], dtype=np.float64)
                return self._position_servo_delta(env, target, gripper_target=0.0, posture_bias=PREGRASP_QPOS)
            drop_xy = env._dropzone_position()[:2]
            if env._dropzone_xy_distance() > 0.085:
                target = np.asarray([drop_xy[0], drop_xy[1], 0.115], dtype=np.float64)
                return self._position_servo_delta(env, target, gripper_target=0.0, posture_bias=HOME_QPOS)
            target = np.asarray([drop_xy[0], drop_xy[1], 0.060], dtype=np.float64)
            return self._position_servo_delta(env, target, gripper_target=1.10, posture_bias=HOME_QPOS)
        if env.visibility_score() < 0.12 or env.center_error_px() > self._scaled_px(env, 48.0):
            return self._observe_action(env)
        if env.ee_target_distance() > 0.065 or env.center_error_px() > self._scaled_px(env, 18.0):
            target = env._target_grasp_position() + np.asarray([0.0, 0.0, 0.030], dtype=np.float64)
            return self._position_servo_delta(env, target, gripper_target=1.10, posture_bias=APPROACH_QPOS)
        if env._ear_grasp_contact_count() == 0 and env.grasp_gap() > 0.004:
            target = env._target_grasp_position() + np.asarray([0.0, 0.0, 0.010], dtype=np.float64)
            return self._position_servo_delta(env, target, gripper_target=1.10, posture_bias=None)
        delta = np.zeros(6, dtype=np.float32)
        delta[5] = -self.max_gripper_delta
        return delta

    def act(self, env) -> np.ndarray:
        if env.step_idx == 0 or env.task_name != self.current_task:
            self.reset(env.task_name)
        if env.task_name == "level1_verify":
            action = self._level1_action(env)
        elif env.task_name == "level2_approach":
            action = self._level2_action(env)
        elif env.task_name == "level3_pick_place":
            action = self._level3_action(env)
        else:
            action = self._joint_target_delta(env, HOME_QPOS.copy())
        self.phase_steps += 1
        return action.astype(np.float32)

from __future__ import annotations

import mujoco
import numpy as np

from .skills import (
    APPROACH_QPOS,
    HOME_QPOS,
    LIFT_QPOS,
    OBS_CENTER_QPOS,
    OBS_LEFT_QPOS,
    OBS_RIGHT_QPOS,
    PLACE_RELEASE_QPOS,
    PREGRASP_QPOS,
    TRANSPORT_QPOS,
)


CANONICAL_LEVEL12_TARGET_BODY = np.asarray([0.300, 0.000, 0.040], dtype=np.float64)
CANONICAL_LEVEL3_TARGET_BODY = np.asarray([0.285, 0.000, 0.040], dtype=np.float64)
CANONICAL_LEVEL3_DROP_BODY = np.asarray([0.245, -0.115, 0.045], dtype=np.float64)


class ContinuousWaypointExpert:
    """Task-conditioned continuous teacher anchored to validated primitive end states.

    The teacher still emits Jacobian-based continuous actions, but each stage's
    target is derived from the user-validated primitive final joint states in
    ``skills.py``. This keeps the continuous teacher on the same geometric
    targets as the validated scripted primitive flow.
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
        self.phase_steps: int = 0

    def reset(self, task_name: str | None = None) -> None:
        self.current_task = task_name
        self.phase_steps = 0

    @staticmethod
    def _scaled_px(env, reference_px_at_84: float) -> float:
        return float(reference_px_at_84 * (float(env.cfg["image_size"]) / 84.0))

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

    def _canonical_target_body(self, env) -> np.ndarray:
        if env.task_name == "level3_pick_place":
            return CANONICAL_LEVEL3_TARGET_BODY.copy()
        return CANONICAL_LEVEL12_TARGET_BODY.copy()

    def _canonical_target_grasp(self, env) -> np.ndarray:
        target_local = env._target_grasp_position() - env._target_body_position()
        return self._canonical_target_body(env) + target_local

    def _ee_position_for_qpos(self, env, q_target: np.ndarray) -> np.ndarray:
        saved_qpos = env.data.qpos[:6].copy()
        saved_ctrl = env.data.ctrl[:6].copy()
        env.data.qpos[:6] = np.asarray(q_target, dtype=np.float64)
        env.data.ctrl[:6] = np.asarray(q_target, dtype=np.float64)
        mujoco.mj_forward(env.model, env.data)
        ee = env._ee_position().copy()
        env.data.qpos[:6] = saved_qpos
        env.data.ctrl[:6] = saved_ctrl
        mujoco.mj_forward(env.model, env.data)
        return ee

    def _target_relative_anchor(self, env, q_target: np.ndarray) -> np.ndarray:
        anchor_ee = self._ee_position_for_qpos(env, q_target)
        canonical_offset = anchor_ee - self._canonical_target_grasp(env)
        return env._target_grasp_position() + canonical_offset

    def _drop_relative_anchor(self, env, q_target: np.ndarray) -> np.ndarray:
        anchor_ee = self._ee_position_for_qpos(env, q_target)
        canonical_offset = anchor_ee - CANONICAL_LEVEL3_DROP_BODY
        return env._dropzone_position() + canonical_offset

    def _observe_pose(self, env) -> np.ndarray:
        yaw_error = env.target_yaw_error()
        if yaw_error > 0.10:
            return OBS_LEFT_QPOS
        if yaw_error < -0.10:
            return OBS_RIGHT_QPOS
        return OBS_CENTER_QPOS

    def _observe_action(self, env) -> np.ndarray:
        return self._joint_target_delta(env, self._observe_pose(env))

    def _level1_candidate_metrics(self, env, q_target: np.ndarray) -> dict[str, float]:
        saved_qpos = env.data.qpos[:6].copy()
        saved_ctrl = env.data.ctrl[:6].copy()
        try:
            env.data.qpos[:6] = np.asarray(q_target, dtype=np.float64)
            env.data.ctrl[:6] = np.asarray(q_target, dtype=np.float64)
            mujoco.mj_forward(env.model, env.data)
            return {
                "visibility": float(env.visibility_score()),
                "center_error": float(env.center_error_px()),
                "keypoint_ratio": float(env.target_keypoint_visibility_ratio()),
                "components": float(env.target_visible_component_count()),
                "pixels": float(env.target_visible_pixels()),
                "occlusion": float(env.target_occlusion_ratio()),
                "intrusion": float(env.gripper_intrusion_ratio()),
            }
        finally:
            env.data.qpos[:6] = saved_qpos
            env.data.ctrl[:6] = saved_ctrl
            mujoco.mj_forward(env.model, env.data)

    def _level1_pose_variants(self) -> list[np.ndarray]:
        base = OBS_CENTER_QPOS.astype(np.float64).copy()
        variants: list[np.ndarray] = []
        for shoulder_delta, elbow_delta, wrist_delta in (
            (-0.02, +0.08, -0.10),
            (-0.04, +0.12, -0.16),
            (-0.06, +0.18, -0.24),
            (-0.08, +0.22, -0.30),
        ):
            q_target = base.copy()
            # User-validated joint semantics:
            # - shoulder closer to 0 raises the arm
            # - larger elbow tucks the arm in
            # - more negative wrist_pitch adds gripper droop
            q_target[1] = max(0.02, float(q_target[1]) + shoulder_delta)
            q_target[2] = min(2.90, float(q_target[2]) + elbow_delta)
            q_target[3] = max(-0.78, float(q_target[3]) + wrist_delta)
            variants.append(q_target)
        return variants

    def _level1_observe_qpos(self, env) -> np.ndarray:
        centered_yaw = float(
            np.clip(
                0.90 * env.target_yaw_error(),
                float(OBS_RIGHT_QPOS[0]),
                float(OBS_LEFT_QPOS[0]),
            )
        )
        observe_seed = float(self._observe_pose(env)[0])
        yaw_seeds = []
        for seed in (centered_yaw, observe_seed):
            if not any(abs(seed - existing) < 1e-6 for existing in yaw_seeds):
                yaw_seeds.append(seed)
        candidate_offsets = (-0.16, -0.12, -0.08, -0.05, -0.025, 0.0, 0.025, 0.05, 0.08, 0.12, 0.16)
        best_qpos: np.ndarray | None = None
        best_key: tuple[float, ...] | None = None
        for variant_idx, base_pose in enumerate(self._level1_pose_variants()):
            for yaw_seed in yaw_seeds:
                for offset in candidate_offsets:
                    q_target = base_pose.copy()
                    q_target[0] = float(
                        np.clip(
                            yaw_seed + offset,
                            float(OBS_RIGHT_QPOS[0]),
                            float(OBS_LEFT_QPOS[0]),
                        )
                    )
                    metrics = self._level1_candidate_metrics(env, q_target)
                    complete = (
                        metrics["keypoint_ratio"] >= 1.0
                        and metrics["components"] >= 4.0
                        and metrics["pixels"] >= 30.0
                    )
                    low_occlusion = complete and metrics["occlusion"] <= 0.12
                    # Prefer fully visible, low-occlusion candidates. Once that
                    # constraint is satisfied, recenter as much as possible.
                    key = (
                        1.0 if low_occlusion else 0.0,
                        1.0 if complete else 0.0,
                        -metrics["occlusion"],
                        -metrics["center_error"],
                        metrics["pixels"],
                        -abs(offset),
                        -abs(yaw_seed - centered_yaw),
                        float(variant_idx),
                    )
                    if best_key is None or key > best_key:
                        best_key = key
                        best_qpos = q_target
        if best_qpos is None:
            return OBS_CENTER_QPOS.astype(np.float64).copy()
        return best_qpos

    def _level1_action(self, env) -> np.ndarray:
        if env.clear_view_ready():
            return np.zeros(6, dtype=np.float32)
        q_target = self._level1_observe_qpos(env)
        return self._joint_target_delta(env, q_target)

    def _level2_action(self, env) -> np.ndarray:
        if env.visibility_score() < 0.12 or env.center_error_px() > self._scaled_px(env, 40.0):
            return self._observe_action(env)
        target = self._target_relative_anchor(env, PREGRASP_QPOS)
        return self._position_servo_delta(
            env,
            target,
            gripper_target=float(PREGRASP_QPOS[5]),
            posture_bias=PREGRASP_QPOS,
        )

    def _level3_action(self, env) -> np.ndarray:
        if env.object_attached:
            if not env.lifted:
                target = self._target_relative_anchor(env, LIFT_QPOS)
                return self._position_servo_delta(
                    env,
                    target,
                    gripper_target=float(LIFT_QPOS[5]),
                    posture_bias=LIFT_QPOS,
                )
            if env._target_dropzone_xy_distance() > 0.030 or env._target_body_position()[2] > 0.070:
                target = self._drop_relative_anchor(env, TRANSPORT_QPOS)
                return self._position_servo_delta(
                    env,
                    target,
                    gripper_target=float(TRANSPORT_QPOS[5]),
                    posture_bias=TRANSPORT_QPOS,
                )
            target = self._drop_relative_anchor(env, PLACE_RELEASE_QPOS)
            if env._target_body_position()[2] > 0.070:
                return self._position_servo_delta(
                    env,
                    target,
                    gripper_target=float(PLACE_RELEASE_QPOS[5]),
                    posture_bias=PLACE_RELEASE_QPOS,
                )
            delta = np.zeros(6, dtype=np.float32)
            delta[5] = self.max_gripper_delta
            return delta

        if env.visibility_score() < 0.12 or env.center_error_px() > self._scaled_px(env, 48.0):
            return self._observe_action(env)
        if env.ee_target_distance() > 0.075 or env.center_error_px() > self._scaled_px(env, 16.0):
            target = self._target_relative_anchor(env, APPROACH_QPOS)
            return self._position_servo_delta(
                env,
                target,
                gripper_target=float(APPROACH_QPOS[5]),
                posture_bias=APPROACH_QPOS,
            )
        if env._ear_grasp_contact_count() == 0 and env.grasp_gap() > 0.004:
            target = self._target_relative_anchor(env, PREGRASP_QPOS)
            return self._position_servo_delta(
                env,
                target,
                gripper_target=float(PREGRASP_QPOS[5]),
                posture_bias=PREGRASP_QPOS,
            )
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

from __future__ import annotations

import mujoco
import numpy as np

from .skills import (
    APPROACH_QPOS,
    HOME_QPOS,
    L1_FIXED_OBSERVE_QPOS,
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
        self.visual_lock_count: int = 0
        self.search_pose_index: int = 0
        self.search_pose_steps: int = 0
        self.level1_scan_index: int = 0
        self.level1_scan_steps: int = 0
        self.level1_tracking_active: bool = False
        self.level1_center_hold_steps: int = 0
        self.level1_last_visible_dx: float = 0.0
        self.level2_approach_active: bool = False
        self.level2_cached_approach_qpos: np.ndarray | None = None
        self.level2_cached_pregrasp_qpos: np.ndarray | None = None
        self.level2_bridge_start_qpos: np.ndarray | None = None
        self.level2_bridge_steps_remaining: int = 0
        self.level2_bridge_total_steps: int = 3
        self.level3_cached_lift_qpos: np.ndarray | None = None
        self.level3_cached_transport_qpos: np.ndarray | None = None
        self.level3_cached_place_qpos: np.ndarray | None = None
        self.level3_place_active: bool = False
        self.search_poses = (
            OBS_LEFT_QPOS.astype(np.float64).copy(),
            OBS_CENTER_QPOS.astype(np.float64).copy(),
            OBS_RIGHT_QPOS.astype(np.float64).copy(),
            OBS_CENTER_QPOS.astype(np.float64).copy(),
        )

    def reset(self, task_name: str | None = None) -> None:
        self.current_task = task_name
        self.phase_steps = 0
        self.visual_lock_count = 0
        self.search_pose_index = 0
        self.search_pose_steps = 0
        self.level1_scan_index = 0
        self.level1_scan_steps = 0
        self.level1_tracking_active = False
        self.level1_center_hold_steps = 0
        self.level1_last_visible_dx = 0.0
        self.level2_approach_active = False
        self.level2_cached_approach_qpos = None
        self.level2_cached_pregrasp_qpos = None
        self.level2_bridge_start_qpos = None
        self.level2_bridge_steps_remaining = 0
        self.level3_cached_lift_qpos = None
        self.level3_cached_transport_qpos = None
        self.level3_cached_place_qpos = None
        self.level3_place_active = False

    @staticmethod
    def _scaled_px(env, reference_px_at_84: float) -> float:
        return float(reference_px_at_84 * (float(env.cfg["image_size"]) / 84.0))

    def _joint_target_delta(self, env, q_target: np.ndarray) -> np.ndarray:
        current = env.data.qpos[:6].astype(np.float32).copy()
        delta = q_target.astype(np.float32) - current
        delta[:5] = np.clip(delta[:5], -self.max_arm_delta, self.max_arm_delta)
        delta[5] = np.clip(delta[5], -self.max_gripper_delta, self.max_gripper_delta)
        return delta.astype(np.float32)

    def _control_action_from_delta(self, env, delta: np.ndarray) -> np.ndarray:
        delta = np.asarray(delta, dtype=np.float32).reshape(6)
        if getattr(env, "control_mode", "joint_delta") == "joint_target":
            target = env.data.qpos[:6].astype(np.float32).copy() + delta
            target = np.clip(target, env.target_low, env.target_high)
            return target.astype(np.float32)
        return delta.astype(np.float32)

    def _joint_target_action(self, env, q_target: np.ndarray) -> np.ndarray:
        q_target = np.asarray(q_target, dtype=np.float32).reshape(6)
        if getattr(env, "control_mode", "joint_delta") == "joint_target":
            return np.clip(q_target, env.target_low, env.target_high).astype(np.float32)
        return self._control_action_from_delta(env, self._joint_target_delta(env, q_target))

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

    def _position_servo_action(
        self,
        env,
        target_pos: np.ndarray,
        *,
        gripper_target: float | None = None,
        posture_bias: np.ndarray | None = None,
    ) -> np.ndarray:
        return self._control_action_from_delta(
            env,
            self._position_servo_delta(
                env,
                target_pos,
                gripper_target=gripper_target,
                posture_bias=posture_bias,
            ),
        )

    def _canonical_target_body(self, env) -> np.ndarray:
        if env.task_name == "level3_pick_place":
            return CANONICAL_LEVEL3_TARGET_BODY.copy()
        return CANONICAL_LEVEL12_TARGET_BODY.copy()

    def _canonical_target_grasp(self, env) -> np.ndarray:
        target_local = env._target_grasp_position() - env._target_body_position()
        return self._canonical_target_body(env) + target_local

    def _canonical_target_ear_center(self, env) -> np.ndarray:
        target_local = env._target_ear_center_position() - env._target_body_position()
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

    def _gripper_center_position_for_qpos(self, env, q_target: np.ndarray) -> np.ndarray:
        saved_qpos = env.data.qpos[:6].copy()
        saved_ctrl = env.data.ctrl[:6].copy()
        env.data.qpos[:6] = np.asarray(q_target, dtype=np.float64)
        env.data.ctrl[:6] = np.asarray(q_target, dtype=np.float64)
        mujoco.mj_forward(env.model, env.data)
        center = env._gripper_center_position().copy()
        env.data.qpos[:6] = saved_qpos
        env.data.ctrl[:6] = saved_ctrl
        mujoco.mj_forward(env.model, env.data)
        return center

    def _ee_rotation_for_qpos(self, env, q_target: np.ndarray) -> np.ndarray:
        saved_qpos = env.data.qpos[:6].copy()
        saved_ctrl = env.data.ctrl[:6].copy()
        site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        env.data.qpos[:6] = np.asarray(q_target, dtype=np.float64)
        env.data.ctrl[:6] = np.asarray(q_target, dtype=np.float64)
        mujoco.mj_forward(env.model, env.data)
        rot = env.data.site_xmat[site_id].reshape(3, 3).copy()
        env.data.qpos[:6] = saved_qpos
        env.data.ctrl[:6] = saved_ctrl
        mujoco.mj_forward(env.model, env.data)
        return rot

    def _finger_height_gap_for_qpos(self, env, q_target: np.ndarray) -> float:
        saved_qpos = env.data.qpos[:6].copy()
        saved_ctrl = env.data.ctrl[:6].copy()
        upper_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "gripper_upper_finger_geom")
        lower_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "gripper_lower_finger_geom")
        env.data.qpos[:6] = np.asarray(q_target, dtype=np.float64)
        env.data.ctrl[:6] = np.asarray(q_target, dtype=np.float64)
        mujoco.mj_forward(env.model, env.data)
        upper = env.data.geom_xpos[upper_id].copy()
        lower = env.data.geom_xpos[lower_id].copy()
        env.data.qpos[:6] = saved_qpos
        env.data.ctrl[:6] = saved_ctrl
        mujoco.mj_forward(env.model, env.data)
        return float(abs(upper[2] - lower[2]))

    def _solve_level2_horizontal_wrist_pitch(self, env, seed_qpos: np.ndarray) -> float:
        seed_q3 = float(seed_qpos[3])
        # User-validated geometry: more negative q3 raises the gripper and keeps
        # the fingers closer to table-horizontal. Search broadly in the
        # negative direction, but only allow a small positive drift from the
        # primitive seed so we preserve the original approach family.
        low = max(float(env.target_low[3]), -1.15)
        high = min(float(env.target_high[3]), seed_q3 + 0.05)
        best_q3 = float(seed_qpos[3])
        best_score = float("inf")
        for wrist_pitch in np.linspace(low, high, num=161, dtype=np.float64):
            q_target = np.asarray(seed_qpos, dtype=np.float64).copy()
            q_target[3] = float(wrist_pitch)
            finger_height_gap = self._finger_height_gap_for_qpos(env, q_target)
            score = finger_height_gap + 0.01 * abs(float(wrist_pitch) - seed_q3)
            if score < best_score:
                best_score = score
                best_q3 = float(wrist_pitch)
        return best_q3

    def _level2_horizontalized_pose(self, env, seed_qpos: np.ndarray) -> np.ndarray:
        pose = np.asarray(seed_qpos, dtype=np.float64).copy()
        pose[3] = -0.25
        pose[4] = 0.0
        return pose

    def _target_relative_anchor(self, env, q_target: np.ndarray) -> np.ndarray:
        anchor_ee = self._ee_position_for_qpos(env, q_target)
        canonical_offset = anchor_ee - self._canonical_target_grasp(env)
        return env._target_grasp_position() + canonical_offset

    def _target_relative_anchor_to_ear_center(self, env, q_target: np.ndarray) -> np.ndarray:
        anchor_ee = self._ee_position_for_qpos(env, q_target)
        canonical_offset = anchor_ee - self._canonical_target_ear_center(env)
        return env._target_ear_center_position() + canonical_offset

    def _ee_ear_center_distance_for_qpos(self, env, q_target: np.ndarray) -> float:
        center = self._gripper_center_position_for_qpos(env, q_target)
        return float(np.linalg.norm(center - env._target_ear_center_position()))

    def _drop_relative_anchor(self, env, q_target: np.ndarray) -> np.ndarray:
        anchor_ee = self._ee_position_for_qpos(env, q_target)
        canonical_offset = anchor_ee - CANONICAL_LEVEL3_DROP_BODY
        return env._dropzone_position() + canonical_offset

    def _level3_drop_yaw(self, env) -> float:
        drop = env._dropzone_position()
        yaw = float(np.arctan2(drop[1], max(drop[0], 1e-6)))
        return float(np.clip(yaw, float(env.target_low[0]), float(env.target_high[0])))

    def _level3_lift_pose(self, env) -> np.ndarray:
        if self.level3_cached_lift_qpos is not None:
            return self.level3_cached_lift_qpos

        q_current = env.data.qpos[:6].astype(np.float64).copy()
        ee_current = env._ee_position().copy()
        target_z = min(float(ee_current[2] + 0.050), 0.24)
        q0 = float(q_current[0])
        q3 = float(q_current[3])
        q4 = float(q_current[4])
        q5 = 0.0
        q1_low = float(env.target_low[1])
        q1_high = float(env.target_high[1])
        q2_low = float(env.target_low[2])
        q2_high = float(env.target_high[2])

        best_q = q_current.copy()
        best_score = float("inf")
        for q1 in np.linspace(q1_low, q1_high, num=21, dtype=np.float64):
            for q2 in np.linspace(q2_low, q2_high, num=25, dtype=np.float64):
                cand = q_current.copy()
                cand[0] = q0
                cand[1] = float(q1)
                cand[2] = float(q2)
                cand[3] = q3
                cand[4] = q4
                cand[5] = q5
                ee = self._ee_position_for_qpos(env, cand)
                dz_shortfall = max(target_z - float(ee[2]), 0.0)
                dz_overshoot = max(float(ee[2]) - target_z, 0.0)
                xy_drift = float(np.linalg.norm(ee[:2] - ee_current[:2]))
                score = 6.0 * dz_shortfall + 1.2 * dz_overshoot + 1.5 * xy_drift
                score += 0.10 * abs(float(q1) - float(q_current[1]))
                score += 0.06 * abs(float(q2) - float(q_current[2]))
                if score < best_score:
                    best_score = float(score)
                    best_q = cand

        self.level3_cached_lift_qpos = best_q
        return best_q

    def _level3_transport_pose(self, env) -> np.ndarray:
        current_q = env.data.qpos[:6].astype(np.float64).copy()
        q_target = self._solve_level3_drop_stage_qpos(
            env,
            body_z_offset=0.060,
            q1_seed=float(current_q[1]),
            q2_seed=float(current_q[2]),
            q3_seed=float(current_q[3]),
            anchor_qpos=current_q,
            q0_window=0.12,
            q1_window=0.24,
            q2_window=0.32,
            q3_window=0.14,
            seed_penalty_scale=0.35,
            anchor_penalty_scale=0.15,
        )
        self.level3_cached_transport_qpos = q_target
        return q_target

    def _level3_place_pose(self, env) -> np.ndarray:
        current_q = env.data.qpos[:6].astype(np.float64).copy()
        q_target = self._solve_level3_drop_stage_qpos(
            env,
            body_z_offset=0.020,
            q1_seed=float(current_q[1]),
            q2_seed=float(current_q[2]),
            q3_seed=float(current_q[3]),
            anchor_qpos=current_q,
            q0_window=0.06,
            q1_window=0.16,
            q2_window=0.20,
            q3_window=0.10,
            seed_penalty_scale=0.20,
            anchor_penalty_scale=0.10,
        )
        self.level3_cached_place_qpos = q_target
        return q_target

    def _level3_close_pose(self, env) -> np.ndarray:
        if self.level2_cached_pregrasp_qpos is None:
            self.level2_cached_pregrasp_qpos = self._solve_level2_target_qpos(env, PREGRASP_QPOS, near=True)
        q_target = self.level2_cached_pregrasp_qpos.astype(np.float64).copy()
        q_target[5] = float(env.target_low[5])
        return q_target

    def _level3_gripper_fully_closed(self, env) -> bool:
        return bool(float(env.data.qpos[5]) <= float(env.target_low[5]) + 0.01)

    def _solve_level3_drop_stage_qpos(
        self,
        env,
        *,
        body_z_offset: float,
        q1_seed: float,
        q2_seed: float,
        q3_seed: float,
        anchor_qpos: np.ndarray | None = None,
        q0_window: float = 0.10,
        q1_window: float = 0.30,
        q2_window: float = 0.45,
        q3_window: float = 0.16,
        seed_penalty_scale: float = 1.0,
        anchor_penalty_scale: float = 1.0,
    ) -> np.ndarray:
        q_current = env.data.qpos[:6].astype(np.float64).copy()
        drop = env._dropzone_position().copy()
        body_target = np.asarray([drop[0], drop[1], drop[2] + body_z_offset], dtype=np.float64)
        target = body_target - env.active_grasp_local_offset
        anchor = None if anchor_qpos is None else np.asarray(anchor_qpos, dtype=np.float64).copy()
        q0_seed = float(anchor[0]) if anchor is not None else self._level3_drop_yaw(env)
        q4 = 0.0
        q5 = 0.0
        q1_low = float(env.target_low[1])
        q1_high = float(env.target_high[1])
        q2_low = float(env.target_low[2])
        q2_high = float(env.target_high[2])
        q3_low = float(env.target_low[3])
        q3_high = float(env.target_high[3])

        def stage_seed(q0_value: float) -> np.ndarray:
            q = (anchor.copy() if anchor is not None else q_current.copy())
            q[0] = float(np.clip(q0_value, float(env.target_low[0]), float(env.target_high[0])))
            q[1] = float(np.clip(q1_seed, q1_low, q1_high))
            q[2] = float(np.clip(q2_seed, q2_low, q2_high))
            q[3] = float(np.clip(q3_seed, q3_low, q3_high))
            q[4] = q4
            q[5] = q5
            return q

        def clamp_pose(q_target: np.ndarray) -> np.ndarray:
            q_target = np.asarray(q_target, dtype=np.float64).copy()
            q_target[0] = float(np.clip(q_target[0], float(env.target_low[0]), float(env.target_high[0])))
            q_target[1] = float(np.clip(q_target[1], q1_low, q1_high))
            q_target[2] = float(np.clip(q_target[2], q2_low, q2_high))
            q_target[3] = float(np.clip(q_target[3], q3_low, q3_high))
            q_target[4] = q4
            q_target[5] = q5
            return q_target

        def pose_score(q_target: np.ndarray) -> float:
            ee = self._ee_position_for_qpos(env, q_target)
            err = ee - target
            score = 2.5 * float(np.linalg.norm(err[:2])) + 2.0 * abs(float(err[2]))
            score += seed_penalty_scale * 0.03 * abs(float(q_target[0]) - q0_seed)
            score += seed_penalty_scale * 0.02 * abs(float(q_target[1]) - float(q1_seed))
            score += seed_penalty_scale * 0.02 * abs(float(q_target[2]) - float(q2_seed))
            score += seed_penalty_scale * 0.02 * abs(float(q_target[3]) - float(q3_seed))
            if anchor is not None:
                score += anchor_penalty_scale * 0.10 * abs(float(q_target[0]) - float(anchor[0]))
                score += anchor_penalty_scale * 0.06 * abs(float(q_target[1]) - float(anchor[1]))
                score += anchor_penalty_scale * 0.05 * abs(float(q_target[2]) - float(anchor[2]))
                score += anchor_penalty_scale * 0.05 * abs(float(q_target[3]) - float(anchor[3]))
            return score

        seed_candidates = [
            stage_seed(q0_seed),
            stage_seed(float((anchor[0] if anchor is not None else q_current[0]))),
            stage_seed(
                float(
                    np.clip(
                        0.5 * (q0_seed + float(anchor[0] if anchor is not None else q_current[0])),
                        float(env.target_low[0]),
                        float(env.target_high[0]),
                    )
                )
            ),
        ]
        q = min(seed_candidates, key=pose_score).copy()

        coarse_best = q.copy()
        coarse_best_score = pose_score(q)
        for q0 in np.linspace(max(float(env.target_low[0]), q0_seed - q0_window), min(float(env.target_high[0]), q0_seed + q0_window), num=7, dtype=np.float64):
            for q1 in np.linspace(max(q1_low, q1_seed - q1_window), min(q1_high, q1_seed + q1_window), num=9, dtype=np.float64):
                for q2 in np.linspace(max(q2_low, q2_seed - q2_window), min(q2_high, q2_seed + q2_window), num=11, dtype=np.float64):
                    for q3 in np.linspace(max(q3_low, q3_seed - q3_window), min(q3_high, q3_seed + q3_window), num=7, dtype=np.float64):
                        cand = q.copy()
                        cand[0] = float(q0)
                        cand[1] = float(q1)
                        cand[2] = float(q2)
                        cand[3] = float(q3)
                        score = pose_score(cand)
                        if score < coarse_best_score:
                            coarse_best_score = score
                            coarse_best = cand
        q = clamp_pose(coarse_best)

        for _ in range(18):
            ee = self._ee_position_for_qpos(env, q)
            err = target - ee
            if float(np.linalg.norm(err)) < 0.006:
                break

            eps = np.asarray([0.004, 0.006, 0.008, 0.010], dtype=np.float64)
            jac = np.zeros((3, 4), dtype=np.float64)
            for col, (joint_idx, step) in enumerate(zip((0, 1, 2, 3), eps, strict=False)):
                q_plus = clamp_pose(q.copy())
                q_minus = clamp_pose(q.copy())
                q_plus[joint_idx] += float(step)
                q_minus[joint_idx] -= float(step)
                q_plus = clamp_pose(q_plus)
                q_minus = clamp_pose(q_minus)
                p_plus = self._ee_position_for_qpos(env, q_plus)
                p_minus = self._ee_position_for_qpos(env, q_minus)
                jac[:, col] = (p_plus - p_minus) / (2.0 * float(step))

            dq = np.linalg.pinv(jac, rcond=1e-4) @ err
            dq = np.asarray(dq, dtype=np.float64)
            dq *= 0.80
            dq[0] = float(np.clip(dq[0], -min(0.06, q0_window), min(0.06, q0_window)))
            dq[1] = float(np.clip(dq[1], -min(0.12, q1_window), min(0.12, q1_window)))
            dq[2] = float(np.clip(dq[2], -min(0.16, q2_window), min(0.16, q2_window)))
            dq[3] = float(np.clip(dq[3], -min(0.08, q3_window), min(0.08, q3_window)))

            improved = False
            best_local_q = q
            best_local_score = pose_score(q)
            for scale in (1.0, 0.5, 0.25):
                cand = clamp_pose(q.copy())
                cand[:4] += dq * scale
                cand = clamp_pose(cand)
                score = pose_score(cand)
                if score + 1e-9 < best_local_score:
                    best_local_q = cand
                    best_local_score = score
                    improved = True
            q = best_local_q
            if not improved:
                break
        return clamp_pose(q)

    def _visual_contact_signal(self, env) -> bool:
        return (
            env.visibility_score() >= 0.10
            and env.target_keypoint_visibility_ratio() >= 0.8
            and env.target_visible_pixels() >= 24
        )

    def _visual_lock_signal(self, env) -> bool:
        return (
            env.visibility_score() >= 0.14
            and env.target_keypoint_visibility_ratio() >= 1.0
            and env.target_visible_component_count() >= 4
            and env.target_visible_pixels() >= 30
            and env.target_occlusion_ratio() <= 0.18
            and env.center_error_px() <= self._scaled_px(env, 20.0)
        )

    def _update_visual_lock(self, env) -> None:
        if self._visual_lock_signal(env):
            self.visual_lock_count = min(self.visual_lock_count + 1, 6)
        elif self._visual_contact_signal(env):
            self.visual_lock_count = max(self.visual_lock_count - 1, 0)
        else:
            self.visual_lock_count = 0

    def _has_visual_lock(self, env, *, required_steps: int = 2) -> bool:
        self._update_visual_lock(env)
        return self.visual_lock_count >= required_steps

    def _search_pose(self, env) -> np.ndarray:
        pose = self.search_poses[self.search_pose_index].copy()
        arm_error = float(np.linalg.norm(env.data.qpos[:5] - pose[:5]))
        self.search_pose_steps += 1
        if arm_error < 0.10 or self.search_pose_steps >= 3:
            self.search_pose_index = (self.search_pose_index + 1) % len(self.search_poses)
            self.search_pose_steps = 0
        return pose

    def _observe_pose(self, env) -> np.ndarray:
        return self._search_pose(env)

    def _observe_action(self, env) -> np.ndarray:
        return self._joint_target_action(env, self._observe_pose(env))

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

    def _target_offset_for_qpos(self, env, q_target: np.ndarray) -> tuple[bool, float, float]:
        saved_qpos = env.data.qpos[:6].copy()
        saved_ctrl = env.data.ctrl[:6].copy()
        try:
            env.data.qpos[:6] = np.asarray(q_target, dtype=np.float64)
            env.data.ctrl[:6] = np.asarray(q_target, dtype=np.float64)
            mujoco.mj_forward(env.model, env.data)
            return env.target_image_offset_px()
        finally:
            env.data.qpos[:6] = saved_qpos
            env.data.ctrl[:6] = saved_ctrl
            mujoco.mj_forward(env.model, env.data)

    def _solve_level1_yaw(self, env, pose: np.ndarray) -> float:
        target = env._target_grasp_position()
        yaw_geom = float(np.arctan2(target[1], max(target[0], 1e-6)))
        return float(np.clip(yaw_geom, float(env.target_low[0]), float(env.target_high[0])))

    def _solve_level1_elbow(self, env, pose: np.ndarray, yaw: float) -> float:
        elbow_low = float(env.target_low[2])
        elbow_high = float(env.target_high[2])
        desired_upper_third_dy = -float(env.cfg["image_size"]) / 6.0
        best_elbow = float(pose[2])
        best_score = float("inf")
        for elbow in np.linspace(elbow_low, elbow_high, num=121, dtype=np.float64):
            q_target = pose.copy()
            q_target[0] = float(yaw)
            q_target[2] = float(elbow)
            visible, dx, dy = self._target_offset_for_qpos(env, q_target)
            if not visible:
                continue
            vertical_error = abs(float(dy) - desired_upper_third_dy)
            score = vertical_error + 0.02 * abs(float(dx))
            if score < best_score:
                best_score = score
                best_elbow = float(elbow)
        return best_elbow

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

    def _level1_scan_base_pose(self) -> np.ndarray:
        # User-specified fixed L1 observation pose. During L1 we steer q0 for
        # horizontal yaw and q2 for vertical framing via forearm depth.
        pose = L1_FIXED_OBSERVE_QPOS.astype(np.float64).copy()
        return pose

    def _level1_scan_qpos(self, env) -> np.ndarray:
        pose = self._level1_scan_base_pose()
        sweep = np.linspace(float(OBS_LEFT_QPOS[0]), float(OBS_RIGHT_QPOS[0]), num=9, dtype=np.float64)
        pose[0] = float(sweep[self.level1_scan_index])
        arm_error = float(np.linalg.norm(env.data.qpos[:5] - pose[:5]))
        self.level1_scan_steps += 1
        if arm_error < 0.10 or self.level1_scan_steps >= 2:
            self.level1_scan_index = (self.level1_scan_index + 1) % len(sweep)
            self.level1_scan_steps = 0
        return pose

    def _level1_at_scan_pose(self, env) -> bool:
        pose = self._level1_scan_base_pose()
        arm_error = float(np.linalg.norm(env.data.qpos[:5] - pose[:5]))
        return arm_error < 0.16

    def _level1_search_direction(self, env) -> int:
        visible, dx, _ = env.target_image_offset_px()
        if visible:
            self.level1_last_visible_dx = float(dx)
            return -1 if dx < 0.0 else 1
        if abs(self.level1_last_visible_dx) > 1e-6:
            return -1 if self.level1_last_visible_dx < 0.0 else 1
        return -1 if self.level1_scan_index < 4 else 1

    def _level1_center_qpos(self, env) -> np.ndarray:
        pose = self._level1_scan_base_pose()
        visible, dx, _ = env.target_image_offset_px()
        if visible:
            yaw = self._solve_level1_yaw(env, pose)
            elbow = self._solve_level1_elbow(env, pose, yaw)
            pose[0] = float(np.clip(yaw, float(env.target_low[0]), float(env.target_high[0])))
            pose[2] = float(np.clip(elbow, float(env.target_low[2]), float(env.target_high[2])))
            self.level1_last_visible_dx = float(dx)
        else:
            pose[0] = float(env.data.qpos[0])
        return pose

    def _level1_observe_qpos(self, env) -> np.ndarray:
        if self._visual_contact_signal(env):
            return self._level1_center_qpos(env)
        return self._level1_scan_qpos(env)

    def _level1_stage_complete(self, env) -> bool:
        return bool(env.level1_observation_pose_reached and env.clear_view_ready())

    def _level2_phase_gate_ready(self, env) -> bool:
        return bool(
            env.level1_observation_pose_reached
            and env.visibility_score() > 0.14
            and env.target_completeness_ready()
            and abs(env.target_yaw_error()) <= np.deg2rad(2.5)
        )

    def _level3_close_gate_ready(self, env) -> bool:
        return bool(
            env.visibility_score() > 0.10
            and env.target_completeness_ready()
            and env.ee_ear_center_distance() < 0.020
        )

    def _solve_level2_yaw(self, env, pose: np.ndarray) -> float:
        target = env._target_ear_center_position()
        yaw = float(np.arctan2(target[1], max(target[0], 1e-6)))
        return float(np.clip(yaw, float(env.target_low[0]), float(env.target_high[0])))

    def _score_level2_q12(
        self,
        env,
        q_target: np.ndarray,
        *,
        desired_dist: float,
        visible_bonus: float,
        exact_center: bool,
    ) -> float:
        _ = visible_bonus
        dist = self._ee_ear_center_distance_for_qpos(env, q_target)
        if exact_center:
            return dist
        return abs(dist - desired_dist)

    def _solve_level2_target_qpos(self, env, seed_qpos: np.ndarray, *, near: bool) -> np.ndarray:
        q = env.data.qpos[:6].astype(np.float64).copy()
        q3_center = -0.25
        q[3] = q3_center
        q[4] = 0.0
        q[5] = float(seed_qpos[5])
        desired_dist = 0.0 if near else 0.050
        q0_anchor = float(q[0])
        q0_target_geom = self._solve_level2_yaw(env, q)
        q0_span = 0.06 if near else 0.07
        q0_low = max(float(env.target_low[0]), q0_anchor - q0_span)
        q0_high = min(float(env.target_high[0]), q0_anchor + q0_span)
        q[0] = float(np.clip(q0_target_geom, q0_low, q0_high))
        q[1] = float(np.clip(float(seed_qpos[1]), float(env.target_low[1]), float(env.target_high[1])))
        q[2] = float(np.clip(float(seed_qpos[2]), float(env.target_low[2]), float(env.target_high[2])))
        q1_low = float(env.target_low[1])
        q1_high = float(env.target_high[1])
        q2_low = float(env.target_low[2])
        q2_high = float(env.target_high[2])
        q3_low = max(float(env.target_low[3]), -0.40 if near else -0.32)
        q3_high = min(float(env.target_high[3]), -0.10 if near else -0.18)
        ear = env._target_ear_center_position().copy()

        def target_point(center: np.ndarray) -> np.ndarray:
            if near:
                return ear
            delta = center - ear
            norm = float(np.linalg.norm(delta))
            if norm <= 1e-8:
                return ear + np.asarray([desired_dist, 0.0, 0.0], dtype=np.float64)
            return ear + delta * (desired_dist / norm)

        def clamp_pose(q_target: np.ndarray) -> np.ndarray:
            q_target = np.asarray(q_target, dtype=np.float64).copy()
            q_target[0] = float(np.clip(q_target[0], q0_low, q0_high))
            q_target[1] = float(np.clip(q_target[1], q1_low, q1_high))
            q_target[2] = float(np.clip(q_target[2], q2_low, q2_high))
            q_target[3] = float(np.clip(q_target[3], q3_low, q3_high))
            q_target[4] = 0.0
            q_target[5] = float(seed_qpos[5])
            return q_target

        if near:
            coarse_best = q.copy()
            coarse_best_score = float("inf")
            for q0 in np.linspace(q0_low, q0_high, num=7, dtype=np.float64):
                for q1 in np.linspace(q1_low, q1_high, num=15, dtype=np.float64):
                    for q2 in np.linspace(q2_low, q2_high, num=21, dtype=np.float64):
                        for q3 in np.linspace(q3_low, q3_high, num=7, dtype=np.float64):
                            cand = q.copy()
                            cand[0] = float(q0)
                            cand[1] = float(q1)
                            cand[2] = float(q2)
                            cand[3] = float(q3)
                            score = self._ee_ear_center_distance_for_qpos(env, cand)
                            score += 0.006 * abs(float(q0) - q0_anchor)
                            score += 0.010 * abs(float(q3) - q3_center)
                            if score < coarse_best_score:
                                coarse_best_score = float(score)
                                coarse_best = cand
            q = clamp_pose(coarse_best)

        for _ in range(18 if near else 12):
            center = self._gripper_center_position_for_qpos(env, q)
            err = target_point(center) - center
            dist = float(np.linalg.norm(center - ear))
            if near and dist < 0.008:
                break
            if (not near) and abs(dist - desired_dist) < 0.006:
                break

            eps = np.asarray([0.004, 0.006, 0.008, 0.010], dtype=np.float64)
            jac = np.zeros((3, 4), dtype=np.float64)
            for col, (joint_idx, step) in enumerate(zip((0, 1, 2, 3), eps, strict=False)):
                q_plus = clamp_pose(q.copy())
                q_minus = clamp_pose(q.copy())
                q_plus[joint_idx] += float(step)
                q_minus[joint_idx] -= float(step)
                q_plus = clamp_pose(q_plus)
                q_minus = clamp_pose(q_minus)
                p_plus = self._gripper_center_position_for_qpos(env, q_plus)
                p_minus = self._gripper_center_position_for_qpos(env, q_minus)
                jac[:, col] = (p_plus - p_minus) / (2.0 * float(step))

            dq = np.linalg.pinv(jac, rcond=1e-4) @ err
            dq = np.asarray(dq, dtype=np.float64)
            dq *= 0.85 if near else 0.75
            dq[0] = float(np.clip(dq[0], -0.015 if near else -0.020, 0.015 if near else 0.020))
            dq[1] = float(np.clip(dq[1], -0.10, 0.10))
            dq[2] = float(np.clip(dq[2], -0.14, 0.14))
            dq[3] = float(np.clip(dq[3], -0.08 if near else -0.04, 0.08 if near else 0.04))
            q[:4] += dq
            q = clamp_pose(q)

        return q

    def _solve_level2_continuous_qpos(self, env) -> np.ndarray:
        q = env.data.qpos[:6].astype(np.float64).copy()
        q3_center = -0.25
        q[3] = float(np.clip(q[3], -0.40, -0.10))
        q[4] = 0.0
        q[5] = float(PREGRASP_QPOS[5])
        ear = env._target_ear_center_position().copy()
        center = self._gripper_center_position_for_qpos(env, q)
        current_dist = float(np.linalg.norm(center - ear))
        desired_dist = max(0.0, current_dist - 0.040)

        if current_dist <= 1e-8:
            target = ear.copy()
        else:
            direction = (center - ear) / current_dist
            target = ear + direction * desired_dist

        q0_geom = self._solve_level2_yaw(env, q)
        q0_center = float(np.clip(q0_geom, float(env.target_low[0]), float(env.target_high[0])))
        q0_low = max(float(env.target_low[0]), float(q[0]) - 0.05, q0_center - 0.08)
        q0_high = min(float(env.target_high[0]), float(q[0]) + 0.05, q0_center + 0.08)
        q1_low = float(env.target_low[1])
        q1_high = float(env.target_high[1])
        q2_low = float(env.target_low[2])
        q2_high = float(env.target_high[2])
        q3_low = max(float(env.target_low[3]), -0.40)
        q3_high = min(float(env.target_high[3]), -0.10)

        def clamp_pose(q_target: np.ndarray) -> np.ndarray:
            q_target = np.asarray(q_target, dtype=np.float64).copy()
            q_target[0] = float(np.clip(q_target[0], q0_low, q0_high))
            q_target[1] = float(np.clip(q_target[1], q1_low, q1_high))
            q_target[2] = float(np.clip(q_target[2], q2_low, q2_high))
            q_target[3] = float(np.clip(q_target[3], q3_low, q3_high))
            q_target[4] = 0.0
            q_target[5] = float(PREGRASP_QPOS[5])
            return q_target

        q = clamp_pose(q)
        for _ in range(12):
            center = self._gripper_center_position_for_qpos(env, q)
            err = target - center
            if float(np.linalg.norm(err)) < 0.004:
                break

            eps = np.asarray([0.004, 0.006, 0.008, 0.010], dtype=np.float64)
            jac = np.zeros((3, 4), dtype=np.float64)
            for col, (joint_idx, step) in enumerate(zip((0, 1, 2, 3), eps, strict=False)):
                q_plus = clamp_pose(q.copy())
                q_minus = clamp_pose(q.copy())
                q_plus[joint_idx] += float(step)
                q_minus[joint_idx] -= float(step)
                q_plus = clamp_pose(q_plus)
                q_minus = clamp_pose(q_minus)
                p_plus = self._gripper_center_position_for_qpos(env, q_plus)
                p_minus = self._gripper_center_position_for_qpos(env, q_minus)
                jac[:, col] = (p_plus - p_minus) / (2.0 * float(step))

            dq = np.linalg.pinv(jac, rcond=1e-4) @ err
            dq = np.asarray(dq, dtype=np.float64) * 0.95
            dq[0] = float(np.clip(dq[0], -0.016, 0.016))
            dq[1] = float(np.clip(dq[1], -0.10, 0.10))
            dq[2] = float(np.clip(dq[2], -0.14, 0.14))
            dq[3] = float(np.clip(dq[3], -0.06, 0.06))
            q[:4] += dq
            q = clamp_pose(q)

        return q

    def _level1_action(self, env) -> np.ndarray:
        if self._level1_stage_complete(env):
            self.level1_tracking_active = False
            self.level1_center_hold_steps = 0
            if getattr(env, "control_mode", "joint_delta") == "joint_target":
                return np.clip(env.data.qpos[:6].astype(np.float32).copy(), env.target_low, env.target_high)
            return np.zeros(6, dtype=np.float32)
        # Hard gate requested by the user: L1 must first truly arrive at the
        # fixed observation pose before any target-driven scan or tracking is
        # allowed to start.
        if not env.level1_observation_pose_reached:
            self.level1_tracking_active = False
            self.level1_center_hold_steps = 0
            centered = self._level1_scan_base_pose()
            centered[0] = float(OBS_CENTER_QPOS[0])
            return self._joint_target_action(env, centered)
        if not self.level1_tracking_active:
            if self._visual_contact_signal(env):
                self.level1_tracking_active = True
                return self._joint_target_action(env, self._level1_center_qpos(env))
            return self._joint_target_action(env, self._level1_scan_qpos(env))
        if not self._visual_contact_signal(env):
            self.level1_tracking_active = False
            self.visual_lock_count = 0
            self.level1_center_hold_steps = 0
            direction = self._level1_search_direction(env)
            q_target = self._level1_scan_base_pose()
            q_target[0] = float(
                np.clip(
                    env.data.qpos[0] + 0.08 * direction,
                    float(OBS_RIGHT_QPOS[0]),
                    float(OBS_LEFT_QPOS[0]),
                )
            )
            return self._joint_target_action(env, q_target)
        q_target = self._level1_center_qpos(env)
        return self._joint_target_action(env, q_target)

    def _level2_action(self, env) -> np.ndarray:
        if not self.level2_approach_active and not self._level2_phase_gate_ready(env):
            return self._level1_action(env)
        if not self.level2_approach_active:
            self.level2_approach_active = True
            self.level1_tracking_active = False
            self.level1_center_hold_steps = 0
            self.visual_lock_count = 0
            self.level2_bridge_start_qpos = env.data.qpos[:6].astype(np.float64).copy()
            self.level2_bridge_steps_remaining = self.level2_bridge_total_steps
        if self.level2_cached_approach_qpos is None:
            self.level2_cached_approach_qpos = self._solve_level2_target_qpos(env, APPROACH_QPOS, near=False)
        if self.level2_cached_pregrasp_qpos is None:
            self.level2_cached_pregrasp_qpos = self._solve_level2_target_qpos(env, PREGRASP_QPOS, near=True)
        approach_pose = self.level2_cached_approach_qpos
        pregrasp_pose = self.level2_cached_pregrasp_qpos
        if self.level2_bridge_steps_remaining > 0 and self.level2_bridge_start_qpos is not None:
            progress = (self.level2_bridge_total_steps - self.level2_bridge_steps_remaining + 1) / float(
                self.level2_bridge_total_steps
            )
            bridge_pose = self.level2_bridge_start_qpos + progress * (approach_pose - self.level2_bridge_start_qpos)
            bridge_pose = np.asarray(bridge_pose, dtype=np.float64)
            bridge_pose[3] = approach_pose[3]
            bridge_pose[4] = approach_pose[4]
            bridge_pose[5] = approach_pose[5]
            self.level2_bridge_steps_remaining -= 1
            return self._joint_target_action(env, bridge_pose)
        if env.staged_reset_active("level2_approach"):
            return self._joint_target_action(env, pregrasp_pose)
        continuous_pose = self._solve_level2_continuous_qpos(env)
        if env.ee_ear_center_distance() <= 0.030:
            q_target = pregrasp_pose.copy()
            q_target[:4] = continuous_pose[:4]
            q_target[5] = pregrasp_pose[5]
            return self._joint_target_action(env, q_target)
        return self._joint_target_action(env, continuous_pose)

    def _level3_action(self, env) -> np.ndarray:
        if env.object_attached:
            if (not env.lifted) and (not self._level3_gripper_fully_closed(env)):
                return self._joint_target_action(env, self._level3_close_pose(env))
            if not env.lifted:
                return self._joint_target_action(env, self._level3_lift_pose(env))
            if self.level3_place_active or env._target_dropzone_xy_distance() <= 0.035:
                self.level3_place_active = True
            if not self.level3_place_active and env._target_dropzone_xy_distance() > 0.030:
                return self._joint_target_action(env, self._level3_transport_pose(env))
            if env._target_body_position()[2] > 0.110:
                return self._joint_target_action(env, self._level3_place_pose(env))
            delta = np.zeros(6, dtype=np.float32)
            delta[5] = self.max_gripper_delta
            return self._control_action_from_delta(env, delta)

        self.level3_cached_lift_qpos = None
        self.level3_cached_transport_qpos = None
        self.level3_cached_place_qpos = None
        self.level3_place_active = False
        if not self._level3_close_gate_ready(env):
            return self._level2_action(env)
        return self._joint_target_action(env, self._level3_close_pose(env))

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
            action = self._joint_target_action(env, HOME_QPOS.copy())
        self.phase_steps += 1
        return action.astype(np.float32)

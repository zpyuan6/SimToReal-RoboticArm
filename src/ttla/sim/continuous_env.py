from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .context import context_vector
from .env import RoArmSimEnv
from .skills import PREGRASP_QPOS
from .task_defs import TASK_TO_ID


@dataclass
class ContinuousTransition:
    image: np.ndarray
    proprio: np.ndarray
    action: np.ndarray
    next_image: np.ndarray
    next_proprio: np.ndarray
    task_id: int
    success: int
    context: np.ndarray


TASK_TEXT = {
    "level1_verify": "center the target object in the camera view",
    "level2_approach": "move the gripper into a stable pre-grasp approach state",
    "level3_pick_place": "pick up the object and place it in the blue drop zone",
}


class ContinuousRoArmSimEnv(RoArmSimEnv):
    """Continuous-control companion environment sharing the same MuJoCo world.

    The legacy primitive environment remains untouched. This class provides the
    new route-B interface for continuous/chunk backbones:

        observation -> continuous joint action -> next observation
    """

    def __init__(
        self,
        sim_cfg: dict,
        seed: int = 0,
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
        control_mode: str = "joint_delta",
    ) -> None:
        self.control_mode = str(control_mode)
        self._staged_reset_expert = None
        self._continuous_init_ready = False
        super().__init__(sim_cfg, seed=seed)
        self.action_low = np.asarray(action_low if action_low is not None else [-0.25] * 6, dtype=np.float32)
        self.action_high = np.asarray(action_high if action_high is not None else [0.25] * 6, dtype=np.float32)
        if self.action_low.shape != (6,) or self.action_high.shape != (6,):
            raise ValueError("ContinuousRoArmSimEnv expects 6-D joint action clamps.")
        ctrl_range = np.asarray(self.model.actuator_ctrlrange[:6], dtype=np.float32)
        self.target_low = ctrl_range[:, 0].copy()
        self.target_high = ctrl_range[:, 1].copy()
        self._continuous_init_ready = True

    def _staged_reset_prereq(self, task_name: str) -> str | None:
        staged = self.cfg.get("staged_resets", {})
        if not isinstance(staged, dict):
            return None
        prereq = staged.get(task_name)
        return str(prereq) if prereq else None

    def staged_reset_active(self, task_name: str | None = None) -> bool:
        name = str(task_name if task_name is not None else self.task_name)
        return self._staged_reset_prereq(name) is not None

    def _ensure_staged_reset_expert(self):
        if self._staged_reset_expert is None:
            from .continuous_expert import ContinuousWaypointExpert

            self._staged_reset_expert = ContinuousWaypointExpert()
        return self._staged_reset_expert

    def _set_direct_joint_pose(self, qpos: np.ndarray) -> None:
        qpos = np.asarray(qpos, dtype=np.float64).reshape(6)
        self.data.qpos[:6] = qpos
        self.data.ctrl[:6] = qpos
        mujoco.mj_forward(self.model, self.data)

    def _direct_level1_success_pose(self) -> np.ndarray:
        expert = self._ensure_staged_reset_expert()
        expert.reset(task_name="level1_verify")
        q_target = np.asarray(expert._level1_scan_base_pose(), dtype=np.float64).copy()
        self._set_direct_joint_pose(q_target)
        q_target = np.asarray(expert._level1_center_qpos(self), dtype=np.float64).copy()
        q_target[2] = min(float(q_target[2]), 2.55)
        self._set_direct_joint_pose(q_target)
        if not self.clear_view_ready():
            q_target = np.asarray(expert._level1_center_qpos(self), dtype=np.float64).copy()
            q_target[2] = min(float(q_target[2]), 2.55)
        return np.clip(q_target, self.target_low.astype(np.float64), self.target_high.astype(np.float64))

    def _direct_level2_success_pose(self) -> np.ndarray:
        expert = self._ensure_staged_reset_expert()
        q_l1 = self._direct_level1_success_pose()
        self._set_direct_joint_pose(q_l1)
        self.level1_observation_pose_reached = True
        self.level1_observation_pose_hold_steps = 1
        expert.reset(task_name="level2_approach")
        q_target = np.asarray(expert._solve_level2_target_qpos(self, PREGRASP_QPOS, near=True), dtype=np.float64).copy()
        self._set_direct_joint_pose(q_target)
        if not self.approach_success_ready():
            q_target = np.asarray(expert._solve_level2_target_qpos(self, PREGRASP_QPOS, near=True), dtype=np.float64).copy()
        return np.clip(q_target, self.target_low.astype(np.float64), self.target_high.astype(np.float64))

    def _direct_stage_pose(self, requested_task: str, prereq_task: str) -> np.ndarray:
        if prereq_task == "level1_verify":
            return self._direct_level1_success_pose()
        if prereq_task == "level2_approach":
            return self._direct_level2_success_pose()
        raise KeyError(f"Unsupported staged prerequisite '{prereq_task}' for task '{requested_task}'")

    def _finalize_staged_reset(self, task_name: str) -> dict[str, np.ndarray]:
        self.task_name = str(task_name)
        self.step_idx = 0
        self.verified = self.verified_status()
        self.placed = False
        self.object_attached = False
        self.lifted = False
        self.grasp_reference_ee_z = None
        self.release_counter = 0
        self.recent_ear_contact = 0
        self.action_delay_queue.clear()
        self.data.ctrl[:6] = self.data.qpos[:6].copy()
        mujoco.mj_forward(self.model, self.data)
        return self._observation()

    def reset(self, task_name: str | None = None, context: dict[str, float] | None = None) -> dict[str, np.ndarray]:
        if not getattr(self, "_continuous_init_ready", False):
            return super().reset(task_name=task_name, context=context)
        requested_task = str(task_name if task_name is not None else self.task_name)
        prereq_task = self._staged_reset_prereq(requested_task)
        obs = super().reset(task_name=requested_task, context=context)
        if prereq_task is None:
            return obs
        q_target = self._direct_stage_pose(requested_task, prereq_task)
        self._set_direct_joint_pose(q_target)
        if prereq_task in {"level1_verify", "level2_approach"}:
            self.level1_observation_pose_reached = True
            self.level1_observation_pose_hold_steps = 1
        return self._finalize_staged_reset(requested_task)

    def task_text(self) -> str:
        return TASK_TEXT[self.task_name]

    def _settle_released_target(self) -> None:
        target_xy = self._target_body_position()[:2].copy()
        target_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.model.body_pos[target_body] = np.asarray([target_xy[0], target_xy[1], 0.040], dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)

    def _target_in_dropzone(self) -> bool:
        return bool(self._target_xy_in_dropzone() and self._target_body_position()[2] <= 0.050)

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (6,):
            raise ValueError(f"Expected 6-D continuous action, got shape {action.shape}")
        if self.control_mode == "joint_target":
            return np.clip(action, self.target_low, self.target_high)
        return np.clip(action, self.action_low, self.action_high)

    def _target_qpos_from_action(self, action: np.ndarray) -> np.ndarray:
        action = self._clip_action(action)
        if self.control_mode == "joint_delta":
            target_qpos = self.data.qpos[:6].copy() + action.astype(np.float64)
        elif self.control_mode == "joint_target":
            target_qpos = action.astype(np.float64)
        else:
            raise KeyError(f"Unsupported control mode: {self.control_mode}")
        target_qpos = np.clip(target_qpos, self.target_low.astype(np.float64), self.target_high.astype(np.float64))
        return target_qpos

    def _apply_continuous_action(self, target_qpos: np.ndarray) -> None:
        self._apply_target_pose(target_qpos, dwell=1)

    def _update_continuous_manipulation_state(
        self,
        was_attached: bool,
        gripper_command_delta: float,
    ) -> None:
        if not self.object_attached:
            if self._attach_capture_ready():
                self.active_grasp_local_offset = -self._grasp_site_local_positions()[self._nearest_grasp_site_name()]
                self.object_attached = True
                self.lifted = False
                self.grasp_reference_ee_z = float(self._ee_position()[2])
                self.release_counter = 0
                self._update_attached_object_pose()
        elif (
            float(gripper_command_delta) > 0.12
            and self._target_dropzone_xy_distance() <= 0.035
            and self._target_body_position()[2] <= 0.110
        ):
            self.object_attached = False
            self.grasp_reference_ee_z = None
            self.release_counter = 0
        self._update_lifted_from_reference()
        if was_attached and not self.object_attached:
            self._settle_released_target()
            self.placed = self._target_in_dropzone()

    def step_action(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, dict]:
        action = self._clip_action(action)
        obs = self._observation()
        was_attached = bool(self.object_attached)
        before_qpos = self.data.qpos[:6].copy()
        target_qpos = self._target_qpos_from_action(action)
        gripper_command_delta = float(target_qpos[5] - before_qpos[5])
        self._apply_continuous_action(target_qpos)
        self._update_continuous_manipulation_state(was_attached, gripper_command_delta)
        if self.task_name in {"level1_verify", "level2_approach", "level3_pick_place"}:
            if self.level1_fixed_observation_reached():
                self.level1_observation_pose_hold_steps = 1
                self.level1_observation_pose_reached = True
            else:
                self.level1_observation_pose_hold_steps = 0
        self.verified = self.verified_status()
        self.step_idx += 1
        next_obs = self._observation()
        success = self.task_success()
        done = bool(success or self.step_idx >= self.cfg["episode_horizon"])
        reward = self._reward(success)
        info = {
            "task": self.task_name,
            "task_text": self.task_text(),
            "success": success,
            "visibility": self.visibility_score(),
            "center_error": self.center_error_px(),
            "verified": int(self.verified),
            "level1_pose_reached": int(self.level1_observation_pose_reached),
            "level1_pose_hold_steps": int(self.level1_observation_pose_hold_steps),
            "grasped": int(self.object_attached),
            "lifted": int(self.lifted),
            "placed": int(self.placed),
            "ear_contact_count": self._ear_grasp_contact_count(),
            "ee_target_distance": self.ee_target_distance(),
            "ee_ear_center_distance": self.ee_ear_center_distance(),
            "grasp_gap": self.grasp_gap(),
            "dropzone_distance": self.dropzone_distance(),
            "context": context_vector(self.context),
            "transition": ContinuousTransition(
                image=obs["image"],
                proprio=obs["state"],
                action=action.copy(),
                next_image=next_obs["image"],
                next_proprio=next_obs["state"],
                task_id=int(TASK_TO_ID[self.task_name]),
                success=success,
                context=context_vector(self.context),
            ),
        }
        return next_obs, reward, done, info

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .context import context_vector
from .env import RoArmSimEnv


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
        super().__init__(sim_cfg, seed=seed)
        self.control_mode = str(control_mode)
        self.action_low = np.asarray(action_low if action_low is not None else [-0.25] * 6, dtype=np.float32)
        self.action_high = np.asarray(action_high if action_high is not None else [0.25] * 6, dtype=np.float32)
        if self.action_low.shape != (6,) or self.action_high.shape != (6,):
            raise ValueError("ContinuousRoArmSimEnv expects 6-D joint action clamps.")

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
        return np.clip(action, self.action_low, self.action_high)

    def _apply_continuous_action(self, action: np.ndarray) -> None:
        action = self._clip_action(action)
        if self.control_mode == "joint_delta":
            target_qpos = self.data.qpos[:6].copy() + action.astype(np.float64)
        elif self.control_mode == "joint_target":
            target_qpos = action.astype(np.float64)
        else:
            raise KeyError(f"Unsupported control mode: {self.control_mode}")
        self._apply_target_pose(target_qpos, dwell=1)

    def _update_continuous_manipulation_state(self, was_attached: bool, before_z: float, action: np.ndarray) -> None:
        if not self.object_attached:
            if (
                self._gripper_firmly_closed()
                and (
                    self._ear_grasp_contact_count() > 0
                    or self.grasp_gap() < 0.004
                    or (
                        self.ee_target_distance() < 0.055
                        and self.center_error_px() < self._scaled_px(7.0)
                    )
                )
            ):
                self.active_grasp_local_offset = -self._grasp_site_local_positions()[self._nearest_grasp_site_name()]
                self.object_attached = True
                self.lifted = False
                self.release_counter = 0
                self._update_attached_object_pose()
        elif (
            float(action[5]) > 0.12
            and self._target_dropzone_xy_distance() <= 0.030
            and self._target_body_position()[2] <= 0.070
        ):
            self.object_attached = False
            self.release_counter = 0
        if self.object_attached and self._ee_position()[2] > before_z + 0.015:
            self.lifted = True
        if was_attached and not self.object_attached:
            self._settle_released_target()
            self.placed = self._target_in_dropzone()

    def step_action(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, dict]:
        action = self._clip_action(action)
        obs = self._observation()
        was_attached = bool(self.object_attached)
        before_z = float(self._ee_position()[2])
        self._apply_continuous_action(action)
        self._update_continuous_manipulation_state(was_attached, before_z, action)
        self.verified = self.verify_ready()
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
            "grasped": int(self.object_attached),
            "lifted": int(self.lifted),
            "placed": int(self.placed),
            "ear_contact_count": self._ear_grasp_contact_count(),
            "ee_target_distance": self.ee_target_distance(),
            "grasp_gap": self.grasp_gap(),
            "dropzone_distance": self.dropzone_distance(),
            "context": context_vector(self.context),
            "transition": ContinuousTransition(
                image=obs["image"],
                proprio=obs["state"],
                action=action.copy(),
                next_image=next_obs["image"],
                next_proprio=next_obs["state"],
                task_id=int(self._state_vector()[-2]),
                success=success,
                context=context_vector(self.context),
            ),
        }
        return next_obs, reward, done, info

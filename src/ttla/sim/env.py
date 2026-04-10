from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np

from .context import ContextConfig, context_vector, sample_context
from .skills import skill_to_joint_delta


TASK_TO_ID = {"approach": 0, "verify": 1, "observe_then_act": 2}


@dataclass
class Transition:
    image: np.ndarray
    state: np.ndarray
    action: int
    next_image: np.ndarray
    next_state: np.ndarray
    task_id: int
    success: int
    context: np.ndarray


class RoArmSimEnv:
    def __init__(self, sim_cfg: dict, seed: int = 0) -> None:
        self.cfg = sim_cfg
        self.rng = np.random.default_rng(seed)
        xml_path = Path(__file__).resolve().parent / "mjcf" / "roarm_simplified.xml"
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.obs_renderer = mujoco.Renderer(self.model, height=self.cfg["image_size"], width=self.cfg["image_size"])
        self.debug_size = max(256, int(self.cfg["image_size"]) * 4)
        self.debug_renderer = mujoco.Renderer(self.model, height=self.debug_size, width=self.debug_size)
        self.context_cfg = ContextConfig(**sim_cfg["context"])
        self.action_delay_queue: deque[np.ndarray] = deque()
        self.task_name = "approach"
        self.step_idx = 0
        self.context = sample_context(self.context_cfg, self.rng)
        self.reset()

    def reset(self, task_name: str | None = None, context: dict[str, float] | None = None) -> dict[str, np.ndarray]:
        if task_name is not None:
            self.task_name = task_name
        self.context = context if context is not None else sample_context(self.context_cfg, self.rng)
        self.step_idx = 0
        self.action_delay_queue.clear()
        mujoco.mj_resetData(self.model, self.data)
        # Match the official roarm_ws MoveIt fake-system initial pose:
        # base_link_to_link1: 0
        # link1_to_link2: 0
        # link2_to_link3: 2.618
        # link3_to_link4: -1.0472
        # link4_to_link5: 0
        # link5_to_gripper_link: 0
        qpos = np.asarray([0.0, 0.0, 2.618, -1.0472, 0.0, 0.0], dtype=np.float64)
        self.data.qpos[:6] = qpos
        self.data.ctrl[:6] = qpos
        self._sample_task_layout()
        mujoco.mj_forward(self.model, self.data)
        return self._observation()

    def _sample_task_layout(self) -> None:
        target_x = self.rng.uniform(0.22, 0.40)
        target_y = self.rng.uniform(-0.16, 0.16)
        target_z = 0.08
        target_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.model.body_pos[target_body] = np.asarray([target_x, target_y, target_z], dtype=np.float64)
        distractor_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "distractor")
        self.model.body_pos[distractor_body] = np.asarray(
            [self.rng.uniform(0.20, 0.42), self.rng.uniform(-0.18, 0.18), 0.08],
            dtype=np.float64,
        )

    def _apply_skill(self, action: int) -> None:
        command = skill_to_joint_delta(action)
        joint_target = self.data.ctrl[:6].copy()
        delta = np.zeros(6, dtype=np.float64)
        delta[:6] = command.joint_delta.astype(np.float64) * self.context["action_gain"]
        delta[np.abs(delta) < self.context["deadzone"]] = 0.0
        joint_target += delta
        act_delay = self.context["action_delay"]
        self.action_delay_queue.append(joint_target)
        if len(self.action_delay_queue) > act_delay:
            applied = self.action_delay_queue.popleft()
        else:
            applied = self.data.ctrl[:6].copy()
        low = self.model.actuator_ctrlrange[:6, 0]
        high = self.model.actuator_ctrlrange[:6, 1]
        self.data.ctrl[:6] = np.clip(applied, low, high)
        for _ in range(self.cfg["action_repeat"] * command.dwell):
            mujoco.mj_step(self.model, self.data)

    def _camera_image(self) -> np.ndarray:
        image = self._render_camera("forearm_cam", self.obs_renderer)
        image = self._apply_context_appearance(image)
        return image

    def render_debug_view(self, camera_name: str = "overview_cam") -> np.ndarray:
        image = self._render_camera(camera_name, self.debug_renderer)
        return self._apply_context_appearance(image, include_noise=False)

    def _render_camera(self, camera_name: str, renderer: mujoco.Renderer) -> np.ndarray:
        renderer.update_scene(self.data, camera=camera_name)
        image = renderer.render()
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def _apply_context_appearance(self, image: np.ndarray, include_noise: bool = True) -> np.ndarray:
        image = image.copy()
        shift_x = int(round(self.context["cam_yaw"] * image.shape[1] * 0.35))
        shift_y = int(round((self.context["cam_pitch"] + self.context["cam_z"] * 8.0) * image.shape[0] * 0.18))
        scale = 1.0 + self.context["fov_bias"] / 140.0 + self.context["cam_x"] * 1.6
        angle = float(np.rad2deg(self.context["cam_roll"]) * 1.2)
        center = (image.shape[1] / 2.0, image.shape[0] / 2.0)
        affine = cv2.getRotationMatrix2D(center, angle, max(scale, 0.75))
        affine[0, 2] += shift_x
        affine[1, 2] += shift_y
        image = cv2.warpAffine(image, affine, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REFLECT_101)
        image = np.clip(image.astype(np.float32) * self.context["light_gain"], 0, 255).astype(np.uint8)
        sigma = self.context["blur_sigma"]
        if sigma > 1e-5:
            k = max(1, int(round(sigma * 3)) * 2 + 1)
            image = cv2.GaussianBlur(image, (k, k), sigmaX=sigma)
        noise = self.context["noise_std"] if include_noise else self.context["noise_std"] * 0.35
        if include_noise and noise > 0:
            gauss = self.rng.normal(0.0, noise * 255.0, image.shape).astype(np.float32)
            image = np.clip(image.astype(np.float32) + gauss, 0, 255).astype(np.uint8)
        return image

    def _camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "forearm_cam")
        cam_pos = self.data.cam_xpos[cam_id].copy()
        cam_rot = self.data.cam_xmat[cam_id].reshape(3, 3).copy()
        cam_pos += np.asarray([self.context["cam_x"], self.context["cam_y"], self.context["cam_z"]], dtype=np.float64)
        roll = self.context["cam_roll"]
        pitch = self.context["cam_pitch"]
        yaw = self.context["cam_yaw"]
        rx = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(roll), -np.sin(roll)],
                [0.0, np.sin(roll), np.cos(roll)],
            ],
            dtype=np.float64,
        )
        ry = np.asarray(
            [
                [np.cos(pitch), 0.0, np.sin(pitch)],
                [0.0, 1.0, 0.0],
                [-np.sin(pitch), 0.0, np.cos(pitch)],
            ],
            dtype=np.float64,
        )
        rz = np.asarray(
            [
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        cam_rot = cam_rot @ rz @ ry @ rx
        return cam_pos, cam_rot

    def _project_object(self, position: np.ndarray) -> tuple[bool, int, int, float]:
        image_size = self.cfg["image_size"]
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "forearm_cam")
        cam_pos, cam_rot = self._camera_pose()
        rel = position - cam_pos
        cam_rel = cam_rot.T @ rel
        depth = -cam_rel[2]
        if depth <= 1e-6:
            return False, 0, 0, 0.0
        fovy = np.deg2rad(float(self.model.cam_fovy[cam_id]) + self.context["fov_bias"])
        tan_half = max(np.tan(fovy / 2.0), 1e-6)
        u = 0.5 + cam_rel[0] / (2.0 * depth * tan_half)
        v = 0.5 - cam_rel[1] / (2.0 * depth * tan_half)
        radius = 3.0 + 16.0 / (1.0 + 10.0 * depth)
        px = int(np.clip(u * image_size, 0, image_size - 1))
        py = int(np.clip(v * image_size, 0, image_size - 1))
        visible = 0.02 < u < 0.98 and 0.02 < v < 0.98
        return visible, px, py, radius

    def _observation(self) -> dict[str, np.ndarray]:
        return {"image": self._camera_image(), "state": self._state_vector()}

    def observe(self) -> dict[str, np.ndarray]:
        return self._observation()

    def _state_vector(self) -> np.ndarray:
        qpos = self.data.qpos[:6].astype(np.float32)
        qvel = self.data.qvel[:6].astype(np.float32)
        ee = self._ee_position().astype(np.float32)
        target = self._target_position().astype(np.float32)
        task_id = np.asarray([TASK_TO_ID[self.task_name]], dtype=np.float32)
        step_frac = np.asarray([self.step_idx / max(1, self.cfg["episode_horizon"])], dtype=np.float32)
        return np.concatenate([qpos, qvel, ee, target[:2], task_id, step_frac], dtype=np.float32)

    def _ee_position(self) -> np.ndarray:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        return self.data.site_xpos[site_id].copy()

    def _target_position(self) -> np.ndarray:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        return self.data.xpos[body_id].copy()

    def visibility_score(self) -> float:
        visible, _, _, radius = self._project_object(self._target_position())
        if not visible:
            return 0.0
        return float(min(radius / 10.0, 1.0))

    def task_success(self) -> int:
        ee = self._ee_position()
        target = self._target_position()
        dist = np.linalg.norm(ee - target)
        visible = self.visibility_score()
        if self.task_name == "approach":
            return int(dist < 0.09)
        if self.task_name == "verify":
            return int(visible > 0.22)
        return int(dist < 0.12 and visible > 0.18)

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, dict]:
        obs = self._observation()
        self._apply_skill(action)
        self.step_idx += 1
        next_obs = self._observation()
        success = self.task_success()
        done = bool(success or self.step_idx >= self.cfg["episode_horizon"] or action == 6)
        reward = 1.0 if success else -0.01
        info = {
            "task": self.task_name,
            "success": success,
            "visibility": self.visibility_score(),
            "context": context_vector(self.context),
            "transition": Transition(
                image=obs["image"],
                state=obs["state"],
                action=action,
                next_image=next_obs["image"],
                next_state=next_obs["state"],
                task_id=TASK_TO_ID[self.task_name],
                success=success,
                context=context_vector(self.context),
            ),
        }
        return next_obs, reward, done, info

    def idle_step(self, frames: int = 1) -> dict[str, np.ndarray]:
        for _ in range(max(1, frames)):
            mujoco.mj_step(self.model, self.data)
        return self._observation()

    def close(self) -> None:
        self.obs_renderer.close()
        self.debug_renderer.close()

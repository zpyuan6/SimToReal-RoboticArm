from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np

from .context import ContextConfig, context_vector, sample_context
from .skills import (
    ABORT_ID,
    APPROACH_COARSE_ID,
    APPROACH_FINE_ID,
    CARRY_QPOS,
    DROPZONE_QPOS,
    GRASP_EXECUTE_ID,
    HOME_QPOS,
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
    allowed_primitives,
    observe_pose,
    primitive_action,
    primitive_name,
)
from .task_defs import TASK_TO_ID, task_spec


@dataclass
class Transition:
    image: np.ndarray
    state: np.ndarray
    primitive_id: int
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
        self.task_name = self.cfg["tasks"][0]
        self.context = sample_context(self.context_cfg, self.rng)
        self.step_idx = 0
        self.object_attached = False
        self.verified = False
        self.placed = False
        self.lifted = False
        self.last_primitive = OBS_CENTER_ID
        self.object_grasp_offset = np.asarray([0.0, 0.0, -0.02], dtype=np.float64)
        self.reset()

    def reset(self, task_name: str | None = None, context: dict[str, float] | None = None) -> dict[str, np.ndarray]:
        if task_name is not None:
            self.task_name = task_name
        self.context = context if context is not None else sample_context(self.context_cfg, self.rng)
        self.step_idx = 0
        self.object_attached = False
        self.verified = False
        self.placed = False
        self.lifted = False
        self.last_primitive = OBS_CENTER_ID
        self.action_delay_queue.clear()
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:6] = HOME_QPOS.astype(np.float64)
        self.data.ctrl[:6] = HOME_QPOS.astype(np.float64)
        self._sample_task_layout()
        mujoco.mj_forward(self.model, self.data)
        return self._observation()

    def _sample_task_layout(self) -> None:
        target_x = self.rng.uniform(0.22, 0.38)
        target_y = self.rng.uniform(-0.12, 0.12)
        drop_x = self.rng.uniform(0.18, 0.32)
        drop_y = self.rng.uniform(-0.20, -0.08)
        target_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        drop_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "distractor")
        self.model.body_pos[target_body] = np.asarray([target_x, target_y, 0.08], dtype=np.float64)
        self.model.body_pos[drop_body] = np.asarray([drop_x, drop_y, 0.045], dtype=np.float64)

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
        if include_noise and self.context["noise_std"] > 0:
            gauss = self.rng.normal(0.0, self.context["noise_std"] * 255.0, image.shape).astype(np.float32)
            image = np.clip(image.astype(np.float32) + gauss, 0, 255).astype(np.uint8)
        return image

    def _camera_image(self) -> np.ndarray:
        return self._apply_context_appearance(self._render_camera("forearm_cam", self.obs_renderer))

    def render_debug_view(self, camera_name: str = "overview_cam") -> np.ndarray:
        return self._apply_context_appearance(self._render_camera(camera_name, self.debug_renderer), include_noise=False)

    def _camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "forearm_cam")
        cam_pos = self.data.cam_xpos[cam_id].copy()
        cam_rot = self.data.cam_xmat[cam_id].reshape(3, 3).copy()
        cam_pos += np.asarray([self.context["cam_x"], self.context["cam_y"], self.context["cam_z"]], dtype=np.float64)
        roll = self.context["cam_roll"]
        pitch = self.context["cam_pitch"]
        yaw = self.context["cam_yaw"]
        rx = np.asarray([[1.0, 0.0, 0.0], [0.0, np.cos(roll), -np.sin(roll)], [0.0, np.sin(roll), np.cos(roll)]], dtype=np.float64)
        ry = np.asarray([[np.cos(pitch), 0.0, np.sin(pitch)], [0.0, 1.0, 0.0], [-np.sin(pitch), 0.0, np.cos(pitch)]], dtype=np.float64)
        rz = np.asarray([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        return cam_pos, cam_rot @ rz @ ry @ rx

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
        flags = np.asarray(
            [
                float(self.object_attached),
                float(self.verified),
                float(self.lifted),
                float(self.placed),
                float(TASK_TO_ID[self.task_name]),
                float(self.step_idx / max(1, self.cfg["episode_horizon"])),
            ],
            dtype=np.float32,
        )
        return np.concatenate([qpos, qvel, flags], dtype=np.float32)

    def _target_position(self) -> np.ndarray:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        return self.data.xpos[body_id].copy()

    def _dropzone_position(self) -> np.ndarray:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "distractor")
        return self.data.xpos[body_id].copy()

    def _ee_position(self) -> np.ndarray:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        return self.data.site_xpos[site_id].copy()

    def visibility_score(self) -> float:
        visible, _, _, radius = self._project_object(self._target_position())
        if not visible:
            return 0.0
        return float(min(radius / 10.0, 1.0))

    def center_error_px(self) -> float:
        visible, px, py, _ = self._project_object(self._target_position())
        if not visible:
            return float(self.cfg["image_size"])
        center = self.cfg["image_size"] / 2.0
        return float(np.sqrt((px - center) ** 2 + (py - center) ** 2))

    def target_yaw_error(self) -> float:
        target = self._target_position()
        yaw_target = np.arctan2(target[1], max(target[0], 1e-6))
        return float(np.clip(yaw_target - self.data.qpos[0], -0.9, 0.9))

    def ee_target_distance(self) -> float:
        return float(np.linalg.norm(self._ee_position() - self._target_position()))

    def dropzone_distance(self) -> float:
        return float(np.linalg.norm(self._ee_position() - self._dropzone_position()))

    def _dropzone_xy_distance(self) -> float:
        ee_xy = self._ee_position()[:2]
        drop_xy = self._dropzone_position()[:2]
        return float(np.linalg.norm(ee_xy - drop_xy))

    def pregrasp_ready(self) -> bool:
        return self.visibility_score() > 0.14 and self.center_error_px() < 18.0 and self.ee_target_distance() < 0.18

    def approach_success_ready(self) -> bool:
        return (
            self.visibility_score() > 0.16
            and self.center_error_px() < 14.0
            and self.ee_target_distance() < 0.15
        )

    def task_success(self) -> int:
        if self.task_name == "level1_verify":
            return int(self.verified)
        if self.task_name == "level2_approach":
            return int(self.verified and self.approach_success_ready())
        return int(self.placed)

    def _apply_target_pose(self, target_qpos: np.ndarray, dwell: int = 1) -> None:
        current = self.data.ctrl[:6].copy()
        target = np.asarray(target_qpos, dtype=np.float64).copy()
        target[:5] += self.context["joint_bias"]
        desired = current + self.context["action_gain"] * (target - current)
        low = self.model.actuator_ctrlrange[:6, 0]
        high = self.model.actuator_ctrlrange[:6, 1]
        desired = np.clip(desired, low, high)
        self.action_delay_queue.append(desired.copy())
        if len(self.action_delay_queue) > self.context["action_delay"]:
            applied = self.action_delay_queue.popleft()
        else:
            applied = self.data.ctrl[:6].copy()
        for _ in range(max(1, dwell) * self.cfg["action_repeat"]):
            self.data.ctrl[:6] = applied
            mujoco.mj_step(self.model, self.data)
            if self.object_attached:
                self._update_attached_object_pose()
        mujoco.mj_forward(self.model, self.data)

    def _update_attached_object_pose(self) -> None:
        target_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.model.body_pos[target_body] = self._ee_position() + self.object_grasp_offset
        mujoco.mj_forward(self.model, self.data)

    def _prealign_pose(self, target_position: np.ndarray) -> np.ndarray:
        pose = self._pregrasp_anchor_pose(target_position)
        pose[1] = min(pose[1], 0.05)
        pose[2] = min(pose[2], 2.34)
        pose[3] = min(pose[3], -0.26)
        pose[5] = 0.98
        return pose

    def _carry_pose(self, destination: np.ndarray) -> np.ndarray:
        pose = CARRY_QPOS.copy()
        pose[0] = float(np.clip(np.arctan2(destination[1], max(destination[0], 1e-6)), -1.20, 1.20))
        return pose

    def _pregrasp_anchor_pose(self, target_position: np.ndarray) -> np.ndarray:
        pose = PREALIGN_BASE_QPOS.copy()
        pose[0] = float(np.clip(np.arctan2(target_position[1], max(target_position[0], 1e-6)), -1.10, 1.10))
        # Coarse-searched anchor that reaches the tabletop target band more
        # consistently before the local servo loop starts.
        pose[1] = 0.15
        pose[2] = 2.40
        pose[3] = -0.20
        pose[5] = 0.86
        return pose

    def _dropzone_pose(self) -> np.ndarray:
        destination = self._dropzone_position()
        pose = DROPZONE_QPOS.copy()
        pose[0] = float(np.clip(np.arctan2(destination[1], max(destination[0], 1e-6)), -1.30, 1.30))
        # Coarse-searched transport pose template for the drop zone.
        pose[1] = 0.10
        pose[2] = 2.425
        pose[3] = -0.20
        pose[5] = 0.18
        return pose

    def _execute_observe(self, primitive_id_value: int) -> None:
        self._apply_target_pose(observe_pose(primitive_id_value), dwell=2)

    def _execute_verify(self) -> None:
        vis_scores = []
        center_scores = []
        for _ in range(3):
            self._apply_target_pose(self.data.ctrl[:6], dwell=1)
            vis_scores.append(self.visibility_score())
            center_scores.append(self.center_error_px())
        self.verified = bool(np.mean(vis_scores) > 0.11 and np.mean(center_scores) < 25.0)

    def _execute_prealign(self) -> None:
        self._apply_target_pose(self._prealign_pose(self._target_position()), dwell=2)

    def _servo_step_to_point(
        self,
        point: np.ndarray,
        *,
        forward_gain: float,
        vertical_gain: float,
        yaw_gain: float,
        forward_clip: tuple[float, float],
        shoulder_clip: tuple[float, float],
        wrist_clip: tuple[float, float],
        gripper_open: float,
        dwell: int,
    ) -> None:
        q_target = self.data.qpos[:6].copy()
        rel = point - self._ee_position()
        yaw_error = float(np.arctan2(point[1], max(point[0], 1e-6)) - self.data.qpos[0])
        q_target[0] += float(np.clip(yaw_gain * yaw_error, -0.10, 0.10))
        q_target[1] += float(
            np.clip(
                -vertical_gain * rel[2] - 0.18 * (rel[0] - 0.07),
                shoulder_clip[0],
                shoulder_clip[1],
            )
        )
        q_target[2] += float(np.clip(-forward_gain * rel[0], forward_clip[0], forward_clip[1]))
        q_target[3] += float(
            np.clip(
                0.60 * rel[2] - 0.40 * rel[0],
                wrist_clip[0],
                wrist_clip[1],
            )
        )
        q_target[5] = gripper_open
        self._apply_target_pose(q_target, dwell=dwell)

    def _servo_step_to_target(
        self,
        *,
        forward_gain: float,
        vertical_gain: float,
        yaw_gain: float,
        forward_clip: tuple[float, float],
        shoulder_clip: tuple[float, float],
        wrist_clip: tuple[float, float],
        gripper_open: float,
        dwell: int,
    ) -> None:
        self._servo_step_to_point(
            self._target_position(),
            forward_gain=forward_gain,
            vertical_gain=vertical_gain,
            yaw_gain=yaw_gain,
            forward_clip=forward_clip,
            shoulder_clip=shoulder_clip,
            wrist_clip=wrist_clip,
            gripper_open=gripper_open,
            dwell=dwell,
        )

    def _execute_approach(self, fine: bool) -> None:
        if fine:
            for _ in range(3):
                self._servo_step_to_target(
                    forward_gain=0.85,
                    vertical_gain=0.75,
                    yaw_gain=0.80,
                    forward_clip=(-0.08, 0.03),
                    shoulder_clip=(-0.05, 0.05),
                    wrist_clip=(-0.04, 0.04),
                    gripper_open=0.92,
                    dwell=1,
                )
                if self.pregrasp_ready():
                    break
            return
        self._apply_target_pose(self._prealign_pose(self._target_position()), dwell=1)
        for _ in range(3):
            self._servo_step_to_target(
                forward_gain=1.15,
                vertical_gain=0.95,
                yaw_gain=1.00,
                forward_clip=(-0.14, 0.06),
                shoulder_clip=(-0.07, 0.07),
                wrist_clip=(-0.05, 0.05),
                gripper_open=0.96,
                dwell=1,
            )
            if self.ee_target_distance() < 0.14:
                break

    def _execute_retreat(self) -> None:
        q_target = self.data.qpos[:6].copy()
        q_target[1] += 0.10
        q_target[2] += 0.14
        q_target[3] -= 0.06
        q_target[5] = 1.10
        self._apply_target_pose(q_target, dwell=2)

    def _execute_pregrasp_servo(self) -> None:
        self._apply_target_pose(self._pregrasp_anchor_pose(self._target_position()), dwell=2)
        for _ in range(16):
            self._servo_step_to_target(
                forward_gain=1.05,
                vertical_gain=0.85,
                yaw_gain=0.95,
                forward_clip=(-0.10, 0.04),
                shoulder_clip=(-0.05, 0.05),
                wrist_clip=(-0.05, 0.05),
                gripper_open=0.86,
                dwell=1,
            )
            if self.pregrasp_ready():
                break

    def _execute_grasp(self) -> None:
        for _ in range(3):
            self._servo_step_to_target(
                forward_gain=0.48,
                vertical_gain=0.50,
                yaw_gain=0.55,
                forward_clip=(-0.03, 0.015),
                shoulder_clip=(-0.025, 0.025),
                wrist_clip=(-0.025, 0.025),
                gripper_open=0.82,
                dwell=1,
            )
        for _ in range(2):
            q_target = self.data.qpos[:6].copy()
            q_target[1] -= 0.035
            q_target[2] -= 0.045
            q_target[3] += 0.018
            self._apply_target_pose(q_target, dwell=1)
        q_target = self.data.qpos[:6].copy()
        q_target[5] = 0.18
        self._apply_target_pose(q_target, dwell=2)
        if self.ee_target_distance() < 0.24 and self.center_error_px() < 55.0:
            self.object_attached = True
            self.lifted = False
            self._update_attached_object_pose()

    def _execute_lift(self) -> None:
        q_target = self._carry_pose(self._dropzone_position())
        q_target[5] = 0.18
        self._apply_target_pose(q_target, dwell=3)
        if self.object_attached:
            self.lifted = True

    def _execute_transport(self) -> None:
        destination = self._dropzone_position()
        hover = destination + np.asarray([0.0, 0.0, 0.11], dtype=np.float64)
        self._apply_target_pose(self._carry_pose(destination), dwell=1)
        for _ in range(6):
            self._servo_step_to_point(
                hover,
                forward_gain=0.92,
                vertical_gain=0.82,
                yaw_gain=0.90,
                forward_clip=(-0.08, 0.05),
                shoulder_clip=(-0.05, 0.05),
                wrist_clip=(-0.04, 0.04),
                gripper_open=0.18,
                dwell=1,
            )
            if self._dropzone_xy_distance() < 0.10:
                break
        self._apply_target_pose(self._dropzone_pose(), dwell=1)

    def _execute_place(self) -> None:
        destination = self._dropzone_position()
        hover = destination + np.asarray([0.0, 0.0, 0.11], dtype=np.float64)
        place_point = destination + np.asarray([0.0, 0.0, 0.055], dtype=np.float64)
        self._apply_target_pose(self._dropzone_pose(), dwell=1)
        for _ in range(3):
            self._servo_step_to_point(
                hover,
                forward_gain=0.70,
                vertical_gain=0.70,
                yaw_gain=0.80,
                forward_clip=(-0.05, 0.03),
                shoulder_clip=(-0.04, 0.04),
                wrist_clip=(-0.03, 0.03),
                gripper_open=0.18,
                dwell=1,
            )
        for _ in range(4):
            self._servo_step_to_point(
                place_point,
                forward_gain=0.60,
                vertical_gain=0.78,
                yaw_gain=0.75,
                forward_clip=(-0.04, 0.02),
                shoulder_clip=(-0.03, 0.03),
                wrist_clip=(-0.03, 0.03),
                gripper_open=0.18,
                dwell=1,
            )
            if self._dropzone_xy_distance() < 0.09:
                break
        place_distance = self.dropzone_distance()
        place_xy_distance = self._dropzone_xy_distance()
        if self.object_attached and place_distance < 0.28 and place_xy_distance < 0.16:
            self.object_attached = False
            self.lifted = True
            self.placed = True
            target_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
            self.model.body_pos[target_body] = self._dropzone_position() + np.asarray([0.0, 0.0, 0.035], dtype=np.float64)
            mujoco.mj_forward(self.model, self.data)
        q_target = self.data.qpos[:6].copy()
        q_target[5] = 1.10
        self._apply_target_pose(q_target, dwell=2)
        q_target = self._carry_pose(destination)
        q_target[5] = 1.10
        self._apply_target_pose(q_target, dwell=1)

    def _execute_abort(self) -> None:
        self._apply_target_pose(HOME_QPOS, dwell=2)

    def _execute_primitive(self, primitive_id_value: int) -> None:
        if primitive_id_value in (OBS_LEFT_ID, OBS_RIGHT_ID, OBS_CENTER_ID):
            self._execute_observe(primitive_id_value)
            return
        if primitive_id_value == VERIFY_TARGET_ID:
            self._execute_verify()
            return
        if primitive_id_value == PREALIGN_GRASP_ID:
            self._execute_prealign()
            return
        if primitive_id_value == APPROACH_COARSE_ID:
            self._execute_approach(fine=False)
            return
        if primitive_id_value == APPROACH_FINE_ID:
            self._execute_approach(fine=True)
            return
        if primitive_id_value == RETREAT_ID:
            self._execute_retreat()
            return
        if primitive_id_value == REOBSERVE_ID:
            self._execute_observe(OBS_CENTER_ID)
            return
        if primitive_id_value == PREGRASP_SERVO_ID:
            self._execute_pregrasp_servo()
            return
        if primitive_id_value == GRASP_EXECUTE_ID:
            self._execute_grasp()
            return
        if primitive_id_value == LIFT_OBJECT_ID:
            self._execute_lift()
            return
        if primitive_id_value == TRANSPORT_TO_DROPZONE_ID:
            self._execute_transport()
            return
        if primitive_id_value == PLACE_OBJECT_ID:
            self._execute_place()
            return
        if primitive_id_value == ABORT_ID:
            self._execute_abort()
            return
        raise KeyError(primitive_id_value)

    def step(self, action: int | dict[str, int] | str) -> tuple[dict[str, np.ndarray], float, bool, dict]:
        primitive_id_value = primitive_action(action)
        if primitive_id_value not in allowed_primitives(task_spec(self.task_name).level):
            primitive_id_value = ABORT_ID
        obs = self._observation()
        self._execute_primitive(primitive_id_value)
        self.step_idx += 1
        self.last_primitive = primitive_id_value
        next_obs = self._observation()
        success = self.task_success()
        done = bool(success or self.step_idx >= self.cfg["episode_horizon"] or primitive_id_value == ABORT_ID)
        reward = self._reward(success)
        info = {
            "task": self.task_name,
            "success": success,
            "visibility": self.visibility_score(),
            "center_error": self.center_error_px(),
            "primitive_id": primitive_id_value,
            "primitive_name": primitive_name(primitive_id_value),
            "verified": int(self.verified),
            "grasped": int(self.object_attached),
            "lifted": int(self.lifted),
            "placed": int(self.placed),
            "ee_target_distance": self.ee_target_distance(),
            "dropzone_distance": self.dropzone_distance(),
            "context": context_vector(self.context),
            "transition": Transition(
                image=obs["image"],
                state=obs["state"],
                primitive_id=primitive_id_value,
                next_image=next_obs["image"],
                next_state=next_obs["state"],
                task_id=TASK_TO_ID[self.task_name],
                success=success,
                context=context_vector(self.context),
            ),
        }
        return next_obs, reward, done, info

    def _reward(self, success: int) -> float:
        if success:
            return 1.0
        if self.task_name == "level3_pick_place" and self.object_attached:
            return 0.15
        return -0.01

    def idle_step(self, frames: int = 1) -> dict[str, np.ndarray]:
        for _ in range(max(1, frames)):
            mujoco.mj_step(self.model, self.data)
            if self.object_attached:
                self._update_attached_object_pose()
        return self._observation()

    def close(self) -> None:
        self.obs_renderer.close()
        self.debug_renderer.close()

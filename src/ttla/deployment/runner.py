from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch

from ..models import TTLAModel
from ..sim.skills import HOME_QPOS, primitive_name
from ..task_runtime import build_runtime_state
from ..utils.io import ensure_dir, write_json
from .camera import USBCamera
from .primitives import PrimitiveExecutor
from .roarm_serial import RoArmSerialClient


class DeploymentRunner:
    def __init__(self, deploy_cfg: dict, model: TTLAModel | None = None, device: str = "cpu") -> None:
        self.cfg = deploy_cfg
        self.camera = USBCamera(**deploy_cfg["camera"])
        self.robot = RoArmSerialClient(**deploy_cfg["serial"])
        self.executor = PrimitiveExecutor(self.robot, deploy_cfg.get("runtime", {}))
        self.log_dir = ensure_dir(deploy_cfg["runtime"]["log_dir"])
        self.model = model
        self.device = torch.device(device)

    def run_probe_episode(self, episode_name: str = "probe_episode") -> Path:
        episode_dir = ensure_dir(self.log_dir / episode_name)
        if self.cfg["safety"]["reset_before_episode"]:
            self.robot.reset_pose()
            time.sleep(2.0)
        for step in range(self.cfg["runtime"]["probe_steps"]):
            frame = self.camera.read()
            cv2.imwrite(str(episode_dir / f"frame_{step:03d}.png"), frame)
            primitive_id = 0 if step % 2 == 0 else 1
            result = self.executor.run(primitive_id)
            write_json(episode_dir / f"step_{step:03d}.json", result.info)
        write_json(episode_dir / "meta.json", {"mode": "probe", "timestamp": time.time()})
        return episode_dir

    def run_primitive_sequence(self, primitive_ids: list[int], episode_name: str = "primitive_sequence") -> Path:
        episode_dir = ensure_dir(self.log_dir / episode_name)
        self.robot.reset_pose()
        time.sleep(1.5)
        for step, primitive_id in enumerate(primitive_ids):
            frame = self.camera.read()
            cv2.imwrite(str(episode_dir / f"frame_before_{step:03d}.png"), frame)
            result = self.executor.run(primitive_id)
            frame_after = self.camera.read()
            cv2.imwrite(str(episode_dir / f"frame_after_{step:03d}.png"), frame_after)
            write_json(
                episode_dir / f"step_{step:03d}.json",
                {
                    "primitive_id": primitive_id,
                    "primitive_name": primitive_name(primitive_id),
                    **result.info,
                },
            )
            if result.done:
                break
        return episode_dir

    def run_policy_episode(self, task_id: int, episode_name: str = "policy_episode") -> Path:
        if self.model is None:
            raise ValueError("DeploymentRunner.run_policy_episode requires a loaded model.")
        episode_dir = ensure_dir(self.log_dir / episode_name)
        self.robot.reset_pose()
        time.sleep(1.5)
        current_q = HOME_QPOS.copy()
        runtime_state = self.model.init_runtime_state(batch_size=1, device=self.device)
        for step in range(self.cfg["runtime"].get("max_steps", 8)):
            frame = self.camera.read()
            state = build_runtime_state(current_q=current_q, attached=False, verified=False, task_id=task_id, step_idx=step, horizon=self.cfg["runtime"].get("max_steps", 8))
            image_t = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
            state_t = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
            task_ids = torch.tensor([task_id], dtype=torch.long, device=self.device)
            with torch.no_grad():
                primitive_tensor, runtime_state, _ = self.model.act(image_t, state_t, runtime_state, use_adapter=True, task_ids=task_ids)
                primitive_id = int(primitive_tensor.item())
            result = self.executor.run(primitive_id)
            current_q = self.executor.current_q.copy()
            cv2.imwrite(str(episode_dir / f"frame_{step:03d}.png"), frame)
            write_json(
                episode_dir / f"step_{step:03d}.json",
                {"primitive_id": primitive_id, "primitive_name": primitive_name(primitive_id), **result.info},
            )
            if result.done:
                break
        return episode_dir

    def close(self) -> None:
        self.camera.close()
        self.robot.close()

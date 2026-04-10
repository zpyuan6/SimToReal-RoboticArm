from __future__ import annotations

import time
from pathlib import Path

import cv2

from ..utils.io import ensure_dir, write_json
from .camera import USBCamera
from .roarm_serial import RoArmSerialClient


class DeploymentRunner:
    def __init__(self, deploy_cfg: dict) -> None:
        self.cfg = deploy_cfg
        self.camera = USBCamera(**deploy_cfg["camera"])
        self.robot = RoArmSerialClient(**deploy_cfg["serial"])
        self.log_dir = ensure_dir(deploy_cfg["runtime"]["log_dir"])

    def run_probe_episode(self, episode_name: str = "probe_episode") -> Path:
        episode_dir = ensure_dir(self.log_dir / episode_name)
        if self.cfg["safety"]["reset_before_episode"]:
            self.robot.reset_pose()
            time.sleep(2.0)
        for step in range(self.cfg["runtime"]["probe_steps"]):
            frame = self.camera.read()
            cv2.imwrite(str(episode_dir / f"frame_{step:03d}.png"), frame)
            if step % 2 == 0:
                self.robot.move_joints(0.1, 0.0, 1.4, -0.15, 0.20, 2.90)
            else:
                self.robot.move_joints(-0.1, 0.0, 1.4, 0.15, -0.20, 2.60)
            time.sleep(0.8)
        write_json(episode_dir / "meta.json", {"mode": "probe", "timestamp": time.time()})
        return episode_dir

    def close(self) -> None:
        self.camera.close()
        self.robot.close()

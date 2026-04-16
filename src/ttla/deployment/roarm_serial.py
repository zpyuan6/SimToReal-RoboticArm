from __future__ import annotations

import json
import time

import serial

import numpy as np


class RoArmSerialClient:
    def __init__(self, port: str, baudrate: int, timeout: float = 0.2) -> None:
        self.serial = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
        self.last_command: str | None = None
        time.sleep(1.0)

    def send(self, payload: dict) -> str:
        message = json.dumps(payload, separators=(",", ":")) + "\n"
        self.serial.write(message.encode("utf-8"))
        self.serial.flush()
        self.last_command = message.strip()
        return self.last_command

    def reset_pose(self) -> None:
        self.send({"T": 100})

    def request_feedback(self) -> str | None:
        self.send({"T": 105})
        time.sleep(0.05)
        return self.read_line()

    def move_joints(self, b: float, s: float, e: float, t: float = 0.0, r: float = 0.0, h: float = 3.14) -> None:
        angles_deg = np.rad2deg(np.asarray([b, s, e, t, r, h], dtype=float))
        self.send(
            {
                "T": 122,
                "b": float(angles_deg[0]),
                "s": float(angles_deg[1]),
                "e": float(angles_deg[2]),
                "t": float(angles_deg[3]),
                "r": float(angles_deg[4]),
                "h": float(angles_deg[5]),
                "spd": 10,
                "acc": 10,
            }
        )

    def move_joint_vector(self, joints: np.ndarray) -> None:
        q = np.asarray(joints, dtype=float).reshape(-1)
        if q.size != 6:
            raise ValueError("Expected 6 joint values for RoArm primitive execution.")
        self.move_joints(q[0], q[1], q[2], q[3], q[4], q[5])

    def read_line(self) -> str | None:
        raw = self.serial.readline()
        if not raw:
            return None
        return raw.decode("utf-8", errors="replace").strip()

    def close(self) -> None:
        self.serial.close()

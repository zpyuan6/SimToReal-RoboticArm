from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ttla.config import load_config
from ttla.deployment.camera import USBCamera
from ttla.deployment.roarm_serial import RoArmSerialClient
from ttla.utils.io import ensure_dir


WINDOW_NAME = "RoArm Live Monitor"
DEFAULT_TARGET = {"b": 0.0, "s": 0.0, "e": 1.4}
JOINT_LIMITS = {
    "b": (-1.0, 1.0),
    "s": (-0.7, 0.8),
    "e": (0.6, 1.8),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--camera-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _clamp(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


def _apply_delta(targets: dict[str, float], key: str, delta: float) -> None:
    low, high = JOINT_LIMITS[key]
    targets[key] = _clamp(targets[key] + delta, low, high)


def _append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _snapshot_path(session_dir: Path, count: int) -> Path:
    return session_dir / f"snapshot_{count:03d}.png"


def _draw_panel(
    frame: np.ndarray,
    session_dir: Path,
    targets: dict[str, float],
    last_command: str,
    last_feedback: str,
    serial_enabled: bool,
    snapshot_count: int,
) -> np.ndarray:
    frame = cv2.resize(frame, (640, 480))
    cv2.putText(frame, "USB Camera", (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (25, 25, 25), 2)

    panel = np.full((480, 360, 3), 248, dtype=np.uint8)
    lines = [
        f"Serial: {'enabled' if serial_enabled else 'camera-only'}",
        f"Base target: {targets['b']:.2f}",
        f"Shoulder target: {targets['s']:.2f}",
        f"Elbow target: {targets['e']:.2f}",
        f"Snapshots: {snapshot_count}",
        f"Session: {session_dir.name}",
        "Controls:",
        "a/d base -, +",
        "w/s shoulder -, +",
        "z/x elbow -, +",
        "f feedback, r reset, p snapshot, q quit",
    ]
    y = 30
    for line in lines:
        cv2.putText(panel, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (35, 35, 35), 1)
        y += 24

    cv2.putText(panel, "Last command", (12, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (35, 35, 35), 1)
    cv2.putText(panel, last_command[:44], (12, 346), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (55, 55, 55), 1)
    cv2.putText(panel, last_command[44:88], (12, 366), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (55, 55, 55), 1)

    cv2.putText(panel, "Last feedback", (12, 404), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (35, 35, 35), 1)
    cv2.putText(panel, last_feedback[:44], (12, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (55, 55, 55), 1)
    cv2.putText(panel, last_feedback[44:88], (12, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (55, 55, 55), 1)
    return np.concatenate([frame, panel], axis=1)


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    session_dir = ensure_dir(Path(cfg["runtime"]["log_dir"]) / f"live_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    log_path = session_dir / "monitor_log.jsonl"

    if args.dry_run:
        print(f"session_dir={session_dir}")
        return

    camera = USBCamera(**cfg["camera"])
    robot = None if args.camera_only else RoArmSerialClient(**cfg["serial"])
    serial_enabled = robot is not None
    targets = dict(DEFAULT_TARGET)
    last_command = "none"
    last_feedback = "none"
    snapshot_count = 0
    delay = max(1, int(1000 / max(args.fps, 1e-3)))

    try:
        while True:
            frame = camera.read()
            dashboard = _draw_panel(frame, session_dir, targets, last_command, last_feedback, serial_enabled, snapshot_count)
            cv2.imshow(WINDOW_NAME, dashboard)
            key = cv2.waitKey(delay) & 0xFF

            if serial_enabled:
                unsolicited = robot.read_line()
                if unsolicited:
                    last_feedback = unsolicited
                    _append_jsonl(log_path, {"ts": time.time(), "event": "serial_in", "payload": unsolicited})

            if key == 255:
                continue
            if key == ord("q"):
                break
            if key == ord("p"):
                path = _snapshot_path(session_dir, snapshot_count)
                cv2.imwrite(str(path), frame)
                snapshot_count += 1
                _append_jsonl(log_path, {"ts": time.time(), "event": "snapshot", "path": str(path)})
                continue
            if key == ord("r"):
                targets = dict(DEFAULT_TARGET)
                if serial_enabled:
                    last_command = robot.send({"T": 100})
                    _append_jsonl(log_path, {"ts": time.time(), "event": "command", "payload": last_command})
                continue
            if key == ord("f") and serial_enabled:
                feedback = robot.request_feedback()
                if feedback:
                    last_feedback = feedback
                    _append_jsonl(log_path, {"ts": time.time(), "event": "feedback", "payload": feedback})
                continue

            changed = False
            if key == ord("a"):
                _apply_delta(targets, "b", -0.10)
                changed = True
            elif key == ord("d"):
                _apply_delta(targets, "b", 0.10)
                changed = True
            elif key == ord("w"):
                _apply_delta(targets, "s", -0.08)
                changed = True
            elif key == ord("s"):
                _apply_delta(targets, "s", 0.08)
                changed = True
            elif key == ord("z"):
                _apply_delta(targets, "e", -0.08)
                changed = True
            elif key == ord("x"):
                _apply_delta(targets, "e", 0.08)
                changed = True

            if changed and serial_enabled:
                robot.move_joints(
                    targets["b"],
                    targets["s"],
                    targets["e"],
                    0.0,
                    0.0,
                    3.14,
                )
                last_command = robot.last_command or "none"
                _append_jsonl(
                    log_path,
                    {
                        "ts": time.time(),
                        "event": "command",
                        "payload": last_command,
                        "targets": targets,
                    },
                )
    finally:
        camera.close()
        if robot is not None:
            robot.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

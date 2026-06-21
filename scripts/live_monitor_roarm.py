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
    # RoArm base is documented over roughly -180..180 deg.
    "b": (-float(np.pi), float(np.pi)),
    # RoArm shoulder is documented over roughly -90..90 deg.
    "s": (-float(np.pi / 2.0), float(np.pi / 2.0)),
    # RoArm elbow is documented over roughly 0..180 deg; keep the manual
    # monitor aligned with that range instead of the older narrow debug band.
    "e": (0.0, float(np.pi)),
}
FEEDBACK_KEYS = ("b", "s", "e", "t", "r", "h")


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


def _parse_feedback_joints(payload: str) -> dict[str, float] | None:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    joints: dict[str, float] = {}
    for key in FEEDBACK_KEYS:
        value = data.get(key)
        if isinstance(value, (int, float)):
            joints[key] = float(value)
    if "e" in joints:
        # The RoArm feedback for elbow is reported in the opposite sense from
        # the command-side angle convention we use in deployment poses. Convert
        # it here so the live monitor shows angles in the same convention that
        # primitive constants use.
        joints["e"] = 180.0 - joints["e"]
    return joints or None


def _snapshot_path(session_dir: Path, count: int) -> Path:
    return session_dir / f"snapshot_{count:03d}.png"


def _draw_panel(
    frame: np.ndarray,
    session_dir: Path,
    targets: dict[str, float],
    last_command: str,
    last_feedback_joints: dict[str, float] | None,
    serial_enabled: bool,
    snapshot_count: int,
) -> np.ndarray:
    frame = cv2.resize(frame, (640, 480))
    cv2.putText(frame, "USB Camera", (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (25, 25, 25), 2)

    panel = np.full((480, 360, 3), 248, dtype=np.uint8)
    lines = [
        f"Serial: {'enabled' if serial_enabled else 'camera-only'}",
        f"Base target: {np.rad2deg(targets['b']):.1f} deg",
        f"Shoulder target: {np.rad2deg(targets['s']):.1f} deg",
        f"Elbow target: {np.rad2deg(targets['e']):.1f} deg",
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

    if last_feedback_joints:
        feedback_lines = [
            f"Feedback b: {last_feedback_joints.get('b', float('nan')):.1f} deg",
            f"Feedback s: {last_feedback_joints.get('s', float('nan')):.1f} deg",
            f"Feedback e: {last_feedback_joints.get('e', float('nan')):.1f} deg",
        ]
        for line in feedback_lines:
            cv2.putText(panel, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (24, 92, 56), 1)
            y += 22
    else:
        cv2.putText(panel, "Feedback b/s/e: unavailable", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (120, 120, 120), 1)
        y += 22

    cv2.putText(panel, "Last command", (12, 372), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (35, 35, 35), 1)
    cv2.putText(panel, last_command[:44], (12, 398), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (55, 55, 55), 1)
    cv2.putText(panel, last_command[44:88], (12, 418), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (55, 55, 55), 1)
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
    last_feedback_joints: dict[str, float] | None = None
    snapshot_count = 0
    delay = max(1, int(1000 / max(args.fps, 1e-3)))

    try:
        while True:
            frame = camera.read()
            dashboard = _draw_panel(
                frame,
                session_dir,
                targets,
                last_command,
                last_feedback_joints,
                serial_enabled,
                snapshot_count,
            )
            cv2.imshow(WINDOW_NAME, dashboard)
            key = cv2.waitKey(delay) & 0xFF

            if serial_enabled:
                unsolicited = robot.read_line()
                if unsolicited:
                    last_feedback = unsolicited
                    parsed_feedback = _parse_feedback_joints(unsolicited)
                    if parsed_feedback:
                        last_feedback_joints = parsed_feedback
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
                    parsed_feedback = _parse_feedback_joints(feedback)
                    if parsed_feedback:
                        last_feedback_joints = parsed_feedback
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

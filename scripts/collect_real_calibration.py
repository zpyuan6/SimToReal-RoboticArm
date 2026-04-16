from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from ttla.config import load_config
from ttla.deployment import DeploymentRunner
from ttla.sim.skills import primitive_name
from ttla.task_runtime import build_runtime_state
from ttla.utils.io import ensure_dir, save_npz, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--deploy-config", default="configs/deployment.yaml")
    parser.add_argument("--output", default="data/real/calibration.npz")
    parser.add_argument("--primitives", default="0,1,2,3")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--task-id", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config(args.config)
    deploy_cfg = load_config(args.deploy_config)
    runner = DeploymentRunner(deploy_cfg)
    primitive_ids = [int(item.strip()) for item in args.primitives.split(",") if item.strip()]
    output = Path(args.output)
    ensure_dir(output.parent)
    session_dir = ensure_dir(output.parent / f"{output.stem}_session")
    frames_dir = ensure_dir(session_dir / "frames")
    images = []
    states = []
    next_images = []
    next_states = []
    recorded_primitives = []
    recorded_tasks = []
    timestamps = []
    primitive_names = []
    frame_paths = []
    next_frame_paths = []
    frame_size: tuple[int, int] | None = None

    def _write_frame(filename: str, frame: np.ndarray) -> str:
        path = frames_dir / filename
        cv2.imwrite(str(path), frame)
        return str(path)

    try:
        current_q = runner.executor.current_q.copy()
        for repeat in range(args.repeats):
            if deploy_cfg.get("safety", {}).get("reset_before_episode", True):
                runner.robot.reset_pose()
                time.sleep(1.5)
                current_q = runner.executor.current_q.copy()
            for step, primitive_id in enumerate(primitive_ids):
                frame = runner.camera.read()
                if frame_size is None:
                    frame_size = (int(frame.shape[1]), int(frame.shape[0]))
                state = build_runtime_state(current_q=current_q, attached=False, verified=False, task_id=args.task_id, step_idx=step, horizon=len(primitive_ids))
                result = runner.executor.run(primitive_id)
                current_q = runner.executor.current_q.copy()
                next_frame = runner.camera.read()
                next_state = build_runtime_state(current_q=current_q, attached=False, verified=False, task_id=args.task_id, step_idx=step + 1, horizon=len(primitive_ids))
                sample_idx = len(recorded_primitives)
                primitive_label = primitive_name(primitive_id)
                before_rel = _write_frame(
                    f"sample_{sample_idx:04d}_repeat_{repeat:03d}_step_{step:03d}_before_{primitive_label}.jpg",
                    frame,
                )
                after_rel = _write_frame(
                    f"sample_{sample_idx:04d}_repeat_{repeat:03d}_step_{step:03d}_after_{primitive_label}.jpg",
                    next_frame,
                )
                images.append(frame)
                states.append(state)
                next_images.append(next_frame)
                next_states.append(next_state)
                recorded_primitives.append(primitive_id)
                recorded_tasks.append(args.task_id)
                timestamps.append(time.time())
                primitive_names.append(primitive_label)
                frame_paths.append(before_rel)
                next_frame_paths.append(after_rel)
                if result.done:
                    break
    finally:
        runner.close()
    save_npz(
        output,
        images=np.asarray(images, dtype=np.uint8),
        states=np.asarray(states, dtype=np.float32),
        primitive_ids=np.asarray(recorded_primitives, dtype=np.int64),
        next_images=np.asarray(next_images, dtype=np.uint8),
        next_states=np.asarray(next_states, dtype=np.float32),
        tasks=np.asarray(recorded_tasks, dtype=np.int64),
    )
    metadata = {
        "created_at": time.time(),
        "output_npz": str(output),
        "session_dir": str(session_dir),
        "task_id": args.task_id,
        "repeats": args.repeats,
        "primitives": primitive_ids,
        "primitive_names": primitive_names,
        "timestamps": timestamps,
        "frame_paths": frame_paths,
        "next_frame_paths": next_frame_paths,
        "frame_size": None if frame_size is None else {"width": frame_size[0], "height": frame_size[1]},
    }
    write_json(session_dir / "meta.json", metadata)

    if images and frame_size is not None:
        video_path = session_dir / "preview.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            4.0,
            frame_size,
        )
        try:
            for before, after, primitive_label in zip(images, next_images, primitive_names):
                before_annotated = before.copy()
                after_annotated = after.copy()
                cv2.putText(before_annotated, f"before: {primitive_label}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(after_annotated, f"after: {primitive_label}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                writer.write(before_annotated)
                writer.write(after_annotated)
        finally:
            writer.release()
    print(f"saved_real_calibration={output}")
    print(f"saved_session_dir={session_dir}")


if __name__ == "__main__":
    main()

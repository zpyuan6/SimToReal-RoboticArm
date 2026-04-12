from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from ttla.config import load_config
from ttla.deployment import DeploymentRunner
from ttla.task_runtime import build_runtime_state
from ttla.utils.io import ensure_dir, save_npz


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
    images = []
    states = []
    next_images = []
    next_states = []
    recorded_primitives = []
    recorded_tasks = []
    try:
        current_q = runner.executor.current_q.copy()
        for repeat in range(args.repeats):
            for step, primitive_id in enumerate(primitive_ids):
                frame = runner.camera.read()
                state = build_runtime_state(current_q=current_q, attached=False, verified=False, task_id=args.task_id, step_idx=step, horizon=len(primitive_ids))
                result = runner.executor.run(primitive_id)
                current_q = runner.executor.current_q.copy()
                next_frame = runner.camera.read()
                next_state = build_runtime_state(current_q=current_q, attached=False, verified=False, task_id=args.task_id, step_idx=step + 1, horizon=len(primitive_ids))
                images.append(frame)
                states.append(state)
                next_images.append(next_frame)
                next_states.append(next_state)
                recorded_primitives.append(primitive_id)
                recorded_tasks.append(args.task_id)
                if result.done:
                    break
    finally:
        runner.close()
    output = Path(args.output)
    ensure_dir(output.parent)
    save_npz(
        output,
        images=np.asarray(images, dtype=np.uint8),
        states=np.asarray(states, dtype=np.float32),
        primitive_ids=np.asarray(recorded_primitives, dtype=np.int64),
        next_images=np.asarray(next_images, dtype=np.uint8),
        next_states=np.asarray(next_states, dtype=np.float32),
        tasks=np.asarray(recorded_tasks, dtype=np.int64),
    )
    print(f"saved_real_calibration={output}")


if __name__ == "__main__":
    main()

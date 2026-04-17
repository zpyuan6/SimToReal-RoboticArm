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


def _prompt_episode_label(attempt_idx: int, success_count: int, target_successes: int) -> str:
    while True:
        answer = input(
            f"[attempt {attempt_idx}] success count {success_count}/{target_successes} | "
            "Mark episode as success (y), failure (n), redo (r), or quit (q): "
        ).strip().lower()
        if answer in {"y", "n", "r", "q"}:
            return answer
        print("Please enter one of: y / n / r / q")


def _append_episode(container: dict[str, list], episode: dict[str, object]) -> None:
    transitions = int(episode["transition_count"])
    for key in ("images", "states", "next_images", "next_states", "primitive_ids", "tasks"):
        values = episode[key]
        if transitions:
            container[key].extend(values)  # type: ignore[arg-type]
    container["episodes"].append(
        {
            "attempt_idx": episode["attempt_idx"],
            "transition_count": transitions,
            "primitive_names": episode["primitive_names"],
            "primitive_ids": episode["primitive_ids"],
            "timestamps": episode["timestamps"],
            "frame_paths": episode["frame_paths"],
            "next_frame_paths": episode["next_frame_paths"],
            "step_results": episode["step_results"],
            "label": episode["label"],
        }
    )


def _save_dataset(output_path: Path, session_dir: Path, dataset: dict[str, list], metadata: dict[str, object]) -> None:
    ensure_dir(output_path.parent)
    images = np.asarray(dataset["images"], dtype=np.uint8)
    states = np.asarray(dataset["states"], dtype=np.float32)
    primitive_ids = np.asarray(dataset["primitive_ids"], dtype=np.int64)
    next_images = np.asarray(dataset["next_images"], dtype=np.uint8)
    next_states = np.asarray(dataset["next_states"], dtype=np.float32)
    tasks = np.asarray(dataset["tasks"], dtype=np.int64)
    save_npz(
        output_path,
        images=images,
        states=states,
        primitive_ids=primitive_ids,
        next_images=next_images,
        next_states=next_states,
        tasks=tasks,
    )
    write_json(session_dir / "meta.json", metadata)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--deploy-config", default="configs/deployment_l3.yaml")
    parser.add_argument("--output-prefix", default="data/real/v3_pick_place_p1_center")
    parser.add_argument("--primitives", default="8,9,10,11,12,13")
    parser.add_argument("--task-id", type=int, default=2)
    parser.add_argument("--target-successes", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=12)
    args = parser.parse_args()

    cfg = load_config(args.config)
    deploy_cfg = load_config(args.deploy_config)
    runner = DeploymentRunner(deploy_cfg)
    primitive_ids = [int(item.strip()) for item in args.primitives.split(",") if item.strip()]

    output_prefix = Path(args.output_prefix)
    ensure_dir(output_prefix.parent)
    session_root = ensure_dir(output_prefix.parent / f"{output_prefix.stem}_session")
    frames_dir = ensure_dir(session_root / "frames")
    success_session_dir = ensure_dir(session_root / "success")
    failure_session_dir = ensure_dir(session_root / "failure")

    success_dataset: dict[str, list] = {
        "images": [],
        "states": [],
        "next_images": [],
        "next_states": [],
        "primitive_ids": [],
        "tasks": [],
        "episodes": [],
    }
    failure_dataset: dict[str, list] = {
        "images": [],
        "states": [],
        "next_images": [],
        "next_states": [],
        "primitive_ids": [],
        "tasks": [],
        "episodes": [],
    }

    frame_size: tuple[int, int] | None = None
    success_count = 0
    attempt_count = 0

    def _write_frame(filename: str, frame: np.ndarray) -> str:
        path = frames_dir / filename
        cv2.imwrite(str(path), frame)
        return str(path)

    try:
        while success_count < args.target_successes and attempt_count < args.max_attempts:
            attempt_count += 1
            print(f"Starting L3 attempt {attempt_count} / target successes {args.target_successes}")
            if deploy_cfg.get("safety", {}).get("reset_before_episode", True):
                runner.robot.reset_pose()
                time.sleep(1.5)
            current_q = runner.executor.current_q.copy()

            episode = {
                "attempt_idx": attempt_count,
                "images": [],
                "states": [],
                "next_images": [],
                "next_states": [],
                "primitive_ids": [],
                "tasks": [],
                "primitive_names": [],
                "timestamps": [],
                "frame_paths": [],
                "next_frame_paths": [],
                "step_results": [],
                "transition_count": 0,
                "label": "",
            }

            for step, primitive_id in enumerate(primitive_ids):
                frame = runner.camera.read()
                if frame_size is None:
                    frame_size = (int(frame.shape[1]), int(frame.shape[0]))
                state = build_runtime_state(
                    current_q=current_q,
                    attached=False,
                    verified=False,
                    task_id=args.task_id,
                    step_idx=step,
                    horizon=len(primitive_ids),
                )
                result = runner.executor.run(primitive_id)
                current_q = runner.executor.current_q.copy()
                next_frame = runner.camera.read()
                next_state = build_runtime_state(
                    current_q=current_q,
                    attached=False,
                    verified=False,
                    task_id=args.task_id,
                    step_idx=step + 1,
                    horizon=len(primitive_ids),
                )
                sample_idx = len(success_dataset["primitive_ids"]) + len(failure_dataset["primitive_ids"]) + len(episode["primitive_ids"])
                primitive_label = primitive_name(primitive_id)
                before_rel = _write_frame(
                    f"attempt_{attempt_count:03d}_step_{step:03d}_before_{primitive_label}.jpg",
                    frame,
                )
                after_rel = _write_frame(
                    f"attempt_{attempt_count:03d}_step_{step:03d}_after_{primitive_label}.jpg",
                    next_frame,
                )
                episode["images"].append(frame)
                episode["states"].append(state)
                episode["next_images"].append(next_frame)
                episode["next_states"].append(next_state)
                episode["primitive_ids"].append(primitive_id)
                episode["tasks"].append(args.task_id)
                episode["primitive_names"].append(primitive_label)
                episode["timestamps"].append(time.time())
                episode["frame_paths"].append(before_rel)
                episode["next_frame_paths"].append(after_rel)
                episode["step_results"].append({"primitive_id": primitive_id, "primitive_name": primitive_label, "done": bool(result.done)})
                episode["transition_count"] += 1
                if result.done:
                    break

            label = _prompt_episode_label(attempt_count, success_count, args.target_successes)
            if label == "q":
                print("Stopping collection early at user request.")
                break
            if label == "r":
                print("Redo requested. This attempt will not be counted.")
                continue

            episode["label"] = "success" if label == "y" else "failure"
            if label == "y":
                success_count += 1
                _append_episode(success_dataset, episode)
                print(f"Recorded success episode {success_count}/{args.target_successes}.")
            else:
                _append_episode(failure_dataset, episode)
                print("Recorded failure episode.")
    finally:
        runner.close()

    success_output = output_prefix.with_name(f"{output_prefix.stem}_success.npz")
    failure_output = output_prefix.with_name(f"{output_prefix.stem}_failure.npz")

    common_metadata = {
        "created_at": time.time(),
        "output_prefix": str(output_prefix),
        "session_root": str(session_root),
        "task_id": args.task_id,
        "primitives": primitive_ids,
        "target_successes": args.target_successes,
        "max_attempts": args.max_attempts,
        "attempts_recorded": attempt_count,
        "frame_size": None if frame_size is None else {"width": frame_size[0], "height": frame_size[1]},
    }

    _save_dataset(
        success_output,
        success_session_dir,
        success_dataset,
        {
            **common_metadata,
            "label": "success",
            "success_count": len(success_dataset["episodes"]),
            "failure_count": len(failure_dataset["episodes"]),
            "episodes": success_dataset["episodes"],
        },
    )
    _save_dataset(
        failure_output,
        failure_session_dir,
        failure_dataset,
        {
            **common_metadata,
            "label": "failure",
            "success_count": len(success_dataset["episodes"]),
            "failure_count": len(failure_dataset["episodes"]),
            "episodes": failure_dataset["episodes"],
        },
    )

    print(f"saved_l3_success={success_output}")
    print(f"saved_l3_failure={failure_output}")
    print(f"saved_l3_session_root={session_root}")
    print(f"recorded_successes={len(success_dataset['episodes'])}")
    print(f"recorded_failures={len(failure_dataset['episodes'])}")


if __name__ == "__main__":
    main()

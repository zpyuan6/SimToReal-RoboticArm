from __future__ import annotations

import argparse
import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ttla.config import load_config
from ttla.deployment import DeploymentRunner
from ttla.deployment.primitives import REAL_HOME_QPOS
from ttla.sim.skills import primitive_id, primitive_name
from ttla.sim.task_defs import TASK_TO_ID, supervision_stage_id
from ttla.task_runtime import build_runtime_state
from ttla.utils.io import ensure_dir, save_npz, write_json


@dataclass
class RuntimeFlags:
    attached: bool = False
    lifted: bool = False
    placed: bool = False

    def apply(self, primitive_idx: int) -> None:
        label = primitive_name(int(primitive_idx))
        if label == "grasp_execute":
            self.attached = True
            self.placed = False
        elif label == "lift_object":
            if self.attached:
                self.lifted = True
        elif label == "place_object":
            self.placed = True
            self.attached = False
            self.lifted = False
        elif label == "abort":
            self.attached = False
            self.lifted = False
            self.placed = False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--deploy-config", default="configs/deployment.yaml")
    parser.add_argument("--plan", default=None, help="YAML collection plan.")
    parser.add_argument("--session", default=None, help="Session key inside the plan file.")
    parser.add_argument("--output-root", default=None, help="Override output root from plan.")
    parser.add_argument("--task", default=None, choices=sorted(TASK_TO_ID.keys()))
    parser.add_argument("--split-role", default=None, choices=["calibration", "heldout", "debug"])
    parser.add_argument("--layout-tag", default=None)
    parser.add_argument("--primitives", default=None, help="Comma-separated primitive names or ids for manual mode.")
    parser.add_argument("--sequence-name", default="manual_sequence")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--operator", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--session-tag", default="")
    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument("--auto-accept", action="store_true")
    parser.add_argument("--save-preview", action="store_true")
    return parser.parse_args()


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _parse_primitive_sequence(values: list[Any] | str) -> list[int]:
    if isinstance(values, str):
        raw_items = [item.strip() for item in values.split(",") if item.strip()]
    else:
        raw_items = list(values)
    sequence: list[int] = []
    for item in raw_items:
        if isinstance(item, int):
            sequence.append(int(item))
            continue
        text = str(item).strip()
        if text.isdigit():
            sequence.append(int(text))
        else:
            sequence.append(primitive_id(text))
    return sequence


def _load_plan_session(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if args.plan is None:
        if args.task is None or args.primitives is None or args.split_role is None:
            raise ValueError("Manual mode requires --task, --primitives, and --split-role.")
        session_key = args.session_tag or f"manual_{args.task}_{_timestamp()}"
        session_spec = {
            "task": args.task,
            "split_role": args.split_role,
            "layout_tag": args.layout_tag or "unspecified",
            "repeats": args.repeats or 1,
            "auto_start": args.auto_start,
            "save_preview": args.save_preview,
            "notes": args.notes,
            "operator": args.operator,
            "sequences": [
                {
                    "name": args.sequence_name,
                    "primitives": _parse_primitive_sequence(args.primitives),
                }
            ],
        }
        return session_spec, session_key

    plan = load_config(args.plan)
    sessions = plan.get("sessions", {})
    if not sessions:
        raise ValueError(f"No sessions defined in plan: {args.plan}")
    if args.session is None:
        raise ValueError("Plan mode requires --session.")
    if args.session not in sessions:
        raise KeyError(f"Unknown session key '{args.session}' in {args.plan}")
    shared = copy.deepcopy(plan.get("shared", {}))
    session_spec = copy.deepcopy(sessions[args.session])
    merged = {**shared, **session_spec}
    merged["sequences"] = copy.deepcopy(session_spec.get("sequences", shared.get("sequences", [])))
    if not merged.get("sequences"):
        raise ValueError(f"Session '{args.session}' does not define any sequences.")
    if args.output_root:
        merged["output_root"] = args.output_root
    if args.operator:
        merged["operator"] = args.operator
    if args.notes:
        merged["notes"] = args.notes
    if args.auto_start:
        merged["auto_start"] = True
    if args.auto_accept:
        merged["auto_accept"] = True
    if args.save_preview:
        merged["save_preview"] = True
    return merged, args.session


def _expand_episodes(session_spec: dict[str, Any]) -> list[dict[str, Any]]:
    default_repeats = int(session_spec.get("repeats", 1))
    expanded: list[dict[str, Any]] = []
    for sequence in session_spec.get("sequences", []):
        repeats = int(sequence.get("repeats", default_repeats))
        primitive_ids = _parse_primitive_sequence(sequence["primitives"])
        for repeat_idx in range(repeats):
            expanded.append(
                {
                    "name": sequence["name"],
                    "repeat_idx": repeat_idx,
                    "layout_tag": sequence.get("layout_tag", session_spec.get("layout_tag", "unspecified")),
                    "notes": sequence.get("notes", ""),
                    "primitive_ids": primitive_ids,
                }
            )
    return expanded


def _prompt_continue(prompt: str) -> bool:
    while True:
        answer = input(prompt).strip().lower()
        if answer in {"", "y", "yes"}:
            return True
        if answer in {"q", "quit", "n", "no"}:
            return False
        print("Please enter Enter/y to continue or q/n to stop.")


def _prompt_accept() -> str:
    while True:
        answer = input("Keep this episode [k], redo [r], or quit [q]? ").strip().lower()
        if answer in {"", "k", "keep"}:
            return "keep"
        if answer in {"r", "redo"}:
            return "redo"
        if answer in {"q", "quit"}:
            return "quit"
        print("Please enter k / r / q.")


def _write_preview(session_dir: Path, entries: list[dict[str, Any]], frame_size: tuple[int, int]) -> None:
    if not entries:
        return
    preview_path = session_dir / "preview.mp4"
    writer = cv2.VideoWriter(
        str(preview_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        4.0,
        frame_size,
    )
    try:
        for entry in entries:
            before = cv2.imread(entry["before_path"])
            after = cv2.imread(entry["after_path"])
            if before is None or after is None:
                continue
            primitive_label = entry["primitive_name"]
            before_annotated = before.copy()
            after_annotated = after.copy()
            cv2.putText(before_annotated, f"before: {primitive_label}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(after_annotated, f"after: {primitive_label}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            writer.write(before_annotated)
            writer.write(after_annotated)
    finally:
        writer.release()


def main() -> None:
    args = _parse_args()
    session_spec, session_key = _load_plan_session(args)
    _cfg = load_config(session_spec.get("config", args.config))
    deploy_cfg = load_config(session_spec.get("deploy_config", args.deploy_config))
    deploy_cfg.setdefault("runtime", {})
    if "primitive_sleep_s" in session_spec:
        deploy_cfg["runtime"]["primitive_sleep_s"] = float(session_spec["primitive_sleep_s"])
    output_root = ensure_dir(session_spec.get("output_root", "data/real_v2/transitions"))
    session_dir = ensure_dir(output_root / f"{session_key}_{_timestamp()}")
    frames_dir = ensure_dir(session_dir / "frames")

    task_name = str(session_spec["task"])
    task_id = int(TASK_TO_ID[task_name])
    split_role = str(session_spec["split_role"])
    episodes = _expand_episodes(session_spec)
    auto_start = bool(session_spec.get("auto_start", False))
    auto_accept = bool(session_spec.get("auto_accept", False))
    save_preview = bool(session_spec.get("save_preview", False))
    reset_between_episodes = bool(session_spec.get("reset_between_episodes", deploy_cfg.get("safety", {}).get("reset_before_episode", True)))
    operator = str(session_spec.get("operator", ""))
    notes = str(session_spec.get("notes", ""))

    images: list[np.ndarray] = []
    states: list[np.ndarray] = []
    next_images: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    primitive_ids: list[int] = []
    tasks: list[int] = []
    episode_ids: list[int] = []
    step_ids: list[int] = []
    stage_ids: list[int] = []
    contexts: list[np.ndarray] = []
    success_flags: list[float] = []
    preview_entries: list[dict[str, Any]] = []
    episode_records: list[dict[str, Any]] = []
    frame_size: tuple[int, int] | None = None

    runner = DeploymentRunner(deploy_cfg)
    next_episode_id = 0

    try:
        for episode_index, episode_spec in enumerate(episodes):
            episode_name = f"{episode_spec['name']}_r{episode_spec['repeat_idx']:02d}"
            layout_tag = str(episode_spec.get("layout_tag", "unspecified"))
            primitive_sequence = list(episode_spec["primitive_ids"])
            if not auto_start:
                should_continue = _prompt_continue(
                    f"[{episode_name}] layout={layout_tag} | task={task_name} | "
                    f"steps={len(primitive_sequence)}. Press Enter to execute or q to stop: "
                )
                if not should_continue:
                    break

            if reset_between_episodes:
                runner.robot.reset_pose()
                time.sleep(1.5)
                runner.executor.current_q = REAL_HOME_QPOS.copy()
            current_q = runner.executor.current_q.copy()
            flags = RuntimeFlags()

            local_images: list[np.ndarray] = []
            local_states: list[np.ndarray] = []
            local_next_images: list[np.ndarray] = []
            local_next_states: list[np.ndarray] = []
            local_primitives: list[int] = []
            local_stage_ids: list[int] = []
            local_preview: list[dict[str, Any]] = []
            local_frame_paths: list[str] = []
            local_next_frame_paths: list[str] = []
            local_step_records: list[dict[str, Any]] = []

            for step_index, primitive_idx in enumerate(primitive_sequence):
                before = runner.camera.read()
                if frame_size is None:
                    frame_size = (int(before.shape[1]), int(before.shape[0]))
                stage_before = supervision_stage_id(task_id, primitive_idx)
                state = build_runtime_state(
                    current_q=current_q,
                    task_id=task_id,
                    step_idx=step_index,
                    horizon=len(primitive_sequence),
                )
                result = runner.executor.run(primitive_idx)
                current_q = runner.executor.current_q.copy()
                flags.apply(primitive_idx)
                after = runner.camera.read()
                next_state = build_runtime_state(
                    current_q=current_q,
                    task_id=task_id,
                    step_idx=step_index + 1,
                    horizon=len(primitive_sequence),
                )
                primitive_label = primitive_name(primitive_idx)
                before_path = frames_dir / f"episode_{episode_index:04d}_step_{step_index:03d}_before_{primitive_label}.jpg"
                after_path = frames_dir / f"episode_{episode_index:04d}_step_{step_index:03d}_after_{primitive_label}.jpg"
                cv2.imwrite(str(before_path), before)
                cv2.imwrite(str(after_path), after)
                local_images.append(before)
                local_states.append(state)
                local_next_images.append(after)
                local_next_states.append(next_state)
                local_primitives.append(int(primitive_idx))
                local_stage_ids.append(int(stage_before))
                local_frame_paths.append(str(before_path))
                local_next_frame_paths.append(str(after_path))
                local_step_records.append(
                    {
                        "step_index": step_index,
                        "primitive_id": int(primitive_idx),
                        "primitive_name": primitive_label,
                        "done": bool(result.done),
                        "result_info": result.info,
                    }
                )
                local_preview.append(
                    {
                        "before_path": str(before_path),
                        "after_path": str(after_path),
                        "primitive_name": primitive_label,
                    }
                )
                if result.done:
                    break

            decision = "keep" if auto_accept else _prompt_accept()
            if decision == "quit":
                break
            if decision == "redo":
                print(f"redo_episode={episode_name}")
                continue

            for step_index, primitive_idx in enumerate(local_primitives):
                images.append(local_images[step_index])
                states.append(local_states[step_index])
                next_images.append(local_next_images[step_index])
                next_states.append(local_next_states[step_index])
                primitive_ids.append(primitive_idx)
                tasks.append(task_id)
                episode_ids.append(next_episode_id)
                step_ids.append(step_index)
                stage_ids.append(local_stage_ids[step_index])
                contexts.append(np.zeros(8, dtype=np.float32))
                success_flags.append(0.0)
            preview_entries.extend(local_preview)
            episode_records.append(
                {
                    "episode_id": next_episode_id,
                    "episode_name": episode_name,
                    "layout_tag": layout_tag,
                    "sequence_notes": episode_spec.get("notes", ""),
                    "primitive_ids": local_primitives,
                    "primitive_names": [primitive_name(pid) for pid in local_primitives],
                    "frame_paths": local_frame_paths,
                    "next_frame_paths": local_next_frame_paths,
                    "steps_executed": len(local_primitives),
                    "task": task_name,
                }
            )
            next_episode_id += 1
    finally:
        runner.close()

    dataset_path = session_dir / "session_dataset.npz"
    camera_width = int(deploy_cfg.get("camera", {}).get("width", 640))
    camera_height = int(deploy_cfg.get("camera", {}).get("height", 480))
    if primitive_ids:
        save_npz(
            dataset_path,
            images=np.asarray(images, dtype=np.uint8),
            states=np.asarray(states, dtype=np.float32),
            primitive_ids=np.asarray(primitive_ids, dtype=np.int64),
            next_images=np.asarray(next_images, dtype=np.uint8),
            next_states=np.asarray(next_states, dtype=np.float32),
            tasks=np.asarray(tasks, dtype=np.int64),
            contexts=np.asarray(contexts, dtype=np.float32),
            success=np.asarray(success_flags, dtype=np.float32),
            episode_ids=np.asarray(episode_ids, dtype=np.int64),
            step_ids=np.asarray(step_ids, dtype=np.int64),
            stage_ids=np.asarray(stage_ids, dtype=np.int64),
        )
    else:
        save_npz(
            dataset_path,
            images=np.zeros((0, camera_height, camera_width, 3), dtype=np.uint8),
            states=np.zeros((0, 18), dtype=np.float32),
            primitive_ids=np.zeros((0,), dtype=np.int64),
            next_images=np.zeros((0, camera_height, camera_width, 3), dtype=np.uint8),
            next_states=np.zeros((0, 18), dtype=np.float32),
            tasks=np.zeros((0,), dtype=np.int64),
            contexts=np.zeros((0, 8), dtype=np.float32),
            success=np.zeros((0,), dtype=np.float32),
            episode_ids=np.zeros((0,), dtype=np.int64),
            step_ids=np.zeros((0,), dtype=np.int64),
            stage_ids=np.zeros((0,), dtype=np.int64),
        )

    meta = {
        "created_at": time.time(),
        "plan_path": args.plan,
        "config_path": session_spec.get("config", args.config),
        "deploy_config_path": session_spec.get("deploy_config", args.deploy_config),
        "primitive_sleep_s": float(deploy_cfg.get("runtime", {}).get("primitive_sleep_s", 0.8)),
        "session_key": session_key,
        "session_dir": str(session_dir),
        "session_dataset_path": str(dataset_path),
        "split_role": split_role,
        "task": task_name,
        "task_id": task_id,
        "layout_tag": session_spec.get("layout_tag", "unspecified"),
        "operator": operator,
        "notes": notes,
        "object_name": session_spec.get("object_name", ""),
        "dropzone_name": session_spec.get("dropzone_name", ""),
        "scene_constraints": session_spec.get("scene_constraints", []),
        "video_global_expectations": session_spec.get("video_global_expectations", []),
        "placement_guide": session_spec.get("placement_guide", []),
        "acceptance_criteria": session_spec.get("acceptance_criteria", []),
        "video_expectations": session_spec.get("video_expectations", []),
        "primitive_vocabulary": "legacy",
        "episodes_planned": len(episodes),
        "episodes_collected": len(episode_records),
        "transitions_collected": len(primitive_ids),
        "frame_size": None if frame_size is None else {"width": frame_size[0], "height": frame_size[1]},
        "episode_records": episode_records,
    }
    write_json(session_dir / "meta.json", meta)

    if save_preview and frame_size is not None:
        _write_preview(session_dir, preview_entries, frame_size)

    print(f"saved_transition_session={session_dir}")
    print(f"saved_transition_dataset={dataset_path}")
    print(f"episodes_collected={len(episode_records)}")
    print(f"transitions_collected={len(primitive_ids)}")


if __name__ == "__main__":
    main()

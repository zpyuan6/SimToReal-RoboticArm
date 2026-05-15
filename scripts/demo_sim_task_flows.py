from __future__ import annotations

import argparse
from pathlib import Path

from sim_demo_common import (
    TASK_PRESET_FLOWS,
    TASK_SPECS,
    add_shared_args,
    canonical_context,
    load_env,
    run_trace,
    save_contact_sheet,
    save_video,
    set_canonical_layout,
    task_output_dir,
    write_manifest,
)
from ttla.sim.skills import primitive_name


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the preset primitive flow for each task and save videos/contact sheets."
    )
    add_shared_args(parser, default_output_root="results/sim_task_flow_demos")
    parser.add_argument(
        "--tasks",
        default="level1_verify,level2_approach,level3_pick_place",
        help="Comma-separated task names.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _, env = load_env(args.config, seed=args.seed)
    output_root = Path(args.output_root)
    selected_tasks = [token.strip() for token in args.tasks.split(",") if token.strip()]
    manifest: dict[str, object] = {
        "config": args.config,
        "seed": args.seed,
        "fps": args.fps,
        "primitive_vocabulary": env.primitive_vocabulary,
        "tasks": [],
    }

    try:
        for task_name in selected_tasks:
            if task_name not in TASK_SPECS:
                raise KeyError(f"Unknown task: {task_name}")
            env.reset(task_name=task_name, context=canonical_context())
            set_canonical_layout(env, task_name)
            flow = list(TASK_PRESET_FLOWS[task_name])
            task_dir = task_output_dir(output_root, task_name)
            all_frames = []
            all_forearm_frames = []
            all_overview_frames = []
            step_summaries = []
            for step_idx, primitive_id_value in enumerate(flow, start=1):
                primitive_label = primitive_name(primitive_id_value, primitive_vocabulary=env.primitive_vocabulary)
                subtitle = f"task={task_name} | step={step_idx}/{len(flow)} | primitive={primitive_label}"
                trace = run_trace(
                    env,
                    primitive_id_value,
                    title=f"Task Flow: {task_name}",
                    subtitle=subtitle,
                    extra_lines=[
                        f"flow_step={step_idx}/{len(flow)}",
                        f"task_public_label={TASK_SPECS[task_name].public_label}",
                    ],
                )
                all_frames.extend(trace["frames"])
                all_forearm_frames.extend(trace["forearm_frames"])
                all_overview_frames.extend(trace["overview_frames"])
                info = trace["info"]
                step_summaries.append(
                    {
                        "step_index": step_idx,
                        "primitive_id": primitive_id_value,
                        "primitive_name": primitive_label,
                        "micro_steps": trace["micro_steps"],
                        "reward": trace["reward"],
                        "done": trace["done"],
                        "success": int(info["success"]),
                        "executor_primitive_id": int(info["executor_primitive_id"]),
                        "executor_primitive_name": str(info["executor_primitive_name"]),
                        "visibility": float(info["visibility"]),
                        "center_error": float(info["center_error"]),
                        "grasp_gap": float(info["grasp_gap"]),
                        "ee_target_distance": float(info["ee_target_distance"]),
                        "dropzone_distance": float(info["dropzone_distance"]),
                        "flags": {
                            "verified": int(info["verified"]),
                            "grasped": int(info["grasped"]),
                            "lifted": int(info["lifted"]),
                            "placed": int(info["placed"]),
                        },
                    }
                )
                if trace["done"]:
                    break

            final_success = int(env.task_success())
            save_video(task_dir / "dashboard_flow.mp4", all_frames, fps=args.fps)
            save_video(task_dir / "forearm_flow.mp4", all_forearm_frames, fps=args.fps)
            save_video(task_dir / "overview_flow.mp4", all_overview_frames, fps=args.fps)
            save_contact_sheet(
                task_dir / "dashboard_contact_sheet.png",
                all_frames,
                labels=["start", "mid", "final"],
            )
            save_contact_sheet(
                task_dir / "forearm_contact_sheet.png",
                all_forearm_frames,
                labels=["start", "mid", "final"],
            )
            save_contact_sheet(
                task_dir / "overview_contact_sheet.png",
                all_overview_frames,
                labels=["start", "mid", "final"],
            )
            write_manifest(
                task_dir / "meta.json",
                {
                    "config": args.config,
                    "seed": args.seed,
                    "task": task_name,
                    "task_public_label": TASK_SPECS[task_name].public_label,
                    "task_description": TASK_SPECS[task_name].description,
                    "preset_flow_ids": flow,
                    "preset_flow_names": [primitive_name(pid) for pid in flow],
                    "final_success": final_success,
                    "final_flags": {
                        "verified": int(env.verified),
                        "grasped": int(env.object_attached),
                        "lifted": int(env.lifted),
                        "placed": int(env.placed),
                    },
                    "step_summaries": step_summaries,
                    "artifacts": {
                        "dashboard_video": "dashboard_flow.mp4",
                        "forearm_video": "forearm_flow.mp4",
                        "overview_video": "overview_flow.mp4",
                        "dashboard_contact_sheet": "dashboard_contact_sheet.png",
                        "forearm_contact_sheet": "forearm_contact_sheet.png",
                        "overview_contact_sheet": "overview_contact_sheet.png",
                    },
                },
            )
            manifest["tasks"].append(
                {
                    "task": task_name,
                    "output_dir": str(task_dir),
                    "final_success": final_success,
                }
            )
    finally:
        env.close()

    write_manifest(output_root / "manifest.json", manifest)
    print(f"saved={output_root}")


if __name__ == "__main__":
    main()

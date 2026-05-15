from __future__ import annotations

import argparse
from pathlib import Path

from sim_demo_common import (
    PRIMITIVE_CANONICAL_TASK,
    PRIMITIVE_SETUP_PREFIXES,
    add_shared_args,
    canonical_context,
    load_env,
    parse_primitive_tokens,
    primitive_output_dir,
    run_trace,
    save_contact_sheet,
    save_video,
    set_canonical_layout,
    write_manifest,
)
from ttla.sim.skills import primitive_name


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render one simulated demonstration for each primitive and save videos/contact sheets."
    )
    add_shared_args(parser, default_output_root="results/sim_primitive_demos")
    parser.add_argument(
        "--primitives",
        help="Comma-separated primitive names or ids. Defaults to the full legacy primitive list.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg, env = load_env(args.config, seed=args.seed)
    output_root = Path(args.output_root)
    manifest: dict[str, object] = {
        "config": args.config,
        "seed": args.seed,
        "fps": args.fps,
        "primitive_vocabulary": env.primitive_vocabulary,
        "demos": [],
    }
    primitive_ids = parse_primitive_tokens(args.primitives, env.primitive_vocabulary)

    try:
        for primitive_id_value in primitive_ids:
            task_name = PRIMITIVE_CANONICAL_TASK[primitive_id_value]
            setup_prefix = list(PRIMITIVE_SETUP_PREFIXES[primitive_id_value])
            env.reset(task_name=task_name, context=canonical_context())
            set_canonical_layout(env, task_name)
            for setup_primitive in setup_prefix:
                env.step(setup_primitive)

            primitive_label = primitive_name(primitive_id_value, primitive_vocabulary=env.primitive_vocabulary)
            demo_dir = primitive_output_dir(output_root, primitive_id_value, env.primitive_vocabulary)
            subtitle = f"task={task_name} | setup={','.join(primitive_name(pid) for pid in setup_prefix) or 'none'}"
            trace = run_trace(
                env,
                primitive_id_value,
                title=f"Primitive Demo: {primitive_label}",
                subtitle=subtitle,
                extra_lines=[f"setup_prefix={','.join(primitive_name(pid) for pid in setup_prefix) or 'none'}"],
            )
            save_video(demo_dir / "dashboard.mp4", trace["frames"], fps=args.fps)
            save_video(demo_dir / "forearm_cam.mp4", trace["forearm_frames"], fps=args.fps)
            save_video(demo_dir / "overview_cam.mp4", trace["overview_frames"], fps=args.fps)
            save_contact_sheet(
                demo_dir / "dashboard_contact_sheet.png",
                trace["frames"],
                labels=["before", "internal", "after"],
            )
            save_contact_sheet(
                demo_dir / "forearm_contact_sheet.png",
                trace["forearm_frames"],
                labels=["before", "internal", "after"],
            )
            save_contact_sheet(
                demo_dir / "overview_contact_sheet.png",
                trace["overview_frames"],
                labels=["before", "internal", "after"],
            )
            info = trace["info"]
            write_manifest(
                demo_dir / "meta.json",
                {
                    "config": args.config,
                    "seed": args.seed,
                    "task": task_name,
                    "primitive_id": primitive_id_value,
                    "primitive_name": primitive_label,
                    "setup_prefix_ids": setup_prefix,
                    "setup_prefix_names": [primitive_name(pid) for pid in setup_prefix],
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
                    "artifacts": {
                        "dashboard_video": "dashboard.mp4",
                        "forearm_video": "forearm_cam.mp4",
                        "overview_video": "overview_cam.mp4",
                        "dashboard_contact_sheet": "dashboard_contact_sheet.png",
                        "forearm_contact_sheet": "forearm_contact_sheet.png",
                        "overview_contact_sheet": "overview_contact_sheet.png",
                    },
                },
            )
            manifest["demos"].append(
                {
                    "primitive_id": primitive_id_value,
                    "primitive_name": primitive_label,
                    "task": task_name,
                    "output_dir": str(demo_dir),
                }
            )
    finally:
        env.close()

    write_manifest(output_root / "manifest.json", manifest)
    print(f"saved={output_root}")


if __name__ == "__main__":
    main()

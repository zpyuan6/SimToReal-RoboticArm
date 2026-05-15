from __future__ import annotations

import argparse

import cv2
import mujoco.viewer

from sim_demo_common import (
    ACCENT,
    SUCCESS,
    TASK_PRESET_FLOWS,
    WARN,
    add_shared_args,
    canonical_context,
    execute_with_animation,
    load_env,
    panel_frame,
    set_canonical_layout,
)
from ttla.sim.skills import primitive_name
from ttla.sim.task_defs import TASK_SPECS


WINDOW_NAME = "TTLA Sim Task Flow Inspector"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive MuJoCo inspector for preset simulated task flows."
    )
    add_shared_args(parser, default_output_root="results/sim_task_flow_demos")
    parser.add_argument(
        "--tasks",
        default="level1_verify,level2_approach,level3_pick_place",
        help="Comma-separated task names.",
    )
    parser.add_argument("--frame-sleep-s", type=float, default=0.03)
    return parser.parse_args()


def _viewer_overlay(handle, env, task_name: str, step_cursor: int, flow: list[int], phase: str, micro_step: int) -> None:
    next_name = primitive_name(flow[step_cursor], primitive_vocabulary=env.primitive_vocabulary) if step_cursor < len(flow) else "none"
    texts = [
        (
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            "TTLA Task Flow Inspector",
            (
                f"task={task_name} | next={next_name} | flow_step={step_cursor + 1}/{len(flow)}\n"
                f"phase={phase} | internal_step={micro_step}\n"
                f"vis={env.visibility_score():.3f} center={env.center_error_px():.1f}px grasp_gap={env.grasp_gap():+.3f}m"
            ),
        ),
        (
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPRIGHT,
            "controls",
            "Space/Enter next primitive\nB step back\nR reset flow\nN next task | P previous task\nQ quit",
        ),
    ]
    handle.set_texts(texts)


def _preview(env, task_name: str, flow: list[int], step_cursor: int, executed: list[int], status: str, status_color):
    obs = env.observe()
    upcoming = primitive_name(flow[step_cursor], primitive_vocabulary=env.primitive_vocabulary) if step_cursor < len(flow) else "none"
    lines = [
        f"task={task_name}",
        f"public_label={TASK_SPECS[task_name].public_label}",
        f"executed={','.join(primitive_name(pid) for pid in executed) or 'none'}",
        f"next={upcoming}",
        f"flow={','.join(primitive_name(pid) for pid in flow)}",
        f"visibility={env.visibility_score():.3f}",
        f"center_error_px={env.center_error_px():.1f}",
        f"grasp_gap={env.grasp_gap():+.3f}",
        f"ee_target_distance={env.ee_target_distance():.3f}",
        f"flags verified={int(env.verified)} grasped={int(env.object_attached)} lifted={int(env.lifted)} placed={int(env.placed)}",
        "controls: Space next | B back | R reset | N next task | P prev task | Q quit",
    ]
    return panel_frame(
        env,
        obs,
        title=f"Task Flow Inspector: {task_name}",
        subtitle=TASK_SPECS[task_name].description,
        lines=lines,
        status_label=status,
        status_color=status_color,
    )


def main() -> None:
    args = _parse_args()
    _, env = load_env(args.config, seed=args.seed)
    selected_tasks = [token.strip() for token in args.tasks.split(",") if token.strip()]
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(WINDOW_NAME, 1320, 900)
    viewer = mujoco.viewer.launch_passive(env.model, env.data, show_left_ui=True, show_right_ui=True)

    task_index = 0
    task_name = selected_tasks[task_index]
    flow = list(TASK_PRESET_FLOWS[task_name])
    step_cursor = 0
    executed: list[int] = []
    status = "READY"
    status_color = ACCENT

    def rebuild_state(replay_steps: int = 0) -> None:
        nonlocal task_name, flow, step_cursor, executed, status, status_color
        task_name = selected_tasks[task_index]
        flow = list(TASK_PRESET_FLOWS[task_name])
        env.reset(task_name=task_name, context=canonical_context())
        set_canonical_layout(env, task_name)
        executed = []
        step_cursor = 0
        for _ in range(replay_steps):
            if step_cursor >= len(flow):
                break
            env.step(flow[step_cursor])
            executed.append(flow[step_cursor])
            step_cursor += 1
        status, status_color = "READY", ACCENT

    rebuild_state()

    try:
        while viewer.is_running():
            _viewer_overlay(viewer, env, task_name, min(step_cursor, max(len(flow) - 1, 0)), flow, "idle", 0)
            viewer.sync()
            cv2.imshow(WINDOW_NAME, _preview(env, task_name, flow, step_cursor, executed, status, status_color))
            key = cv2.waitKey(15) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("n"):
                task_index = (task_index + 1) % len(selected_tasks)
                rebuild_state()
                continue
            if key == ord("p"):
                task_index = (task_index - 1) % len(selected_tasks)
                rebuild_state()
                continue
            if key == ord("r"):
                rebuild_state()
                status, status_color = "RESET", WARN
                continue
            if key == ord("b"):
                replay_steps = max(step_cursor - 1, 0)
                rebuild_state(replay_steps=replay_steps)
                status, status_color = "BACK", WARN
                continue
            if key not in (ord(" "), 13):
                continue
            if step_cursor >= len(flow):
                status, status_color = "FLOW DONE", SUCCESS
                continue

            primitive_id_value = flow[step_cursor]

            def on_frame(phase: str, micro_step: int) -> None:
                if not viewer.is_running():
                    return
                _viewer_overlay(viewer, env, task_name, step_cursor, flow, phase, micro_step)
                viewer.sync()
                label = "RUNNING" if phase == "internal" else phase.upper()
                cv2.imshow(WINDOW_NAME, _preview(env, task_name, flow, step_cursor, executed, label, ACCENT))
                cv2.waitKey(1)

            (_, _, done, info), _ = execute_with_animation(
                env,
                primitive_id_value,
                on_frame=on_frame,
                frame_sleep_s=args.frame_sleep_s,
            )
            executed.append(primitive_id_value)
            step_cursor += 1
            if info["success"]:
                status, status_color = "SUCCESS", SUCCESS
            elif done:
                status, status_color = "DONE", WARN
            else:
                status, status_color = "READY", ACCENT
    finally:
        viewer.close()
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

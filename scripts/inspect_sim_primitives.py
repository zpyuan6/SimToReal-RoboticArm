from __future__ import annotations

import argparse
from typing import Callable

import cv2
import mujoco.viewer
import numpy as np

from sim_demo_common import (
    ACCENT,
    PRIMITIVE_CANONICAL_TASK,
    PRIMITIVE_SETUP_PREFIXES,
    SUCCESS,
    WARN,
    add_shared_args,
    canonical_context,
    execute_with_animation,
    load_env,
    panel_frame,
    parse_primitive_tokens,
    set_canonical_layout,
)
from ttla.sim.skills import primitive_name


WINDOW_NAME = "TTLA Sim Primitive Inspector"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive MuJoCo inspector for individual simulated primitives."
    )
    add_shared_args(parser, default_output_root="results/sim_primitive_demos")
    parser.add_argument(
        "--primitives",
        help="Comma-separated primitive names or ids. Defaults to all legacy primitives.",
    )
    parser.add_argument("--frame-sleep-s", type=float, default=0.03)
    return parser.parse_args()


def _viewer_overlay(handle, env, primitive_id_value: int, phase: str, micro_step: int) -> None:
    texts = [
        (
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            "TTLA Primitive Inspector",
            (
                f"task={env.task_name} | primitive={primitive_name(primitive_id_value, primitive_vocabulary=env.primitive_vocabulary)}\n"
                f"phase={phase} | internal_step={micro_step}\n"
                f"vis={env.visibility_score():.3f} center={env.center_error_px():.1f}px grasp_gap={env.grasp_gap():+.3f}m"
            ),
        ),
        (
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPRIGHT,
            "controls",
            "Space/Enter execute\nN next primitive\nP previous primitive\nR reset scene\nQ quit",
        ),
    ]
    handle.set_texts(texts)


def _preview(env, primitive_id_value: int, task_name: str, setup_prefix: list[int], status: str, status_color) -> np.ndarray:
    obs = env.observe()
    lines = [
        f"task={task_name}",
        f"primitive={primitive_name(primitive_id_value, primitive_vocabulary=env.primitive_vocabulary)}",
        f"setup_prefix={','.join(primitive_name(pid) for pid in setup_prefix) or 'none'}",
        f"visibility={env.visibility_score():.3f}",
        f"center_error_px={env.center_error_px():.1f}",
        f"grasp_gap={env.grasp_gap():+.3f}",
        f"ee_target_distance={env.ee_target_distance():.3f}",
        f"flags verified={int(env.verified)} grasped={int(env.object_attached)} lifted={int(env.lifted)} placed={int(env.placed)}",
        "controls: Space execute | N next | P previous | R reset | Q quit",
    ]
    return panel_frame(
        env,
        obs,
        title=f"Primitive Inspector: {primitive_name(primitive_id_value, primitive_vocabulary=env.primitive_vocabulary)}",
        subtitle=f"task={task_name}",
        lines=lines,
        status_label=status,
        status_color=status_color,
    )


def main() -> None:
    args = _parse_args()
    _, env = load_env(args.config, seed=args.seed)
    primitive_ids = parse_primitive_tokens(args.primitives, env.primitive_vocabulary)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(WINDOW_NAME, 1320, 900)
    viewer = mujoco.viewer.launch_passive(env.model, env.data, show_left_ui=True, show_right_ui=True)

    primitive_index = 0

    def reset_current() -> tuple[str, list[int]]:
        primitive_id_value = primitive_ids[primitive_index]
        task_name = PRIMITIVE_CANONICAL_TASK[primitive_id_value]
        setup_prefix = list(PRIMITIVE_SETUP_PREFIXES[primitive_id_value])
        env.reset(task_name=task_name, context=canonical_context())
        set_canonical_layout(env, task_name)
        for setup_primitive in setup_prefix:
            env.step(setup_primitive)
        return task_name, setup_prefix

    task_name, setup_prefix = reset_current()
    status = "READY"
    status_color = ACCENT

    try:
        while viewer.is_running():
            primitive_id_value = primitive_ids[primitive_index]
            _viewer_overlay(viewer, env, primitive_id_value, "idle", 0)
            viewer.sync()
            cv2.imshow(WINDOW_NAME, _preview(env, primitive_id_value, task_name, setup_prefix, status, status_color))
            key = cv2.waitKey(15) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("n"):
                primitive_index = (primitive_index + 1) % len(primitive_ids)
                task_name, setup_prefix = reset_current()
                status, status_color = "READY", ACCENT
                continue
            if key == ord("p"):
                primitive_index = (primitive_index - 1) % len(primitive_ids)
                task_name, setup_prefix = reset_current()
                status, status_color = "READY", ACCENT
                continue
            if key == ord("r"):
                task_name, setup_prefix = reset_current()
                status, status_color = "RESET", WARN
                continue
            if key not in (ord(" "), 13):
                continue

            def on_frame(phase: str, micro_step: int) -> None:
                if not viewer.is_running():
                    return
                _viewer_overlay(viewer, env, primitive_id_value, phase, micro_step)
                viewer.sync()
                label = "RUNNING" if phase == "internal" else phase.upper()
                cv2.imshow(WINDOW_NAME, _preview(env, primitive_id_value, task_name, setup_prefix, label, ACCENT))
                cv2.waitKey(1)

            (_, reward, done, info), _ = execute_with_animation(
                env,
                primitive_id_value,
                on_frame=on_frame,
                frame_sleep_s=args.frame_sleep_s,
            )
            if info["success"]:
                status, status_color = "SUCCESS", SUCCESS
            elif done:
                status, status_color = "DONE", WARN
            else:
                status, status_color = f"REWARD {reward:+.2f}", ACCENT
    finally:
        viewer.close()
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

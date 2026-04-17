from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ttla.config import load_config
from ttla.deployment import DeploymentRunner
from ttla.sim import RoArmSimEnv
from ttla.sim.skills import ABORT_ID, PRIMITIVE_NAMES, primitive_name
from ttla.utils.io import ensure_dir, write_json


WINDOW_NAME = "TTLA Action Validator"
WINDOW_FLAGS = cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED
BG = (242, 244, 248)
CARD = (251, 252, 254)
BORDER = (218, 223, 232)
TEXT = (32, 37, 48)
SUBTLE = (98, 108, 125)
ACCENT = (54, 106, 255)
SUCCESS = (44, 142, 86)
WARN = (204, 129, 54)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--deploy-config", default="configs/deployment.yaml")
    parser.add_argument("--task", default="level1_verify", choices=["level1_verify", "level2_approach", "level3_pick_place"])
    parser.add_argument("--primitives", required=True, help="Comma-separated primitive ids, e.g. 2,0,1,2,3")
    parser.add_argument("--save-dir", default="data/raw/action_validation")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--step-delay", type=float, default=0.5, help="Pause after a real action before reading the after-frame.")
    parser.add_argument("--auto-advance", action="store_true", help="Advance automatically instead of waiting for a key between steps.")
    parser.add_argument("--disable-gui", action="store_true")
    return parser.parse_args()


def _put(canvas: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.58, color: tuple[int, int, int] = TEXT, thickness: int = 1) -> None:
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, lineType=cv2.LINE_AA)


def _card(canvas: np.ndarray, top_left: tuple[int, int], bottom_right: tuple[int, int], title: str, subtitle: str | None = None) -> None:
    x1, y1 = top_left
    x2, y2 = bottom_right
    cv2.rectangle(canvas, (x1, y1), (x2, y2), CARD, -1, lineType=cv2.LINE_AA)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), BORDER, 1, lineType=cv2.LINE_AA)
    _put(canvas, title, (x1 + 18, y1 + 28), 0.72, TEXT, 2)
    if subtitle:
        _put(canvas, subtitle, (x1 + 18, y1 + 52), 0.46, SUBTLE, 1)


def _badge(canvas: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = origin
    width = max(100, 18 + len(text) * 10)
    cv2.rectangle(canvas, (x, y), (x + width, y + 28), color, -1, lineType=cv2.LINE_AA)
    _put(canvas, text, (x + 10, y + 20), 0.48, (255, 255, 255), 1)


def _compose_dashboard(
    task_name: str,
    step_index: int,
    primitive_id: int,
    sim_before: np.ndarray,
    sim_after: np.ndarray,
    real_before: np.ndarray,
    real_after: np.ndarray,
    sim_info: dict,
    real_info: dict,
) -> np.ndarray:
    canvas = np.full((980, 1280, 3), BG, dtype=np.uint8)
    _put(canvas, "TTLA Sim-to-Real Action Validator", (28, 34), 0.92, TEXT, 2)
    _put(canvas, "Compare expected simulated motion and real robot execution for the same primitive.", (30, 60), 0.48, SUBTLE, 1)
    _badge(canvas, f"STEP {step_index}", (988, 22), ACCENT)
    _badge(canvas, primitive_name(primitive_id), (1098, 22), SUCCESS if primitive_id != ABORT_ID else WARN)

    _card(canvas, (24, 84), (620, 462), "Sim Before", f"Task: {task_name}")
    _card(canvas, (660, 84), (1256, 462), "Sim After", f"Primitive: {primitive_name(primitive_id)}")
    _card(canvas, (24, 500), (620, 878), "Real Before")
    _card(canvas, (660, 500), (1256, 878), "Real After")
    _card(canvas, (24, 900), (1256, 956), "Summary", None)

    sim_before_view = cv2.resize(sim_before, (560, 280), interpolation=cv2.INTER_CUBIC)
    sim_after_view = cv2.resize(sim_after, (560, 280), interpolation=cv2.INTER_CUBIC)
    real_before_view = cv2.resize(real_before, (560, 280), interpolation=cv2.INTER_CUBIC)
    real_after_view = cv2.resize(real_after, (560, 280), interpolation=cv2.INTER_CUBIC)
    canvas[150:430, 44:604] = sim_before_view
    canvas[150:430, 680:1240] = sim_after_view
    canvas[566:846, 44:604] = real_before_view
    canvas[566:846, 680:1240] = real_after_view

    sim_summary = (
        f"sim success={int(sim_info.get('success', 0))} "
        f"vis={float(sim_info.get('visibility', 0.0)):.3f} "
        f"center={float(sim_info.get('center_error', 0.0)):.1f}px"
    )
    real_summary = real_info.get("primitive_name", primitive_name(primitive_id))
    _put(canvas, f"Sim: {sim_summary}", (44, 934), 0.5, SUBTLE, 1)
    _put(canvas, f"Real: {real_summary}", (640, 934), 0.5, SUBTLE, 1)
    return canvas


def _save_step_artifacts(
    step_dir: Path,
    primitive_id: int,
    sim_before: np.ndarray,
    sim_after: np.ndarray,
    real_before: np.ndarray,
    real_after: np.ndarray,
    sim_info: dict,
    real_info: dict,
) -> None:
    cv2.imwrite(str(step_dir / "sim_before.png"), sim_before)
    cv2.imwrite(str(step_dir / "sim_after.png"), sim_after)
    cv2.imwrite(str(step_dir / "real_before.png"), real_before)
    cv2.imwrite(str(step_dir / "real_after.png"), real_after)
    write_json(
        step_dir / "meta.json",
        {
            "primitive_id": primitive_id,
            "primitive_name": primitive_name(primitive_id),
            "sim_info": sim_info,
            "real_info": real_info,
        },
    )


def _await_next_step(auto_advance: bool, disable_gui: bool) -> bool:
    if disable_gui or auto_advance:
        return True
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            return False
        if key in (ord("n"), ord(" "), 13):
            return True


def main() -> None:
    args = _parse_args()
    primitive_ids = [int(item.strip()) for item in args.primitives.split(",") if item.strip()]
    if not primitive_ids:
        raise ValueError("No primitives were provided.")

    cfg = load_config(args.config)
    deploy_cfg = load_config(args.deploy_config)
    sim_env = RoArmSimEnv(cfg["sim"], seed=args.seed)
    runner = DeploymentRunner(deploy_cfg)
    session_dir = ensure_dir(Path(args.save_dir) / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    if not args.disable_gui:
        cv2.namedWindow(WINDOW_NAME, WINDOW_FLAGS)
        cv2.resizeWindow(WINDOW_NAME, 1320, 1020)

    obs = sim_env.reset(task_name=args.task)
    if deploy_cfg.get("safety", {}).get("reset_before_episode", True):
        runner.robot.reset_pose()
        time.sleep(1.5)

    try:
        for step_index, primitive_id in enumerate(primitive_ids, start=1):
            step_dir = ensure_dir(session_dir / f"step_{step_index:03d}_{primitive_name(primitive_id)}")

            sim_before = obs["image"].copy()
            real_before = runner.camera.read()

            next_obs, _reward, _done, sim_info = sim_env.step(primitive_id)
            real_result = runner.executor.run(primitive_id)
            time.sleep(max(args.step_delay, 0.0))
            real_after = runner.camera.read()
            sim_after = next_obs["image"].copy()

            dashboard = _compose_dashboard(
                args.task,
                step_index,
                primitive_id,
                sim_before,
                sim_after,
                real_before,
                real_after,
                sim_info,
                real_result.info,
            )
            if not args.disable_gui:
                cv2.imshow(WINDOW_NAME, dashboard)
                cv2.waitKey(1)

            _save_step_artifacts(
                step_dir,
                primitive_id,
                sim_before,
                sim_after,
                real_before,
                real_after,
                {
                    "task": args.task,
                    "success": int(sim_info.get("success", 0)),
                    "visibility": float(sim_info.get("visibility", 0.0)),
                    "center_error": float(sim_info.get("center_error", 0.0)),
                    "primitive_name": sim_info.get("primitive_name", primitive_name(primitive_id)),
                },
                real_result.info,
            )
            cv2.imwrite(str(step_dir / "comparison.png"), dashboard)

            obs = next_obs
            if not _await_next_step(args.auto_advance, args.disable_gui):
                break
    finally:
        write_json(
            session_dir / "session_meta.json",
            {
                "task": args.task,
                "primitives": primitive_ids,
                "primitive_names": [PRIMITIVE_NAMES[idx] for idx in primitive_ids],
            },
        )
        runner.close()
        sim_env.close()
        cv2.destroyAllWindows()

    print(f"saved_validation_session={session_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path
from typing import Any

import cv2
import torch

from ttla.config import load_config
from ttla.deployment import DeploymentRunner
from ttla.models import TTLAModel, load_model_state
from ttla.sim.skills import primitive_id, primitive_name
from ttla.sim.task_defs import TASK_SPECS, TASK_TO_ID
from ttla.utils.io import ensure_dir, write_json


EPISODE_STATUS_CHOICES = {
    "s": "success",
    "f": "fail",
    "p": "partial",
    "r": "retry",
    "q": "quit",
}

PRIMITIVE_STATUS_CHOICES = {
    "s": "success",
    "f": "fail",
    "p": "partial",
    "u": "unsafe",
    "r": "retry",
    "q": "quit",
}


def _load_model(cfg: dict[str, Any], checkpoint_path: str, device: str) -> TTLAModel:
    from ttla.training import build_model

    payload = torch.load(checkpoint_path, map_location=device)
    model = build_model(cfg).to(device)
    load_model_state(model, payload["model_state"])
    model.eval()
    return model


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _parse_primitives(primitives_arg: str | None, task_name: str) -> list[int]:
    if not primitives_arg:
        return list(TASK_SPECS[task_name].primary_primitives)
    parsed: list[int] = []
    for item in primitives_arg.split(","):
        value = item.strip()
        if not value:
            continue
        if value.isdigit():
            parsed.append(int(value))
        else:
            parsed.append(primitive_id(value))
    return parsed


def _load_step_records(episode_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(episode_dir.glob("step_*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["_path"] = str(path)
        records.append(payload)
    return records


def _prompt_choice(prompt: str, choices: dict[str, str]) -> str:
    while True:
        raw = input(prompt).strip().lower()
        if raw in choices:
            return choices[raw]
        print(f"Invalid choice. Expected one of: {', '.join(sorted(choices.keys()))}")


def _prompt_notes() -> str:
    return input("Notes (optional, press Enter to skip): ").strip()


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _summarize_episode_labels(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counted = [row for row in rows if row["status"] not in {"retry", "quit"}]
    success = sum(row["status"] == "success" for row in counted)
    fail = sum(row["status"] == "fail" for row in counted)
    partial = sum(row["status"] == "partial" for row in counted)
    total = len(counted)
    return {
        "episodes_total": total,
        "episodes_success": success,
        "episodes_fail": fail,
        "episodes_partial": partial,
        "episode_success_rate": float(success / total) if total else 0.0,
    }


def _summarize_primitive_labels(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counted = [row for row in rows if row["status"] not in {"retry", "quit"}]
    success = sum(row["status"] == "success" for row in counted)
    fail = sum(row["status"] == "fail" for row in counted)
    partial = sum(row["status"] == "partial" for row in counted)
    unsafe = sum(row["status"] == "unsafe" for row in counted)
    total = len(counted)
    return {
        "primitives_total": total,
        "primitives_success": success,
        "primitives_fail": fail,
        "primitives_partial": partial,
        "primitives_unsafe": unsafe,
        "primitive_success_rate": float(success / total) if total else 0.0,
    }


def _run_sequence_episode(
    runner: DeploymentRunner,
    primitive_ids: list[int],
    episode_name: str,
    sequence_rows: list[dict[str, Any]],
    task_name: str,
    checkpoint_path: str | None,
) -> tuple[Path, bool]:
    episode_dir = ensure_dir(runner.log_dir / episode_name)
    if runner.cfg["safety"].get("reset_before_episode", True):
        runner.robot.reset_pose()
        time.sleep(1.5)
    quit_requested = False
    for step, primitive_idx in enumerate(primitive_ids):
        primitive_label = primitive_name(primitive_idx)
        before_frame = runner.camera.read()
        cv2.imwrite(str(episode_dir / f"frame_before_{step:03d}.png"), before_frame)
        result = runner.executor.run(primitive_idx)
        after_frame = runner.camera.read()
        cv2.imwrite(str(episode_dir / f"frame_after_{step:03d}.png"), after_frame)
        record = {
            "primitive_id": primitive_idx,
            "primitive_name": primitive_label,
            **result.info,
        }
        write_json(episode_dir / f"step_{step:03d}.json", record)
        while True:
            status = _prompt_choice(
                f"[{episode_name}] Primitive {step} ({primitive_label}) [s/f/p/u/r/q]: ",
                PRIMITIVE_STATUS_CHOICES,
            )
            notes = _prompt_notes()
            if status == "retry":
                result = runner.executor.run(primitive_idx)
                after_frame = runner.camera.read()
                cv2.imwrite(str(episode_dir / f"frame_after_{step:03d}_retry.png"), after_frame)
                continue
            sequence_rows.append(
                {
                    "episode": episode_name,
                    "task": task_name,
                    "checkpoint": checkpoint_path or "",
                    "step": step,
                    "primitive_id": primitive_idx,
                    "primitive_name": primitive_label,
                    "status": status,
                    "notes": notes,
                    "episode_dir": str(episode_dir),
                }
            )
            if status == "quit":
                quit_requested = True
            break
        if quit_requested or result.done:
            break
    return episode_dir, quit_requested


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fixed_protocol.yaml")
    parser.add_argument("--deploy-config", default="configs/deployment.yaml")
    parser.add_argument("--mode", default="policy", choices=["policy", "sequence"])
    parser.add_argument("--checkpoint", default="results/checkpoints/adapter_calibrated.pt")
    parser.add_argument(
        "--task",
        default="level3_pick_place",
        choices=sorted(TASK_SPECS.keys()),
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--primitives", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", default="results/real_deployment_eval")
    parser.add_argument("--operator", default="")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    deploy_cfg = load_config(args.deploy_config)
    task_id = TASK_TO_ID[args.task]
    primitive_ids = _parse_primitives(args.primitives, args.task) if args.mode == "sequence" else []
    run_name = args.run_name or f"{args.mode}_{args.task}_{_timestamp()}"
    run_dir = ensure_dir(Path(args.output_root) / run_name)
    episodes_dir = ensure_dir(run_dir / "episodes")

    deploy_cfg = copy.deepcopy(deploy_cfg)
    deploy_cfg.setdefault("runtime", {})
    deploy_cfg["runtime"]["log_dir"] = str(episodes_dir)

    model = _load_model(cfg, args.checkpoint, cfg["train"]["device"]) if args.mode == "policy" else None
    runner = DeploymentRunner(deploy_cfg, model=model, device=cfg["train"]["device"])

    episode_rows: list[dict[str, Any]] = []
    primitive_rows: list[dict[str, Any]] = []
    metadata = {
        "mode": args.mode,
        "task": args.task,
        "task_id": task_id,
        "checkpoint": args.checkpoint if args.mode == "policy" else "",
        "operator": args.operator,
        "notes": args.notes,
        "episodes_requested": args.episodes,
        "primitives": primitive_ids,
        "timestamp": time.time(),
    }
    write_json(run_dir / "meta.json", metadata)
    write_json(run_dir / "config_snapshot.json", {"config": cfg, "deploy_config": deploy_cfg})

    quit_requested = False
    try:
        for episode_idx in range(args.episodes):
            attempt_idx = 0
            while True:
                episode_name = f"episode_{episode_idx:03d}_attempt_{attempt_idx:02d}"
                if args.mode == "policy":
                    episode_dir = runner.run_policy_episode(task_id=task_id, episode_name=episode_name)
                else:
                    episode_dir, quit_requested = _run_sequence_episode(
                        runner=runner,
                        primitive_ids=primitive_ids,
                        episode_name=episode_name,
                        sequence_rows=primitive_rows,
                        task_name=args.task,
                        checkpoint_path=args.checkpoint if args.mode == "policy" else None,
                    )
                step_records = _load_step_records(episode_dir)
                primitive_trace = [record.get("primitive_name", "") for record in step_records]
                status = _prompt_choice(
                    f"[{episode_name}] Episode outcome [s/f/p/r/q]: ",
                    EPISODE_STATUS_CHOICES,
                )
                notes = _prompt_notes()
                if status == "retry":
                    attempt_idx += 1
                    continue
                episode_rows.append(
                    {
                        "episode_index": episode_idx,
                        "attempt_index": attempt_idx,
                        "episode_name": episode_name,
                        "mode": args.mode,
                        "task": args.task,
                        "task_id": task_id,
                        "checkpoint": args.checkpoint if args.mode == "policy" else "",
                        "steps_executed": len(step_records),
                        "primitive_trace": " | ".join(primitive_trace),
                        "status": status,
                        "notes": notes,
                        "episode_dir": str(episode_dir),
                    }
                )
                if status == "quit":
                    quit_requested = True
                break
            if quit_requested:
                break
    finally:
        runner.close()

    episode_summary = _summarize_episode_labels(episode_rows)
    primitive_summary = _summarize_primitive_labels(primitive_rows)
    summary = {
        **metadata,
        **episode_summary,
        **primitive_summary,
    }
    write_json(run_dir / "summary.json", summary)

    episode_fieldnames = [
        "episode_index",
        "attempt_index",
        "episode_name",
        "mode",
        "task",
        "task_id",
        "checkpoint",
        "steps_executed",
        "primitive_trace",
        "status",
        "notes",
        "episode_dir",
    ]
    _write_csv(run_dir / "episode_summary.csv", episode_rows, episode_fieldnames)

    if primitive_rows:
        primitive_fieldnames = [
            "episode",
            "task",
            "checkpoint",
            "step",
            "primitive_id",
            "primitive_name",
            "status",
            "notes",
            "episode_dir",
        ]
        _write_csv(run_dir / "primitive_summary.csv", primitive_rows, primitive_fieldnames)

    print(f"run_dir={run_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import yaml
from serial.tools import list_ports


ROOT = Path(__file__).resolve().parents[1]


METHODS = {
    "no_adaptation": ROOT / "results" / "fixed_protocol" / "backbone_suite" / "feedforward" / "checkpoints" / "best_model.pt",
    "static_adapter": ROOT / "results" / "fixed_protocol" / "backbone_suite_real" / "feedforward" / "checkpoints" / "static_adapter_calibrated.pt",
    "few_shot_finetuning": ROOT / "results" / "fixed_protocol" / "backbone_suite_real" / "feedforward" / "checkpoints" / "few_shot_finetuned.pt",
    "plica": ROOT / "results" / "fixed_protocol" / "backbone_suite_real" / "feedforward" / "checkpoints" / "adapter_calibrated.pt",
}


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _parse_methods(raw: str) -> list[str]:
    methods = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in methods if item not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Expected one of: {sorted(METHODS)}")
    return methods


def _preflight_camera(camera_index: int) -> tuple[bool, str]:
    cap = cv2.VideoCapture(camera_index)
    opened = cap.isOpened()
    ok, _ = cap.read() if opened else (False, None)
    cap.release()
    if opened and ok:
        return True, f"camera index {camera_index} is readable"
    if opened:
        return False, f"camera index {camera_index} opened but no frame was returned"
    return False, f"camera index {camera_index} is not available"


def _preflight_serial(serial_port: str) -> tuple[bool, str]:
    ports = sorted(port.device for port in list_ports.comports())
    if serial_port in ports:
        return True, f"serial port {serial_port} is visible"
    return False, f"serial port {serial_port} is not visible; available ports: {ports or 'none'}"


def _prepare_deploy_config(
    deploy_config_path: Path,
    serial_port: str | None,
    camera_index: int | None,
) -> tuple[Path, Path | None]:
    if serial_port is None and camera_index is None:
        return deploy_config_path, None

    with deploy_config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if serial_port is not None:
        payload.setdefault("serial", {})
        payload["serial"]["port"] = serial_port
    if camera_index is not None:
        payload.setdefault("camera", {})
        payload["camera"]["index"] = camera_index

    temp_dir = ROOT / "results" / "real_deployment_eval" / "_runtime_configs"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"deployment_l3_policy_{_timestamp()}.yaml"
    with temp_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return temp_path, temp_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated-trial real L3 policy-mode baseline evaluation.")
    parser.add_argument("--config", default="configs/fixed_protocol.yaml")
    parser.add_argument("--deploy-config", default="configs/deployment.yaml")
    parser.add_argument("--task", default="level3_pick_place")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--methods", default="no_adaptation,static_adapter,few_shot_finetuning,plica")
    parser.add_argument("--output-root", default="results/real_deployment_eval")
    parser.add_argument("--operator", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--serial-port", default=None)
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--run-prefix", default="l3_policy")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    methods = _parse_methods(args.methods)
    config_path = ROOT / args.config
    deploy_config_path = ROOT / args.deploy_config
    output_root = ROOT / args.output_root

    effective_camera_index = args.camera_index
    effective_serial_port = args.serial_port

    if effective_camera_index is None:
        with deploy_config_path.open("r", encoding="utf-8") as handle:
            deploy_cfg = yaml.safe_load(handle)
        effective_camera_index = int(deploy_cfg.get("camera", {}).get("index", 0))
        effective_serial_port = effective_serial_port or str(deploy_cfg.get("serial", {}).get("port", ""))

    camera_ok, camera_msg = _preflight_camera(int(effective_camera_index))
    serial_ok, serial_msg = _preflight_serial(str(effective_serial_port))
    print(f"[preflight] {camera_msg}")
    print(f"[preflight] {serial_msg}")

    if not args.dry_run and (not camera_ok or not serial_ok):
        raise SystemExit(
            "Hardware preflight failed. Fix camera/serial availability or rerun with corrected "
            "--serial-port / --camera-index."
        )

    effective_deploy_config, temp_config = _prepare_deploy_config(
        deploy_config_path=deploy_config_path,
        serial_port=effective_serial_port,
        camera_index=int(effective_camera_index),
    )

    try:
        for method in methods:
            checkpoint = METHODS[method]
            if not checkpoint.exists():
                raise FileNotFoundError(f"Checkpoint not found for {method}: {checkpoint}")
            run_name = f"{args.run_prefix}_{method}_{_timestamp()}"
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "run_real_deployment_eval.py"),
                "--config",
                str(config_path),
                "--deploy-config",
                str(effective_deploy_config),
                "--mode",
                "policy",
                "--checkpoint",
                str(checkpoint),
                "--task",
                args.task,
                "--episodes",
                str(args.episodes),
                "--run-name",
                run_name,
                "--output-root",
                str(output_root),
                "--operator",
                args.operator,
                "--notes",
                f"{args.notes} baseline={method}".strip(),
            ]
            print(f"[run] method={method}")
            print(" ".join(f'"{item}"' if " " in item else item for item in cmd))
            if args.dry_run:
                continue
            result = subprocess.run(cmd, cwd=str(ROOT))
            if result.returncode != 0:
                raise SystemExit(f"Run failed for {method} with exit code {result.returncode}")
    finally:
        if temp_config is not None and temp_config.exists():
            temp_config.unlink()


if __name__ == "__main__":
    main()

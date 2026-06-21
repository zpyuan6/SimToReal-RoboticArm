from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from run_continuous_sim2real_stress import PROFILE_CONTEXTS


DEFAULT_CONFIGS = {
    "act": "configs/continuous_act_jointtarget_staged_frozen_best.yaml",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run continuous sim-to-real bridge experiments across profiles/configs.")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=[f"{label}={path}" for label, path in DEFAULT_CONFIGS.items()],
        help="Config specs as label=path. Defaults to frozen ACT.",
    )
    parser.add_argument(
        "--profiles",
        default="neutral,visual,camera,actuation,combined_mild,combined_hard",
        help=f"Comma-separated profiles. Available: {','.join(PROFILE_CONTEXTS)}",
    )
    parser.add_argument("--policy-device", default=None)
    parser.add_argument("--output-root", default="results/continuous_sim2real_bridge_suite")
    parser.add_argument("--calibration-episodes", type=int, default=6)
    parser.add_argument("--heldout-episodes", type=int, default=12)
    parser.add_argument("--fresh-episodes-per-task", type=int, default=4)
    parser.add_argument("--exact-num-per-task", type=int, default=4)
    parser.add_argument("--adapter", choices=("none", "task_bias", "task_affine"), default="task_bias")
    parser.add_argument("--adapter-blend", type=float, default=0.25)
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true", help="Only aggregate existing run directories.")
    return parser.parse_args()


def _parse_configs(raw: list[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for item in raw:
        if "=" not in item:
            path = item
            label = Path(path).stem
        else:
            label, path = item.split("=", 1)
            label = label.strip()
            path = path.strip()
        if not label or not path:
            raise ValueError(f"Invalid config spec: {item!r}")
        parsed.append((label, path))
    return parsed


def _parse_profiles(raw: str) -> list[str]:
    names = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(names).difference(PROFILE_CONTEXTS))
    if unknown:
        raise KeyError(f"Unknown profiles: {unknown}. Available profiles: {sorted(PROFILE_CONTEXTS)}")
    return names


def _run_one(args: argparse.Namespace, label: str, config_path: str, profile: str, output_dir: Path) -> None:
    cmd = [
        sys.executable,
        "scripts/run_continuous_sim2real_bridge.py",
        "--config",
        config_path,
        "--profile",
        profile,
        "--calibration-episodes",
        str(int(args.calibration_episodes)),
        "--heldout-episodes",
        str(int(args.heldout_episodes)),
        "--fresh-episodes-per-task",
        str(int(args.fresh_episodes_per_task)),
        "--exact-num-per-task",
        str(int(args.exact_num_per_task)),
        "--output-dir",
        str(output_dir),
        "--adapter",
        str(args.adapter),
        "--adapter-blend",
        str(float(args.adapter_blend)),
        "--ridge",
        str(float(args.ridge)),
        "--seed",
        str(int(args.seed)),
    ]
    if args.policy_device:
        cmd.extend(["--policy-device", str(args.policy_device)])
    if args.force_regenerate:
        cmd.append("--force-regenerate")
    print(f"[suite] running label={label} profile={profile} output_dir={output_dir}", flush=True)
    subprocess.run(cmd, check=True)


def _aggregate(output_root: Path, runs: list[tuple[str, str, Path]]) -> tuple[Path, Path]:
    transition_frames: list[pd.DataFrame] = []
    success_frames: list[pd.DataFrame] = []
    for label, profile, run_dir in runs:
        transition_path = run_dir / "heldout_transition_metrics.csv"
        success_path = run_dir / "combined_success_summary.csv"
        if transition_path.exists():
            transition = pd.read_csv(transition_path)
            transition.insert(0, "profile", profile)
            transition.insert(0, "config_label", label)
            transition_frames.append(transition)
        if success_path.exists():
            success = pd.read_csv(success_path)
            success.insert(0, "config_label", label)
            success_frames.append(success)
    transition_out = output_root / "suite_transition_metrics.csv"
    success_out = output_root / "suite_success_summary.csv"
    if transition_frames:
        pd.concat(transition_frames, ignore_index=True).to_csv(transition_out, index=False)
    if success_frames:
        pd.concat(success_frames, ignore_index=True).to_csv(success_out, index=False)
    return transition_out, success_out


def main() -> None:
    args = _parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    configs = _parse_configs(args.configs)
    profiles = _parse_profiles(args.profiles)
    runs: list[tuple[str, str, Path]] = []
    for label, config_path in configs:
        for profile in profiles:
            run_dir = output_root / label / profile
            if not args.aggregate_only:
                _run_one(args, label, config_path, profile, run_dir)
            runs.append((label, profile, run_dir))
    transition_out, success_out = _aggregate(output_root, runs)
    print(f"suite_transition_metrics_csv={transition_out}")
    print(f"suite_success_summary_csv={success_out}")


if __name__ == "__main__":
    main()

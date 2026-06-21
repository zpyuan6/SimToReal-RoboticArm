from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd

from ttla.config import load_config
from ttla.evaluation import evaluate_continuous_backbone


NEUTRAL_CONTEXT = {
    "camera_translation_jitter": 0.0,
    "camera_rotation_jitter": 0.0,
    "fov_jitter": 0.0,
    "light_jitter": 0.0,
    "blur_sigma": 0.0,
    "noise_std": 0.0,
    "action_gain_low": 1.0,
    "action_gain_high": 1.0,
    "action_delay_max": 0,
    "joint_bias": 0.0,
}


PROFILE_CONTEXTS = {
    "neutral": NEUTRAL_CONTEXT,
    "appearance_shift": {
        "camera_translation_jitter": 0.010,
        "camera_rotation_jitter": 0.02,
        "fov_jitter": 2.0,
        "light_jitter": 0.85,
        "blur_sigma": 2.1,
        "noise_std": 0.075,
        "action_gain_low": 0.93,
        "action_gain_high": 1.07,
        "action_delay_max": 1,
        "joint_bias": 0.03,
    },
    "embodiment_shift": {
        "camera_translation_jitter": 0.040,
        "camera_rotation_jitter": 0.08,
        "fov_jitter": 13.0,
        "light_jitter": 0.15,
        "blur_sigma": 0.35,
        "noise_std": 0.01,
        "action_gain_low": 0.58,
        "action_gain_high": 1.38,
        "action_delay_max": 4,
        "joint_bias": 0.17,
    },
    "joint_shift": {
        "camera_translation_jitter": 0.038,
        "camera_rotation_jitter": 0.08,
        "fov_jitter": 12.0,
        "light_jitter": 0.80,
        "blur_sigma": 2.0,
        "noise_std": 0.07,
        "action_gain_low": 0.62,
        "action_gain_high": 1.34,
        "action_delay_max": 4,
        "joint_bias": 0.16,
    },
    "train_like": {
        "camera_translation_jitter": 0.015,
        "camera_rotation_jitter": 0.03,
        "fov_jitter": 6.0,
        "light_jitter": 0.35,
        "blur_sigma": 1.0,
        "noise_std": 0.03,
        "action_gain_low": 0.78,
        "action_gain_high": 1.18,
        "action_delay_max": 2,
        "joint_bias": 0.09,
    },
    "visual": {
        "camera_translation_jitter": 0.0,
        "camera_rotation_jitter": 0.0,
        "fov_jitter": 0.0,
        "light_jitter": 0.90,
        "blur_sigma": 2.2,
        "noise_std": 0.08,
        "action_gain_low": 1.0,
        "action_gain_high": 1.0,
        "action_delay_max": 0,
        "joint_bias": 0.0,
    },
    "camera": {
        "camera_translation_jitter": 0.038,
        "camera_rotation_jitter": 0.08,
        "fov_jitter": 10.0,
        "light_jitter": 0.0,
        "blur_sigma": 0.0,
        "noise_std": 0.0,
        "action_gain_low": 1.0,
        "action_gain_high": 1.0,
        "action_delay_max": 0,
        "joint_bias": 0.0,
    },
    "actuation": {
        "camera_translation_jitter": 0.0,
        "camera_rotation_jitter": 0.0,
        "fov_jitter": 0.0,
        "light_jitter": 0.0,
        "blur_sigma": 0.0,
        "noise_std": 0.0,
        "action_gain_low": 0.62,
        "action_gain_high": 1.34,
        "action_delay_max": 4,
        "joint_bias": 0.16,
    },
    "combined_mild": {
        "camera_translation_jitter": 0.025,
        "camera_rotation_jitter": 0.025,
        "fov_jitter": 9.0,
        "light_jitter": 0.55,
        "blur_sigma": 1.5,
        "noise_std": 0.05,
        "action_gain_low": 0.72,
        "action_gain_high": 1.24,
        "action_delay_max": 3,
        "joint_bias": 0.12,
    },
    "combined_hard": {
        "camera_translation_jitter": 0.035,
        "camera_rotation_jitter": 0.035,
        "fov_jitter": 12.0,
        "light_jitter": 0.75,
        "blur_sigma": 2.0,
        "noise_std": 0.07,
        "action_gain_low": 0.65,
        "action_gain_high": 1.32,
        "action_delay_max": 4,
        "joint_bias": 0.16,
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run continuous policy stress tests under pseudo-real sim shifts.")
    parser.add_argument("--config", required=True, help="Continuous policy config.")
    parser.add_argument("--policy-path", default=None, help="Pretrained policy path. Defaults to frozen_baseline.policy_path.")
    parser.add_argument("--policy-device", default=None, help="Policy device override.")
    parser.add_argument("--episodes-per-task", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument(
        "--profiles",
        default="neutral,train_like,visual,camera,actuation,combined_mild,combined_hard",
        help=f"Comma-separated profiles. Available: {','.join(PROFILE_CONTEXTS)}",
    )
    parser.add_argument("--output-dir", default="results/continuous_sim2real_stress/act_frozen")
    return parser.parse_args()


def _profile_names(raw: str) -> list[str]:
    names = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(names).difference(PROFILE_CONTEXTS))
    if unknown:
        raise KeyError(f"Unknown profiles: {unknown}. Available profiles: {sorted(PROFILE_CONTEXTS)}")
    return names


def _policy_path(cfg: dict, override: str | None) -> str:
    if override:
        return override
    frozen = cfg.get("frozen_baseline", {})
    policy_path = frozen.get("policy_path")
    if not policy_path:
        raise ValueError("No --policy-path was provided and config has no frozen_baseline.policy_path.")
    return str(policy_path)


def _cfg_for_profile(base_cfg: dict, profile_name: str) -> dict:
    cfg = deepcopy(base_cfg)
    cfg["sim"]["context"] = dict(PROFILE_CONTEXTS[profile_name])
    cfg["sim"]["task_context_rescaling"] = False
    return cfg


def _flatten_summary(profile_name: str, summary_path: Path) -> list[dict[str, object]]:
    summary = pd.read_csv(summary_path)
    rows: list[dict[str, object]] = []
    for _, row in summary.iterrows():
        rows.append(
            {
                "profile": profile_name,
                "split": row["split"],
                "task": row["task"],
                "success": float(row["success"]),
                "steps": float(row["steps"]),
                "visibility": float(row["visibility"]),
                "center_error": float(row["center_error"]),
                "verified": float(row["verified"]),
                "grasped": float(row["grasped"]),
                "lifted": float(row["lifted"]),
                "placed": float(row["placed"]),
                "final_ee_ear_center_distance": float(row["final_ee_ear_center_distance"]),
                "final_ee_target_distance": float(row["final_ee_target_distance"]),
                "final_grasp_gap": float(row["final_grasp_gap"]),
                "final_dropzone_distance": float(row["final_dropzone_distance"]),
                "summary_csv": str(summary_path),
            }
        )
    return rows


def main() -> None:
    args = _parse_args()
    base_cfg = load_config(args.config)
    policy_path = _policy_path(base_cfg, args.policy_path)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    for profile_name in _profile_names(args.profiles):
        cfg = _cfg_for_profile(base_cfg, profile_name)
        profile_root = output_root / profile_name
        _, summary_path = evaluate_continuous_backbone(
            cfg,
            policy_path=policy_path,
            output_dir=profile_root,
            episodes_per_task=int(args.episodes_per_task),
            policy_device=args.policy_device,
            seed=int(args.seed),
        )
        all_rows.extend(_flatten_summary(profile_name, summary_path))
        print(f"profile={profile_name} summary_csv={summary_path}")

    combined = pd.DataFrame(all_rows)
    combined_path = output_root / "combined_summary.csv"
    combined.to_csv(combined_path, index=False)

    overall = combined.loc[combined["split"] == "overall"].sort_values("success", ascending=False)
    overall_path = output_root / "overall_summary.csv"
    overall.to_csv(overall_path, index=False)
    print(f"combined_summary_csv={combined_path}")
    print(f"overall_summary_csv={overall_path}")


if __name__ == "__main__":
    main()

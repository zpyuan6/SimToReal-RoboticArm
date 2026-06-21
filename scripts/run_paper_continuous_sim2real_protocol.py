from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from run_continuous_sim2real_bridge import _cfg_for_profile, _generate_split
from run_continuous_sim2real_stress import PROFILE_CONTEXTS
from ttla.config import load_config


DEFAULT_CONFIGS = {
    "act": "configs/continuous_act_jointtarget_staged_frozen_best.yaml",
    "diffusion": "configs/continuous_diffusion_jointdelta_staged_frozen_best.yaml",
}

DEFAULT_METHODS = (
    "no_adaptation,"
    "input_normalization,"
    "probe_feature_alignment,"
    "static_adapter,"
    "few_shot_finetuning,"
    "ours_proxy,"
    "ours_action_representation_adapter"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "IEEE-style continuous pseudo-real sim-to-real protocol runner. "
            "Generates disjoint calibration/validation/test splits and evaluates frozen methods without "
            "reusing validation/test outcomes for adaptation."
        )
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=[f"{label}={path}" for label, path in DEFAULT_CONFIGS.items()],
        help="Config specs as label=path.",
    )
    parser.add_argument(
        "--profiles",
        default="appearance_shift,embodiment_shift,joint_shift",
        help=f"Comma-separated profiles. Available: {','.join(PROFILE_CONTEXTS)}",
    )
    parser.add_argument(
        "--seeds",
        default="20260610,20260611,20260612",
        help="Comma-separated experiment seeds. Use at least 3 for paper-grade pseudo-real results.",
    )
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--output-root", default="results/paper_continuous_sim2real_protocol")
    parser.add_argument(
        "--split-source-root",
        default=None,
        help="Optional existing paper-protocol root to reuse calibration/validation/test split files from.",
    )
    parser.add_argument("--policy-device", default=None)
    parser.add_argument("--calibration-episodes", type=int, default=30)
    parser.add_argument("--validation-episodes", type=int, default=60)
    parser.add_argument("--test-episodes", type=int, default=60)
    parser.add_argument("--exact-num-per-task", type=int, default=20)
    parser.add_argument("--fresh-episodes-per-task", type=int, default=20)
    parser.add_argument("--adapter-blend", type=float, default=0.25)
    parser.add_argument(
        "--adapter-task-blends",
        default=None,
        help="Optional comma-separated per-task post-adapter blend for level1,level2,level3.",
    )
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    parser.add_argument("--latent-adapter-hidden-dim", type=int, default=64)
    parser.add_argument("--latent-adapter-transition-hidden-dim", type=int, default=128)
    parser.add_argument("--latent-adapter-action-decoder-hidden-dim", type=int, default=128)
    parser.add_argument("--latent-adapter-epochs", type=int, default=60)
    parser.add_argument("--latent-adapter-transition-epochs", type=int, default=40)
    parser.add_argument("--latent-adapter-action-decoder-epochs", type=int, default=40)
    parser.add_argument("--latent-adapter-batch-size", type=int, default=128)
    parser.add_argument("--latent-adapter-lr", type=float, default=1.0e-3)
    parser.add_argument("--latent-adapter-transition-lr", type=float, default=1.0e-3)
    parser.add_argument("--latent-adapter-action-decoder-lr", type=float, default=1.0e-3)
    parser.add_argument("--latent-adapter-weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--latent-adapter-reg-weight", type=float, default=0.5)
    parser.add_argument("--latent-adapter-action-loss-weight", type=float, default=0.25)
    parser.add_argument("--latent-adapter-chunk-action-loss-weight", type=float, default=0.0)
    parser.add_argument("--latent-adapter-source-stat-weight", type=float, default=0.0)
    parser.add_argument("--latent-adapter-scale", type=float, default=0.05)
    parser.add_argument("--latent-adapter-alignment-blend", type=float, default=0.25)
    parser.add_argument(
        "--latent-adapter-hyperparam-gate",
        choices=["none", "latent_shift"],
        default="none",
        help="Calibration-statistic gate for action-representation adapter hyperparameters.",
    )
    parser.add_argument("--latent-adapter-action-source", choices=["policy", "teacher"], default="teacher")
    parser.add_argument("--latent-adapter-source-max-pairs", type=int, default=4096)
    parser.add_argument("--latent-adapter-calibration-max-pairs", type=int, default=2048)
    parser.add_argument(
        "--action-repr-act-action-loss-backend",
        choices=["policy_head", "source_decoder"],
        default="policy_head",
    )
    parser.add_argument(
        "--action-repr-post-adapter",
        choices=[
            "none",
            "task_bias",
            "task_moment",
            "task_affine",
            "task_regularized_affine",
            "residual_mlp",
            "latent_residual_ridge",
        ],
        default="task_bias",
    )
    parser.add_argument(
        "--action-repr-diffusion-backend",
        choices=["denoise", "trajectory"],
        default="denoise",
    )
    parser.add_argument("--action-repr-fit-post-task-blends", action="store_true")
    parser.add_argument("--action-repr-fit-post-task-blend-max", type=float, default=0.75)
    parser.add_argument("--trajectory-adapter-hidden-dim", type=int, default=64)
    parser.add_argument("--trajectory-adapter-epochs", type=int, default=80)
    parser.add_argument("--trajectory-adapter-batch-size", type=int, default=128)
    parser.add_argument("--trajectory-adapter-lr", type=float, default=1.0e-3)
    parser.add_argument("--trajectory-adapter-weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--trajectory-adapter-scale", type=float, default=0.25)
    parser.add_argument("--trajectory-adapter-first-action-weight", type=float, default=2.0)
    parser.add_argument("--trajectory-adapter-plan-loss-weight", type=float, default=1.0)
    parser.add_argument("--trajectory-adapter-smooth-loss-weight", type=float, default=0.1)
    parser.add_argument("--trajectory-adapter-reg-weight", type=float, default=0.02)
    parser.add_argument("--trajectory-adapter-max-pairs", type=int, default=2048)
    parser.add_argument(
        "--trajectory-adapter-post-adapter",
        choices=[
            "none",
            "task_bias",
            "task_moment",
            "task_affine",
            "task_regularized_affine",
            "residual_mlp",
            "latent_residual_ridge",
        ],
        default="none",
    )
    parser.add_argument("--denoise-adapter-hidden-dim", type=int, default=64)
    parser.add_argument("--denoise-adapter-epochs", type=int, default=30)
    parser.add_argument("--denoise-adapter-batch-size", type=int, default=64)
    parser.add_argument("--denoise-adapter-lr", type=float, default=1.0e-3)
    parser.add_argument("--denoise-adapter-weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--denoise-adapter-scale", type=float, default=0.1)
    parser.add_argument(
        "--denoise-adapter-task-scales",
        default=None,
        help="Optional comma-separated denoise residual scales for level1,level2,level3.",
    )
    parser.add_argument("--denoise-adapter-trajectory-loss-weight", type=float, default=0.0)
    parser.add_argument("--denoise-adapter-first-action-loss-weight", type=float, default=0.0)
    parser.add_argument("--denoise-adapter-action-window-loss-weight", type=float, default=0.0)
    parser.add_argument("--denoise-adapter-reg-weight", type=float, default=0.02)
    parser.add_argument(
        "--denoise-adapter-timestep-sampling",
        choices=["train", "inference"],
        default="train",
        help="Timesteps used when fitting the diffusion denoise adapter.",
    )
    parser.add_argument("--denoise-adapter-max-pairs", type=int, default=1024)
    parser.add_argument(
        "--denoise-adapter-post-adapter",
        choices=[
            "none",
            "task_bias",
            "task_moment",
            "task_affine",
            "task_regularized_affine",
            "residual_mlp",
            "latent_residual_ridge",
        ],
        default="none",
    )
    parser.add_argument(
        "--closed-loop-candidates",
        default="identity,task_moment,trajectory_task_moment,denoise_task_moment",
    )
    parser.add_argument("--closed-loop-selection", choices=["taskwise", "overall"], default="taskwise")
    parser.add_argument("--closed-loop-objective", choices=["exact", "fresh"], default="exact")
    parser.add_argument("--closed-loop-calibration-num-per-task", type=int, default=3)
    parser.add_argument("--max-attempts-per-episode", type=int, default=100)
    parser.add_argument("--eval-splits", default="validation,test", help="Comma-separated among validation,test.")
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--skip-eval", action="store_true", help="Generate splits only.")
    parser.add_argument("--skip-existing-eval", action="store_true", help="Skip eval directories with existing CSV outputs.")
    parser.add_argument(
        "--separate-method-runs",
        action="store_true",
        help="Run each method in its own output directory so long paper protocols can be resumed method-by-method.",
    )
    parser.add_argument("--aggregate-only", action="store_true", help="Only aggregate existing result directories.")
    return parser.parse_args()


def _parse_specs(raw: list[str]) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for item in raw:
        if "=" in item:
            label, path = item.split("=", 1)
        else:
            path = item
            label = Path(path).stem
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"Invalid config spec: {item!r}")
        specs.append((label, path))
    return specs


def _parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _parse_seeds(raw: str) -> list[int]:
    return [int(part) for part in _parse_csv(raw)]


def _validate_profiles(names: list[str]) -> None:
    unknown = sorted(set(names).difference(PROFILE_CONTEXTS))
    if unknown:
        raise KeyError(f"Unknown profiles: {unknown}. Available profiles: {sorted(PROFILE_CONTEXTS)}")


def _split_paths(root: Path) -> dict[str, Path]:
    split_root = root / "splits"
    return {
        "calibration": split_root / "calibration.npz",
        "validation": split_root / "validation.npz",
        "test": split_root / "test.npz",
    }


def _generate_splits(
    *,
    config_path: str,
    profile: str,
    seed: int,
    root: Path,
    calibration_episodes: int,
    validation_episodes: int,
    test_episodes: int,
    force: bool,
    max_attempts_per_episode: int,
) -> dict[str, Path]:
    base_cfg = load_config(config_path)
    cfg = _cfg_for_profile(base_cfg, profile)
    paths = _split_paths(root)
    paths["calibration"].parent.mkdir(parents=True, exist_ok=True)
    split_specs = {
        "calibration": (calibration_episodes, seed + 101),
        "validation": (validation_episodes, seed + 202),
        "test": (test_episodes, seed + 303),
    }
    for split_name, (episodes, split_seed) in split_specs.items():
        _generate_split(
            cfg,
            paths[split_name],
            split_name=f"{profile}_{split_name}_seed{seed}",
            episodes=int(episodes),
            seed=int(split_seed),
            force=bool(force),
            max_attempts_per_episode=int(max_attempts_per_episode),
            l1_terminal_hold_steps=0,
        )
    return paths


def _run_method_comparison(
    *,
    config_path: str,
    profile: str,
    seed: int,
    eval_split: str,
    split_paths: dict[str, Path],
    output_dir: Path,
    methods: str,
    policy_device: str | None,
    exact_num_per_task: int,
    fresh_episodes_per_task: int,
    adapter_blend: float,
    ridge: float,
    latent_adapter_args: argparse.Namespace,
    skip_existing: bool,
) -> None:
    success_path = output_dir / "method_success_summary.csv"
    transition_path = output_dir / "method_transition_metrics.csv"
    if skip_existing and success_path.exists() and transition_path.exists():
        print(f"[paper-protocol] skip existing eval output={output_dir}", flush=True)
        return
    cmd = [
        sys.executable,
        "scripts/run_continuous_sim2real_method_comparison.py",
        "--config",
        config_path,
        "--profile",
        profile,
        "--output-dir",
        str(output_dir),
        "--calibration-data",
        str(split_paths["calibration"]),
        "--heldout-data",
        str(split_paths[eval_split]),
        "--fresh-episodes-per-task",
        str(int(fresh_episodes_per_task)),
        "--exact-num-per-task",
        str(int(exact_num_per_task)),
        "--methods",
        methods,
        "--adapter-blend",
        str(float(adapter_blend)),
    ]
    if latent_adapter_args.adapter_task_blends is not None:
        cmd.extend(["--adapter-task-blends", str(latent_adapter_args.adapter_task_blends)])
    cmd += [
        "--ridge",
        str(float(ridge)),
        "--latent-adapter-hidden-dim",
        str(int(latent_adapter_args.latent_adapter_hidden_dim)),
        "--latent-adapter-transition-hidden-dim",
        str(int(latent_adapter_args.latent_adapter_transition_hidden_dim)),
        "--latent-adapter-action-decoder-hidden-dim",
        str(int(latent_adapter_args.latent_adapter_action_decoder_hidden_dim)),
        "--latent-adapter-epochs",
        str(int(latent_adapter_args.latent_adapter_epochs)),
        "--latent-adapter-transition-epochs",
        str(int(latent_adapter_args.latent_adapter_transition_epochs)),
        "--latent-adapter-action-decoder-epochs",
        str(int(latent_adapter_args.latent_adapter_action_decoder_epochs)),
        "--latent-adapter-batch-size",
        str(int(latent_adapter_args.latent_adapter_batch_size)),
        "--latent-adapter-lr",
        str(float(latent_adapter_args.latent_adapter_lr)),
        "--latent-adapter-transition-lr",
        str(float(latent_adapter_args.latent_adapter_transition_lr)),
        "--latent-adapter-action-decoder-lr",
        str(float(latent_adapter_args.latent_adapter_action_decoder_lr)),
        "--latent-adapter-weight-decay",
        str(float(latent_adapter_args.latent_adapter_weight_decay)),
        "--latent-adapter-reg-weight",
        str(float(latent_adapter_args.latent_adapter_reg_weight)),
        "--latent-adapter-action-loss-weight",
        str(float(latent_adapter_args.latent_adapter_action_loss_weight)),
        "--latent-adapter-chunk-action-loss-weight",
        str(float(latent_adapter_args.latent_adapter_chunk_action_loss_weight)),
        "--latent-adapter-source-stat-weight",
        str(float(latent_adapter_args.latent_adapter_source_stat_weight)),
        "--latent-adapter-scale",
        str(float(latent_adapter_args.latent_adapter_scale)),
        "--latent-adapter-alignment-blend",
        str(float(latent_adapter_args.latent_adapter_alignment_blend)),
        "--latent-adapter-hyperparam-gate",
        str(latent_adapter_args.latent_adapter_hyperparam_gate),
        "--latent-adapter-action-source",
        str(latent_adapter_args.latent_adapter_action_source),
        "--latent-adapter-source-max-pairs",
        str(int(latent_adapter_args.latent_adapter_source_max_pairs)),
        "--latent-adapter-calibration-max-pairs",
        str(int(latent_adapter_args.latent_adapter_calibration_max_pairs)),
        "--action-repr-act-action-loss-backend",
        str(latent_adapter_args.action_repr_act_action_loss_backend),
        "--action-repr-post-adapter",
        str(latent_adapter_args.action_repr_post_adapter),
        "--action-repr-diffusion-backend",
        str(latent_adapter_args.action_repr_diffusion_backend),
        "--action-repr-fit-post-task-blend-max",
        str(float(latent_adapter_args.action_repr_fit_post_task_blend_max)),
        "--trajectory-adapter-hidden-dim",
        str(int(latent_adapter_args.trajectory_adapter_hidden_dim)),
        "--trajectory-adapter-epochs",
        str(int(latent_adapter_args.trajectory_adapter_epochs)),
        "--trajectory-adapter-batch-size",
        str(int(latent_adapter_args.trajectory_adapter_batch_size)),
        "--trajectory-adapter-lr",
        str(float(latent_adapter_args.trajectory_adapter_lr)),
        "--trajectory-adapter-weight-decay",
        str(float(latent_adapter_args.trajectory_adapter_weight_decay)),
        "--trajectory-adapter-scale",
        str(float(latent_adapter_args.trajectory_adapter_scale)),
        "--trajectory-adapter-first-action-weight",
        str(float(latent_adapter_args.trajectory_adapter_first_action_weight)),
        "--trajectory-adapter-plan-loss-weight",
        str(float(latent_adapter_args.trajectory_adapter_plan_loss_weight)),
        "--trajectory-adapter-smooth-loss-weight",
        str(float(latent_adapter_args.trajectory_adapter_smooth_loss_weight)),
        "--trajectory-adapter-reg-weight",
        str(float(latent_adapter_args.trajectory_adapter_reg_weight)),
        "--trajectory-adapter-max-pairs",
        str(int(latent_adapter_args.trajectory_adapter_max_pairs)),
        "--trajectory-adapter-post-adapter",
        str(latent_adapter_args.trajectory_adapter_post_adapter),
        "--denoise-adapter-hidden-dim",
        str(int(latent_adapter_args.denoise_adapter_hidden_dim)),
        "--denoise-adapter-epochs",
        str(int(latent_adapter_args.denoise_adapter_epochs)),
        "--denoise-adapter-batch-size",
        str(int(latent_adapter_args.denoise_adapter_batch_size)),
        "--denoise-adapter-lr",
        str(float(latent_adapter_args.denoise_adapter_lr)),
        "--denoise-adapter-weight-decay",
        str(float(latent_adapter_args.denoise_adapter_weight_decay)),
        "--denoise-adapter-scale",
        str(float(latent_adapter_args.denoise_adapter_scale)),
        "--denoise-adapter-trajectory-loss-weight",
        str(float(latent_adapter_args.denoise_adapter_trajectory_loss_weight)),
        "--denoise-adapter-first-action-loss-weight",
        str(float(latent_adapter_args.denoise_adapter_first_action_loss_weight)),
        "--denoise-adapter-action-window-loss-weight",
        str(float(latent_adapter_args.denoise_adapter_action_window_loss_weight)),
        "--denoise-adapter-reg-weight",
        str(float(latent_adapter_args.denoise_adapter_reg_weight)),
        "--denoise-adapter-timestep-sampling",
        str(latent_adapter_args.denoise_adapter_timestep_sampling),
        "--denoise-adapter-max-pairs",
        str(int(latent_adapter_args.denoise_adapter_max_pairs)),
        "--denoise-adapter-post-adapter",
        str(latent_adapter_args.denoise_adapter_post_adapter),
        "--closed-loop-candidates",
        str(latent_adapter_args.closed_loop_candidates),
        "--closed-loop-selection",
        str(latent_adapter_args.closed_loop_selection),
        "--closed-loop-objective",
        str(latent_adapter_args.closed_loop_objective),
        "--closed-loop-calibration-num-per-task",
        str(int(latent_adapter_args.closed_loop_calibration_num_per_task)),
        "--seed",
        str(int(seed)),
    ]
    if latent_adapter_args.denoise_adapter_task_scales:
        cmd.extend(["--denoise-adapter-task-scales", str(latent_adapter_args.denoise_adapter_task_scales)])
    if bool(latent_adapter_args.action_repr_fit_post_task_blends):
        cmd.append("--action-repr-fit-post-task-blends")
    if policy_device:
        cmd.extend(["--policy-device", str(policy_device)])
    print(
        f"[paper-protocol] eval config={config_path} profile={profile} seed={seed} "
        f"split={eval_split} output={output_dir}",
        flush=True,
    )
    subprocess.run(cmd, check=True)


def _aggregate(output_root: Path) -> tuple[Path, Path, Path]:
    success_frames: list[pd.DataFrame] = []
    transition_frames: list[pd.DataFrame] = []
    manifests: list[dict[str, object]] = []

    def with_metadata(frame: pd.DataFrame, metadata: dict[str, object]) -> pd.DataFrame:
        out = frame.copy()
        for key in metadata:
            if key in out.columns:
                out = out.drop(columns=[key])
        meta = pd.DataFrame({key: [value] * len(out) for key, value in metadata.items()})
        return pd.concat([meta, out], axis=1)

    for success_path in output_root.glob("*/**/method_success_summary.csv"):
        run_dir = success_path.parent
        parts = run_dir.relative_to(output_root).parts
        if len(parts) < 4:
            continue
        config_label, profile, seed_part, eval_split = parts[:4]
        seed = int(str(seed_part).replace("seed_", ""))
        metadata = {
            "config_label": config_label,
            "profile": profile,
            "seed": seed,
            "eval_split": eval_split,
            "run_dir": str(run_dir),
        }
        success = pd.read_csv(success_path)
        success_frames.append(with_metadata(success, metadata))
        transition_path = run_dir / "method_transition_metrics.csv"
        if transition_path.exists():
            transition = pd.read_csv(transition_path)
            transition_frames.append(with_metadata(transition, metadata))
        manifest_path = run_dir / "manifest.yaml"
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = yaml.safe_load(handle) or {}
            manifest.update(
                {
                    "config_label": config_label,
                    "profile": profile,
                    "seed": seed,
                    "eval_split": eval_split,
                    "run_dir": str(run_dir),
                }
            )
            manifests.append(manifest)
    aggregate_root = output_root / "combined"
    aggregate_root.mkdir(parents=True, exist_ok=True)
    success_out = aggregate_root / "paper_success_summary.csv"
    transition_out = aggregate_root / "paper_transition_metrics.csv"
    manifest_out = aggregate_root / "paper_run_manifest.yaml"
    if success_frames:
        pd.concat(success_frames, ignore_index=True).to_csv(success_out, index=False)
    if transition_frames:
        pd.concat(transition_frames, ignore_index=True).to_csv(transition_out, index=False)
    with manifest_out.open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"runs": manifests}, handle, sort_keys=False)
    return success_out, transition_out, manifest_out


def main() -> None:
    args = _parse_args()
    configs = _parse_specs(args.configs)
    profiles = _parse_csv(args.profiles)
    _validate_profiles(profiles)
    seeds = _parse_seeds(args.seeds)
    eval_splits = _parse_csv(args.eval_splits)
    unknown_splits = sorted(set(eval_splits).difference({"validation", "test"}))
    if unknown_splits:
        raise KeyError(f"Unknown eval splits: {unknown_splits}. Expected validation,test.")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    protocol = {
        "configs": [{"label": label, "path": path} for label, path in configs],
        "profiles": profiles,
        "seeds": seeds,
        "methods": _parse_csv(args.methods),
        "calibration_episodes": int(args.calibration_episodes),
        "validation_episodes": int(args.validation_episodes),
        "test_episodes": int(args.test_episodes),
        "exact_num_per_task": int(args.exact_num_per_task),
        "fresh_episodes_per_task": int(args.fresh_episodes_per_task),
        "adapter_blend": float(args.adapter_blend),
        "adapter_task_blends": str(args.adapter_task_blends) if args.adapter_task_blends is not None else None,
        "ridge": float(args.ridge),
        "latent_adapter": {
            "hidden_dim": int(args.latent_adapter_hidden_dim),
            "transition_hidden_dim": int(args.latent_adapter_transition_hidden_dim),
            "action_decoder_hidden_dim": int(args.latent_adapter_action_decoder_hidden_dim),
            "epochs": int(args.latent_adapter_epochs),
            "transition_epochs": int(args.latent_adapter_transition_epochs),
            "action_decoder_epochs": int(args.latent_adapter_action_decoder_epochs),
            "batch_size": int(args.latent_adapter_batch_size),
            "lr": float(args.latent_adapter_lr),
            "transition_lr": float(args.latent_adapter_transition_lr),
            "action_decoder_lr": float(args.latent_adapter_action_decoder_lr),
            "weight_decay": float(args.latent_adapter_weight_decay),
            "reg_weight": float(args.latent_adapter_reg_weight),
            "action_loss_weight": float(args.latent_adapter_action_loss_weight),
            "chunk_action_loss_weight": float(args.latent_adapter_chunk_action_loss_weight),
            "source_stat_weight": float(args.latent_adapter_source_stat_weight),
            "scale": float(args.latent_adapter_scale),
            "alignment_blend": float(args.latent_adapter_alignment_blend),
            "hyperparam_gate": str(args.latent_adapter_hyperparam_gate),
            "action_source": str(args.latent_adapter_action_source),
            "source_max_pairs": int(args.latent_adapter_source_max_pairs),
            "calibration_max_pairs": int(args.latent_adapter_calibration_max_pairs),
        },
        "action_repr": {
            "act_action_loss_backend": str(args.action_repr_act_action_loss_backend),
            "post_adapter": str(args.action_repr_post_adapter),
            "diffusion_backend": str(args.action_repr_diffusion_backend),
            "fit_post_task_blends": bool(args.action_repr_fit_post_task_blends),
            "fit_post_task_blend_max": float(args.action_repr_fit_post_task_blend_max),
        },
        "trajectory_adapter": {
            "hidden_dim": int(args.trajectory_adapter_hidden_dim),
            "epochs": int(args.trajectory_adapter_epochs),
            "batch_size": int(args.trajectory_adapter_batch_size),
            "lr": float(args.trajectory_adapter_lr),
            "weight_decay": float(args.trajectory_adapter_weight_decay),
            "scale": float(args.trajectory_adapter_scale),
            "first_action_weight": float(args.trajectory_adapter_first_action_weight),
            "plan_loss_weight": float(args.trajectory_adapter_plan_loss_weight),
            "smooth_loss_weight": float(args.trajectory_adapter_smooth_loss_weight),
            "reg_weight": float(args.trajectory_adapter_reg_weight),
            "max_pairs": int(args.trajectory_adapter_max_pairs),
            "post_adapter": str(args.trajectory_adapter_post_adapter),
        },
        "denoise_adapter": {
            "hidden_dim": int(args.denoise_adapter_hidden_dim),
            "epochs": int(args.denoise_adapter_epochs),
            "batch_size": int(args.denoise_adapter_batch_size),
            "lr": float(args.denoise_adapter_lr),
            "weight_decay": float(args.denoise_adapter_weight_decay),
            "scale": float(args.denoise_adapter_scale),
            "task_scales": str(args.denoise_adapter_task_scales) if args.denoise_adapter_task_scales else None,
            "trajectory_loss_weight": float(args.denoise_adapter_trajectory_loss_weight),
            "first_action_loss_weight": float(args.denoise_adapter_first_action_loss_weight),
            "action_window_loss_weight": float(args.denoise_adapter_action_window_loss_weight),
            "reg_weight": float(args.denoise_adapter_reg_weight),
            "timestep_sampling": str(args.denoise_adapter_timestep_sampling),
            "max_pairs": int(args.denoise_adapter_max_pairs),
            "post_adapter": str(args.denoise_adapter_post_adapter),
        },
        "closed_loop": {
            "candidates": str(args.closed_loop_candidates),
            "selection": str(args.closed_loop_selection),
            "objective": str(args.closed_loop_objective),
            "calibration_num_per_task": int(args.closed_loop_calibration_num_per_task),
        },
        "max_attempts_per_episode": int(args.max_attempts_per_episode),
        "policy_device": args.policy_device,
        "eval_splits": eval_splits,
        "skip_existing_eval": bool(args.skip_existing_eval),
        "separate_method_runs": bool(args.separate_method_runs),
        "split_source_root": args.split_source_root,
        "rule": "Use validation for method selection only. Report paper claims from test split only.",
    }
    with (output_root / "protocol.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(protocol, handle, sort_keys=False)

    if not args.aggregate_only:
        for config_label, config_path in configs:
            for profile in profiles:
                for seed in seeds:
                    run_root = output_root / config_label / profile / f"seed_{seed}"
                    if args.split_source_root:
                        split_source_root = Path(args.split_source_root) / config_label / profile / f"seed_{seed}"
                        split_paths = _split_paths(split_source_root)
                        missing = [str(path) for path in split_paths.values() if not path.exists()]
                        if missing:
                            raise FileNotFoundError(
                                "Missing source split files for "
                                f"{config_label}/{profile}/seed_{seed}: {missing}"
                            )
                    else:
                        split_paths = _generate_splits(
                            config_path=config_path,
                            profile=profile,
                            seed=int(seed),
                            root=run_root,
                            calibration_episodes=int(args.calibration_episodes),
                            validation_episodes=int(args.validation_episodes),
                            test_episodes=int(args.test_episodes),
                            force=bool(args.force_regenerate),
                            max_attempts_per_episode=int(args.max_attempts_per_episode),
                        )
                    if args.skip_eval:
                        continue
                    for eval_split in eval_splits:
                        method_groups = _parse_csv(args.methods) if args.separate_method_runs else [str(args.methods)]
                        for method_group in method_groups:
                            method_output_dir = run_root / eval_split / method_group if args.separate_method_runs else run_root / eval_split
                            _run_method_comparison(
                                config_path=config_path,
                                profile=profile,
                                seed=int(seed),
                                eval_split=eval_split,
                                split_paths=split_paths,
                                output_dir=method_output_dir,
                                methods=str(method_group),
                                policy_device=args.policy_device,
                                exact_num_per_task=int(args.exact_num_per_task),
                                fresh_episodes_per_task=int(args.fresh_episodes_per_task),
                                adapter_blend=float(args.adapter_blend),
                                ridge=float(args.ridge),
                                latent_adapter_args=args,
                                skip_existing=bool(args.skip_existing_eval),
                            )

    success_out, transition_out, manifest_out = _aggregate(output_root)
    print(f"paper_success_summary_csv={success_out}")
    print(f"paper_transition_metrics_csv={transition_out}")
    print(f"paper_run_manifest_yaml={manifest_out}")


if __name__ == "__main__":
    main()

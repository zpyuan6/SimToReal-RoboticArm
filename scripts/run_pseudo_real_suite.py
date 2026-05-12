from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd
import yaml

from ttla.config import load_config
from ttla.evaluation import evaluate_checkpoint
from ttla.training import calibrate_adapter, calibrate_static_adapter, finetune_few_shot, fit_latent_alignment
from ttla.utils.io import ensure_dir


DEFAULT_BACKBONES = ["feedforward", "recurrent", "chunking", "language", "diffusion"]
DEFAULT_BASELINES = [
    "no_adaptation",
    "input_normalization",
    "probe_feature_alignment",
    "static_adapter",
    "few_shot_finetuning",
    "tent_style",
    "ours",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--backbones", nargs="*", default=DEFAULT_BACKBONES)
    parser.add_argument("--baselines", nargs="*", default=DEFAULT_BASELINES)
    parser.add_argument("--frozen-root", default="results/fixed_protocol/backbone_suite")
    parser.add_argument("--tag", default="")
    parser.add_argument("--adapt-policy-preservation-weight", type=float, default=None)
    parser.add_argument("--adapt-policy-preservation-alpha", type=float, default=None)
    parser.add_argument("--adapt-policy-preservation-stage-max", type=int, default=None)
    parser.add_argument("--adapt-policy-preservation-mode", choices=["full", "family"], default=None)
    parser.add_argument("--adapt-policy-preservation-global-gate", action="store_true")
    parser.add_argument("--adapt-policy-preservation-global-threshold", type=float, default=None)
    parser.add_argument("--adapt-policy-preservation-global-slope", type=float, default=None)
    parser.add_argument("--adapt-policy-preservation-observation-only", action="store_true")
    parser.add_argument("--adapt-policy-preservation-observation-scale", type=float, default=None)
    parser.add_argument("--adapt-policy-preservation-non-observation-scale", type=float, default=None)
    parser.add_argument("--adapt-policy-supervision-weight", type=float, default=None)
    parser.add_argument("--adapt-inverse-supervision-weight", type=float, default=None)
    parser.add_argument("--adapt-lr", type=float, default=None)
    parser.add_argument("--adapt-epochs", type=int, default=None)
    parser.add_argument("--adapt-batch-size", type=int, default=None)
    parser.add_argument("--adapt-reg-weight", type=float, default=None)
    parser.add_argument("--adapt-stage-weight", type=float, default=None)
    parser.add_argument("--adapt-auto-stage-scaling", action="store_true")
    parser.add_argument("--adapt-auto-stage-scale-min", type=float, default=None)
    parser.add_argument("--adapt-auto-stage-scale-max", type=float, default=None)
    parser.add_argument("--adapt-auto-stage-scale-power", type=float, default=None)
    parser.add_argument("--adapt-auto-non-observation-scaling", action="store_true")
    parser.add_argument("--adapt-auto-non-observation-scale-min", type=float, default=None)
    parser.add_argument("--adapt-auto-non-observation-scale-max", type=float, default=None)
    parser.add_argument("--adapt-auto-non-observation-global-threshold", type=float, default=None)
    parser.add_argument("--adapt-auto-non-observation-global-slope", type=float, default=None)
    parser.add_argument("--adapt-auto-latent-affine-scaling", action="store_true")
    parser.add_argument("--adapt-auto-latent-affine-blend-min", type=float, default=None)
    parser.add_argument("--adapt-auto-latent-affine-blend-max", type=float, default=None)
    parser.add_argument("--adapt-auto-latent-affine-global-threshold", type=float, default=None)
    parser.add_argument("--adapt-auto-latent-affine-global-slope", type=float, default=None)
    parser.add_argument("--adapt-static-alignment-weight", type=float, default=None)
    parser.add_argument("--adapt-warm-start-static-adapter", action="store_true")
    parser.add_argument("--adapt-source-alignment-weight", type=float, default=None)
    parser.add_argument("--adapt-source-primitive-alignment-weight", type=float, default=None)
    parser.add_argument("--adapt-source-delta-alignment-weight", type=float, default=None)
    parser.add_argument("--adapt-source-policy-alignment-weight", type=float, default=None)
    parser.add_argument("--adapt-source-anchor-observation-scale", type=float, default=None)
    parser.add_argument("--adapt-source-anchor-non-observation-scale", type=float, default=None)
    parser.add_argument("--adapt-auto-source-anchor-observation-scale", action="store_true")
    parser.add_argument("--adapt-auto-source-anchor-non-observation-balance", action="store_true")
    parser.add_argument("--adapt-auto-source-anchor-balance-min", type=float, default=None)
    parser.add_argument("--adapt-auto-source-anchor-balance-max", type=float, default=None)
    parser.add_argument("--adapt-auto-source-anchor-balance-threshold", type=float, default=None)
    parser.add_argument("--adapt-auto-source-anchor-balance-slope", type=float, default=None)
    parser.add_argument("--adapt-auto-source-anchor-scaling", action="store_true")
    parser.add_argument("--adapt-auto-source-anchor-scale-min", type=float, default=None)
    parser.add_argument("--adapt-auto-source-anchor-scale-max", type=float, default=None)
    parser.add_argument("--adapt-auto-source-anchor-global-threshold", type=float, default=None)
    parser.add_argument("--adapt-auto-source-anchor-global-slope", type=float, default=None)
    parser.add_argument("--adapt-source-local-prototypes-per-primitive", type=int, default=None)
    parser.add_argument("--adapt-source-local-prototype-iters", type=int, default=None)
    parser.add_argument("--adapt-residual-adaptive-reg-alpha", type=float, default=None)
    parser.add_argument("--adapt-residual-adaptive-reg-gamma", type=float, default=None)
    parser.add_argument("--adapt-residual-adaptive-adapt-alpha", type=float, default=None)
    parser.add_argument("--adapt-residual-adaptive-adapt-gamma", type=float, default=None)
    parser.add_argument("--adapt-residual-adaptive-adapt-center", type=float, default=None)
    parser.add_argument("--adapt-residual-adaptive-adapt-min-weight", type=float, default=None)
    parser.add_argument("--adapt-residual-adaptive-adapt-no-normalize", action="store_true")
    parser.add_argument("--adapt-observation-weight", type=float, default=None)
    parser.add_argument("--adapt-precontact-weight", type=float, default=None)
    parser.add_argument("--adapt-postcontact-weight", type=float, default=None)
    parser.add_argument("--adapt-abort-weight", type=float, default=None)
    parser.add_argument("--adapt-task-balance-power", type=float, default=None)
    parser.add_argument("--adapt-stage-balance-power", type=float, default=None)
    parser.add_argument("--adapt-level3-late-stage-boost", type=float, default=None)
    parser.add_argument("--adapt-gate-identity-weight", type=float, default=None)
    parser.add_argument("--adapt-policy-adapter-identity-weight", type=float, default=None)
    parser.add_argument("--adapt-transition-adapter-identity-weight", type=float, default=None)
    parser.add_argument("--model-adapter-progressive-min-scale", type=float, default=None)
    parser.add_argument("--model-adapter-progressive-max-scale", type=float, default=None)
    parser.add_argument("--model-adapter-condition-start-stage", type=int, default=None)
    parser.add_argument("--model-adapter-mode", choices=["full", "legacy_prevprim"], default=None)
    parser.add_argument("--model-adapter-scale", type=float, default=None)
    parser.add_argument("--model-adapter-use-gate", action="store_true")
    parser.add_argument("--model-adapter-disable-gate", action="store_true")
    parser.add_argument("--model-adapter-phase-split", action="store_true")
    parser.add_argument("--model-adapter-condition-observation-only", action="store_true")
    parser.add_argument("--model-adapter-condition-non-observation-scale", type=float, default=None)
    parser.add_argument("--model-latent-affine-enable", action="store_true")
    parser.add_argument("--model-latent-affine-disable", action="store_true")
    parser.add_argument("--model-latent-affine-task-conditioned-enable", action="store_true")
    parser.add_argument("--model-latent-affine-task-conditioned-disable", action="store_true")
    parser.add_argument("--model-latent-affine-max-scale", type=float, default=None)
    parser.add_argument("--model-latent-affine-blend", type=float, default=None)
    parser.add_argument("--model-transition-action-adapter-enable", action="store_true")
    parser.add_argument("--model-transition-action-adapter-disable", action="store_true")
    parser.add_argument("--model-transition-action-adapter-scale", type=float, default=None)
    parser.add_argument("--model-transition-residual-adapter-enable", action="store_true")
    parser.add_argument("--model-transition-residual-adapter-disable", action="store_true")
    parser.add_argument("--model-transition-residual-hidden-dim", type=int, default=None)
    parser.add_argument("--model-transition-residual-scale", type=float, default=None)
    parser.add_argument("--model-transition-residual-phase-split", action="store_true")
    parser.add_argument("--model-transition-residual-observation-scale", type=float, default=None)
    parser.add_argument("--model-transition-residual-non-observation-scale", type=float, default=None)
    parser.add_argument("--model-policy-residual-adapter-enable", action="store_true")
    parser.add_argument("--model-policy-residual-adapter-disable", action="store_true")
    parser.add_argument("--model-policy-residual-hidden-dim", type=int, default=None)
    parser.add_argument("--model-policy-residual-scale", type=float, default=None)
    parser.add_argument("--train-inverse-loss-weight", type=float, default=None)
    parser.add_argument("--model-stage-hard-mask", action="store_true")
    parser.add_argument("--model-recurrent-disable-feature-adapter", action="store_true")
    parser.add_argument("--model-recurrent-disable-latent-adapter", action="store_true")
    return parser.parse_args()


def _suite_root(cfg: dict, tag: str = "") -> Path:
    suffix = "backbone_suite" if not tag else f"backbone_suite_{tag}"
    return ensure_dir(Path(cfg["paths"]["results_root"]) / suffix)


def _deep_update(target: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _prepare_cfg(base_cfg: dict, backbone: str, suite_root: Path) -> tuple[dict, Path]:
    cfg = copy.deepcopy(base_cfg)
    overrides = copy.deepcopy(base_cfg.get("backbone_overrides", {}).get(backbone, {}))
    if overrides:
        _deep_update(cfg, overrides)
    root = suite_root / backbone
    cfg["model"]["backbone_type"] = backbone
    cfg["paths"]["results_root"] = str(root)
    cfg["paths"]["checkpoint_dir"] = str(root / "checkpoints")
    ensure_dir(root)
    ensure_dir(root / "checkpoints")
    return cfg, root


def _write_cfg_snapshot(cfg: dict, root: Path) -> None:
    with (root / "config_snapshot.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def _summarize(csv_paths: list[Path], backbone: str, output_path: Path) -> pd.DataFrame:
    merged = pd.concat([pd.read_csv(path) for path in csv_paths], ignore_index=True)
    merged.insert(0, "backbone", backbone)
    merged.to_csv(output_path, index=False)
    return merged


def main() -> None:
    args = _parse_args()
    base_cfg = load_config(args.config)
    calibration_path = Path(base_cfg["paths"]["data_root"]) / "calibration.npz"
    source_train_path = Path(base_cfg.get("pseudo_real", {}).get("source_train_path", "data/processed/train.npz"))
    suite_root = _suite_root(base_cfg, args.tag)
    frozen_root = Path(args.frozen_root)
    suite_records: list[pd.DataFrame] = []
    checkpoint_records: list[dict[str, str]] = []

    for backbone in args.backbones:
        cfg, backbone_root = _prepare_cfg(base_cfg, backbone, suite_root)
        if args.adapt_policy_preservation_weight is not None:
            cfg.setdefault("adaptation", {})["policy_preservation_weight"] = float(args.adapt_policy_preservation_weight)
        if args.adapt_policy_preservation_alpha is not None:
            cfg.setdefault("adaptation", {})["policy_preservation_alpha"] = float(args.adapt_policy_preservation_alpha)
        if args.adapt_policy_preservation_stage_max is not None:
            cfg.setdefault("adaptation", {})["policy_preservation_stage_max"] = int(args.adapt_policy_preservation_stage_max)
        if args.adapt_policy_preservation_mode is not None:
            cfg.setdefault("adaptation", {})["policy_preservation_mode"] = str(args.adapt_policy_preservation_mode)
        if args.adapt_policy_preservation_global_gate:
            cfg.setdefault("adaptation", {})["policy_preservation_global_gate"] = True
        if args.adapt_policy_preservation_global_threshold is not None:
            cfg.setdefault("adaptation", {})["policy_preservation_global_threshold"] = float(args.adapt_policy_preservation_global_threshold)
        if args.adapt_policy_preservation_global_slope is not None:
            cfg.setdefault("adaptation", {})["policy_preservation_global_slope"] = float(args.adapt_policy_preservation_global_slope)
        if args.adapt_policy_preservation_observation_only:
            cfg.setdefault("adaptation", {})["policy_preservation_observation_only"] = True
        if args.adapt_policy_preservation_observation_scale is not None:
            cfg.setdefault("adaptation", {})["policy_preservation_observation_scale"] = float(
                args.adapt_policy_preservation_observation_scale
            )
        if args.adapt_policy_preservation_non_observation_scale is not None:
            cfg.setdefault("adaptation", {})["policy_preservation_non_observation_scale"] = float(
                args.adapt_policy_preservation_non_observation_scale
            )
        if args.adapt_policy_supervision_weight is not None:
            cfg.setdefault("adaptation", {})["policy_supervision_weight"] = float(
                args.adapt_policy_supervision_weight
            )
        if args.adapt_inverse_supervision_weight is not None:
            cfg.setdefault("adaptation", {})["inverse_supervision_weight"] = float(
                args.adapt_inverse_supervision_weight
            )
        if args.train_inverse_loss_weight is not None:
            cfg.setdefault("train", {})["inverse_loss_weight"] = float(args.train_inverse_loss_weight)
        if args.adapt_lr is not None:
            cfg.setdefault("adaptation", {})["lr"] = float(args.adapt_lr)
        if args.adapt_epochs is not None:
            cfg.setdefault("adaptation", {})["epochs"] = int(args.adapt_epochs)
        if args.adapt_batch_size is not None:
            cfg.setdefault("adaptation", {})["batch_size"] = int(args.adapt_batch_size)
        if args.adapt_reg_weight is not None:
            cfg.setdefault("adaptation", {})["adapter_reg_weight"] = float(args.adapt_reg_weight)
        if args.adapt_stage_weight is not None:
            cfg.setdefault("adaptation", {})["adapter_stage_loss_weight"] = float(args.adapt_stage_weight)
        if args.adapt_auto_stage_scaling:
            cfg.setdefault("adaptation", {})["auto_stage_scaling"] = True
        if args.adapt_auto_stage_scale_min is not None:
            cfg.setdefault("adaptation", {})["auto_stage_scale_min"] = float(args.adapt_auto_stage_scale_min)
        if args.adapt_auto_stage_scale_max is not None:
            cfg.setdefault("adaptation", {})["auto_stage_scale_max"] = float(args.adapt_auto_stage_scale_max)
        if args.adapt_auto_stage_scale_power is not None:
            cfg.setdefault("adaptation", {})["auto_stage_scale_power"] = float(args.adapt_auto_stage_scale_power)
        if args.adapt_auto_non_observation_scaling:
            cfg.setdefault("adaptation", {})["auto_non_observation_scaling"] = True
        if args.adapt_auto_non_observation_scale_min is not None:
            cfg.setdefault("adaptation", {})["auto_non_observation_scale_min"] = float(args.adapt_auto_non_observation_scale_min)
        if args.adapt_auto_non_observation_scale_max is not None:
            cfg.setdefault("adaptation", {})["auto_non_observation_scale_max"] = float(args.adapt_auto_non_observation_scale_max)
        if args.adapt_auto_non_observation_global_threshold is not None:
            cfg.setdefault("adaptation", {})["auto_non_observation_global_threshold"] = float(args.adapt_auto_non_observation_global_threshold)
        if args.adapt_auto_non_observation_global_slope is not None:
            cfg.setdefault("adaptation", {})["auto_non_observation_global_slope"] = float(args.adapt_auto_non_observation_global_slope)
        if args.adapt_auto_latent_affine_scaling:
            cfg.setdefault("adaptation", {})["auto_latent_affine_scaling"] = True
        if args.adapt_auto_latent_affine_blend_min is not None:
            cfg.setdefault("adaptation", {})["auto_latent_affine_blend_min"] = float(args.adapt_auto_latent_affine_blend_min)
        if args.adapt_auto_latent_affine_blend_max is not None:
            cfg.setdefault("adaptation", {})["auto_latent_affine_blend_max"] = float(args.adapt_auto_latent_affine_blend_max)
        if args.adapt_auto_latent_affine_global_threshold is not None:
            cfg.setdefault("adaptation", {})["auto_latent_affine_global_threshold"] = float(args.adapt_auto_latent_affine_global_threshold)
        if args.adapt_auto_latent_affine_global_slope is not None:
            cfg.setdefault("adaptation", {})["auto_latent_affine_global_slope"] = float(args.adapt_auto_latent_affine_global_slope)
        if args.adapt_static_alignment_weight is not None:
            cfg.setdefault("adaptation", {})["static_alignment_weight"] = float(args.adapt_static_alignment_weight)
        if args.adapt_warm_start_static_adapter:
            cfg.setdefault("adaptation", {})["warm_start_static_adapter"] = True
        if args.adapt_source_alignment_weight is not None:
            cfg.setdefault("adaptation", {})["source_alignment_weight"] = float(args.adapt_source_alignment_weight)
        if args.adapt_source_primitive_alignment_weight is not None:
            cfg.setdefault("adaptation", {})["source_primitive_alignment_weight"] = float(args.adapt_source_primitive_alignment_weight)
        if args.adapt_source_delta_alignment_weight is not None:
            cfg.setdefault("adaptation", {})["source_delta_alignment_weight"] = float(args.adapt_source_delta_alignment_weight)
        if args.adapt_source_policy_alignment_weight is not None:
            cfg.setdefault("adaptation", {})["source_policy_alignment_weight"] = float(args.adapt_source_policy_alignment_weight)
        if args.adapt_source_anchor_observation_scale is not None:
            cfg.setdefault("adaptation", {})["source_anchor_observation_scale"] = float(
                args.adapt_source_anchor_observation_scale
            )
        if args.adapt_source_anchor_non_observation_scale is not None:
            cfg.setdefault("adaptation", {})["source_anchor_non_observation_scale"] = float(
                args.adapt_source_anchor_non_observation_scale
            )
        if args.adapt_auto_source_anchor_observation_scale:
            cfg.setdefault("adaptation", {})["auto_source_anchor_observation_scale"] = True
        if args.adapt_auto_source_anchor_non_observation_balance:
            cfg.setdefault("adaptation", {})["auto_source_anchor_non_observation_balance"] = True
        if args.adapt_auto_source_anchor_balance_min is not None:
            cfg.setdefault("adaptation", {})["auto_source_anchor_balance_min"] = float(
                args.adapt_auto_source_anchor_balance_min
            )
        if args.adapt_auto_source_anchor_balance_max is not None:
            cfg.setdefault("adaptation", {})["auto_source_anchor_balance_max"] = float(
                args.adapt_auto_source_anchor_balance_max
            )
        if args.adapt_auto_source_anchor_balance_threshold is not None:
            cfg.setdefault("adaptation", {})["auto_source_anchor_balance_threshold"] = float(
                args.adapt_auto_source_anchor_balance_threshold
            )
        if args.adapt_auto_source_anchor_balance_slope is not None:
            cfg.setdefault("adaptation", {})["auto_source_anchor_balance_slope"] = float(
                args.adapt_auto_source_anchor_balance_slope
            )
        if args.adapt_auto_source_anchor_scaling:
            cfg.setdefault("adaptation", {})["auto_source_anchor_scaling"] = True
        if args.adapt_auto_source_anchor_scale_min is not None:
            cfg.setdefault("adaptation", {})["auto_source_anchor_scale_min"] = float(args.adapt_auto_source_anchor_scale_min)
        if args.adapt_auto_source_anchor_scale_max is not None:
            cfg.setdefault("adaptation", {})["auto_source_anchor_scale_max"] = float(args.adapt_auto_source_anchor_scale_max)
        if args.adapt_auto_source_anchor_global_threshold is not None:
            cfg.setdefault("adaptation", {})["auto_source_anchor_global_threshold"] = float(args.adapt_auto_source_anchor_global_threshold)
        if args.adapt_auto_source_anchor_global_slope is not None:
            cfg.setdefault("adaptation", {})["auto_source_anchor_global_slope"] = float(args.adapt_auto_source_anchor_global_slope)
        if args.adapt_source_local_prototypes_per_primitive is not None:
            cfg.setdefault("adaptation", {})["source_local_prototypes_per_primitive"] = int(args.adapt_source_local_prototypes_per_primitive)
        if args.adapt_source_local_prototype_iters is not None:
            cfg.setdefault("adaptation", {})["source_local_prototype_iters"] = int(args.adapt_source_local_prototype_iters)
        if args.adapt_residual_adaptive_reg_alpha is not None:
            cfg.setdefault("adaptation", {})["residual_adaptive_reg_alpha"] = float(args.adapt_residual_adaptive_reg_alpha)
        if args.adapt_residual_adaptive_reg_gamma is not None:
            cfg.setdefault("adaptation", {})["residual_adaptive_reg_gamma"] = float(args.adapt_residual_adaptive_reg_gamma)
        if args.adapt_residual_adaptive_adapt_alpha is not None:
            cfg.setdefault("adaptation", {})["residual_adaptive_adapt_alpha"] = float(args.adapt_residual_adaptive_adapt_alpha)
        if args.adapt_residual_adaptive_adapt_gamma is not None:
            cfg.setdefault("adaptation", {})["residual_adaptive_adapt_gamma"] = float(args.adapt_residual_adaptive_adapt_gamma)
        if args.adapt_residual_adaptive_adapt_center is not None:
            cfg.setdefault("adaptation", {})["residual_adaptive_adapt_center"] = float(args.adapt_residual_adaptive_adapt_center)
        if args.adapt_residual_adaptive_adapt_min_weight is not None:
            cfg.setdefault("adaptation", {})["residual_adaptive_adapt_min_weight"] = float(args.adapt_residual_adaptive_adapt_min_weight)
        if args.adapt_residual_adaptive_adapt_no_normalize:
            cfg.setdefault("adaptation", {})["residual_adaptive_adapt_normalize"] = False
        if args.adapt_observation_weight is not None:
            cfg.setdefault("adaptation", {})["observation_adapt_weight"] = float(args.adapt_observation_weight)
        if args.adapt_precontact_weight is not None:
            cfg.setdefault("adaptation", {})["precontact_adapt_weight"] = float(args.adapt_precontact_weight)
        if args.adapt_postcontact_weight is not None:
            cfg.setdefault("adaptation", {})["postcontact_adapt_weight"] = float(args.adapt_postcontact_weight)
        if args.adapt_abort_weight is not None:
            cfg.setdefault("adaptation", {})["abort_adapt_weight"] = float(args.adapt_abort_weight)
        if args.adapt_task_balance_power is not None:
            cfg.setdefault("adaptation", {})["task_balance_power"] = float(args.adapt_task_balance_power)
        if args.adapt_stage_balance_power is not None:
            cfg.setdefault("adaptation", {})["stage_balance_power"] = float(args.adapt_stage_balance_power)
        if args.adapt_level3_late_stage_boost is not None:
            cfg.setdefault("adaptation", {})["level3_late_stage_boost"] = float(args.adapt_level3_late_stage_boost)
        if args.adapt_gate_identity_weight is not None:
            cfg.setdefault("adaptation", {})["gate_identity_weight"] = float(args.adapt_gate_identity_weight)
        if args.adapt_policy_adapter_identity_weight is not None:
            cfg.setdefault("adaptation", {})["policy_adapter_identity_weight"] = float(
                args.adapt_policy_adapter_identity_weight
            )
        if args.adapt_transition_adapter_identity_weight is not None:
            cfg.setdefault("adaptation", {})["transition_adapter_identity_weight"] = float(
                args.adapt_transition_adapter_identity_weight
            )
        if args.model_adapter_progressive_min_scale is not None:
            cfg.setdefault("model", {})["adapter_progressive_min_scale"] = float(args.model_adapter_progressive_min_scale)
        if args.model_adapter_progressive_max_scale is not None:
            cfg.setdefault("model", {})["adapter_progressive_max_scale"] = float(args.model_adapter_progressive_max_scale)
        if args.model_adapter_condition_start_stage is not None:
            cfg.setdefault("model", {})["adapter_condition_start_stage"] = int(args.model_adapter_condition_start_stage)
        if args.model_adapter_mode is not None:
            cfg.setdefault("model", {})["adapter_mode"] = str(args.model_adapter_mode)
        if args.model_adapter_scale is not None:
            cfg.setdefault("model", {})["adapter_scale"] = float(args.model_adapter_scale)
        if args.model_adapter_use_gate and args.model_adapter_disable_gate:
            raise ValueError("Cannot enable and disable adapter gate at the same time.")
        if args.model_adapter_use_gate:
            cfg.setdefault("model", {})["adapter_use_gate"] = True
        if args.model_adapter_disable_gate:
            cfg.setdefault("model", {})["adapter_use_gate"] = False
        if args.model_adapter_phase_split:
            cfg.setdefault("model", {})["adapter_phase_split"] = True
        if args.model_adapter_condition_observation_only:
            cfg.setdefault("model", {})["adapter_condition_observation_only"] = True
        if args.model_adapter_condition_non_observation_scale is not None:
            cfg.setdefault("model", {})["adapter_condition_non_observation_scale"] = float(args.model_adapter_condition_non_observation_scale)
        if args.model_latent_affine_enable and args.model_latent_affine_disable:
            raise ValueError("Cannot enable and disable latent affine alignment at the same time.")
        if args.model_latent_affine_enable:
            cfg.setdefault("model", {})["latent_affine_alignment"] = True
        if args.model_latent_affine_disable:
            cfg.setdefault("model", {})["latent_affine_alignment"] = False
        if (
            args.model_latent_affine_task_conditioned_enable
            and args.model_latent_affine_task_conditioned_disable
        ):
            raise ValueError("Cannot enable and disable task-conditioned latent affine alignment at the same time.")
        if args.model_latent_affine_task_conditioned_enable:
            cfg.setdefault("model", {})["latent_affine_task_conditioned"] = True
        if args.model_latent_affine_task_conditioned_disable:
            cfg.setdefault("model", {})["latent_affine_task_conditioned"] = False
        if args.model_latent_affine_max_scale is not None:
            cfg.setdefault("model", {})["latent_affine_max_scale"] = float(args.model_latent_affine_max_scale)
        if args.model_latent_affine_blend is not None:
            cfg.setdefault("model", {})["latent_affine_blend"] = float(args.model_latent_affine_blend)
        if args.model_transition_action_adapter_enable and args.model_transition_action_adapter_disable:
            raise ValueError("Cannot enable and disable transition action adapter at the same time.")
        if args.model_transition_action_adapter_enable:
            cfg.setdefault("model", {})["transition_action_adapter"] = True
        if args.model_transition_action_adapter_disable:
            cfg.setdefault("model", {})["transition_action_adapter"] = False
        if args.model_transition_action_adapter_scale is not None:
            cfg.setdefault("model", {})["transition_action_adapter_scale"] = float(
                args.model_transition_action_adapter_scale
            )
        if args.model_transition_residual_adapter_enable and args.model_transition_residual_adapter_disable:
            raise ValueError("Cannot enable and disable transition residual adapter at the same time.")
        if args.model_transition_residual_adapter_enable:
            cfg.setdefault("model", {})["transition_residual_adapter"] = True
        if args.model_transition_residual_adapter_disable:
            cfg.setdefault("model", {})["transition_residual_adapter"] = False
        if args.model_transition_residual_hidden_dim is not None:
            cfg.setdefault("model", {})["transition_residual_hidden_dim"] = int(
                args.model_transition_residual_hidden_dim
            )
        if args.model_transition_residual_scale is not None:
            cfg.setdefault("model", {})["transition_residual_scale"] = float(
                args.model_transition_residual_scale
            )
        if args.model_transition_residual_phase_split:
            cfg.setdefault("model", {})["transition_residual_phase_split"] = True
        if args.model_transition_residual_observation_scale is not None:
            cfg.setdefault("model", {})["transition_residual_observation_scale"] = float(
                args.model_transition_residual_observation_scale
            )
        if args.model_transition_residual_non_observation_scale is not None:
            cfg.setdefault("model", {})["transition_residual_non_observation_scale"] = float(
                args.model_transition_residual_non_observation_scale
            )
        if args.model_policy_residual_adapter_enable and args.model_policy_residual_adapter_disable:
            raise ValueError("Cannot enable and disable policy residual adapter at the same time.")
        if args.model_policy_residual_adapter_enable:
            cfg.setdefault("model", {})["policy_residual_adapter"] = True
        if args.model_policy_residual_adapter_disable:
            cfg.setdefault("model", {})["policy_residual_adapter"] = False
        if args.model_policy_residual_hidden_dim is not None:
            cfg.setdefault("model", {})["policy_residual_hidden_dim"] = int(args.model_policy_residual_hidden_dim)
        if args.model_policy_residual_scale is not None:
            cfg.setdefault("model", {})["policy_residual_scale"] = float(args.model_policy_residual_scale)
        if args.model_stage_hard_mask:
            cfg.setdefault("model", {})["stage_hard_mask"] = True
        if args.model_recurrent_disable_feature_adapter:
            cfg.setdefault("model", {})["recurrent_adapter_use_feature"] = False
        if args.model_recurrent_disable_latent_adapter:
            cfg.setdefault("model", {})["recurrent_adapter_use_latent"] = False
        _write_cfg_snapshot(cfg, backbone_root)
        frozen_checkpoint = frozen_root / backbone / "checkpoints" / "best_model.pt"
        if not frozen_checkpoint.exists():
            raise FileNotFoundError(f"Missing frozen checkpoint for backbone={backbone}: {frozen_checkpoint}")
        adapter_path: Path | None = None
        static_adapter_path: Path | None = None
        few_shot_path: Path | None = None
        latent_alignment_path: Path | None = None
        warm_start_static = bool(cfg.get("adaptation", {}).get("warm_start_static_adapter", False))
        need_static_adapter = "static_adapter" in args.baselines or warm_start_static
        if need_static_adapter:
            static_adapter_path = calibrate_static_adapter(cfg, frozen_checkpoint, source_train_path, calibration_path)
        if "ours" in args.baselines:
            adapter_init_path = static_adapter_path if (warm_start_static and static_adapter_path is not None) else frozen_checkpoint
            adapter_path = calibrate_adapter(
                cfg,
                adapter_init_path,
                calibration_path,
                static_teacher_path=static_adapter_path if (warm_start_static and static_adapter_path is not None) else None,
            )
        if "few_shot_finetuning" in args.baselines:
            few_shot_path = finetune_few_shot(cfg, frozen_checkpoint, calibration_path)
        if "probe_feature_alignment" in args.baselines:
            latent_alignment_path = fit_latent_alignment(cfg, frozen_checkpoint, source_train_path, calibration_path)
        checkpoint_records.append(
            {
                "backbone": backbone,
                "frozen_checkpoint_path": str(frozen_checkpoint),
                "adapter_path": str(adapter_path) if adapter_path is not None else "",
                "static_adapter_path": str(static_adapter_path) if static_adapter_path is not None else "",
                "few_shot_path": str(few_shot_path) if few_shot_path is not None else "",
                "latent_alignment_path": str(latent_alignment_path) if latent_alignment_path is not None else "",
            }
        )
        csv_paths: list[Path] = []
        for baseline in args.baselines:
            if baseline == "ours":
                model_path = adapter_path
            elif baseline == "static_adapter":
                model_path = static_adapter_path
            elif baseline == "few_shot_finetuning":
                model_path = few_shot_path
            else:
                model_path = frozen_checkpoint
            if model_path is None:
                raise RuntimeError(f"Missing checkpoint for baseline={baseline}, backbone={backbone}")
            csv_path = backbone_root / f"{baseline}.csv"
            artifacts: dict[str, str] = {}
            if baseline == "probe_feature_alignment" and latent_alignment_path is not None:
                artifacts["latent_alignment_path"] = str(latent_alignment_path)
            evaluate_checkpoint(cfg, model_path, baseline, csv_path, baseline_artifacts=artifacts)
            csv_paths.append(csv_path)
        merged = _summarize(csv_paths, backbone, backbone_root / "summary.csv")
        summary = (
            merged.groupby(["backbone", "baseline", "task"], as_index=False)
            .agg(success_rate=("success", "mean"), mean_steps=("steps", "mean"), mean_visibility=("visibility", "mean"))
        )
        summary.to_csv(backbone_root / "summary_metrics.csv", index=False)
        suite_records.append(summary)

    all_summary = pd.concat(suite_records, ignore_index=True)
    all_summary.to_csv(suite_root / "suite_summary_metrics.csv", index=False)
    overall = (
        all_summary.groupby(["backbone", "baseline"], as_index=False)
        .agg(
            mean_success_rate=("success_rate", "mean"),
            mean_steps=("mean_steps", "mean"),
            mean_visibility=("mean_visibility", "mean"),
        )
        .sort_values(["mean_success_rate", "mean_visibility"], ascending=[False, False])
    )
    overall.to_csv(suite_root / "suite_overall_metrics.csv", index=False)
    pd.DataFrame.from_records(checkpoint_records).to_csv(suite_root / "suite_checkpoints.csv", index=False)


if __name__ == "__main__":
    main()

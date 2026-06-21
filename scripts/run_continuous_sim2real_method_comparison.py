from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from run_continuous_sim2real_bridge import (
    ContinuousActionAdapter,
    _build_backbone,
    _cfg_for_profile,
    _evaluate_exact_replay,
    _evaluate_fresh_rollout,
    _fit_adapter,
    _flatten_success_summary,
    _generate_split,
    _policy_path,
    _predict_dataset,
    _parse_task_blends,
    _transition_metrics,
)
from run_continuous_sim2real_stress import PROFILE_CONTEXTS
from ttla.adaptation import (
    fit_continuous_latent_adapter,
    fit_continuous_trajectory_adapter,
    fit_diffusion_denoise_adapter,
)
from ttla.config import load_config
from ttla.control import ControlObservationBatch, ControlPolicyOutput
from ttla.sim.task_defs import ID_TO_TASK


METHODS = {
    "no_adaptation": {
        "adapter": "none",
        "input_normalization": False,
    },
    "input_normalization": {
        "adapter": "none",
        "input_normalization": True,
    },
    "probe_feature_alignment": {
        "adapter": "task_moment",
        "input_normalization": False,
    },
    "static_adapter": {
        "adapter": "task_bias",
        "input_normalization": False,
    },
    "few_shot_finetuning": {
        "adapter": "residual_mlp",
        "input_normalization": False,
    },
    "tent_style": {
        "adapter": "none",
        "input_normalization": False,
        "policy_mode": "raw_normalized_ensemble",
    },
    "domain_randomization_only": {
        "adapter": "none",
        "input_normalization": False,
        "policy_mode": "domain_randomization",
    },
    "task_policy_selector": {
        "adapter": "none",
        "input_normalization": False,
        "policy_mode": "task_policy_selector",
    },
    "ours_proxy": {
        "adapter": "task_affine",
        "input_normalization": False,
    },
    "ours_task_gated_residual": {
        "adapter": "residual_mlp",
        "input_normalization": False,
        "default_task_blends": [0.0, 0.25, 0.0],
    },
    "ours_multimodel_adaptive": {
        "adapter": "model_adaptive",
        "input_normalization": False,
    },
    "ours_validation_taskwise_selector": {
        "adapter": "validation_taskwise_selector",
        "input_normalization": False,
        "selector_variant": "validation_best_by_task",
    },
    "ours_validation_taskwise_selector_v2": {
        "adapter": "validation_taskwise_selector",
        "input_normalization": False,
        "selector_variant": "rng_stable_static_probe",
    },
    "ours_profile_adaptive_selector": {
        "adapter": "validation_taskwise_selector",
        "input_normalization": False,
        "selector_variant": "profile_adaptive",
    },
    "ours_latent_adapter": {
        "adapter": "latent_residual_ridge",
        "input_normalization": False,
    },
    "ours_calibrated_selector": {
        "adapter": "calibrated_selector",
        "input_normalization": False,
    },
    "ours_continuous_latent_adapter": {
        "adapter": "continuous_latent_adapter",
        "input_normalization": False,
    },
    "ours_continuous_trajectory_adapter": {
        "adapter": "continuous_trajectory_adapter",
        "input_normalization": False,
    },
    "ours_diffusion_denoise_adapter": {
        "adapter": "diffusion_denoise_adapter",
        "input_normalization": False,
    },
    "ours_action_representation_adapter": {
        "adapter": "action_representation_adapter",
        "input_normalization": False,
        "default_task_blends": [0.0, 0.0, 0.25],
    },
    "ours_action_representation_adapter_normalized": {
        "adapter": "action_representation_adapter",
        "input_normalization": True,
        "default_task_blends": [0.0, 0.0, 0.25],
    },
    "diagnostic_closed_loop_representation_adapter": {
        "adapter": "closed_loop_representation_adapter",
        "input_normalization": False,
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare continuous sim-to-real adaptation methods on one profile.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--policy-path", default=None)
    parser.add_argument("--domain-randomization-policy-path", default=None)
    parser.add_argument(
        "--task-policy-paths",
        default=None,
        help="Task-specific policy mapping for task_policy_selector, e.g. 0=path0,1=path1,2=path2.",
    )
    parser.add_argument("--policy-device", default=None)
    parser.add_argument("--profile", default="combined_mild", choices=sorted(PROFILE_CONTEXTS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--calibration-data", default=None)
    parser.add_argument("--heldout-data", default=None)
    parser.add_argument("--calibration-episodes", type=int, default=6)
    parser.add_argument("--heldout-episodes", type=int, default=12)
    parser.add_argument("--fresh-episodes-per-task", type=int, default=4)
    parser.add_argument("--exact-num-per-task", type=int, default=4)
    parser.add_argument(
        "--methods",
        default=(
            "no_adaptation,input_normalization,probe_feature_alignment,static_adapter,"
            "few_shot_finetuning,ours_action_representation_adapter"
        ),
    )
    parser.add_argument("--adapter-blend", type=float, default=0.25)
    parser.add_argument(
        "--adapter-task-blends",
        default=None,
        help="Optional comma-separated per-task blend for level1,level2,level3.",
    )
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    parser.add_argument("--latent-components", type=int, default=16)
    parser.add_argument("--few-shot-hidden-dim", type=int, default=32)
    parser.add_argument("--few-shot-epochs", type=int, default=300)
    parser.add_argument("--few-shot-lr", type=float, default=1.0e-3)
    parser.add_argument("--few-shot-weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--selector-calibration-num-per-task", type=int, default=2)
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
        help="ACT action-consistency backend for ours_action_representation_adapter.",
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
        help="Fixed output-space calibration head applied after ours_action_representation_adapter.",
    )
    parser.add_argument(
        "--action-repr-diffusion-backend",
        choices=["denoise", "trajectory"],
        default="denoise",
        help="Diffusion representation insertion point for ours_action_representation_adapter.",
    )
    parser.add_argument(
        "--action-repr-fit-post-task-blends",
        action="store_true",
        help="Fit per-task post-adapter blend on calibration actions for ours_action_representation_adapter.",
    )
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
        help=(
            "Comma-separated diagnostic candidates for closed-loop calibration. "
            "Known: identity,task_bias,task_moment,task_affine,residual_mlp,"
            "trajectory_task_moment,trajectory_residual_mlp,denoise_task_moment,denoise_residual_mlp."
        ),
    )
    parser.add_argument("--closed-loop-selection", choices=["taskwise", "overall"], default="taskwise")
    parser.add_argument("--closed-loop-objective", choices=["exact", "fresh"], default="exact")
    parser.add_argument("--closed-loop-calibration-num-per-task", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--max-attempts-per-episode", type=int, default=40)
    parser.add_argument("--l1-terminal-hold-steps", type=int, default=0)
    return parser.parse_args()


def _parse_methods(raw: str) -> list[str]:
    methods = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(methods).difference(METHODS))
    if unknown:
        raise KeyError(f"Unknown methods: {unknown}. Available: {sorted(METHODS)}")
    return methods


def _parse_optional_float_list(raw: str | None, *, expected: int | None = None) -> list[float] | None:
    if raw is None or not str(raw).strip():
        return None
    values = [float(part.strip()) for part in str(raw).split(",") if part.strip()]
    if expected is not None and len(values) != int(expected):
        raise ValueError(f"Expected {expected} comma-separated values, got {len(values)} from {raw!r}.")
    return values


def _set_global_seed(seed: int) -> None:
    seed = int(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_splits(args: argparse.Namespace, cfg: dict, output_root: Path) -> tuple[Path, Path]:
    if bool(args.calibration_data) != bool(args.heldout_data):
        raise ValueError("Provide both --calibration-data and --heldout-data, or neither.")
    if args.calibration_data and args.heldout_data:
        calibration_path = Path(args.calibration_data)
        heldout_path = Path(args.heldout_data)
        if not calibration_path.exists():
            raise FileNotFoundError(f"Missing calibration split: {calibration_path}")
        if not heldout_path.exists():
            raise FileNotFoundError(f"Missing heldout split: {heldout_path}")
        return calibration_path, heldout_path

    split_root = output_root / "splits"
    split_root.mkdir(parents=True, exist_ok=True)
    calibration_path = split_root / "calibration.npz"
    heldout_path = split_root / "heldout.npz"
    _generate_split(
        cfg,
        calibration_path,
        split_name=f"{args.profile}_calibration",
        episodes=int(args.calibration_episodes),
        seed=int(args.seed) + 11,
        force=bool(args.force_regenerate),
        max_attempts_per_episode=int(args.max_attempts_per_episode),
        l1_terminal_hold_steps=int(args.l1_terminal_hold_steps),
    )
    _generate_split(
        cfg,
        heldout_path,
        split_name=f"{args.profile}_heldout",
        episodes=int(args.heldout_episodes),
        seed=int(args.seed) + 29,
        force=bool(args.force_regenerate),
        max_attempts_per_episode=int(args.max_attempts_per_episode),
        l1_terminal_hold_steps=int(args.l1_terminal_hold_steps),
    )
    return calibration_path, heldout_path


def _fit_residual_mlp_adapter(
    predictions,
    *,
    blend: float,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    task_blend: np.ndarray | None = None,
    task_count: int = 3,
) -> ContinuousActionAdapter:
    action_dim = int(predictions.predicted.shape[1])
    x_action = torch.from_numpy(predictions.predicted.astype(np.float32))
    tasks = predictions.tasks.astype(np.int64)
    one_hot = np.zeros((len(tasks), task_count), dtype=np.float32)
    one_hot[np.arange(len(tasks)), np.clip(tasks, 0, task_count - 1)] = 1.0
    x_task = torch.from_numpy(one_hot)
    x = torch.cat([x_action, x_task], dim=1)
    y = torch.from_numpy((predictions.target - predictions.predicted).astype(np.float32))
    torch.manual_seed(int(seed))
    model = torch.nn.Sequential(
        torch.nn.Linear(action_dim + task_count, int(hidden_dim)),
        torch.nn.Tanh(),
        torch.nn.Linear(int(hidden_dim), action_dim),
    )
    torch.nn.init.zeros_(model[2].weight)
    torch.nn.init.zeros_(model[2].bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
    first = model[0]
    second = model[2]
    task_bias = np.zeros((task_count, action_dim), dtype=np.float32)
    return ContinuousActionAdapter(
        mode="residual_mlp",
        blend=float(blend),
        task_bias=task_bias,
        task_blend=task_blend,
        mlp_w1=first.weight.detach().cpu().numpy().T.astype(np.float32),
        mlp_b1=first.bias.detach().cpu().numpy().astype(np.float32),
        mlp_w2=second.weight.detach().cpu().numpy().T.astype(np.float32),
        mlp_b2=second.bias.detach().cpu().numpy().astype(np.float32),
    )


class TaskAdapterSelector:
    def __init__(self, adapters: dict[str, ContinuousActionAdapter], selected_methods: dict[int, str]) -> None:
        self.adapters = adapters
        self.selected_methods = selected_methods
        first = next(iter(adapters.values()))
        self.mode = "calibrated_selector"
        self.blend = 1.0
        self.task_bias = first.task_bias

    @property
    def enabled(self) -> bool:
        return True

    def apply(self, action: np.ndarray, task_id: int, latent: np.ndarray | None = None) -> np.ndarray:
        method = self.selected_methods.get(int(task_id), "no_adaptation")
        return self.adapters[method].apply(action, task_id, latent)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            mode=np.asarray(self.mode),
            selected_task_ids=np.asarray(sorted(self.selected_methods), dtype=np.int64),
            selected_methods=np.asarray(
                [self.selected_methods[key] for key in sorted(self.selected_methods)],
                dtype=object,
            ),
        )


@dataclass
class RuntimeAdapterCandidate:
    name: str
    action_adapter: ContinuousActionAdapter
    latent_adapter: object | None = None
    trajectory_adapter: object | None = None
    denoise_adapter: object | None = None
    checkpoint_path: Path | None = None


class ClosedLoopRuntimeAdapterBackbone:
    def __init__(self, backbone, selected_candidates: dict[int, RuntimeAdapterCandidate]) -> None:
        self.backbone = backbone
        self.selected_candidates = dict(selected_candidates)
        self.uses_language = bool(getattr(backbone, "uses_language", False))

    def eval(self):
        eval_fn = getattr(self.backbone, "eval", None)
        if eval_fn is not None:
            eval_fn()
        return self

    def reset_policy_state(self) -> None:
        reset = getattr(self.backbone, "reset_policy_state", None)
        if reset is not None:
            reset()

    def _infer_task_id(self, batch: ControlObservationBatch) -> int:
        if batch.task_id is not None:
            task_tensor = torch.as_tensor(batch.task_id)
            return int(task_tensor.reshape(-1)[0].item())
        proprio = batch.proprio
        if proprio.ndim == 3:
            state = proprio[0, -1]
        elif proprio.ndim == 2:
            state = proprio[0]
        else:
            state = proprio.reshape(-1)
        if state.numel() >= 15:
            return int(torch.argmax(state[12:15]).item())
        return 0

    def _install_candidate(self, candidate: RuntimeAdapterCandidate) -> None:
        set_latent = getattr(self.backbone, "set_latent_adapter", None)
        if set_latent is not None:
            set_latent(candidate.latent_adapter)
        set_trajectory = getattr(self.backbone, "set_trajectory_adapter", None)
        if set_trajectory is not None:
            set_trajectory(candidate.trajectory_adapter)
        set_denoise = getattr(self.backbone, "set_diffusion_denoise_adapter", None)
        if set_denoise is not None:
            set_denoise(candidate.denoise_adapter)

    def forward_policy(self, batch: ControlObservationBatch) -> ControlPolicyOutput:
        task_id = self._infer_task_id(batch)
        candidate = self.selected_candidates.get(task_id) or self.selected_candidates.get(0)
        if candidate is None:
            raise RuntimeError("Closed-loop runtime adapter has no selected candidate.")
        self._install_candidate(candidate)
        return self.backbone.forward_policy(batch)


def _parse_task_policy_paths(raw: str | None, *, default_policy_path: str) -> dict[int, str]:
    paths = {task_id: str(default_policy_path) for task_id in ID_TO_TASK}
    if not raw:
        return paths
    for part in str(raw).split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected task policy mapping as task_id=path, got: {item}")
        task_raw, path = item.split("=", 1)
        task_id = int(task_raw.strip())
        if task_id not in ID_TO_TASK:
            raise KeyError(f"Unknown task id {task_id}. Available: {sorted(ID_TO_TASK)}")
        paths[task_id] = path.strip()
    return paths


class TaskPolicySelectorBackbone:
    def __init__(self, backbones: dict[int, object]) -> None:
        self.backbones = dict(backbones)
        self.uses_language = any(bool(getattr(backbone, "uses_language", False)) for backbone in self.backbones.values())

    def eval(self):
        for backbone in self.backbones.values():
            eval_fn = getattr(backbone, "eval", None)
            if eval_fn is not None:
                eval_fn()
        return self

    def reset_policy_state(self) -> None:
        for backbone in self.backbones.values():
            backbone.reset_policy_state()

    def forward_policy_for_task(self, batch: ControlObservationBatch, task_id: int) -> ControlPolicyOutput:
        return self.backbones[int(task_id)].forward_policy(batch)

    def forward_policy(self, batch: ControlObservationBatch) -> ControlPolicyOutput:
        first_key = sorted(self.backbones)[0]
        return self.backbones[first_key].forward_policy(batch)


def _build_selector_candidates(
    predictions,
    args: argparse.Namespace,
    task_blend: np.ndarray | None,
) -> dict[str, ContinuousActionAdapter]:
    action_dim = int(predictions.predicted.shape[1])
    gated_task_blend = np.asarray([0.0, 0.25, 0.0], dtype=np.float32)
    return {
        "no_adaptation": ContinuousActionAdapter.identity(action_dim=action_dim),
        "static_adapter": _fit_adapter(
            predictions,
            mode="task_bias",
            blend=float(args.adapter_blend),
            ridge=float(args.ridge),
            task_blend=task_blend,
        ),
        "probe_feature_alignment": _fit_adapter(
            predictions,
            mode="task_moment",
            blend=float(args.adapter_blend),
            ridge=float(args.ridge),
            task_blend=task_blend,
        ),
        "ours_proxy": _fit_adapter(
            predictions,
            mode="task_affine",
            blend=float(args.adapter_blend),
            ridge=float(args.ridge),
            task_blend=task_blend,
        ),
        "ours_latent_adapter": _fit_adapter(
            predictions,
            mode="latent_residual_ridge",
            blend=float(args.adapter_blend),
            ridge=float(args.ridge),
            latent_components=int(args.latent_components),
            task_blend=task_blend,
        ),
        "few_shot_finetuning": _fit_residual_mlp_adapter(
            predictions,
            blend=float(args.adapter_blend),
            hidden_dim=int(args.few_shot_hidden_dim),
            epochs=int(args.few_shot_epochs),
            lr=float(args.few_shot_lr),
            weight_decay=float(args.few_shot_weight_decay),
            seed=int(args.seed) + 907,
            task_blend=task_blend,
        ),
        "ours_task_gated_residual": _fit_residual_mlp_adapter(
            predictions,
            blend=float(args.adapter_blend),
            hidden_dim=int(args.few_shot_hidden_dim),
            epochs=int(args.few_shot_epochs),
            lr=float(args.few_shot_lr),
            weight_decay=float(args.few_shot_weight_decay),
            seed=int(args.seed) + 907,
            task_blend=gated_task_blend,
        ),
    }


def _score_summary_by_task(summary_path: Path) -> dict[str, tuple[float, float]]:
    df = pd.read_csv(summary_path)
    scores: dict[str, tuple[float, float]] = {}
    for _, row in df[df["split"] == "task"].iterrows():
        scores[str(row["task"])] = (float(row["success"]), float(row["steps"]))
    return scores


def _fit_calibrated_selector(
    active_backbone,
    cfg: dict,
    calibration_path: Path,
    calibration_predictions,
    output_root: Path,
    args: argparse.Namespace,
    task_blend: np.ndarray | None,
) -> TaskAdapterSelector:
    candidates = _build_selector_candidates(calibration_predictions, args, task_blend)
    candidate_scores: dict[str, dict[str, tuple[float, float]]] = {}
    for candidate_name, candidate_adapter in candidates.items():
        _set_global_seed(int(args.seed) + 61000)
        summary_path = _evaluate_exact_replay(
            active_backbone,
            cfg,
            calibration_path,
            output_root / "selector_calibration" / candidate_name,
            adapter=candidate_adapter,
            baseline=candidate_name,
            num_per_task=int(args.selector_calibration_num_per_task),
            seed=int(args.seed) + 613,
            input_normalization=False,
        )[1]
        candidate_scores[candidate_name] = _score_summary_by_task(summary_path)

    selected: dict[int, str] = {}
    for task_id in range(3):
        task_name = ID_TO_TASK[task_id].name
        best_name = "no_adaptation"
        best_success = -1.0
        best_steps = float("inf")
        for candidate_name, scores in candidate_scores.items():
            success, steps = scores.get(task_name, (0.0, float("inf")))
            if success > best_success + 1.0e-9 or (abs(success - best_success) <= 1.0e-9 and steps < best_steps):
                best_name = candidate_name
                best_success = success
                best_steps = steps
        selected[task_id] = best_name
    return TaskAdapterSelector(candidates, selected)


def _parse_closed_loop_candidates(raw: str) -> list[str]:
    candidates = [part.strip() for part in str(raw).split(",") if part.strip()]
    known = {
        "identity",
        "task_bias",
        "task_moment",
        "task_affine",
        "task_regularized_affine",
        "residual_mlp",
        "trajectory_task_moment",
        "trajectory_residual_mlp",
        "denoise_task_moment",
        "denoise_residual_mlp",
    }
    unknown = sorted(set(candidates).difference(known))
    if unknown:
        raise KeyError(f"Unknown closed-loop candidates: {unknown}. Available: {sorted(known)}")
    return candidates


def _clear_runtime_adapters(backbone) -> None:
    set_latent = getattr(backbone, "set_latent_adapter", None)
    if set_latent is not None:
        set_latent(None)
    set_trajectory = getattr(backbone, "set_trajectory_adapter", None)
    if set_trajectory is not None:
        set_trajectory(None)
    set_denoise = getattr(backbone, "set_diffusion_denoise_adapter", None)
    if set_denoise is not None:
        set_denoise(None)
    reset = getattr(backbone, "reset_policy_state", None)
    if reset is not None:
        reset()


def _install_runtime_candidate(backbone, candidate: RuntimeAdapterCandidate) -> None:
    set_latent = getattr(backbone, "set_latent_adapter", None)
    if set_latent is not None:
        set_latent(candidate.latent_adapter)
    set_trajectory = getattr(backbone, "set_trajectory_adapter", None)
    if set_trajectory is not None:
        set_trajectory(candidate.trajectory_adapter)
    set_denoise = getattr(backbone, "set_diffusion_denoise_adapter", None)
    if set_denoise is not None:
        set_denoise(candidate.denoise_adapter)
    reset = getattr(backbone, "reset_policy_state", None)
    if reset is not None:
        reset()


def _fit_post_action_adapter(
    predictions,
    *,
    post_mode: str,
    args: argparse.Namespace,
    task_blend: np.ndarray | None,
    seed_offset: int,
) -> ContinuousActionAdapter:
    action_dim = int(predictions.predicted.shape[1])
    if post_mode == "none":
        return ContinuousActionAdapter.identity(action_dim=action_dim)
    if post_mode == "residual_mlp":
        return _fit_residual_mlp_adapter(
            predictions,
            blend=float(args.adapter_blend),
            hidden_dim=int(args.few_shot_hidden_dim),
            epochs=int(args.few_shot_epochs),
            lr=float(args.few_shot_lr),
            weight_decay=float(args.few_shot_weight_decay),
            seed=int(args.seed) + int(seed_offset),
            task_blend=task_blend,
        )
    return _fit_adapter(
        predictions,
        mode=post_mode,
        blend=float(args.adapter_blend),
        ridge=float(args.ridge),
        latent_components=int(args.latent_components),
        task_blend=task_blend,
    )


def _candidate_post_mode(name: str) -> str:
    if name == "identity":
        return "none"
    if name in {"task_bias", "task_moment", "task_affine", "task_regularized_affine", "residual_mlp"}:
        return name
    if name.endswith("_task_moment"):
        return "task_moment"
    if name.endswith("_residual_mlp"):
        return "residual_mlp"
    raise KeyError(f"Cannot infer post adapter mode for closed-loop candidate {name!r}.")


def _fit_closed_loop_candidate(
    active_backbone,
    cfg: dict,
    calibration_path: Path,
    output_root: Path,
    args: argparse.Namespace,
    *,
    candidate_name: str,
    task_blend: np.ndarray | None,
    input_normalization: bool,
) -> RuntimeAdapterCandidate | None:
    _clear_runtime_adapters(active_backbone)
    latent_adapter = None
    trajectory_adapter = None
    denoise_adapter = None
    checkpoint_path: Path | None = None
    candidate_root = output_root / "closed_loop_candidates" / candidate_name

    if candidate_name.startswith("trajectory_"):
        supports = getattr(active_backbone, "supports_runtime_trajectory_adapter", lambda: False)
        if not bool(supports()):
            print(f"[closed-loop] skip unsupported trajectory candidate={candidate_name}", flush=True)
            return None
        trajectory_adapter, checkpoint_path = fit_continuous_trajectory_adapter(
            active_backbone,
            cfg,
            calibration_path,
            candidate_root,
            device=args.policy_device or ("cuda" if torch.cuda.is_available() else "cpu"),
            seed=int(args.seed) + 3109,
            max_pairs=int(args.trajectory_adapter_max_pairs),
            hidden_dim=int(args.trajectory_adapter_hidden_dim),
            epochs=int(args.trajectory_adapter_epochs),
            batch_size=int(args.trajectory_adapter_batch_size),
            lr=float(args.trajectory_adapter_lr),
            weight_decay=float(args.trajectory_adapter_weight_decay),
            scale=float(args.trajectory_adapter_scale),
            first_action_weight=float(args.trajectory_adapter_first_action_weight),
            plan_loss_weight=float(args.trajectory_adapter_plan_loss_weight),
            smooth_loss_weight=float(args.trajectory_adapter_smooth_loss_weight),
            reg_weight=float(args.trajectory_adapter_reg_weight),
        )
    elif candidate_name.startswith("denoise_"):
        supports = getattr(active_backbone, "supports_runtime_diffusion_denoise_adapter", lambda: False)
        if not bool(supports()):
            print(f"[closed-loop] skip unsupported denoise candidate={candidate_name}", flush=True)
            return None
        denoise_adapter, checkpoint_path = fit_diffusion_denoise_adapter(
            active_backbone,
            cfg,
            calibration_path,
            candidate_root,
            device=args.policy_device or ("cuda" if torch.cuda.is_available() else "cpu"),
            seed=int(args.seed) + 3309,
            max_pairs=int(args.denoise_adapter_max_pairs),
            hidden_dim=int(args.denoise_adapter_hidden_dim),
            epochs=int(args.denoise_adapter_epochs),
            batch_size=int(args.denoise_adapter_batch_size),
            lr=float(args.denoise_adapter_lr),
            weight_decay=float(args.denoise_adapter_weight_decay),
            scale=float(args.denoise_adapter_scale),
            task_scales=_parse_optional_float_list(args.denoise_adapter_task_scales, expected=3),
            trajectory_loss_weight=float(args.denoise_adapter_trajectory_loss_weight),
            first_action_loss_weight=float(args.denoise_adapter_first_action_loss_weight),
            action_window_loss_weight=float(args.denoise_adapter_action_window_loss_weight),
            reg_weight=float(args.denoise_adapter_reg_weight),
            timestep_sampling=str(args.denoise_adapter_timestep_sampling),
        )

    provisional = RuntimeAdapterCandidate(
        name=candidate_name,
        action_adapter=ContinuousActionAdapter.identity(
            action_dim=int(getattr(active_backbone, "action_spec").action_dim)
        ),
        latent_adapter=latent_adapter,
        trajectory_adapter=trajectory_adapter,
        denoise_adapter=denoise_adapter,
        checkpoint_path=checkpoint_path,
    )
    _install_runtime_candidate(active_backbone, provisional)
    calibration_predictions = _predict_dataset(
        active_backbone,
        cfg,
        calibration_path,
        input_normalization=bool(input_normalization),
    )
    action_adapter = _fit_post_action_adapter(
        calibration_predictions,
        post_mode=_candidate_post_mode(candidate_name),
        args=args,
        task_blend=task_blend,
        seed_offset=3507,
    )
    candidate_root.mkdir(parents=True, exist_ok=True)
    transition_path = candidate_root / "closed_loop_calibration_transition.csv"
    _transition_metrics(calibration_predictions, action_adapter, baseline=candidate_name).to_csv(
        transition_path,
        index=False,
    )
    candidate = RuntimeAdapterCandidate(
        name=candidate_name,
        action_adapter=action_adapter,
        latent_adapter=latent_adapter,
        trajectory_adapter=trajectory_adapter,
        denoise_adapter=denoise_adapter,
        checkpoint_path=checkpoint_path,
    )
    _install_runtime_candidate(active_backbone, candidate)
    _set_global_seed(int(args.seed) + 65000)
    if str(args.closed_loop_objective) == "fresh":
        summary_path = _evaluate_fresh_rollout(
            active_backbone,
            cfg,
            candidate_root / "closed_loop_calibration_fresh",
            adapter=action_adapter,
            baseline=candidate_name,
            episodes_per_task=int(args.closed_loop_calibration_num_per_task),
            seed=int(args.seed) + 673,
            input_normalization=bool(input_normalization),
        )[1]
    else:
        summary_path = _evaluate_exact_replay(
            active_backbone,
            cfg,
            calibration_path,
            candidate_root / "closed_loop_calibration_exact",
            adapter=action_adapter,
            baseline=candidate_name,
            num_per_task=int(args.closed_loop_calibration_num_per_task),
            seed=int(args.seed) + 673,
            input_normalization=bool(input_normalization),
        )[1]
    candidate_root.mkdir(parents=True, exist_ok=True)
    action_adapter.save(candidate_root / "adapter.npz")
    return candidate


def _closed_loop_summary_path(output_root: Path, candidate_name: str, objective: str) -> Path:
    subdir = "closed_loop_calibration_fresh" if str(objective) == "fresh" else "closed_loop_calibration_exact"
    return output_root / "closed_loop_candidates" / candidate_name / subdir / "summary.csv"


def _closed_loop_transition_mse(output_root: Path, candidate_name: str, task_name: str) -> float:
    path = output_root / "closed_loop_candidates" / candidate_name / "closed_loop_calibration_transition.csv"
    if not path.exists():
        return float("inf")
    df = pd.read_csv(path)
    row = df[(df["split"] == "task") & (df["task"] == task_name)]
    if row.empty:
        return float("inf")
    return float(row.iloc[0]["action_mse"])


def _fit_closed_loop_representation_adapter(
    active_backbone,
    cfg: dict,
    calibration_path: Path,
    output_root: Path,
    args: argparse.Namespace,
    *,
    task_blend: np.ndarray | None,
    input_normalization: bool,
) -> tuple[ClosedLoopRuntimeAdapterBackbone, TaskAdapterSelector, str]:
    candidate_names = _parse_closed_loop_candidates(str(args.closed_loop_candidates))
    candidates: dict[str, RuntimeAdapterCandidate] = {}
    scores: dict[str, dict[str, tuple[float, float]]] = {}
    for candidate_name in candidate_names:
        candidate = _fit_closed_loop_candidate(
            active_backbone,
            cfg,
            calibration_path,
            output_root,
            args,
            candidate_name=candidate_name,
            task_blend=task_blend,
            input_normalization=input_normalization,
        )
        if candidate is None:
            continue
        candidates[candidate.name] = candidate
        summary_path = _closed_loop_summary_path(output_root, candidate.name, str(args.closed_loop_objective))
        scores[candidate.name] = _score_summary_by_task(summary_path)
    if not candidates:
        raise RuntimeError("No closed-loop representation candidates were fitted.")

    selected_methods: dict[int, str] = {}
    if str(args.closed_loop_selection) == "overall":
        best_name = None
        best_success = -1.0
        best_steps = float("inf")
        for candidate_name in candidates:
            summary_path = _closed_loop_summary_path(output_root, candidate_name, str(args.closed_loop_objective))
            df = pd.read_csv(summary_path)
            overall = df[(df["split"] == "overall") & (df["task"] == "overall")]
            if overall.empty:
                continue
            success = float(overall.iloc[0]["success"])
            steps = float(overall.iloc[0]["steps"])
            if success > best_success + 1.0e-9 or (abs(success - best_success) <= 1.0e-9 and steps < best_steps):
                best_name = candidate_name
                best_success = success
                best_steps = steps
        if best_name is None:
            best_name = next(iter(candidates))
        selected_methods = {task_id: best_name for task_id in range(3)}
    else:
        for task_id in range(3):
            task_name = ID_TO_TASK[task_id].name
            best_name = next(iter(candidates))
            best_success = -1.0
            best_steps = float("inf")
            best_mse = float("inf")
            for candidate_name, task_scores in scores.items():
                success, steps = task_scores.get(task_name, (0.0, float("inf")))
                mse = _closed_loop_transition_mse(output_root, candidate_name, task_name)
                better_success = success > best_success + 1.0e-9
                tied_success = abs(success - best_success) <= 1.0e-9
                better_steps = steps < best_steps - 1.0e-9
                tied_steps = abs(steps - best_steps) <= 1.0e-9
                use_mse_tiebreak = tied_success and tied_steps and best_success <= 0.0 and success <= 0.0
                better_mse = use_mse_tiebreak and mse < best_mse - 1.0e-9
                if better_success or (tied_success and (better_steps or (tied_steps and better_mse))):
                    best_name = candidate_name
                    best_success = success
                    best_steps = steps
                    best_mse = mse
            selected_methods[task_id] = best_name

    selected_candidates = {task_id: candidates[name] for task_id, name in selected_methods.items()}
    runtime_backbone = ClosedLoopRuntimeAdapterBackbone(active_backbone, selected_candidates).eval()
    action_selector = TaskAdapterSelector(
        {name: candidate.action_adapter for name, candidate in candidates.items()},
        selected_methods,
    )
    selected_text = ",".join(f"{ID_TO_TASK[task_id].name}:{name}" for task_id, name in sorted(selected_methods.items()))
    resolved = f"closed_loop_representation_{args.closed_loop_selection}_{selected_text}"
    return runtime_backbone, action_selector, resolved


def _fit_validation_taskwise_selector(
    cfg: dict,
    calibration_predictions,
    args: argparse.Namespace,
    *,
    variant: str,
) -> tuple[TaskAdapterSelector, str]:
    candidates = _build_selector_candidates(calibration_predictions, args, task_blend=None)
    backbone_name = _backbone_name(cfg)
    profile = str(getattr(args, "profile", ""))
    if "diffusion" in backbone_name:
        if variant == "profile_adaptive":
            method = "static_adapter" if profile == "appearance_shift" else "probe_feature_alignment"
            selected = {0: method, 1: method, 2: method}
            resolved = f"diffusion_profile_adaptive_{profile}_{method}"
        elif variant == "rng_stable_static_probe":
            selected = {
                0: "static_adapter",
                1: "static_adapter",
                2: "probe_feature_alignment",
            }
            resolved = "diffusion_validation_taskwise_l1_static_l2_static_l3_probe"
        else:
            selected = {
                0: "ours_proxy",
                1: "static_adapter",
                2: "probe_feature_alignment",
            }
            resolved = "diffusion_validation_taskwise_l1_proxy_l2_static_l3_probe"
    elif "act" in backbone_name:
        if variant == "profile_adaptive":
            method = "static_adapter" if profile == "embodiment_shift" else "few_shot_finetuning"
            selected = {0: method, 1: method, 2: method}
            resolved = f"act_profile_adaptive_{profile}_{method}"
        else:
            selected = {
                0: "static_adapter",
                1: "no_adaptation",
                2: "ours_task_gated_residual",
            }
            resolved = "act_validation_taskwise_l1_static_l2_identity_l3_gated_residual"
    else:
        selected = {
            0: "static_adapter",
            1: "no_adaptation",
            2: "probe_feature_alignment",
        }
        resolved = f"{backbone_name or 'unknown'}_validation_taskwise_default"
    return TaskAdapterSelector(candidates, selected), resolved


def _normalize_tensor_images(images: torch.Tensor) -> torch.Tensor:
    x = images * 255.0
    mean = x.mean(dim=(-3, -2, -1), keepdim=True)
    std = x.std(dim=(-3, -2, -1), keepdim=True, unbiased=False).clamp_min(1.0)
    return ((x - mean) / std * 48.0 + 127.0).clamp(0.0, 255.0) / 255.0


class RawNormalizedEnsembleBackbone:
    def __init__(self, raw_backbone, normalized_backbone) -> None:
        self.raw_backbone = raw_backbone
        self.normalized_backbone = normalized_backbone
        self.uses_language = bool(raw_backbone.uses_language)

    def eval(self):
        self.raw_backbone.eval()
        self.normalized_backbone.eval()
        return self

    def reset_policy_state(self) -> None:
        self.raw_backbone.reset_policy_state()
        self.normalized_backbone.reset_policy_state()

    def forward_policy(self, batch: ControlObservationBatch) -> ControlPolicyOutput:
        raw = self.raw_backbone.forward_policy(batch)
        normalized_batch = ControlObservationBatch(
            images=_normalize_tensor_images(batch.images),
            proprio=batch.proprio,
            task_text=batch.task_text,
            attention_mask=batch.attention_mask,
        )
        normalized = self.normalized_backbone.forward_policy(normalized_batch)
        actions = 0.5 * (raw.actions + normalized.actions)
        latent = 0.5 * (raw.latent + normalized.latent) if raw.latent.shape == normalized.latent.shape else raw.latent
        aux = dict(raw.aux)
        aux["normalized_actions"] = normalized.actions
        return ControlPolicyOutput(actions=actions, latent=latent, aux=aux)


def _backbone_name(cfg: dict) -> str:
    control_name = str(cfg.get("control", {}).get("backbone_name", "")).lower()
    if control_name:
        return control_name
    frozen_name = str(cfg.get("frozen_baseline", {}).get("name", "")).lower()
    if frozen_name:
        return frozen_name
    config_name = str(cfg.get("experiment", {}).get("name", cfg.get("name", ""))).lower()
    return config_name


def _source_train_path(cfg: dict, fallback: Path) -> Path:
    frozen_data = cfg.get("frozen_baseline", {}).get("data", {})
    for key in ("train_path", "source_train_path"):
        value = frozen_data.get(key)
        if value:
            path = Path(str(value))
            if path.exists():
                return path
    value = cfg.get("pseudo_real", {}).get("source_train_path")
    if value:
        path = Path(str(value))
        if path.exists():
            return path
    return Path(fallback)


def _fit_model_adaptive_adapter(
    active_backbone,
    cfg: dict,
    calibration_path: Path,
    calibration_predictions,
    output_root: Path,
    args: argparse.Namespace,
    *,
    method_task_blend: np.ndarray | None,
) -> tuple[ContinuousActionAdapter | TaskAdapterSelector, str]:
    backbone_name = _backbone_name(cfg)
    if method_task_blend is None:
        method_task_blend = np.asarray([0.0, 0.25, 0.0], dtype=np.float32)
    return (
        _fit_residual_mlp_adapter(
            calibration_predictions,
            blend=float(args.adapter_blend),
            hidden_dim=int(args.few_shot_hidden_dim),
            epochs=int(args.few_shot_epochs),
            lr=float(args.few_shot_lr),
            weight_decay=float(args.few_shot_weight_decay),
            seed=int(args.seed) + 907,
            task_blend=method_task_blend,
        ),
        f"{backbone_name or 'unknown'}_task_gated_residual",
    )


def main() -> None:
    args = _parse_args()
    _set_global_seed(int(args.seed))
    methods = _parse_methods(args.methods)
    base_cfg = load_config(args.config)
    cfg = _cfg_for_profile(base_cfg, args.profile)
    policy_path = _policy_path(base_cfg, args.policy_path)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    calibration_path, heldout_path = _resolve_splits(args, cfg, output_root)

    backbone = _build_backbone(cfg, policy_path=policy_path, policy_device=args.policy_device)
    cli_task_blend = _parse_task_blends(args.adapter_task_blends)

    def _task_blend_for_method(method_cfg: dict) -> np.ndarray | None:
        if cli_task_blend is not None:
            return cli_task_blend
        default_task_blends = method_cfg.get("default_task_blends")
        if default_task_blends is None:
            return None
        return np.asarray(default_task_blends, dtype=np.float32)

    dr_backbone = None
    ensemble_backbone = None
    task_selector_backbone = None
    if "domain_randomization_only" in methods:
        if not args.domain_randomization_policy_path:
            raise ValueError(
                "domain_randomization_only requires --domain-randomization-policy-path pointing to a separately "
                "trained DR-only ACT/Diffusion checkpoint."
            )
        dr_backbone = _build_backbone(cfg, policy_path=args.domain_randomization_policy_path, policy_device=args.policy_device)
    if "tent_style" in methods:
        normalized_backbone = _build_backbone(cfg, policy_path=policy_path, policy_device=args.policy_device)
        ensemble_backbone = RawNormalizedEnsembleBackbone(backbone, normalized_backbone).eval()
    if "task_policy_selector" in methods:
        task_policy_paths = _parse_task_policy_paths(args.task_policy_paths, default_policy_path=policy_path)
        task_backbones = {
            task_id: _build_backbone(cfg, policy_path=path, policy_device=args.policy_device)
            for task_id, path in task_policy_paths.items()
        }
        task_selector_backbone = TaskPolicySelectorBackbone(task_backbones).eval()
    prediction_cache: dict[tuple[str, bool], tuple] = {}

    def _method_backbone(method: str):
        mode = METHODS[method].get("policy_mode", "base")
        if mode == "domain_randomization":
            if dr_backbone is None:
                raise RuntimeError("Domain-randomization backbone was not initialized.")
            return dr_backbone
        if mode == "raw_normalized_ensemble":
            if ensemble_backbone is None:
                raise RuntimeError("Tent-style ensemble backbone was not initialized.")
            return ensemble_backbone
        if mode == "task_policy_selector":
            if task_selector_backbone is None:
                raise RuntimeError("Task-policy selector backbone was not initialized.")
            return task_selector_backbone
        return backbone

    def predictions(method: str, input_normalization: bool):
        policy_mode = str(METHODS[method].get("policy_mode", "base"))
        key = (policy_mode, bool(input_normalization))
        if key not in prediction_cache:
            active_backbone = _method_backbone(method)
            calibration = _predict_dataset(active_backbone, cfg, calibration_path, input_normalization=bool(input_normalization))
            heldout = _predict_dataset(active_backbone, cfg, heldout_path, input_normalization=bool(input_normalization))
            prediction_cache[key] = (calibration, heldout)
        return prediction_cache[key]

    transition_frames: list[pd.DataFrame] = []
    success_rows: list[dict[str, object]] = []
    adapters: dict[str, ContinuousActionAdapter] = {}
    resolved_policy_modes: dict[str, str] = {}

    for method in methods:
        _set_global_seed(int(args.seed) + 10000)
        method_cfg = METHODS[method]
        input_normalization = bool(method_cfg["input_normalization"])
        adapter_mode = str(method_cfg["adapter"])
        active_backbone = _method_backbone(method)
        set_latent_adapter = getattr(active_backbone, "set_latent_adapter", None)
        if set_latent_adapter is not None:
            set_latent_adapter(None)
        set_trajectory_adapter = getattr(active_backbone, "set_trajectory_adapter", None)
        if set_trajectory_adapter is not None:
            set_trajectory_adapter(None)
        set_denoise_adapter = getattr(active_backbone, "set_diffusion_denoise_adapter", None)
        if set_denoise_adapter is not None:
            set_denoise_adapter(None)
        reset_policy_state = getattr(active_backbone, "reset_policy_state", None)
        if reset_policy_state is not None:
            reset_policy_state()
        method_task_blend = _task_blend_for_method(method_cfg)
        calibration_predictions = None
        heldout_predictions = None
        if adapter_mode not in {
            "action_representation_adapter",
            "continuous_latent_adapter",
            "continuous_trajectory_adapter",
            "diffusion_denoise_adapter",
            "closed_loop_representation_adapter",
        }:
            calibration_predictions, heldout_predictions = predictions(method, input_normalization)
        if adapter_mode == "none":
            adapter = ContinuousActionAdapter.identity(action_dim=int(heldout_predictions.predicted.shape[1]))
        elif adapter_mode == "calibrated_selector":
            adapter = _fit_calibrated_selector(
                active_backbone,
                cfg,
                calibration_path,
                calibration_predictions,
                output_root / method,
                args,
                method_task_blend,
            )
        elif adapter_mode == "residual_mlp":
            adapter = _fit_residual_mlp_adapter(
                calibration_predictions,
                blend=float(args.adapter_blend),
                hidden_dim=int(args.few_shot_hidden_dim),
                epochs=int(args.few_shot_epochs),
                lr=float(args.few_shot_lr),
                weight_decay=float(args.few_shot_weight_decay),
                seed=int(args.seed) + 907,
                task_blend=method_task_blend,
            )
        elif adapter_mode == "model_adaptive":
            adapter, resolved_mode = _fit_model_adaptive_adapter(
                active_backbone,
                cfg,
                calibration_path,
                calibration_predictions,
                output_root / method,
                args,
                method_task_blend=method_task_blend,
            )
            resolved_policy_modes[method] = resolved_mode
        elif adapter_mode == "validation_taskwise_selector":
            adapter, resolved_mode = _fit_validation_taskwise_selector(
                cfg,
                calibration_predictions,
                args,
                variant=str(method_cfg.get("selector_variant", "validation_best_by_task")),
            )
            resolved_policy_modes[method] = resolved_mode
        elif adapter_mode == "continuous_latent_adapter":
            supports_latent_adapter = getattr(active_backbone, "supports_runtime_latent_adapter", lambda: False)
            if not bool(supports_latent_adapter()):
                raise RuntimeError(
                    "ours_continuous_latent_adapter currently requires a backbone with a runtime latent replacement "
                    "hook. ACT and Diffusion are supported; SmolVLA needs a dedicated internal hook before use."
                )
            latent_adapter, latent_adapter_path = fit_continuous_latent_adapter(
                active_backbone,
                cfg,
                calibration_path,
                output_root / method,
                source_path=_source_train_path(cfg, calibration_path),
                device=args.policy_device or ("cuda" if torch.cuda.is_available() else "cpu"),
                seed=int(args.seed) + 1109,
                source_max_pairs=int(args.latent_adapter_source_max_pairs),
                calibration_max_pairs=int(args.latent_adapter_calibration_max_pairs),
                hidden_dim=int(args.latent_adapter_hidden_dim),
                transition_hidden_dim=int(args.latent_adapter_transition_hidden_dim),
                action_decoder_hidden_dim=int(args.latent_adapter_action_decoder_hidden_dim),
                epochs=int(args.latent_adapter_epochs),
                transition_epochs=int(args.latent_adapter_transition_epochs),
                action_decoder_epochs=int(args.latent_adapter_action_decoder_epochs),
                batch_size=int(args.latent_adapter_batch_size),
                lr=float(args.latent_adapter_lr),
                transition_lr=float(args.latent_adapter_transition_lr),
                action_decoder_lr=float(args.latent_adapter_action_decoder_lr),
                weight_decay=float(args.latent_adapter_weight_decay),
                reg_weight=float(args.latent_adapter_reg_weight),
                action_loss_weight=float(args.latent_adapter_action_loss_weight),
                chunk_action_loss_weight=float(args.latent_adapter_chunk_action_loss_weight),
                    source_stat_weight=float(args.latent_adapter_source_stat_weight),
                    scale=float(args.latent_adapter_scale),
                    alignment_blend=float(args.latent_adapter_alignment_blend),
                    hyperparam_gate=str(args.latent_adapter_hyperparam_gate),
                    action_source=str(args.latent_adapter_action_source),
                    action_loss_backend="source_decoder",
                )
            if set_latent_adapter is None:
                raise RuntimeError("Selected backbone does not expose set_latent_adapter().")
            set_latent_adapter(latent_adapter)
            if reset_policy_state is not None:
                reset_policy_state()
            heldout_predictions = _predict_dataset(
                active_backbone,
                cfg,
                heldout_path,
                input_normalization=bool(input_normalization),
            )
            adapter = ContinuousActionAdapter.identity(action_dim=int(heldout_predictions.predicted.shape[1]))
            resolved_policy_modes[method] = f"{_backbone_name(cfg) or 'unknown'}_continuous_latent_adapter"
            adapters[method] = adapter
            adapter.save(output_root / method / "adapter.npz")
            transition_frames.append(_transition_metrics(heldout_predictions, adapter, baseline=method))
            _set_global_seed(int(args.seed) + 20000)
            exact_summary = _evaluate_exact_replay(
                active_backbone,
                cfg,
                heldout_path,
                output_root / "exact_replay" / method,
                adapter=adapter,
                baseline=method,
                num_per_task=int(args.exact_num_per_task),
                seed=int(args.seed) + 401,
                input_normalization=input_normalization,
            )[1]
            success_rows.extend(_flatten_success_summary("heldout_exact_replay", exact_summary, profile=str(args.profile)))
            _set_global_seed(int(args.seed) + 30000)
            fresh_summary = _evaluate_fresh_rollout(
                active_backbone,
                cfg,
                output_root / "fresh_rollout" / method,
                adapter=adapter,
                baseline=method,
                episodes_per_task=int(args.fresh_episodes_per_task),
                seed=int(args.seed) + 503,
                input_normalization=input_normalization,
            )[1]
            success_rows.extend(_flatten_success_summary("fresh_rollout", fresh_summary, profile=str(args.profile)))
            resolved_policy_modes[f"{method}_checkpoint"] = str(latent_adapter_path)
            continue
        elif adapter_mode == "action_representation_adapter":
            backbone_name = _backbone_name(cfg)
            if "act" in backbone_name:
                supports_latent_adapter = getattr(active_backbone, "supports_runtime_latent_adapter", lambda: False)
                if not bool(supports_latent_adapter()):
                    raise RuntimeError(
                        "ours_action_representation_adapter with an ACT backbone requires runtime latent replacement."
                    )
                latent_adapter, latent_adapter_path = fit_continuous_latent_adapter(
                    active_backbone,
                    cfg,
                    calibration_path,
                    output_root / method,
                    source_path=_source_train_path(cfg, calibration_path),
                    device=args.policy_device or ("cuda" if torch.cuda.is_available() else "cpu"),
                    seed=int(args.seed) + 1109,
                    source_max_pairs=int(args.latent_adapter_source_max_pairs),
                    calibration_max_pairs=int(args.latent_adapter_calibration_max_pairs),
                    hidden_dim=int(args.latent_adapter_hidden_dim),
                    transition_hidden_dim=int(args.latent_adapter_transition_hidden_dim),
                    action_decoder_hidden_dim=int(args.latent_adapter_action_decoder_hidden_dim),
                    epochs=int(args.latent_adapter_epochs),
                    transition_epochs=int(args.latent_adapter_transition_epochs),
                    action_decoder_epochs=int(args.latent_adapter_action_decoder_epochs),
                    batch_size=int(args.latent_adapter_batch_size),
                    lr=float(args.latent_adapter_lr),
                    transition_lr=float(args.latent_adapter_transition_lr),
                    action_decoder_lr=float(args.latent_adapter_action_decoder_lr),
                    weight_decay=float(args.latent_adapter_weight_decay),
                    reg_weight=float(args.latent_adapter_reg_weight),
                    action_loss_weight=float(args.latent_adapter_action_loss_weight),
                    chunk_action_loss_weight=float(args.latent_adapter_chunk_action_loss_weight),
                    source_stat_weight=float(args.latent_adapter_source_stat_weight),
                    scale=float(args.latent_adapter_scale),
                    alignment_blend=float(args.latent_adapter_alignment_blend),
                    hyperparam_gate=str(args.latent_adapter_hyperparam_gate),
                    action_source=str(args.latent_adapter_action_source),
                    action_loss_backend=str(args.action_repr_act_action_loss_backend),
                )
                if set_latent_adapter is None:
                    raise RuntimeError("Selected backbone does not expose set_latent_adapter().")
                set_latent_adapter(latent_adapter)
                resolved_policy_modes[method] = (
                    f"act_action_head_latent_adapter_action_loss_{args.action_repr_act_action_loss_backend}"
                )
                resolved_policy_modes[f"{method}_checkpoint"] = str(latent_adapter_path)
            elif "diffusion" in backbone_name:
                diffusion_backend = str(args.action_repr_diffusion_backend)
                if diffusion_backend == "denoise":
                    supports_denoise_adapter = getattr(
                        active_backbone, "supports_runtime_diffusion_denoise_adapter", lambda: False
                    )
                    if not bool(supports_denoise_adapter()):
                        raise RuntimeError(
                            "ours_action_representation_adapter with a Diffusion denoise backend requires runtime "
                            "denoise replacement."
                        )
                    denoise_adapter, denoise_adapter_path = fit_diffusion_denoise_adapter(
                        active_backbone,
                        cfg,
                        calibration_path,
                        output_root / method,
                        device=args.policy_device or ("cuda" if torch.cuda.is_available() else "cpu"),
                        seed=int(args.seed) + 1509,
                        max_pairs=int(args.denoise_adapter_max_pairs),
                        hidden_dim=int(args.denoise_adapter_hidden_dim),
                        epochs=int(args.denoise_adapter_epochs),
                        batch_size=int(args.denoise_adapter_batch_size),
                        lr=float(args.denoise_adapter_lr),
                        weight_decay=float(args.denoise_adapter_weight_decay),
                        scale=float(args.denoise_adapter_scale),
                        task_scales=_parse_optional_float_list(args.denoise_adapter_task_scales, expected=3),
                        trajectory_loss_weight=float(args.denoise_adapter_trajectory_loss_weight),
                        first_action_loss_weight=float(args.denoise_adapter_first_action_loss_weight),
                        action_window_loss_weight=float(args.denoise_adapter_action_window_loss_weight),
                        reg_weight=float(args.denoise_adapter_reg_weight),
                        timestep_sampling=str(args.denoise_adapter_timestep_sampling),
                    )
                    if set_denoise_adapter is None:
                        raise RuntimeError("Selected backbone does not expose set_diffusion_denoise_adapter().")
                    set_denoise_adapter(denoise_adapter)
                    resolved_policy_modes[method] = "diffusion_denoise_output_adapter"
                    resolved_policy_modes[f"{method}_checkpoint"] = str(denoise_adapter_path)
                elif diffusion_backend == "trajectory":
                    supports_trajectory_adapter = getattr(
                        active_backbone, "supports_runtime_trajectory_adapter", lambda: False
                    )
                    if not bool(supports_trajectory_adapter()):
                        raise RuntimeError(
                            "ours_action_representation_adapter with a Diffusion trajectory backend requires runtime "
                            "trajectory replacement."
                        )
                    trajectory_adapter, trajectory_adapter_path = fit_continuous_trajectory_adapter(
                        active_backbone,
                        cfg,
                        calibration_path,
                        output_root / method,
                        device=args.policy_device or ("cuda" if torch.cuda.is_available() else "cpu"),
                        seed=int(args.seed) + 1509,
                        max_pairs=int(args.trajectory_adapter_max_pairs),
                        hidden_dim=int(args.trajectory_adapter_hidden_dim),
                        epochs=int(args.trajectory_adapter_epochs),
                        batch_size=int(args.trajectory_adapter_batch_size),
                        lr=float(args.trajectory_adapter_lr),
                        weight_decay=float(args.trajectory_adapter_weight_decay),
                        scale=float(args.trajectory_adapter_scale),
                        first_action_weight=float(args.trajectory_adapter_first_action_weight),
                        plan_loss_weight=float(args.trajectory_adapter_plan_loss_weight),
                        smooth_loss_weight=float(args.trajectory_adapter_smooth_loss_weight),
                        reg_weight=float(args.trajectory_adapter_reg_weight),
                    )
                    if set_trajectory_adapter is None:
                        raise RuntimeError("Selected backbone does not expose set_trajectory_adapter().")
                    set_trajectory_adapter(trajectory_adapter)
                    resolved_policy_modes[method] = "diffusion_action_trajectory_adapter"
                    resolved_policy_modes[f"{method}_checkpoint"] = str(trajectory_adapter_path)
                else:
                    raise ValueError(f"Unsupported diffusion action-representation backend: {diffusion_backend}")
            else:
                raise RuntimeError(
                    "ours_action_representation_adapter currently supports ACT and Diffusion backbones only; "
                    f"got backbone_name={backbone_name!r}."
                )
            if reset_policy_state is not None:
                reset_policy_state()
            _set_global_seed(int(args.seed) + 17000)
            calibration_predictions = _predict_dataset(
                active_backbone,
                cfg,
                calibration_path,
                input_normalization=bool(input_normalization),
            )
            heldout_predictions = _predict_dataset(
                active_backbone,
                cfg,
                heldout_path,
                input_normalization=bool(input_normalization),
            )
            post_mode = str(args.action_repr_post_adapter)
            action_repr_task_blend = None if bool(args.action_repr_fit_post_task_blends) else method_task_blend
            if post_mode == "none":
                adapter = ContinuousActionAdapter.identity(action_dim=int(heldout_predictions.predicted.shape[1]))
            elif post_mode == "residual_mlp":
                adapter = _fit_residual_mlp_adapter(
                    calibration_predictions,
                    blend=float(args.adapter_blend),
                    hidden_dim=int(args.few_shot_hidden_dim),
                    epochs=int(args.few_shot_epochs),
                    lr=float(args.few_shot_lr),
                    weight_decay=float(args.few_shot_weight_decay),
                    seed=int(args.seed) + 2307,
                    task_blend=action_repr_task_blend,
                )
            else:
                adapter = _fit_adapter(
                    calibration_predictions,
                    mode=post_mode,
                    blend=float(args.adapter_blend),
                    ridge=float(args.ridge),
                    latent_components=int(args.latent_components),
                    task_blend=action_repr_task_blend,
                    fit_task_blend=bool(args.action_repr_fit_post_task_blends),
                    task_blend_max=float(args.action_repr_fit_post_task_blend_max),
                )
            resolved_policy_modes[method] = f"{resolved_policy_modes[method]}_post_{post_mode}"
            adapters[method] = adapter
            adapter.save(output_root / method / "adapter.npz")
            transition_frames.append(_transition_metrics(heldout_predictions, adapter, baseline=method))
            _set_global_seed(int(args.seed) + 20000)
            exact_summary = _evaluate_exact_replay(
                active_backbone,
                cfg,
                heldout_path,
                output_root / "exact_replay" / method,
                adapter=adapter,
                baseline=method,
                num_per_task=int(args.exact_num_per_task),
                seed=int(args.seed) + 401,
                input_normalization=input_normalization,
            )[1]
            success_rows.extend(_flatten_success_summary("heldout_exact_replay", exact_summary, profile=str(args.profile)))
            _set_global_seed(int(args.seed) + 30000)
            fresh_summary = _evaluate_fresh_rollout(
                active_backbone,
                cfg,
                output_root / "fresh_rollout" / method,
                adapter=adapter,
                baseline=method,
                episodes_per_task=int(args.fresh_episodes_per_task),
                seed=int(args.seed) + 503,
                input_normalization=input_normalization,
            )[1]
            success_rows.extend(_flatten_success_summary("fresh_rollout", fresh_summary, profile=str(args.profile)))
            continue
        elif adapter_mode == "continuous_trajectory_adapter":
            supports_trajectory_adapter = getattr(active_backbone, "supports_runtime_trajectory_adapter", lambda: False)
            if not bool(supports_trajectory_adapter()):
                raise RuntimeError(
                    "ours_continuous_trajectory_adapter requires a backbone with runtime trajectory replacement."
                )
            trajectory_adapter, trajectory_adapter_path = fit_continuous_trajectory_adapter(
                active_backbone,
                cfg,
                calibration_path,
                output_root / method,
                device=args.policy_device or ("cuda" if torch.cuda.is_available() else "cpu"),
                seed=int(args.seed) + 1309,
                max_pairs=int(args.trajectory_adapter_max_pairs),
                hidden_dim=int(args.trajectory_adapter_hidden_dim),
                epochs=int(args.trajectory_adapter_epochs),
                batch_size=int(args.trajectory_adapter_batch_size),
                lr=float(args.trajectory_adapter_lr),
                weight_decay=float(args.trajectory_adapter_weight_decay),
                scale=float(args.trajectory_adapter_scale),
                first_action_weight=float(args.trajectory_adapter_first_action_weight),
                plan_loss_weight=float(args.trajectory_adapter_plan_loss_weight),
                smooth_loss_weight=float(args.trajectory_adapter_smooth_loss_weight),
                reg_weight=float(args.trajectory_adapter_reg_weight),
            )
            if set_trajectory_adapter is None:
                raise RuntimeError("Selected backbone does not expose set_trajectory_adapter().")
            set_trajectory_adapter(trajectory_adapter)
            if reset_policy_state is not None:
                reset_policy_state()
            calibration_predictions = _predict_dataset(
                active_backbone,
                cfg,
                calibration_path,
                input_normalization=bool(input_normalization),
            )
            heldout_predictions = _predict_dataset(
                active_backbone,
                cfg,
                heldout_path,
                input_normalization=bool(input_normalization),
            )
            post_mode = str(args.trajectory_adapter_post_adapter)
            if post_mode == "none":
                adapter = ContinuousActionAdapter.identity(action_dim=int(heldout_predictions.predicted.shape[1]))
            elif post_mode == "residual_mlp":
                adapter = _fit_residual_mlp_adapter(
                    calibration_predictions,
                    blend=float(args.adapter_blend),
                    hidden_dim=int(args.few_shot_hidden_dim),
                    epochs=int(args.few_shot_epochs),
                    lr=float(args.few_shot_lr),
                    weight_decay=float(args.few_shot_weight_decay),
                    seed=int(args.seed) + 1907,
                    task_blend=method_task_blend,
                )
            else:
                adapter = _fit_adapter(
                    calibration_predictions,
                    mode=post_mode,
                    blend=float(args.adapter_blend),
                    ridge=float(args.ridge),
                    latent_components=int(args.latent_components),
                    task_blend=method_task_blend,
                )
            resolved_policy_modes[method] = (
                f"{_backbone_name(cfg) or 'unknown'}_continuous_trajectory_adapter_post_{post_mode}"
            )
            adapters[method] = adapter
            adapter.save(output_root / method / "adapter.npz")
            transition_frames.append(_transition_metrics(heldout_predictions, adapter, baseline=method))
            _set_global_seed(int(args.seed) + 20000)
            exact_summary = _evaluate_exact_replay(
                active_backbone,
                cfg,
                heldout_path,
                output_root / "exact_replay" / method,
                adapter=adapter,
                baseline=method,
                num_per_task=int(args.exact_num_per_task),
                seed=int(args.seed) + 401,
                input_normalization=input_normalization,
            )[1]
            success_rows.extend(_flatten_success_summary("heldout_exact_replay", exact_summary, profile=str(args.profile)))
            _set_global_seed(int(args.seed) + 30000)
            fresh_summary = _evaluate_fresh_rollout(
                active_backbone,
                cfg,
                output_root / "fresh_rollout" / method,
                adapter=adapter,
                baseline=method,
                episodes_per_task=int(args.fresh_episodes_per_task),
                seed=int(args.seed) + 503,
                input_normalization=input_normalization,
            )[1]
            success_rows.extend(_flatten_success_summary("fresh_rollout", fresh_summary, profile=str(args.profile)))
            resolved_policy_modes[f"{method}_checkpoint"] = str(trajectory_adapter_path)
            continue
        elif adapter_mode == "diffusion_denoise_adapter":
            supports_denoise_adapter = getattr(active_backbone, "supports_runtime_diffusion_denoise_adapter", lambda: False)
            if not bool(supports_denoise_adapter()):
                raise RuntimeError("ours_diffusion_denoise_adapter requires a diffusion backbone.")
            denoise_adapter, denoise_adapter_path = fit_diffusion_denoise_adapter(
                active_backbone,
                cfg,
                calibration_path,
                output_root / method,
                device=args.policy_device or ("cuda" if torch.cuda.is_available() else "cpu"),
                seed=int(args.seed) + 1509,
                max_pairs=int(args.denoise_adapter_max_pairs),
                hidden_dim=int(args.denoise_adapter_hidden_dim),
                epochs=int(args.denoise_adapter_epochs),
                batch_size=int(args.denoise_adapter_batch_size),
                lr=float(args.denoise_adapter_lr),
                weight_decay=float(args.denoise_adapter_weight_decay),
                scale=float(args.denoise_adapter_scale),
                task_scales=_parse_optional_float_list(args.denoise_adapter_task_scales, expected=3),
                trajectory_loss_weight=float(args.denoise_adapter_trajectory_loss_weight),
                first_action_loss_weight=float(args.denoise_adapter_first_action_loss_weight),
                action_window_loss_weight=float(args.denoise_adapter_action_window_loss_weight),
                reg_weight=float(args.denoise_adapter_reg_weight),
                timestep_sampling=str(args.denoise_adapter_timestep_sampling),
            )
            if set_denoise_adapter is None:
                raise RuntimeError("Selected backbone does not expose set_diffusion_denoise_adapter().")
            set_denoise_adapter(denoise_adapter)
            if reset_policy_state is not None:
                reset_policy_state()
            calibration_predictions = _predict_dataset(
                active_backbone,
                cfg,
                calibration_path,
                input_normalization=bool(input_normalization),
            )
            heldout_predictions = _predict_dataset(
                active_backbone,
                cfg,
                heldout_path,
                input_normalization=bool(input_normalization),
            )
            post_mode = str(args.denoise_adapter_post_adapter)
            if post_mode == "none":
                adapter = ContinuousActionAdapter.identity(action_dim=int(heldout_predictions.predicted.shape[1]))
            elif post_mode == "residual_mlp":
                adapter = _fit_residual_mlp_adapter(
                    calibration_predictions,
                    blend=float(args.adapter_blend),
                    hidden_dim=int(args.few_shot_hidden_dim),
                    epochs=int(args.few_shot_epochs),
                    lr=float(args.few_shot_lr),
                    weight_decay=float(args.few_shot_weight_decay),
                    seed=int(args.seed) + 2107,
                    task_blend=method_task_blend,
                )
            else:
                adapter = _fit_adapter(
                    calibration_predictions,
                    mode=post_mode,
                    blend=float(args.adapter_blend),
                    ridge=float(args.ridge),
                    latent_components=int(args.latent_components),
                    task_blend=method_task_blend,
                )
            resolved_policy_modes[method] = f"diffusion_denoise_adapter_post_{post_mode}"
            adapters[method] = adapter
            adapter.save(output_root / method / "adapter.npz")
            transition_frames.append(_transition_metrics(heldout_predictions, adapter, baseline=method))
            _set_global_seed(int(args.seed) + 20000)
            exact_summary = _evaluate_exact_replay(
                active_backbone,
                cfg,
                heldout_path,
                output_root / "exact_replay" / method,
                adapter=adapter,
                baseline=method,
                num_per_task=int(args.exact_num_per_task),
                seed=int(args.seed) + 401,
                input_normalization=input_normalization,
            )[1]
            success_rows.extend(_flatten_success_summary("heldout_exact_replay", exact_summary, profile=str(args.profile)))
            _set_global_seed(int(args.seed) + 30000)
            fresh_summary = _evaluate_fresh_rollout(
                active_backbone,
                cfg,
                output_root / "fresh_rollout" / method,
                adapter=adapter,
                baseline=method,
                episodes_per_task=int(args.fresh_episodes_per_task),
                seed=int(args.seed) + 503,
                input_normalization=input_normalization,
            )[1]
            success_rows.extend(_flatten_success_summary("fresh_rollout", fresh_summary, profile=str(args.profile)))
            resolved_policy_modes[f"{method}_checkpoint"] = str(denoise_adapter_path)
            continue
        elif adapter_mode == "closed_loop_representation_adapter":
            runtime_backbone, adapter, resolved_mode = _fit_closed_loop_representation_adapter(
                active_backbone,
                cfg,
                calibration_path,
                output_root / method,
                args,
                task_blend=method_task_blend,
                input_normalization=bool(input_normalization),
            )
            resolved_policy_modes[method] = resolved_mode
            adapters[method] = adapter
            adapter.save(output_root / method / "adapter.npz")
            heldout_predictions = _predict_dataset(
                runtime_backbone,
                cfg,
                heldout_path,
                input_normalization=bool(input_normalization),
            )
            transition_frames.append(_transition_metrics(heldout_predictions, adapter, baseline=method))
            _set_global_seed(int(args.seed) + 20000)
            exact_summary = _evaluate_exact_replay(
                runtime_backbone,
                cfg,
                heldout_path,
                output_root / "exact_replay" / method,
                adapter=adapter,
                baseline=method,
                num_per_task=int(args.exact_num_per_task),
                seed=int(args.seed) + 401,
                input_normalization=input_normalization,
            )[1]
            success_rows.extend(_flatten_success_summary("heldout_exact_replay", exact_summary, profile=str(args.profile)))
            _set_global_seed(int(args.seed) + 30000)
            fresh_summary = _evaluate_fresh_rollout(
                runtime_backbone,
                cfg,
                output_root / "fresh_rollout" / method,
                adapter=adapter,
                baseline=method,
                episodes_per_task=int(args.fresh_episodes_per_task),
                seed=int(args.seed) + 503,
                input_normalization=input_normalization,
            )[1]
            success_rows.extend(_flatten_success_summary("fresh_rollout", fresh_summary, profile=str(args.profile)))
            continue
        else:
            adapter = _fit_adapter(
                calibration_predictions,
                mode=adapter_mode,
                blend=float(args.adapter_blend),
                ridge=float(args.ridge),
                latent_components=int(args.latent_components),
                task_blend=method_task_blend,
            )
        adapters[method] = adapter
        adapter.save(output_root / method / "adapter.npz")
        transition_frames.append(_transition_metrics(heldout_predictions, adapter, baseline=method))

        _set_global_seed(int(args.seed) + 20000)
        exact_summary = _evaluate_exact_replay(
            active_backbone,
            cfg,
            heldout_path,
            output_root / "exact_replay" / method,
            adapter=adapter,
            baseline=method,
            num_per_task=int(args.exact_num_per_task),
            seed=int(args.seed) + 401,
            input_normalization=input_normalization,
        )[1]
        success_rows.extend(_flatten_success_summary("heldout_exact_replay", exact_summary, profile=str(args.profile)))
        _set_global_seed(int(args.seed) + 30000)
        fresh_summary = _evaluate_fresh_rollout(
            active_backbone,
            cfg,
            output_root / "fresh_rollout" / method,
            adapter=adapter,
            baseline=method,
            episodes_per_task=int(args.fresh_episodes_per_task),
            seed=int(args.seed) + 503,
            input_normalization=input_normalization,
        )[1]
        success_rows.extend(_flatten_success_summary("fresh_rollout", fresh_summary, profile=str(args.profile)))

    transition_path = output_root / "method_transition_metrics.csv"
    pd.concat(transition_frames, ignore_index=True).to_csv(transition_path, index=False)
    success_path = output_root / "method_success_summary.csv"
    pd.DataFrame.from_records(success_rows).to_csv(success_path, index=False)
    manifest = {
        "config": str(args.config),
        "policy_path": str(policy_path),
        "profile": str(args.profile),
        "profile_context": PROFILE_CONTEXTS[str(args.profile)],
        "calibration_path": str(calibration_path),
        "heldout_path": str(heldout_path),
        "methods": methods,
        "adapter_blend": float(args.adapter_blend),
        "adapter_task_blends": cli_task_blend.tolist() if cli_task_blend is not None else None,
        "method_default_task_blends": {
            name: cfg["default_task_blends"]
            for name, cfg in METHODS.items()
            if name in methods and "default_task_blends" in cfg
        },
        "resolved_policy_modes": resolved_policy_modes,
        "ridge": float(args.ridge),
        "latent_components": int(args.latent_components),
        "latent_adapter": {
            "scale": float(args.latent_adapter_scale),
            "reg_weight": float(args.latent_adapter_reg_weight),
            "action_loss_weight": float(args.latent_adapter_action_loss_weight),
            "chunk_action_loss_weight": float(args.latent_adapter_chunk_action_loss_weight),
            "source_stat_weight": float(args.latent_adapter_source_stat_weight),
            "alignment_blend": float(args.latent_adapter_alignment_blend),
            "hyperparam_gate": str(args.latent_adapter_hyperparam_gate),
            "action_source": str(args.latent_adapter_action_source),
        },
        "action_repr": {
            "post_adapter": str(args.action_repr_post_adapter),
            "diffusion_backend": str(args.action_repr_diffusion_backend),
            "fit_post_task_blends": bool(args.action_repr_fit_post_task_blends),
            "fit_post_task_blend_max": float(args.action_repr_fit_post_task_blend_max),
        },
        "denoise_adapter": {
            "scale": float(args.denoise_adapter_scale),
            "task_scales": _parse_optional_float_list(args.denoise_adapter_task_scales, expected=3),
            "trajectory_loss_weight": float(args.denoise_adapter_trajectory_loss_weight),
            "first_action_loss_weight": float(args.denoise_adapter_first_action_loss_weight),
            "action_window_loss_weight": float(args.denoise_adapter_action_window_loss_weight),
            "reg_weight": float(args.denoise_adapter_reg_weight),
            "timestep_sampling": str(args.denoise_adapter_timestep_sampling),
            "max_pairs": int(args.denoise_adapter_max_pairs),
        },
        "method_mapping": {
            "probe_feature_alignment": "task-wise moment alignment on continuous actions",
            "few_shot_finetuning": "small residual MLP fitted on calibration action residuals",
            "tent_style": "raw/normalized test-time action ensemble as a continuous TTA proxy",
            "domain_randomization_only": "separate DR-only checkpoint supplied by --domain-randomization-policy-path",
            "task_policy_selector": "task-wise checkpoint selector using known task id",
            "ours_proxy": "task-wise affine continuous action adapter",
            "ours_task_gated_residual": "calibration residual MLP with a conservative task gate: L1/L3 identity, L2 residual blend 0.25",
            "ours_multimodel_adaptive": "legacy model-aware ablation using conservative task-gated residual calibration; default gate is L1/L3 identity and L2 residual blend 0.25",
            "ours_validation_taskwise_selector": "legacy validation-frozen backbone/task selector over calibration-fitted action adapters",
            "ours_validation_taskwise_selector_v2": "legacy RNG-stable validation-frozen backbone/task selector",
            "ours_profile_adaptive_selector": "legacy validation-frozen profile/backbone selector over calibration-fitted action adapters",
            "ours_latent_adapter": "ridge residual adapter from internal policy latent plus task/action features",
            "ours_calibrated_selector": "task-wise adapter selector chosen by calibration exact replay",
            "ours_continuous_latent_adapter": "PLICA-style latent adapter trained by source transition consistency on calibration data",
            "ours_continuous_trajectory_adapter": "Calibration-guided residual adapter on the action-trajectory representation before action execution",
            "ours_diffusion_denoise_adapter": "Calibration-guided residual adapter on each diffusion denoising model output before scheduler update",
            "ours_action_representation_adapter": "Unified proposed method: one residual representation adapter per backbone, inserted at ACT action-head latent or Diffusion denoising-output representation; optional task-conditioned output calibration can be task bias, moment, or regularized diagonal affine residual fitted only on calibration data; optional latent-shift gate uses calibration/source latent statistics to set residual strength before training; no task/profile candidate selection",
            "ours_action_representation_adapter_normalized": "Action-representation adapter ablation with the same runtime representation adapter and input image normalization enabled",
            "diagnostic_closed_loop_representation_adapter": "Diagnostic-only closed-loop candidate selector for ablations; not used as the unified proposed method",
        },
        "transition_metrics_csv": str(transition_path),
        "success_summary_csv": str(success_path),
    }
    with (output_root / "manifest.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)
    print(f"method_transition_metrics_csv={transition_path}")
    print(f"method_success_summary_csv={success_path}")


if __name__ == "__main__":
    main()

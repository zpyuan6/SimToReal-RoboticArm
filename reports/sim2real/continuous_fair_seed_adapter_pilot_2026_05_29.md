# Continuous Sim-to-Real Adapter Pilot with Fair Episode Seeds

Date: 2026-05-29

## What Changed

- The continuous LeRobot wrapper now exposes internal policy latents instead of using flattened action plans:
  - ACT: input token sequence to `model.action_head`, shape `6144` for the frozen ACT checkpoint.
  - Diffusion: `_prepare_global_conditioning()` output, shape `160` for the frozen Diffusion checkpoint.
- Fresh rollout and exact replay in `scripts/run_continuous_sim2real_bridge.py` now use fixed per-record/per-episode seeds. This prevents one method's earlier termination from changing later task initial states or observation noise.
- Added adapter variants:
  - `ours_latent_adapter`: ridge residual correction from internal latent + task + current action.
  - `ours_calibrated_selector`: per-task selector chosen by calibration exact replay.
  - Optional `--adapter-task-blends` for per-task adapter strength.

## Fair-Seed Pilot Results: appearance_shift

Budget: 4 heldout exact replay episodes per task and 4 fresh rollout episodes per task.

### ACT

| Method | Heldout Exact Overall | Fresh Overall | Fresh L1 | Fresh L2 | Fresh L3 |
|---|---:|---:|---:|---:|---:|
| no_adaptation | 0.3333 | 0.4167 | 0.5000 | 0.0000 | 0.7500 |
| input_normalization | 0.2500 | 0.2500 | 0.2500 | 0.5000 | 0.0000 |
| probe_feature_alignment | 0.3333 | 0.2500 | 0.5000 | 0.2500 | 0.0000 |
| static_adapter | 0.3333 | 0.2500 | 0.5000 | 0.0000 | 0.2500 |
| few_shot_finetuning | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 |
| tent_style | 0.3333 | 0.3333 | 0.2500 | 0.5000 | 0.2500 |
| ours_proxy | 0.1667 | 0.2500 | 0.7500 | 0.0000 | 0.0000 |
| ours_latent_adapter | 0.2500 | 0.3333 | 0.5000 | 0.0000 | 0.5000 |

Result: `no_adaptation` is still strongest on fresh rollout for ACT under this profile.

### Diffusion

| Method | Heldout Exact Overall | Fresh Overall | Fresh L1 | Fresh L2 | Fresh L3 |
|---|---:|---:|---:|---:|---:|
| no_adaptation | 0.2500 | 0.1667 | 0.0000 | 0.2500 | 0.2500 |
| input_normalization | 0.0833 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| probe_feature_alignment | 0.0833 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| static_adapter | 0.2500 | 0.0833 | 0.0000 | 0.0000 | 0.2500 |
| few_shot_finetuning | 0.2500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tent_style | 0.0000 | 0.0833 | 0.0000 | 0.0000 | 0.2500 |
| ours_proxy | 0.0833 | 0.0833 | 0.0000 | 0.0000 | 0.2500 |
| ours_latent_adapter | 0.0833 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Result: `no_adaptation` is also strongest on fresh rollout for Diffusion under this profile.

## Interpretation

The action/latent residual adapters often reduce heldout action MSE, but do not improve closed-loop success. This is a concrete sign that one-step action closeness is not a reliable optimization target for these staged manipulation tasks. Small residual corrections can push the controller out of the narrow successful closed-loop trajectory, especially for L2/L3.

The calibrated selector also failed: calibration exact replay selected adapters that reduced action MSE but degraded heldout and fresh success. This makes selector-based adapter tuning unsafe unless it uses substantially more calibration rollouts and a success-oriented selection objective.

## Decision

Do not continue optimizing action-space or post-hoc latent residual adapters as the primary sim-to-real method.

The next meaningful optimization was tested as real domain-randomized ACT checkpoints:

- Generate train/val/test data with randomized visual/camera/actuation contexts.
- Train ACT and Diffusion on that data.
- Compare `domain_randomization_only` against the frozen non-DR checkpoints and the lightweight adapters under the same fair-seed evaluator.

## Domain-Randomization Follow-up

Three ACT variants were evaluated across `appearance_shift`, `embodiment_shift`, and `joint_shift`, always against the same frozen `no_adaptation` checkpoint:

| Variant | Heldout Exact Overall | Fresh Overall |
|---|---:|---:|
| frozen no_adaptation | 0.3333 | 0.3056 |
| scratch DR ACT | 0.2222 | 0.0556 |
| DR-only fine-tune from frozen ACT | 0.1389 | 0.1111 |
| neutral+DR mixed fine-tune from frozen ACT | 0.3056 | 0.1667 |
| neutral+visual/camera DR mixed fine-tune from frozen ACT | 0.2500 | 0.1944 |
| visual/camera DR frozen-backbone head fine-tune, 500 steps | 0.3056 | 0.3056 |
| task-policy selector: L1 head-500, L2/L3 frozen ACT, 4/task pilot | 0.3889 | 0.3056 |

Mixed fine-tuning was the best DR attempt, but still did not beat the frozen checkpoint on fresh rollout. It improved or matched some one-step/heldout measurements, but the closed-loop success dropped, especially for L2/L3.

Mixed fine-tune per-stage mean across the three shift profiles:

| Eval | Method | L1 | L2 | L3 | Overall |
|---|---|---:|---:|---:|---:|
| heldout exact | frozen no_adaptation | 0.3333 | 0.4167 | 0.2500 | 0.3333 |
| heldout exact | neutral+DR mixed fine-tune | 0.6667 | 0.2500 | 0.0000 | 0.3056 |
| fresh rollout | frozen no_adaptation | 0.2500 | 0.2500 | 0.4167 | 0.3056 |
| fresh rollout | neutral+DR mixed fine-tune | 0.3333 | 0.0833 | 0.0833 | 0.1667 |

The narrower visual/camera-only DR run used no actuation randomization (`action_gain=1`, `action_delay=0`, `joint_bias=0`) and a lower fine-tuning learning rate (`policy.optimizer_lr=3e-6`, `policy.optimizer_lr_backbone=1e-6`). It reached `1.0000` train-like rollout success during checkpoint selection, but still did not transfer better than the frozen policy under the stronger shift profiles.

Visual/camera DR mixed fine-tune per-stage mean across the three shift profiles:

| Eval | Method | L1 | L2 | L3 | Overall |
|---|---|---:|---:|---:|---:|
| heldout exact | frozen no_adaptation | 0.3333 | 0.4167 | 0.2500 | 0.3333 |
| heldout exact | visual/camera DR mixed fine-tune | 0.4167 | 0.3333 | 0.0000 | 0.2500 |
| fresh rollout | frozen no_adaptation | 0.2500 | 0.2500 | 0.4167 | 0.3056 |
| fresh rollout | visual/camera DR mixed fine-tune | 0.1667 | 0.1667 | 0.2500 | 0.1944 |

The best partial fine-tuning variant freezes the ResNet backbone by setting `policy.optimizer_lr_backbone=0` and fine-tunes only the non-backbone ACT parameters for 500 steps. Longer 2500-step fine-tuning degraded L3 transfer.

The 4/task pilot suggested a possible conservative task-policy selector:

- L1 uses the 500-step visual/camera DR frozen-backbone head fine-tuned checkpoint.
- L2 and L3 keep the original frozen ACT checkpoint.

Task-policy selector mean across the three shift profiles:

| Eval | Method | L1 | L2 | L3 | Overall |
|---|---|---:|---:|---:|---:|
| heldout exact | frozen no_adaptation | 0.3333 | 0.4167 | 0.2500 | 0.3333 |
| heldout exact | task-policy selector | 0.5000 | 0.4167 | 0.2500 | 0.3889 |
| fresh rollout | frozen no_adaptation | 0.2500 | 0.2500 | 0.4167 | 0.3056 |
| fresh rollout | task-policy selector | 0.2500 | 0.2500 | 0.4167 | 0.3056 |

## Expanded Selector Check

The selector hypothesis was rerun with a larger and cleaner budget: 20 heldout exact replay episodes per task and 20 fresh rollout episodes per task, with newly generated 60-episode heldout splits for each shift profile.

| Method | Heldout Exact Overall | Fresh Overall |
|---|---:|---:|
| frozen no_adaptation | 0.2778 | 0.2611 |
| task-policy selector: L1 head-500, L2/L3 frozen ACT | 0.2833 | 0.2556 |

Expanded heldout exact replay per-stage mean:

| Method | L1 | L2 | L3 | Overall |
|---|---:|---:|---:|---:|
| frozen no_adaptation | 0.4167 | 0.2667 | 0.1500 | 0.2778 |
| task-policy selector | 0.4500 | 0.2667 | 0.1333 | 0.2833 |

Expanded fresh rollout per-stage mean:

| Method | L1 | L2 | L3 | Overall |
|---|---:|---:|---:|---:|
| frozen no_adaptation | 0.2500 | 0.2000 | 0.3333 | 0.2611 |
| task-policy selector | 0.2167 | 0.2167 | 0.3333 | 0.2556 |

Conclusion: broad post-hoc residual adapters and broad DR fine-tuning are not superior to the frozen ACT policy. The 4/task selector improvement was not confirmed by the 20/task expanded check: heldout exact replay improves only by `+0.0056`, while fresh rollout drops by `-0.0055`. The selector is useful as an experimental runtime capability, but should not be promoted as the default sim-to-real policy.

The current default should remain the frozen ACT `no_adaptation` policy. Further model optimization should focus on L2/L3 closed-loop geometry and success-oriented validation, not action-MSE adapters or global DR fine-tuning.

The next optimization should not be another post-hoc residual adapter. If sim-to-real performance must be improved from this point, use a smaller and more targeted policy update:

- visual-only or camera-only randomization first, without actuation randomization;
- lower learning-rate fine-tuning from the frozen checkpoint;
- partial fine-tuning or LoRA/adapters inside the visual/state encoder while keeping the action head stable;
- success-oriented validation rollouts for checkpoint selection, not action MSE alone.

Output artifacts:

- ACT fair pilot: `results/continuous_sim2real_method_comparison_fairpilot/act/appearance_shift`
- Diffusion fair pilot: `results/continuous_sim2real_method_comparison_fairpilot/diffusion/appearance_shift`
- Selector pilot: `results/continuous_sim2real_method_comparison_selectorpilot/act/appearance_shift`
- ACT scratch DR: `results/continuous_sim2real_method_comparison_drpilot/act`
- ACT DR-only fine-tune: `results/continuous_sim2real_method_comparison_drfinepilot/act`
- ACT neutral+DR mixed fine-tune: `results/continuous_sim2real_method_comparison_drmixpilot/act`
- ACT neutral+visual/camera DR mixed fine-tune: `results/continuous_sim2real_method_comparison_visualdrmixpilot/act`
- ACT frozen-backbone head fine-tune, 2500 steps: `results/continuous_sim2real_method_comparison_visualdrheadpilot/act`
- ACT frozen-backbone head fine-tune, 500 steps: `results/continuous_sim2real_method_comparison_visualdrhead500pilot/act`
- ACT task-policy selector: `results/continuous_sim2real_method_comparison_taskselectorpilot/act`
- ACT task-policy selector expanded check: `results/continuous_sim2real_method_comparison_taskselector_eval20_bigsplit/act`
- Runtime task-policy selector config: `configs/continuous_act_jointtarget_staged_task_selector.yaml`
- Runtime task-policy selector smoke test: `results/continuous_task_selector_smoke`

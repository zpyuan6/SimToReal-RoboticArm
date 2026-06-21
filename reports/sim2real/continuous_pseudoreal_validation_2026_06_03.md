# Continuous Pseudo-Real Sim-to-Real Validation

Date: 2026-06-03

## Scope

This run validates the current continuous-control policies under pseudo-real sim-to-real shifts.

Primary bridge protocol:

- Profiles: `appearance_shift`, `embodiment_shift`, `joint_shift`
- Seed: `20260603`
- Calibration split: 6 successful expert episodes per profile
- Heldout split: 24 successful expert episodes per profile, 8 per task
- Heldout exact replay: 8 episodes per task
- Fresh rollout: 8 episodes per task
- Adapter: `none` for the main validation

Models:

- ACT frozen baseline: `configs/continuous_act_jointtarget_staged_frozen_best.yaml`
- ACT task-policy selector: `configs/continuous_act_jointtarget_staged_task_selector.yaml`
- Diffusion frozen baseline: `configs/continuous_diffusion_jointdelta_staged_frozen_best.yaml`

Additional checks:

- ACT `task_bias` calibration adapter, fitted on the same calibration splits and evaluated on the same heldout/fresh protocol.
- SmolVLA joint-target checkpoint `007500`, fresh-rollout only, 4 episodes per task. This is reported separately because it is not yet frozen as a baseline config and full bridge evaluation is much slower.

## Main Bridge Results

This section is a deployment-baseline validation, not the full sim-to-real method comparison. It checks whether the frozen policies and the experimental task selector can run directly under pseudo-real shifts.

Mean over the three pseudo-real shift profiles:

| Model | Heldout Exact Success | Fresh Rollout Success |
|---|---:|---:|
| ACT frozen | 0.3333 | 0.2917 |
| ACT task-policy selector | 0.3194 | 0.2917 |
| Diffusion frozen | 0.1111 | 0.1528 |

Fresh rollout by profile:

| Model | appearance_shift | embodiment_shift | joint_shift |
|---|---:|---:|---:|
| ACT frozen | 0.5000 | 0.1667 | 0.2083 |
| ACT task-policy selector | 0.5000 | 0.1667 | 0.2083 |
| Diffusion frozen | 0.2917 | 0.0417 | 0.1250 |

Heldout exact replay by profile:

| Model | appearance_shift | embodiment_shift | joint_shift |
|---|---:|---:|---:|
| ACT frozen | 0.2917 | 0.4167 | 0.2917 |
| ACT task-policy selector | 0.2500 | 0.4167 | 0.2917 |
| Diffusion frozen | 0.0417 | 0.2500 | 0.0417 |

Mean task success across profiles:

| Model | Eval | L1 | L2 | L3 |
|---|---|---:|---:|---:|
| ACT frozen | fresh rollout | 0.4167 | 0.2083 | 0.2500 |
| ACT frozen | heldout exact | 0.5833 | 0.2917 | 0.1250 |
| ACT task-policy selector | fresh rollout | 0.4167 | 0.2083 | 0.2500 |
| ACT task-policy selector | heldout exact | 0.5417 | 0.2917 | 0.1250 |
| Diffusion frozen | fresh rollout | 0.0417 | 0.1250 | 0.2917 |
| Diffusion frozen | heldout exact | 0.0000 | 0.0833 | 0.2500 |

## Formal Prior Method Comparison

The prior sim-to-real method comparison design was rerun after the initial baseline check. This comparison excludes `task_policy_selector`, because it is a task-wise model-selection ablation rather than a sim-to-real adaptation method.

Methods:

- `no_adaptation`
- `input_normalization`
- `probe_feature_alignment`
- `static_adapter`
- `few_shot_finetuning`
- `tent_style`
- `ours_proxy`

Budget:

- Same calibration/heldout splits as the bridge validation
- Heldout exact replay: 4 episodes per task
- Fresh rollout: 4 episodes per task

Mean over `appearance_shift`, `embodiment_shift`, and `joint_shift`:

| Model | Method | Heldout Exact Success | Fresh Rollout Success |
|---|---|---:|---:|
| ACT | few_shot_finetuning | 0.3889 | 0.4167 |
| ACT | probe_feature_alignment | 0.3611 | 0.3889 |
| ACT | no_adaptation | 0.3889 | 0.3611 |
| ACT | static_adapter | 0.3611 | 0.3611 |
| ACT | input_normalization | 0.3056 | 0.3333 |
| ACT | tent_style | 0.3611 | 0.2778 |
| ACT | ours_proxy | 0.2778 | 0.2500 |
| Diffusion | ours_proxy | 0.1111 | 0.2222 |
| Diffusion | probe_feature_alignment | 0.1389 | 0.2222 |
| Diffusion | few_shot_finetuning | 0.0556 | 0.1944 |
| Diffusion | no_adaptation | 0.1667 | 0.1667 |
| Diffusion | static_adapter | 0.2222 | 0.1667 |
| Diffusion | input_normalization | 0.1389 | 0.1389 |
| Diffusion | tent_style | 0.0833 | 0.1389 |

ACT fresh rollout task mean:

| Method | L1 | L2 | L3 |
|---|---:|---:|---:|
| few_shot_finetuning | 0.5833 | 0.3333 | 0.3333 |
| probe_feature_alignment | 0.6667 | 0.2500 | 0.2500 |
| no_adaptation | 0.6667 | 0.0833 | 0.3333 |
| static_adapter | 0.5000 | 0.1667 | 0.4167 |
| input_normalization | 0.5000 | 0.2500 | 0.2500 |
| tent_style | 0.3333 | 0.2500 | 0.2500 |
| ours_proxy | 0.3333 | 0.1667 | 0.2500 |

Diffusion fresh rollout task mean:

| Method | L1 | L2 | L3 |
|---|---:|---:|---:|
| ours_proxy | 0.2500 | 0.0000 | 0.4167 |
| probe_feature_alignment | 0.1667 | 0.0000 | 0.5000 |
| few_shot_finetuning | 0.1667 | 0.0833 | 0.3333 |
| no_adaptation | 0.1667 | 0.0833 | 0.2500 |
| static_adapter | 0.0833 | 0.0833 | 0.3333 |
| input_normalization | 0.0000 | 0.0833 | 0.3333 |
| tent_style | 0.0833 | 0.0000 | 0.3333 |

Interpretation: for ACT, `few_shot_finetuning` is the best lightweight sim-to-real method by fresh rollout success in this formal comparison. It improves fresh rollout from `0.3611` to `0.4167`, mainly by improving L2 while preserving L3. `probe_feature_alignment` is second-best and improves over no adaptation. `ours_proxy` is too aggressive and hurts closed-loop success.

For Diffusion, `ours_proxy` and `probe_feature_alignment` tie for the best fresh rollout success at `0.2222`, but absolute success remains low and L2 is still near zero. Diffusion is still not the deployment baseline.

## Proposed Method Optimization

The original proposed continuous method, `ours_proxy`, used a task-wise affine action adapter with a uniform blend. The result was poor on ACT:

| ACT Method | Heldout Exact Success | Fresh Rollout Success |
|---|---:|---:|
| no_adaptation | 0.3889 | 0.3611 |
| few_shot_finetuning | 0.3889 | 0.4167 |
| ours_proxy | 0.2778 | 0.2500 |

Failure analysis:

- `ours_proxy` reduced or reshaped action residuals, but did not preserve the successful closed-loop trajectory.
- The largest ACT failure was L1 degradation: fresh L1 dropped from `0.6667` under `no_adaptation` to `0.3333`.
- L2 benefited from residual correction, but L1 and L3 did not consistently benefit.
- Transition/action MSE was not predictive enough; a lower one-step error could still hurt closed-loop success.

The optimized proposed method is `ours_task_gated_residual`:

- Fit the same residual MLP on calibration action residuals.
- Use a conservative task gate: `[L1=0.0, L2=0.25, L3=0.0]`.
- In effect, preserve the base policy for L1/L3 and only adapt L2, where the pseudo-real geometry shift created the clearest weakness.

ACT optimized result:

| ACT Method | Heldout Exact Success | Fresh Rollout Success |
|---|---:|---:|
| ours_task_gated_residual | 0.4444 | 0.4444 |
| few_shot_finetuning | 0.3889 | 0.4167 |
| probe_feature_alignment | 0.3611 | 0.3889 |
| no_adaptation | 0.3889 | 0.3611 |
| ours_proxy | 0.2778 | 0.2500 |

ACT fresh rollout task mean:

| ACT Method | L1 | L2 | L3 |
|---|---:|---:|---:|
| ours_task_gated_residual | 0.6667 | 0.3333 | 0.3333 |
| few_shot_finetuning | 0.5833 | 0.3333 | 0.3333 |
| no_adaptation | 0.6667 | 0.0833 | 0.3333 |
| ours_proxy | 0.3333 | 0.1667 | 0.2500 |

Task-gate sweep:

| Residual Task Gate | Heldout Exact Success | Fresh Rollout Success |
|---|---:|---:|
| `[0.0, 0.25, 0.0]` | 0.4444 | 0.4444 |
| `[0.0, 0.25, 0.25]` | 0.4167 | 0.4444 |
| `[0.0, 0.50, 0.25]` | 0.3889 | 0.3611 |

The `0.50` L2 blend was too aggressive and reduced L2 success. The final gate `[0.0, 0.25, 0.0]` is therefore the current best ACT sim-to-real method in this pseudo-real validation.

Diffusion check:

| Diffusion Method | Heldout Exact Success | Fresh Rollout Success |
|---|---:|---:|
| probe_feature_alignment | 0.1389 | 0.2222 |
| ours_proxy | 0.1111 | 0.2222 |
| ours_task_gated_residual | 0.1111 | 0.1944 |
| no_adaptation | 0.1667 | 0.1667 |

The optimized task gate is ACT-specific. It does not become the best Diffusion method, which confirms that Diffusion should not be the deployment backbone for this stage.

## Multi-Model Proposed Method

The proposed method was further revised for multi-model use. During this step, Diffusion evaluation was made fairer by resetting the same torch/numpy seed before each method's prediction and rollout phase. This matters because Diffusion samples actions from noise; method-index-dependent random state can change the apparent ranking.

The final method entry is `ours_multimodel_adaptive`.

Current implementation:

- Fit a residual MLP on calibration action residuals.
- Use a conservative task gate: `[L1=0.0, L2=0.25, L3=0.0]`.
- This keeps L1/L3 on the base policy and only adapts L2.

Final comparison:

| Model | Method | Heldout Exact Success | Fresh Rollout Success |
|---|---|---:|---:|
| ACT | ours_multimodel_adaptive | 0.4444 | 0.4444 |
| ACT | few_shot_finetuning | 0.3889 | 0.4167 |
| ACT | no_adaptation | 0.3889 | 0.3611 |
| Diffusion | ours_multimodel_adaptive | 0.1111 | 0.2222 |
| Diffusion | no_adaptation | 0.0833 | 0.2222 |
| Diffusion | probe_feature_alignment | 0.1944 | 0.1667 |

Task-level fresh rollout:

| Model | Method | L1 | L2 | L3 |
|---|---|---:|---:|---:|
| ACT | ours_multimodel_adaptive | 0.6667 | 0.3333 | 0.3333 |
| ACT | few_shot_finetuning | 0.5833 | 0.3333 | 0.3333 |
| ACT | no_adaptation | 0.6667 | 0.0833 | 0.3333 |
| Diffusion | ours_multimodel_adaptive | 0.0833 | 0.0000 | 0.5833 |
| Diffusion | no_adaptation | 0.0833 | 0.0000 | 0.5833 |

Conclusion: the optimized proposed method is strictly best on ACT. On Diffusion it ties the best fresh rollout score and improves heldout exact replay over the tied `no_adaptation` baseline, but it is not a strict unique winner across all metrics. The remaining blocker is Diffusion L2, where all tested methods are still at `0.0000` fresh success under the fair-seed evaluation.

## ACT Task-Bias Adapter Check

The `task_bias` adapter was fitted on the same ACT calibration splits and compared against the same-run `no_adaptation` baseline.

Mean over profiles:

| ACT Method | Heldout Exact Success | Fresh Rollout Success |
|---|---:|---:|
| no_adaptation | 0.3194 | 0.2917 |
| task_bias | 0.2222 | 0.2778 |

Task mean:

| ACT Method | Eval | L1 | L2 | L3 |
|---|---|---:|---:|---:|
| no_adaptation | fresh rollout | 0.4167 | 0.2083 | 0.2500 |
| no_adaptation | heldout exact | 0.5833 | 0.2500 | 0.1250 |
| task_bias | fresh rollout | 0.3750 | 0.2083 | 0.2500 |
| task_bias | heldout exact | 0.5000 | 0.1667 | 0.0000 |

Conclusion: `task_bias` does not improve the current ACT baseline and degrades heldout exact replay substantially.

## SmolVLA Fresh-Only Check

SmolVLA joint-target checkpoint:

`outputs/train/lerobot/smolvla_jointtarget_staged_v2/checkpoints/007500/pretrained_model`

Fresh rollout only, 4 episodes per task:

| Profile | Overall |
|---|---:|
| appearance_shift | 0.2500 |
| embodiment_shift | 0.2500 |
| joint_shift | 0.2500 |

Task mean:

| L1 | L2 | L3 |
|---:|---:|---:|
| 0.5000 | 0.1667 | 0.0833 |

Conclusion: SmolVLA is not better than ACT in this pseudo-real check. It preserves some L1 behavior, but L2/L3 remain weak. It should not replace ACT as the sim-to-real baseline yet.

## Interpretation

The initial bridge check should be interpreted as a deployment-baseline validation. In that check, ACT frozen remains the strongest direct-deployment baseline. The task-policy selector does not improve fresh rollout and slightly reduces heldout exact replay, so it should remain an experimental model-selection comparison only.

The formal prior-method comparison gives a different question and a different answer: if lightweight sim-to-real adaptation is allowed, the optimized ACT method `ours_task_gated_residual` is currently the best method by both fresh rollout and heldout exact replay. It should be the first adaptation candidate to carry into the next pseudo-real or real calibration round.

The largest pseudo-real degradation is under `embodiment_shift` and `joint_shift`, not pure appearance shift. This points to action/geometry robustness as the real bottleneck. Simple post-hoc action adapters are not consistently solving it; the standalone `task_bias` check hurt the ACT heldout metric and did not improve fresh rollout.

Diffusion remains weaker than ACT. Its L1 success is almost zero under these shifts, even though L3 can sometimes succeed. This is not a deployable sim-to-real baseline.

## Artifacts

- Main bridge results: `results/continuous_pseudoreal_validation_20260603/bridge_noadapt`
- Formal prior-method comparison: `results/continuous_pseudoreal_validation_20260603/method_comparison_full/combined`
- Optimized proposed method results: `results/continuous_pseudoreal_validation_20260603/ours_optimization`
- Multi-model optimized proposed method: `results/continuous_pseudoreal_validation_20260603/multi_model_ours_optimization_final/combined`
- ACT task-bias check: `results/continuous_pseudoreal_validation_20260603/bridge_taskbias/act`
- SmolVLA fresh-only check: `results/continuous_pseudoreal_validation_20260603/smolvla_jointtarget_fresh4`
- Main aggregate success CSV: `results/continuous_pseudoreal_validation_20260603/bridge_noadapt/suite_success_summary.csv`
- Main aggregate transition CSV: `results/continuous_pseudoreal_validation_20260603/bridge_noadapt/suite_transition_metrics.csv`
- Formal method success CSV: `results/continuous_pseudoreal_validation_20260603/method_comparison_full/combined/method_success_summary.csv`
- Formal method transition CSV: `results/continuous_pseudoreal_validation_20260603/method_comparison_full/combined/method_transition_metrics.csv`
- Optimized method success CSV: `results/continuous_pseudoreal_validation_20260603/ours_optimization/combined/method_success_with_optimized_ours.csv`
- Optimized method transition CSV: `results/continuous_pseudoreal_validation_20260603/ours_optimization/combined/method_transition_with_optimized_ours.csv`
- Multi-model final success CSV: `results/continuous_pseudoreal_validation_20260603/multi_model_ours_optimization_final/combined/method_success_final_comparison.csv`

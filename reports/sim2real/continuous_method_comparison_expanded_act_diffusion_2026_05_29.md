# Expanded Continuous Sim-to-Real Method Comparison - ACT and Diffusion

Date: 2026-05-29

## Added Methods

The first continuous method comparison only included the adopted paper subset:

- `no_adaptation`
- `input_normalization`
- `static_adapter`
- `ours_proxy`

This expanded run adds two more methods from the prior comparison pool:

- `probe_feature_alignment`
- `few_shot_finetuning`

Because the official ACT and Diffusion wrappers still do not expose the final
PLICA latent hooks, both are implemented as continuous action-space proxies:

| Method | Continuous implementation |
| --- | --- |
| `probe_feature_alignment` | task-wise moment alignment between calibration policy actions and expert actions |
| `few_shot_finetuning` | small residual MLP fitted on calibration action residuals |

These are not substitutes for the final continuous latent PLICA adapter. They
are lightweight comparable baselines that can run on the frozen official ACT and
Diffusion policies.

## Not Yet Implemented

`domain_randomization_only` still requires separate ACT/Diffusion checkpoints
trained with domain-randomized data. It should not be reported unless those
checkpoints exist.

`tent_style` does not have a direct continuous-policy equivalent here because
the old implementation adapts from classification entropy. Official ACT and
Diffusion output continuous actions, not primitive logits. A fair version needs
a different unsupervised objective.

## Outputs

- Expanded transition metrics:
  `results/continuous_sim2real_method_comparison_expanded/combined/method_transition_metrics.csv`
- Expanded success summary:
  `results/continuous_sim2real_method_comparison_expanded/combined/method_success_summary.csv`

## Aggregate Results

Mean over `appearance_shift`, `embodiment_shift`, and `joint_shift`.

| Model | Method | Transition MSE | Heldout exact success | Fresh rollout success |
| --- | --- | ---: | ---: | ---: |
| ACT | no_adaptation | 0.020411 | 0.2222 | 0.2500 |
| ACT | input_normalization | 0.018092 | 0.2500 | 0.2500 |
| ACT | probe_feature_alignment | 0.022627 | 0.3056 | 0.3333 |
| ACT | static_adapter | 0.020299 | 0.3056 | 0.2500 |
| ACT | few_shot_finetuning | 0.018627 | 0.2500 | 0.3056 |
| ACT | ours_proxy | 0.020211 | 0.3056 | 0.1667 |
| Diffusion | no_adaptation | 0.007888 | 0.1111 | 0.0833 |
| Diffusion | input_normalization | 0.007398 | 0.0833 | 0.1667 |
| Diffusion | probe_feature_alignment | 0.008046 | 0.1111 | 0.1389 |
| Diffusion | static_adapter | 0.007830 | 0.1389 | 0.0833 |
| Diffusion | few_shot_finetuning | 0.007400 | 0.0833 | 0.1389 |
| Diffusion | ours_proxy | 0.007549 | 0.1389 | 0.0833 |

## Interpretation

For ACT, `probe_feature_alignment` is now the best lightweight migration
method by fresh rollout success, and it matches the best heldout exact replay
success. `few_shot_finetuning` also improves fresh rollout over no adaptation.
The action-space `ours_proxy` remains too aggressive for closed-loop rollout.

For Diffusion, `input_normalization` remains the best fresh rollout method.
`probe_feature_alignment` and `few_shot_finetuning` help relative to no
adaptation, but the absolute success rate remains low.

The expanded comparison strengthens the earlier conclusion: action-space
calibration baselines can help, especially on ACT, but they do not replace the
planned continuous latent adapter.

## Reproduction

Example command for the added ACT methods on one shift:

```powershell
python scripts\run_continuous_sim2real_method_comparison.py --config configs\continuous_act_jointtarget_staged_frozen_best.yaml --profile appearance_shift --policy-device cuda --output-dir results\continuous_sim2real_method_comparison_extra\act\appearance_shift --calibration-data results\continuous_sim2real_method_comparison\act\appearance_shift\splits\calibration.npz --heldout-data results\continuous_sim2real_method_comparison\act\appearance_shift\splits\heldout.npz --fresh-episodes-per-task 4 --exact-num-per-task 4 --methods probe_feature_alignment,few_shot_finetuning --adapter-blend 0.25
```

Example command for Diffusion:

```powershell
python scripts\run_continuous_sim2real_method_comparison.py --config configs\continuous_diffusion_jointdelta_staged_frozen_best.yaml --profile appearance_shift --policy-device cuda --output-dir results\continuous_sim2real_method_comparison_extra\diffusion\appearance_shift --calibration-data results\continuous_sim2real_method_comparison\diffusion\appearance_shift\splits\calibration.npz --heldout-data results\continuous_sim2real_method_comparison\diffusion\appearance_shift\splits\heldout.npz --fresh-episodes-per-task 4 --exact-num-per-task 4 --methods probe_feature_alignment,few_shot_finetuning --adapter-blend 0.25
```

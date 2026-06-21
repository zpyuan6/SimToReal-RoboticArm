# Full Continuous Sim-to-Real Method Comparison - ACT and Diffusion

Date: 2026-05-29

## Methods Included

This run extends the continuous ACT/Diffusion comparison to the practical method
pool that can be evaluated on the frozen official policies:

- `no_adaptation`
- `input_normalization`
- `probe_feature_alignment`
- `static_adapter`
- `few_shot_finetuning`
- `tent_style`
- `ours_proxy`

`domain_randomization_only` is implemented as a script entry, but it requires a
separately trained DR-only checkpoint through
`--domain-randomization-policy-path`. No such verified continuous ACT/Diffusion
checkpoint exists in the current frozen baseline set, so it is not reported.

## Continuous Method Mapping

| Prior method | Continuous implementation |
| --- | --- |
| `no_adaptation` | frozen official policy |
| `input_normalization` | legacy per-image normalization before policy input |
| `probe_feature_alignment` | task-wise action moment alignment on calibration split |
| `static_adapter` | task-conditioned action-bias adapter |
| `few_shot_finetuning` | small residual MLP fitted on calibration action residuals |
| `tent_style` | raw/normalized test-time action ensemble as a continuous TTA proxy |
| `ours_proxy` | task-conditioned affine action adapter |

The true continuous PLICA latent adapter is still not implemented here. These
are action/input-space baselines that can run on the current official
ACT/Diffusion wrappers.

## Outputs

- Full transition metrics:
  `results/continuous_sim2real_method_comparison_full/combined/method_transition_metrics.csv`
- Full success summary:
  `results/continuous_sim2real_method_comparison_full/combined/method_success_summary.csv`

## Aggregate Results

Mean over `appearance_shift`, `embodiment_shift`, and `joint_shift`.

| Model | Method | Transition MSE | Heldout exact success | Fresh rollout success |
| --- | --- | ---: | ---: | ---: |
| ACT | no_adaptation | 0.020411 | 0.2222 | 0.2500 |
| ACT | input_normalization | 0.018092 | 0.2500 | 0.2500 |
| ACT | probe_feature_alignment | 0.022627 | 0.3056 | 0.3333 |
| ACT | static_adapter | 0.020299 | 0.3056 | 0.2500 |
| ACT | few_shot_finetuning | 0.018627 | 0.2500 | 0.3056 |
| ACT | tent_style | 0.018610 | 0.3611 | 0.2500 |
| ACT | ours_proxy | 0.020211 | 0.3056 | 0.1667 |
| Diffusion | no_adaptation | 0.007888 | 0.1111 | 0.0833 |
| Diffusion | input_normalization | 0.007398 | 0.0833 | 0.1667 |
| Diffusion | probe_feature_alignment | 0.008046 | 0.1111 | 0.1389 |
| Diffusion | static_adapter | 0.007830 | 0.1389 | 0.0833 |
| Diffusion | few_shot_finetuning | 0.007400 | 0.0833 | 0.1389 |
| Diffusion | tent_style | 0.006855 | 0.1944 | 0.0833 |
| Diffusion | ours_proxy | 0.007549 | 0.1389 | 0.0833 |

## Interpretation

For ACT, `probe_feature_alignment` is the best current lightweight migration
method by fresh rollout success. `tent_style` is strongest on heldout exact
replay, but does not improve fresh rollout over no adaptation.

For Diffusion, `input_normalization` is still the best method by fresh rollout.
`tent_style` gives the best heldout exact replay and lowest transition MSE, but
that does not transfer into closed-loop success.

Overall, the results reinforce the main point: transition/action error is useful
as a diagnostic, but closed-loop rollout must remain the primary deployment
metric. Action-space proxies help, but the missing piece is still the planned
continuous latent PLICA adapter.

## Reproduction

Example `tent_style` run for ACT:

```powershell
python scripts\run_continuous_sim2real_method_comparison.py --config configs\continuous_act_jointtarget_staged_frozen_best.yaml --profile appearance_shift --policy-device cuda --output-dir results\continuous_sim2real_method_comparison_extra2\act\appearance_shift --calibration-data results\continuous_sim2real_method_comparison\act\appearance_shift\splits\calibration.npz --heldout-data results\continuous_sim2real_method_comparison\act\appearance_shift\splits\heldout.npz --fresh-episodes-per-task 4 --exact-num-per-task 4 --methods tent_style --adapter-blend 0.25
```

Example `domain_randomization_only` invocation once a DR-only checkpoint exists:

```powershell
python scripts\run_continuous_sim2real_method_comparison.py --config configs\continuous_act_jointtarget_staged_frozen_best.yaml --profile appearance_shift --policy-device cuda --domain-randomization-policy-path outputs\train\lerobot\<dr_only_run>\checkpoints\<step>\pretrained_model --output-dir results\continuous_sim2real_method_comparison_dr\act\appearance_shift --calibration-data results\continuous_sim2real_method_comparison\act\appearance_shift\splits\calibration.npz --heldout-data results\continuous_sim2real_method_comparison\act\appearance_shift\splits\heldout.npz --fresh-episodes-per-task 4 --exact-num-per-task 4 --methods domain_randomization_only
```

# Continuous Sim-to-Real Method Comparison - ACT and Diffusion

Date: 2026-05-29

## Prior Comparison Settings Checked

The prior adopted pseudo-real comparison uses these baselines:

- `no_adaptation`
- `input_normalization`
- `static_adapter`
- `ours`

The adopted shift families are:

- `appearance_shift`
- `embodiment_shift`
- `joint_shift`

The original primitive-policy configs use large calibration/evaluation budgets,
for example `144-160` calibration episodes, `216-240` heldout episodes, and
`96` rollout episodes per task. This continuous comparison uses a lightweight
budget to keep ACT and Diffusion experiments tractable:

- calibration episodes: 6
- heldout episodes: 12
- exact replay episodes per task: 4
- fresh rollout episodes per task: 4

## Continuous Method Mapping

The current official ACT/Diffusion wrappers do not yet expose the final PLICA
latent adapter. The continuous comparison therefore uses these method mappings:

| Prior method | Continuous method used here |
| --- | --- |
| `no_adaptation` | frozen policy, no correction |
| `input_normalization` | legacy per-image normalization before policy input |
| `static_adapter` | task-conditioned action-bias adapter fitted on calibration split |
| `ours` | `ours_proxy`: task-conditioned affine action adapter fitted on calibration split |

`ours_proxy` is not the final latent PLICA method. It is a stronger continuous
action-space proxy used to compare method behavior on the new ACT/Diffusion
backbones.

## Outputs

- Combined transition metrics:
  `results/continuous_sim2real_method_comparison/combined/method_transition_metrics.csv`
- Combined success summary:
  `results/continuous_sim2real_method_comparison/combined/method_success_summary.csv`

## Aggregate Results

Mean over `appearance_shift`, `embodiment_shift`, and `joint_shift`.

| Model | Method | Transition MSE | Heldout exact success | Fresh rollout success |
| --- | --- | ---: | ---: | ---: |
| ACT | no_adaptation | 0.020411 | 0.2222 | 0.2500 |
| ACT | input_normalization | 0.018092 | 0.2500 | 0.2500 |
| ACT | static_adapter | 0.020299 | 0.3056 | 0.2500 |
| ACT | ours_proxy | 0.020211 | 0.3056 | 0.1667 |
| Diffusion | no_adaptation | 0.007888 | 0.1111 | 0.0833 |
| Diffusion | input_normalization | 0.007398 | 0.0833 | 0.1667 |
| Diffusion | static_adapter | 0.007830 | 0.1389 | 0.0833 |
| Diffusion | ours_proxy | 0.007549 | 0.1389 | 0.0833 |

## Interpretation

For ACT, `static_adapter` is the best conservative method in heldout exact
replay and does not hurt fresh rollout on average. `ours_proxy` improves
heldout exact replay but hurts fresh rollout, so action-space affine correction
is too aggressive for closed-loop ACT deployment.

For Diffusion, `input_normalization` gives the best fresh rollout average, while
`static_adapter` and `ours_proxy` improve heldout exact replay. Absolute
Diffusion success remains low under the old shift profiles.

Across both models, transition MSE improvements do not reliably predict
closed-loop success. The final sim-to-real method should therefore use the
documented continuous latent adapter rather than relying on action-space
correction alone.

## Reproduction

Example ACT run:

```powershell
python scripts\run_continuous_sim2real_method_comparison.py --config configs\continuous_act_jointtarget_staged_frozen_best.yaml --profile appearance_shift --policy-device cuda --output-dir results\continuous_sim2real_method_comparison\act\appearance_shift --calibration-episodes 6 --heldout-episodes 12 --fresh-episodes-per-task 4 --exact-num-per-task 4 --adapter-blend 0.25 --force-regenerate
```

Example Diffusion run:

```powershell
python scripts\run_continuous_sim2real_method_comparison.py --config configs\continuous_diffusion_jointdelta_staged_frozen_best.yaml --profile appearance_shift --policy-device cuda --output-dir results\continuous_sim2real_method_comparison\diffusion\appearance_shift --calibration-episodes 6 --heldout-episodes 12 --fresh-episodes-per-task 4 --exact-num-per-task 4 --adapter-blend 0.25 --force-regenerate
```

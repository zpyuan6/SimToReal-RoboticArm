# Continuous Pseudo-Real Sim-to-Real Paper Results V3

Date: 2026-06-06

This report updates the previous paper-grade result after redesigning the proposed method.

## Proposed Method

New method: `ours_profile_adaptive_selector`.

The method is a validation-frozen profile/backbone selector over calibration-fitted action adapters. It first fits the same candidate adapters on the calibration split, then selects one adapter family for each `(backbone, pseudo-real profile)` according to validation results. The test split is not used for method selection.

Frozen selector:

- ACT + `appearance_shift`: `few_shot_finetuning`
- ACT + `embodiment_shift`: `static_adapter`
- ACT + `joint_shift`: `few_shot_finetuning`
- Diffusion + `appearance_shift`: `static_adapter`
- Diffusion + `embodiment_shift`: `probe_feature_alignment`
- Diffusion + `joint_shift`: `probe_feature_alignment`

This design avoids the previous taskwise selector's Diffusion issue, where choosing different adapters across L1/L2/L3 changed stochastic rollout RNG consumption and could degrade later tasks.

## Completed Evaluation

- Source split root: `results/paper_continuous_sim2real_protocol`
- Proposed method root: `results/paper_continuous_sim2real_protocol_v3`
- Combined comparison root: `results/paper_continuous_sim2real_protocol_v3_combined`
- Completed eval units: 36/36
- stderr logs: empty

## Main Test Fresh-Rollout Result

Primary metric: test fresh-rollout overall success, averaged over 9 repeated units `(3 profiles x 3 seeds)`.

| Backbone | Method | Success mean | 95% CI | Rank |
|---|---:|---:|---:|---:|
| ACT | ours_profile_adaptive_selector | 0.2685 | [0.1864, 0.3506] | 1 |
| ACT | few_shot_finetuning | 0.2481 | [0.1563, 0.3400] | 2 |
| ACT | no_adaptation | 0.2481 | [0.1616, 0.3347] | 2 |
| ACT | static_adapter | 0.2463 | [0.1609, 0.3317] | 4 |
| Diffusion | ours_profile_adaptive_selector | 0.1296 | [0.0780, 0.1812] | 1 |
| Diffusion | static_adapter | 0.1093 | [0.0491, 0.1694] | 2 |
| Diffusion | probe_feature_alignment | 0.0963 | [0.0784, 0.1142] | 3 |
| Diffusion | ours_proxy | 0.0926 | [0.0560, 0.1292] | 4 |

## Task-Level Test Fresh-Rollout Result

| Backbone | L1 | L2 | L3 | Overall |
|---|---:|---:|---:|---:|
| ACT | 0.2611 | 0.3111 | 0.2333 | 0.2685 |
| Diffusion | 0.0333 | 0.0944 | 0.2611 | 0.1296 |

## Pairwise Comparisons

For ACT fresh rollout, `ours_profile_adaptive_selector` has the best mean. It is clearly better than input normalization and probe/static by bootstrap CI, but the improvement over no adaptation and few-shot is not statistically decisive with 9 paired units.

For Diffusion fresh rollout, `ours_profile_adaptive_selector` has the best mean. It improves over static_adapter by +0.0204, with bootstrap CI [0.0037, 0.0389]. The paired sign-flip p-value is 0.0967, so the mean improvement is positive but still needs more seeds for a strong p < 0.05 claim.

## Interpretation

The redesign works better because the adaptation target is matched to the pseudo-real shift type:

- ACT benefits from few-shot residuals under appearance/joint shifts, but from a static bias under embodiment shift.
- Diffusion benefits from static calibration under appearance shift, but from probe/moment alignment under embodiment and joint shifts.
- A single global adapter is too brittle because the shift mechanism changes the failure mode.

For the manuscript, the defensible claim is:

> A validation-frozen profile-aware adaptation selector improves mean pseudo-real fresh-rollout success over all single adaptation baselines on both ACT and Diffusion.

The stronger claim of statistically significant superiority over every baseline still needs more repeated seeds, especially for ACT against no adaptation/few-shot and Diffusion against static_adapter.

# Continuous Pseudo-Real Sim-to-Real Paper Results

Date: 2026-06-05

Protocol:

- Backbones: ACT and Diffusion frozen baselines.
- Profiles: `appearance_shift`, `embodiment_shift`, `joint_shift`.
- Seeds: `20260610`, `20260611`, `20260612`.
- Splits: disjoint calibration, validation, and test per `(backbone, profile, seed)`.
- Methods: `no_adaptation`, `input_normalization`, `probe_feature_alignment`, `static_adapter`, `few_shot_finetuning`, `ours_proxy`, `ours_multimodel_adaptive`.
- Completed eval units: 36/36.
- Aggregated files: `results/paper_continuous_sim2real_protocol/combined/`.

## Main Finding

The formal paper-grade run does not support claiming that `ours_multimodel_adaptive` is the best method.

- ACT test fresh-rollout: `ours_multimodel_adaptive` is close to no adaptation and few-shot residual fitting, but it is not ranked first.
- Diffusion test fresh-rollout: `static_adapter` and `probe_feature_alignment` outperform `ours_multimodel_adaptive`.
- Therefore, the current proposed calibration method should not be used as the headline IEEE Transactions contribution without further method redesign or a clearer claim.

## Test Overall Success

Primary metric: test fresh-rollout overall success across 9 repeated units `(3 profiles x 3 seeds)`.

| Backbone | Method | Fresh success mean | 95% CI | Rank |
|---|---:|---:|---:|---:|
| ACT | few_shot_finetuning | 0.2481 | [0.1563, 0.3400] | 1 |
| ACT | no_adaptation | 0.2481 | [0.1616, 0.3347] | 1 |
| ACT | static_adapter | 0.2463 | [0.1609, 0.3317] | 3 |
| ACT | ours_multimodel_adaptive | 0.2444 | [0.1578, 0.3310] | 4 |
| ACT | probe_feature_alignment | 0.2389 | [0.1585, 0.3193] | 5 |
| ACT | ours_proxy | 0.2370 | [0.1632, 0.3108] | 6 |
| ACT | input_normalization | 0.1833 | [0.1121, 0.2545] | 7 |
| Diffusion | static_adapter | 0.1093 | [0.0491, 0.1694] | 1 |
| Diffusion | probe_feature_alignment | 0.0963 | [0.0784, 0.1142] | 2 |
| Diffusion | ours_proxy | 0.0926 | [0.0560, 0.1292] | 3 |
| Diffusion | few_shot_finetuning | 0.0870 | [0.0460, 0.1281] | 4 |
| Diffusion | input_normalization | 0.0796 | [0.0405, 0.1188] | 5 |
| Diffusion | ours_multimodel_adaptive | 0.0722 | [0.0456, 0.0989] | 6 |
| Diffusion | no_adaptation | 0.0611 | [0.0344, 0.0878] | 7 |

## Proposed Method Task-Level Test Results

`ours_multimodel_adaptive`, test split.

| Backbone | Eval type | L1 | L2 | L3 | Overall |
|---|---|---:|---:|---:|---:|
| ACT | heldout exact replay | 0.4833 | 0.2833 | 0.0889 | 0.2852 |
| ACT | fresh rollout | 0.2056 | 0.2944 | 0.2333 | 0.2444 |
| Diffusion | heldout exact replay | 0.0500 | 0.0333 | 0.2000 | 0.0944 |
| Diffusion | fresh rollout | 0.0000 | 0.0667 | 0.1500 | 0.0722 |

## Pairwise Test Comparisons

Paired comparisons use `(profile, seed)` as the repeated unit.

- ACT fresh rollout: `ours_multimodel_adaptive` beats only `input_normalization` significantly. It is statistically inconclusive against no adaptation, few-shot, static adapter, probe alignment, and ours_proxy.
- ACT heldout exact replay: `ours_multimodel_adaptive` is lower than few-shot; it is not a robust improvement over standard baselines.
- Diffusion fresh rollout: `ours_multimodel_adaptive` is lower than `probe_feature_alignment` and `ours_proxy`; it is not the best method.
- Diffusion heldout exact replay: `probe_feature_alignment` is significantly better than `ours_multimodel_adaptive`.

## Interpretation

The current pseudo-real experiment shows that action-space residual calibration is fragile. On ACT, the base policy is already near the best achievable level under these shifts, so residual adaptation provides little gain and can slightly hurt exact replay. On Diffusion, the baseline policy is weak and task-dependent; conservative residual gating does not repair the main failures, especially L1 fresh rollout.

For a publication-grade contribution, the next method iteration should not be framed as the current `ours_multimodel_adaptive` adapter. Better options are:

- Use `probe_feature_alignment` or `static_adapter` as stronger adaptation baselines.
- Redesign the proposed method around task-aware geometry/state calibration rather than action residual fitting alone.
- Add real or pseudo-real calibration observations that directly identify L1 visual alignment failures and Diffusion stochastic rollout drift.

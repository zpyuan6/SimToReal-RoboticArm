# Seed Extension, Ablation, and Robustness Results

Date: 2026-06-08

This report covers the first three follow-up items after the V3 proposed method:

1. Add repeated seeds.
2. Run method ablations.
3. Report robustness by pseudo-real shift profile.

## Seed Extension

Added seeds:

- `20260613`
- `20260614`
- `20260615`
- `20260616`
- `20260617`

Combined with the original seeds `20260610-20260612`, the key-method test comparison now uses 8 seeds per profile, or 24 repeated units per backbone.

Evaluated methods:

- `no_adaptation`
- `few_shot_finetuning`
- `static_adapter`
- `probe_feature_alignment`
- `ours_profile_adaptive_selector`

Completed extension eval units: 30/30.

All extension stderr logs are empty.

## 8-Seed Main Result

Primary metric: test fresh-rollout overall success.

| Backbone | Method | Mean | 95% CI | Rank |
|---|---:|---:|---:|---:|
| ACT | ours_profile_adaptive_selector | 0.2451 | [0.2027, 0.2876] | 1 |
| ACT | probe_feature_alignment | 0.2444 | [0.1995, 0.2894] | 2 |
| ACT | static_adapter | 0.2437 | [0.1954, 0.2921] | 3 |
| ACT | few_shot_finetuning | 0.2417 | [0.1976, 0.2857] | 4 |
| ACT | no_adaptation | 0.2375 | [0.1905, 0.2845] | 5 |
| Diffusion | ours_profile_adaptive_selector | 0.1104 | [0.0859, 0.1350] | 1 |
| Diffusion | probe_feature_alignment | 0.1000 | [0.0839, 0.1161] | 2 |
| Diffusion | few_shot_finetuning | 0.0924 | [0.0717, 0.1130] | 3 |
| Diffusion | static_adapter | 0.0910 | [0.0637, 0.1183] | 4 |
| Diffusion | no_adaptation | 0.0736 | [0.0574, 0.0898] | 5 |

## Pairwise Result

`ours_profile_adaptive_selector` remains mean-rank 1 for both ACT and Diffusion, but the statistical strength differs by backbone.

ACT:

- vs `no_adaptation`: +0.0076, bootstrap CI [-0.0042, 0.0194], sign-flip p = 0.2555.
- vs `few_shot_finetuning`: +0.0035, bootstrap CI [-0.0069, 0.0160], sign-flip p = 0.5675.
- vs `probe_feature_alignment`: +0.0007, bootstrap CI [-0.0174, 0.0174], sign-flip p = 0.9093.
- Conclusion: ACT is mean-best but effectively tied with strong baselines under 8 seeds.

Diffusion:

- vs `no_adaptation`: +0.0368, bootstrap CI [0.0146, 0.0597], sign-flip p = 0.0047.
- vs `few_shot_finetuning`: +0.0181, bootstrap CI [0.0021, 0.0333], sign-flip p = 0.0476.
- vs `static_adapter`: +0.0194, bootstrap CI [0.0090, 0.0312], sign-flip p = 0.0008.
- vs `probe_feature_alignment`: +0.0104, bootstrap CI [-0.0069, 0.0292], sign-flip p = 0.2813.
- Conclusion: Diffusion improvement is statistically supported against no-adaptation, few-shot, and static adapter, but not against probe alignment.

## Ablation

The 3-seed ablation shows the profile-aware design is better than the earlier global and taskwise designs.

| Backbone | Variant | Test fresh overall |
|---|---:|---:|
| ACT | no_adaptation | 0.2481 |
| ACT | global residual, `ours_multimodel_adaptive` | 0.2444 |
| ACT | taskwise selector, `ours_validation_taskwise_selector` | 0.2630 |
| ACT | profile-aware selector, `ours_profile_adaptive_selector` | 0.2685 |
| Diffusion | no_adaptation | 0.0611 |
| Diffusion | global residual, `ours_multimodel_adaptive` | 0.0722 |
| Diffusion | taskwise selector, `ours_validation_taskwise_selector` | 0.1074 |
| Diffusion | profile-aware selector, `ours_profile_adaptive_selector` | 0.1296 |

Interpretation:

- Global residual adaptation is too brittle.
- Taskwise selection improves over a single residual, but can disrupt Diffusion stochastic rollout behavior because different tasks consume different rollout trajectories.
- Profile-aware selection is the most stable of the tested proposed variants.

## Robustness Breakdown

8-seed test fresh-rollout overall success by profile.

ACT:

| Profile | Best method | Best mean | Proposed mean |
|---|---:|---:|---:|
| appearance_shift | static_adapter | 0.4000 | 0.3833 |
| embodiment_shift | ours_profile_adaptive_selector | 0.1937 | 0.1937 |
| joint_shift | few_shot_finetuning / proposed tie | 0.1583 | 0.1583 |

Diffusion:

| Profile | Best method | Best mean | Proposed mean |
|---|---:|---:|---:|
| appearance_shift | ours_profile_adaptive_selector | 0.1687 | 0.1687 |
| embodiment_shift | probe_feature_alignment | 0.0937 | 0.0917 |
| joint_shift | ours_profile_adaptive_selector / probe tie | 0.0708 | 0.0708 |

## Current Conclusion

The proposed V3 method is still the best mean method across both backbones after increasing to 8 seeds, but the claim should be phrased carefully:

> A validation-frozen profile-aware adaptation selector improves mean fresh-rollout success over strong single-method baselines on ACT and Diffusion, with statistically supported gains on Diffusion and a mean-best but statistically tied result on ACT.

The next necessary improvement is to refine the ACT profile selector, especially `appearance_shift`, where the 8-seed result shows the validation-frozen choice underperformed `static_adapter`.

## Artifacts

- 8-seed merged success summary: `results/paper_continuous_sim2real_protocol_8seed_keymethods/combined/paper_success_summary.csv`
- 8-seed success stats: `results/paper_continuous_sim2real_protocol_8seed_keymethods/combined/paper_success_stats.csv`
- 8-seed pairwise comparisons: `results/paper_continuous_sim2real_protocol_8seed_keymethods/combined/paper_pairwise_success_comparisons.csv`
- Profile breakdown: `results/paper_continuous_sim2real_protocol_8seed_keymethods/combined/paper_success_stats_by_profile.csv`
- Ablation summary: `results/paper_continuous_sim2real_protocol_8seed_keymethods/combined/paper_ablation_3seed_summary.csv`

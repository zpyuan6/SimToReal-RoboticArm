# Continuous Sim-to-Real Bridge Results - ACT and Diffusion

Date: 2026-05-28

## Scope

This run evaluates the current frozen continuous ACT and Diffusion baselines
under pseudo-real sim-to-real shifts using the continuous bridge protocol.

The bridge follows the documented calibration discipline:

- generate explicit pseudo-real `calibration.npz`
- generate separate pseudo-real `heldout.npz`
- fit a small calibration adapter only on calibration data
- evaluate heldout transition error separately from closed-loop success
- evaluate both heldout exact replay and fresh rollout

This is not the final latent PLICA adapter. It is a conservative
task-conditioned continuous action-bias adapter used to validate the
continuous calibration/evaluation harness.

## Inputs

- ACT config: `configs/continuous_act_jointtarget_staged_frozen_best.yaml`
- ACT policy: `outputs/train/lerobot/act_jointtarget_staged_quicktrial/checkpoints/020000/pretrained_model`
- Diffusion config: `configs/continuous_diffusion_jointdelta_staged_frozen_best.yaml`
- Diffusion policy: `outputs/train/lerobot/diffusion_jointdelta_staged_nodrop_compact/checkpoints/012500/pretrained_model`

Profiles:

- `neutral`
- `visual`
- `camera`
- `actuation`
- `combined_mild`
- `combined_hard`

Per profile:

- calibration episodes: 6
- heldout episodes: 12
- exact replay episodes per task: 4
- fresh rollout episodes per task: 4
- adapter: `task_bias`
- adapter blend: 0.25

## Outputs

- ACT transition metrics:
  `results/continuous_sim2real_bridge_suite/act_full/suite_transition_metrics.csv`
- ACT success summary:
  `results/continuous_sim2real_bridge_suite/act_full/suite_success_summary.csv`
- Diffusion transition metrics:
  `results/continuous_sim2real_bridge_suite/diffusion_full/suite_transition_metrics.csv`
- Diffusion success summary:
  `results/continuous_sim2real_bridge_suite/diffusion_full/suite_success_summary.csv`
- Combined transition metrics:
  `results/continuous_sim2real_bridge_suite/combined_act_diffusion/suite_transition_metrics.csv`
- Combined success summary:
  `results/continuous_sim2real_bridge_suite/combined_act_diffusion/suite_success_summary.csv`

## Aggregate Results

Mean over all six profiles.

| Model | Baseline | Heldout transition MSE | Heldout exact success | Fresh rollout success |
| --- | --- | ---: | ---: | ---: |
| ACT | no_adaptation | 0.015625 | 0.4444 | 0.3472 |
| ACT | task_bias | 0.015408 | 0.4028 | 0.3750 |
| Diffusion | no_adaptation | 0.006726 | 0.1250 | 0.2222 |
| Diffusion | task_bias | 0.006654 | 0.1806 | 0.2778 |

## Interpretation

The bridge harness works end to end for both frozen ACT and frozen Diffusion.

For ACT, the task-bias adapter slightly improves transition error and fresh
rollout success, but it hurts average heldout exact replay. The gain is most
visible under `combined_mild`; it does not help `combined_hard`.

For Diffusion, the task-bias adapter gives small improvements in transition
error, heldout exact replay, and fresh rollout. The absolute success rate
remains below ACT, so Diffusion is still a weaker deployment candidate.

The action-bias adapter is therefore useful as a diagnostic calibration
baseline, but it is not strong enough to be the final sim-to-real method.
The next research step should be the documented continuous latent adapter:

- ACT: adapter on transformer/chunk latent
- Diffusion: adapter on condition latent

## Reproduction

ACT:

```powershell
python scripts\run_continuous_sim2real_bridge_suite.py --configs act=configs\continuous_act_jointtarget_staged_frozen_best.yaml --profiles neutral,visual,camera,actuation,combined_mild,combined_hard --policy-device cuda --output-root results\continuous_sim2real_bridge_suite\act_full --calibration-episodes 6 --heldout-episodes 12 --fresh-episodes-per-task 4 --exact-num-per-task 4 --adapter-blend 0.25 --force-regenerate
```

Diffusion:

```powershell
python scripts\run_continuous_sim2real_bridge_suite.py --configs diffusion=configs\continuous_diffusion_jointdelta_staged_frozen_best.yaml --profiles neutral,visual,camera,actuation,combined_mild,combined_hard --policy-device cuda --output-root results\continuous_sim2real_bridge_suite\diffusion_full --calibration-episodes 6 --heldout-episodes 12 --fresh-episodes-per-task 4 --exact-num-per-task 4 --adapter-blend 0.25 --force-regenerate
```

Aggregate existing runs only:

```powershell
python scripts\run_continuous_sim2real_bridge_suite.py --configs act=configs\continuous_act_jointtarget_staged_frozen_best.yaml --profiles neutral,visual,camera,actuation,combined_mild,combined_hard --output-root results\continuous_sim2real_bridge_suite\act_full --aggregate-only
```

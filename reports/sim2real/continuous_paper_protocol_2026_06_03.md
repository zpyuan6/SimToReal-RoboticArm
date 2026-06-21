# Continuous Pseudo-Real Sim-to-Real Paper Protocol

Date: 2026-06-03

This protocol upgrades the previous pilot comparisons into a paper-grade experimental pipeline for IEEE Transactions-style reporting. The pilot results remain useful for method development, but paper claims should be made only from the held-out test split produced by this protocol.

## Frozen Scope

Default evaluated backbones:

- ACT: `configs/continuous_act_jointtarget_staged_frozen_best.yaml`
- Diffusion: `configs/continuous_diffusion_jointdelta_staged_frozen_best.yaml`

Optional backbones can be added through `--configs label=config.yaml` if they support the same continuous control evaluation interface.

Default evaluated methods:

- `no_adaptation`
- `input_normalization`
- `probe_feature_alignment`
- `static_adapter`
- `few_shot_finetuning`
- `ours_proxy`
- `ours_multimodel_adaptive`

The proposed method is frozen as `ours_multimodel_adaptive`: conservative task-gated residual calibration with L1/L3 identity and L2 residual blend 0.25.

## Data Splits

Each `(backbone, pseudo-real profile, seed)` run creates three disjoint splits:

- `calibration`: used only to fit adapters or few-shot residuals.
- `validation`: used for method/hyperparameter selection and ablation discussion.
- `test`: used for final paper claims.

The default paper budget is 3 profiles x 3 seeds x 2 backbones, with 30 calibration, 60 validation, and 60 test episodes per run. This gives 9 repeated pseudo-real units per backbone before task-level breakdown.

## Metrics

Primary metric:

- Test fresh-rollout overall success.

Secondary metrics:

- Test heldout exact-replay overall success.
- Test task-level success for L1/L2/L3.
- Validation success for development analysis only.
- Steps-to-completion.
- Transition action MSE/MAE as diagnostic metrics, not final task performance.

## Statistical Reporting

The summarizer treats each `(profile, seed)` as a paired repeated unit. It reports:

- Mean, standard deviation, SEM, and normal-approximation 95% CI.
- Paired bootstrap confidence intervals for proposed-minus-baseline success differences.
- Paired sign-flip p-values for method comparisons.

For the manuscript, report test split results first. Validation split results should be marked as model-selection or ablation evidence.

## Smoke Command

```powershell
python scripts\run_paper_continuous_sim2real_protocol.py --configs act=configs\continuous_act_jointtarget_staged_frozen_best.yaml --profiles appearance_shift --seeds 20260610 --methods no_adaptation,ours_multimodel_adaptive --output-root results\paper_continuous_sim2real_smoke --policy-device cuda --calibration-episodes 2 --validation-episodes 2 --test-episodes 2 --exact-num-per-task 1 --fresh-episodes-per-task 1 --max-attempts-per-episode 20 --eval-splits validation --force-regenerate
```

```powershell
python scripts\summarize_paper_continuous_sim2real.py --input-root results\paper_continuous_sim2real_smoke --primary-method ours_multimodel_adaptive --bootstrap-samples 200
```

## Full Paper Command

```powershell
python scripts\run_paper_continuous_sim2real_protocol.py --configs act=configs\continuous_act_jointtarget_staged_frozen_best.yaml diffusion=configs\continuous_diffusion_jointdelta_staged_frozen_best.yaml --profiles appearance_shift,embodiment_shift,joint_shift --seeds 20260610,20260611,20260612 --methods no_adaptation,input_normalization,probe_feature_alignment,static_adapter,few_shot_finetuning,ours_proxy,ours_multimodel_adaptive --output-root results\paper_continuous_sim2real_protocol --policy-device cuda --calibration-episodes 30 --validation-episodes 60 --test-episodes 60 --exact-num-per-task 20 --fresh-episodes-per-task 20 --max-attempts-per-episode 100 --eval-splits validation,test --force-regenerate
```

```powershell
python scripts\summarize_paper_continuous_sim2real.py --input-root results\paper_continuous_sim2real_protocol --primary-method ours_multimodel_adaptive --bootstrap-samples 10000
```

## Resume Command

Use this after an interrupted full run. It reuses existing split files and skips eval directories that already contain both success and transition CSVs.

```powershell
python scripts\run_paper_continuous_sim2real_protocol.py --configs act=configs\continuous_act_jointtarget_staged_frozen_best.yaml diffusion=configs\continuous_diffusion_jointdelta_staged_frozen_best.yaml --profiles appearance_shift,embodiment_shift,joint_shift --seeds 20260610,20260611,20260612 --methods no_adaptation,input_normalization,probe_feature_alignment,static_adapter,few_shot_finetuning,ours_proxy,ours_multimodel_adaptive --output-root results\paper_continuous_sim2real_protocol --policy-device cuda --calibration-episodes 30 --validation-episodes 60 --test-episodes 60 --exact-num-per-task 20 --fresh-episodes-per-task 20 --max-attempts-per-episode 100 --eval-splits validation,test --skip-existing-eval
```

# Continuous Real Transfer Frozen v1

Date: 2026-06-17

This freezes the current ACT/Diffusion continuous L1-L3 setup before real-world data collection. The purpose is to prevent method drift while moving from pseudo-real validation to real calibration and real rollout.

## Frozen Models

### ACT

- Config: `configs/continuous_act_jointtarget_staged_frozen_best.yaml`
- Policy: `outputs/train/lerobot/act_jointtarget_staged_quicktrial/checkpoints/020000/pretrained_model`
- Action representation: `joint_target`
- Proposed adapter: ACT action-head latent adapter
- Post adapter: `task_bias`
- Key parameters: scale `0.05`, reg `0.5`, action loss `0.25`, task blends `[0.0, 0.0, 0.25]`

### Diffusion

- Config: `configs/continuous_diffusion_jointdelta_staged_frozen_best.yaml`
- Policy: `outputs/train/lerobot/diffusion_jointdelta_staged_nodrop_compact/checkpoints/012500/pretrained_model`
- Action representation: `joint_delta`
- Proposed adapter: Diffusion action-trajectory representation adapter
- Post adapter: `none`
- Key parameters: scale `0.2`, first action weight `2.0`, plan loss `1.0`, smooth loss `0.1`, reg `0.02`

## Frozen Pseudo-Real Results

| Model/Method | Split | Exact Replay | Fresh Rollout |
|---|---:|---:|---:|
| ACT proposed | validation | 32.0% | 25.9% |
| ACT proposed | test | 30.7% | 25.7% |
| Diffusion proposed | validation | 13.33% | 12.78% |
| Diffusion proposed | test | 15.74% | 12.59% |

Diffusion proposed result files:

- `results/paper_diffusion_action_repr_trajectory_nopost_validation/combined/paper_success_summary.csv`
- `results/paper_diffusion_action_repr_trajectory_nopost_test/combined/paper_success_summary.csv`

## Reproduction Commands

Validation:

```powershell
.venv\Scripts\python.exe scripts\run_paper_continuous_sim2real_protocol.py --configs diffusion=configs\continuous_diffusion_jointdelta_staged_frozen_best.yaml --profiles appearance_shift,embodiment_shift,joint_shift --seeds 20260610,20260611,20260612 --methods ours_action_representation_adapter --eval-splits validation --split-source-root results\paper_continuous_sim2real_protocol --output-root results\paper_diffusion_action_repr_trajectory_nopost_validation --action-repr-diffusion-backend trajectory --action-repr-post-adapter none --trajectory-adapter-epochs 80 --trajectory-adapter-max-pairs 2048 --trajectory-adapter-scale 0.2 --trajectory-adapter-first-action-weight 2.0 --trajectory-adapter-plan-loss-weight 1.0 --trajectory-adapter-smooth-loss-weight 0.1 --trajectory-adapter-reg-weight 0.02 --exact-num-per-task 20 --fresh-episodes-per-task 20
```

Test:

```powershell
.venv\Scripts\python.exe scripts\run_paper_continuous_sim2real_protocol.py --configs diffusion=configs\continuous_diffusion_jointdelta_staged_frozen_best.yaml --profiles appearance_shift,embodiment_shift,joint_shift --seeds 20260610,20260611,20260612 --methods ours_action_representation_adapter --eval-splits test --split-source-root results\paper_continuous_sim2real_protocol --output-root results\paper_diffusion_action_repr_trajectory_nopost_test --action-repr-diffusion-backend trajectory --action-repr-post-adapter none --trajectory-adapter-epochs 80 --trajectory-adapter-max-pairs 2048 --trajectory-adapter-scale 0.2 --trajectory-adapter-first-action-weight 2.0 --trajectory-adapter-plan-loss-weight 1.0 --trajectory-adapter-smooth-loss-weight 0.1 --trajectory-adapter-reg-weight 0.02 --exact-num-per-task 20 --fresh-episodes-per-task 20
```

## Real Collection Handoff

Use:

- Plan: `configs/continuous_real_collection_plan_v1.yaml`
- Collector: `scripts/collect_continuous_real_calibration.py`
- Merger: `scripts/merge_continuous_real_sessions.py`
- Operator protocol: `docs/continuous_real_collection_protocol_v1.md`

The collector saves both action formats from the same physical session:

- `joint_target` for ACT
- `joint_delta` for Diffusion

These are the expected merged files:

- `data/real_continuous_v1/merged/calibration_joint_target.npz`
- `data/real_continuous_v1/merged/calibration_joint_delta.npz`
- `data/real_continuous_v1/merged/heldout_joint_target.npz`
- `data/real_continuous_v1/merged/heldout_joint_delta.npz`

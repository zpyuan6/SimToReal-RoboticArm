# Frozen Baseline: ACT Joint Target Staged

Frozen on: 2026-05-21

This is the current best unified ACT baseline for `level1_verify`, `level2_approach`, and `level3_pick_place`.

## Fixed Artifacts

- Config: `configs/continuous_act_jointtarget_staged_frozen_best.yaml`
- Source config: `configs/continuous_act_jointtarget_staged_quicktrial.yaml`
- Train data: `data/continuous_jointtarget_staged_quicktrial/train.npz`
- Val data: `data/continuous_jointtarget_staged_quicktrial/val.npz`
- Test data: `data/continuous_jointtarget_staged_quicktrial/test.npz`
- LeRobot train data: `data/lerobot/roarm_continuous_train_jointtarget_staged_quicktrial`
- Policy checkpoint: `outputs/train/lerobot/act_jointtarget_staged_quicktrial/checkpoints/020000/pretrained_model`

## Training Strategy

- Backbone: official LeRobot ACT
- Control mode: `joint_target`
- Image input: forearm camera, `224x224x3`
- State input: 16-D vector, `qpos6 + qvel6 + task_one_hot3 + progress1`
- Task conditioning: one-hot task id in state
- History length: 2
- Action horizon: 2
- ACT chunk size: 12
- ACT action steps: 2
- Batch size: 8
- Max steps: 30000
- Checkpoint selection: validation loss during early-stop loop, then fresh rollout ranking
- Staged resets: L2 starts from direct L1 success pose, L3 starts from direct L2 success pose

## Metrics

Fresh rollout, selected checkpoint `020000`:

| Task | Success |
|---|---:|
| level1_verify | 0.4167 |
| level2_approach | 0.6667 |
| level3_pick_place | 0.5833 |
| overall | 0.5556 |

Train exact replay:

| Task | Success |
|---|---:|
| level1_verify | 0.9267 |
| level2_approach | 0.4500 |
| level3_pick_place | 0.2733 |
| overall | 0.5500 |

Val exact replay:

| Task | Success |
|---|---:|
| level1_verify | 0.9333 |
| level2_approach | 0.4000 |
| level3_pick_place | 0.2667 |
| overall | 0.5333 |

## Decision

This version is frozen as the current best unified baseline. The L2 subgoal, L2 geometry-state, and L2 joint-delta exploration artifacts were cleaned because they did not improve the frozen unified fresh rollout baseline.

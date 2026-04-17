# Real Data Collection Plan

This document records the formal real-world collection plan to use **after**
the validator baseline was stabilized at git tag:

- `stable-validation-v1`

Use this plan for the new formal dataset. Earlier data collected before this
stable baseline should be treated as:

- legacy data
- debugging data
- pre-calibration data

and should not be mixed into the new main dataset by default.

## Scope

This plan assumes that the only environmental factor that can be controlled
reliably is:

- target position on the tabletop

It does **not** assume stable control over:

- lighting
- background
- other scene appearance factors

Those factors may vary naturally and can be recorded in notes, but they are not
used as deliberate collection variables in this version.

## Stable Baseline

This plan is based on the validated stable sim-to-real alignment at:

- `stable-validation-v1`

At this baseline, the following have been manually checked and accepted:

- L1 validation
- L2 validation
- L3 step-by-step validation
- real / sim initial pose alignment
- gripper open / close direction
- tabletop alignment in the simulator

## Collection Principles

- Re-start formal collection from L1 using the stable baseline.
- Keep target placement fixed within each batch.
- Move the target only once between batches.
- Preserve raw-resolution images.
- Keep `.npz` as the main structured dataset format.
- Keep `frames/`, `meta.json`, and `preview.mp4` for audit and debugging.

## Primitive Reference

### L1 / task-id 0

- `2` = `obs_center`
- `0` = `obs_left`
- `1` = `obs_right`
- `3` = `verify_target`

Recommended L1 sequence:

- `2,0,1,2,3`

### L2 / task-id 1

- `4` = `prealign_grasp`
- `5` = `approach_coarse`
- `6` = `approach_fine`
- `7` = `retreat`

Recommended L2 sequence:

- `2,3,4,5,6,7`

### L3 / task-id 2

- `8` = `reobserve`
- `9` = `pregrasp_servo`
- `10` = `grasp_execute`
- `11` = `lift_object`
- `12` = `transport_to_dropzone`
- `13` = `place_object`

Recommended L3 sequence:

- `8,9,10,11,12,13`

## Target Placements

Use one center reference placement and only move the target between these five
placements:

- `P1`: center
- `P2`: left
- `P3`: right
- `P4`: front
- `P5`: back

Recommended displacement:

- left / right: `3-5 cm`
- front / back: `3-5 cm`

Do not intentionally vary object orientation unless it shifts naturally while
being moved.

## Dataset Size Target

This version aims to support both:

- fine-tuning / adaptation
- offline testing

Recommended first formal dataset size:

- L1: 5 placements x 10 repeats
- L2: 5 placements x 6 repeats
- L3: 3 placements x 3 repeats

Approximate transition count:

- L1: `5 x 10 x 5 = 250`
- L2: `5 x 6 x 6 = 180`
- L3: `3 x 3 x 6 = 54`

Total:

- about `484` transitions

This gives a practical first full dataset while keeping L3 conservative.

## Collection Table

| Stage | Batch Name | Task ID | How To Place The Object | Need To Move The Target? | How To Move The Target | Command To Run |
|---|---|---:|---|---|---|---|
| 0 | `smoke_verify` | 0 | Place the target at the center reference placement and confirm it is clearly visible. | No | Do not move the target. | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/smoke_verify.npz --primitives 2,3 --repeats 1 --task-id 0` |
| 1 | `v1_verify_p1_center` | 0 | Place the target at the center reference placement. | No | Keep it fixed for all 10 repeats. | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v1_verify_p1_center.npz --primitives 2,0,1,2,3 --repeats 10 --task-id 0` |
| 1 | `v1_verify_p2_left` | 0 | Place the target slightly left of center while keeping it fully visible. | Yes | Move it once, about `3-5 cm` left from center, before the run. | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v1_verify_p2_left.npz --primitives 2,0,1,2,3 --repeats 10 --task-id 0` |
| 1 | `v1_verify_p3_right` | 0 | Place the target slightly right of center while keeping it fully visible. | Yes | Move it once, about `3-5 cm` right from center, before the run. | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v1_verify_p3_right.npz --primitives 2,0,1,2,3 --repeats 10 --task-id 0` |
| 1 | `v1_verify_p4_front` | 0 | Place the target slightly forward from center while keeping it visible. | Yes | Move it once, about `3-5 cm` toward the front direction, before the run. | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v1_verify_p4_front.npz --primitives 2,0,1,2,3 --repeats 10 --task-id 0` |
| 1 | `v1_verify_p5_back` | 0 | Place the target slightly backward from center while keeping it visible. | Yes | Move it once, about `3-5 cm` toward the back direction, before the run. | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v1_verify_p5_back.npz --primitives 2,0,1,2,3 --repeats 10 --task-id 0` |
| 2 | `v2_approach_p1_center` | 1 | Place the target at the center reference placement with a clear approach path. | No | Keep it fixed for all 6 repeats. | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v2_approach_p1_center.npz --primitives 2,3,4,5,6,7 --repeats 6 --task-id 1` |
| 2 | `v2_approach_p2_left` | 1 | Place the target slightly left of center while keeping it in a safe approach region. | Yes | Move it once, about `3-5 cm` left from center, before the run. | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v2_approach_p2_left.npz --primitives 2,3,4,5,6,7 --repeats 6 --task-id 1` |
| 2 | `v2_approach_p3_right` | 1 | Place the target slightly right of center while keeping it in a safe approach region. | Yes | Move it once, about `3-5 cm` right from center, before the run. | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v2_approach_p3_right.npz --primitives 2,3,4,5,6,7 --repeats 6 --task-id 1` |
| 2 | `v2_approach_p4_front` | 1 | Place the target slightly forward from center while keeping it reachable and safe. | Yes | Move it once, about `3-5 cm` forward from center, before the run. | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v2_approach_p4_front.npz --primitives 2,3,4,5,6,7 --repeats 6 --task-id 1` |
| 2 | `v2_approach_p5_back` | 1 | Place the target slightly backward from center while keeping it reachable and safe. | Yes | Move it once, about `3-5 cm` backward from center, before the run. | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v2_approach_p5_back.npz --primitives 2,3,4,5,6,7 --repeats 6 --task-id 1` |
| 3 | `v3_pick_place_p1_center` | 2 | Place the target at the center reference placement and ensure the pick-and-place path is clear. | No | Keep it fixed for all 3 repeats. Re-seat the object if it is displaced after each repeat. | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v3_pick_place_p1_center.npz --primitives 8,9,10,11,12,13 --repeats 3 --task-id 2` |
| 3 | `v3_pick_place_p2_left` | 2 | Place the target slightly left of center while keeping the full pick-and-place path safe. | Yes | Move it once, about `3-5 cm` left from center, before the run. Re-seat after each repeat as needed. | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v3_pick_place_p2_left.npz --primitives 8,9,10,11,12,13 --repeats 3 --task-id 2` |
| 3 | `v3_pick_place_p3_right` | 2 | Place the target slightly right of center while keeping the full pick-and-place path safe. | Yes | Move it once, about `3-5 cm` right from center, before the run. Re-seat after each repeat as needed. | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v3_pick_place_p3_right.npz --primitives 8,9,10,11,12,13 --repeats 3 --task-id 2` |

## Recommended Execution Order

1. `smoke_verify`
2. `v1_verify_p1_center`
3. `v1_verify_p2_left`
4. `v1_verify_p3_right`
5. `v1_verify_p4_front`
6. `v1_verify_p5_back`
7. `v2_approach_p1_center`
8. `v2_approach_p2_left`
9. `v2_approach_p3_right`
10. `v2_approach_p4_front`
11. `v2_approach_p5_back`
12. `v3_pick_place_p1_center`
13. `v3_pick_place_p2_left`
14. `v3_pick_place_p3_right`

## Recommended Data Split

Two simple split strategies are recommended.

### Split by placement

- Use `P1 / P2 / P3 / P4` for adaptation / fine-tuning
- Hold out `P5` for offline testing in L1 / L2
- In L3, hold out one full placement such as `P3`

### Split by repeat

- Use earlier repeats for adaptation / fine-tuning
- Hold out the last `2` repeats per placement for offline testing
- For L3, hold out the last repeat per placement

If you want smaller train/test distribution shift, split-by-repeat is usually
the safer choice.

## Post-Run Checklist

After every batch:

1. Verify the `.npz` file:

```bash
uv run python -c "import numpy as np; d=np.load('data/real/<batch_name>.npz'); print(d.files); print(d['images'].shape, d['states'].shape, d['primitive_ids'].shape)"
```

2. Inspect the session directory:

- `meta.json`
- `frames/`
- `preview.mp4`

3. Confirm:

- the target stays visible
- the image quality is acceptable
- the primitive sequence looks correct
- the output file name matches the intended batch

## L3 Notes

- Re-seat the object between repeats if it is moved or dropped outside the
  intended start pose.
- Keep the drop zone clear and fixed during a batch.
- Use the same validated hardware baseline as `stable-validation-v1`.
- If any L3 batch shows unstable behavior, stop and re-validate primitives
  before continuing collection.


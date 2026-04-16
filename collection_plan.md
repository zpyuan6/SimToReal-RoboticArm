# Real Data Collection Plan

This document records a practical real-world collection plan for the RoArm-M3
setup when only the tabletop target position can be controlled reliably.

The current plan focuses on:

- L1 / `task-id 0`: observation and verification
- L2 / `task-id 1`: pre-alignment, approach, and recovery

It intentionally does **not** depend on controlled lighting, controlled
background swaps, or other environment factors that may be difficult to manage
consistently during data collection.

## Scope

This plan is built around one controllable factor only:

- target position on the tabletop

The target is moved between a small number of discrete placements. During each
batch, the target stays fixed and multiple repeats are collected from that same
placement.

This plan currently covers:

- low-risk L1 observation data
- medium-risk L2 approach data
- target-position variation only

This plan does not yet include L3 grasp / lift / transport / place as a main
collection target.

## Dataset Size Target

If the goal is only smoke testing or very small few-shot experiments, a smaller
dataset is enough.

However, if the goal is to support both:

- fine-tuning / adaptation
- offline testing

then a larger first-round dataset is recommended.

The recommended target for this version is:

- L1: 5 placements with `10` repeats each
- L2: 5 placements with `6` repeats each

Rough transition count:

- L1: `5 x 10 x 5 = 250` transitions
- L2: `5 x 6 x 6 = 180` transitions

Total:

- about `430` transitions

This scale is more appropriate for:

- first-round fine-tuning / adaptation
- holding out part of the data for offline testing
- train / offline-test splits by placement or repeat

## Primitive Reference

Low-risk observation primitives:

- `2` = `obs_center`
- `0` = `obs_left`
- `1` = `obs_right`
- `3` = `verify_target`

Medium-risk approach primitives:

- `4` = `prealign_grasp`
- `5` = `approach_coarse`
- `6` = `approach_fine`
- `7` = `retreat`

## Target Placements

Define one center reference placement, then only move the target between these
small offsets:

- `P1`: center
- `P2`: left
- `P3`: right
- `P4`: front
- `P5`: back

Recommended movement size:

- left / right: about `3-5 cm`
- front / back: about `3-5 cm`

Keep object orientation unchanged unless it changes naturally while you move it.
Do not intentionally add lighting or background variation in this version.

## Collection Table

| Stage | Batch Name | Task ID | How To Place The Object | Need To Move The Target? | How To Move The Target | Command To Run |
|---|---|---:|---|---|---|---|
| 0 | `smoke_verify` | 0 | Place the target at the center reference placement. Keep it clearly visible in the forearm camera view. | No | Do not move the target. | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/smoke_verify.npz --primitives 2,3 --repeats 1 --task-id 0` |
| 1 | `v1_verify_p1_center` | 0 | Place the target at the center reference placement. | No | Keep the target fixed for all 10 repeats. | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v1_verify_p1_center.npz --primitives 2,0,1,2,3 --repeats 10 --task-id 0` |
| 1 | `v1_verify_p2_left` | 0 | Start from the center reference and place the target slightly left while keeping it fully visible. | Yes | Move it once before the run, about `3-5 cm` left from center. Keep it fixed during the 10 repeats. | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v1_verify_p2_left.npz --primitives 2,0,1,2,3 --repeats 10 --task-id 0` |
| 1 | `v1_verify_p3_right` | 0 | Start from the center reference and place the target slightly right while keeping it fully visible. | Yes | Move it once before the run, about `3-5 cm` right from center. Keep it fixed during the 10 repeats. | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v1_verify_p3_right.npz --primitives 2,0,1,2,3 --repeats 10 --task-id 0` |
| 1 | `v1_verify_p4_front` | 0 | Start from the center reference and place the target slightly forward while keeping it visible and safe. | Yes | Move it once before the run, about `3-5 cm` in the front direction. Keep it fixed during the 10 repeats. | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v1_verify_p4_front.npz --primitives 2,0,1,2,3 --repeats 10 --task-id 0` |
| 1 | `v1_verify_p5_back` | 0 | Start from the center reference and place the target slightly backward while keeping it visible and safe. | Yes | Move it once before the run, about `3-5 cm` in the back direction. Keep it fixed during the 10 repeats. | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v1_verify_p5_back.npz --primitives 2,0,1,2,3 --repeats 10 --task-id 0` |
| 2 | `v2_approach_p1_center` | 1 | Place the target at the center reference placement, making sure the approach path is clear and safe. | No | Keep the target fixed for all 6 repeats. | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v2_approach_p1_center.npz --primitives 2,3,4,5,6,7 --repeats 6 --task-id 1` |
| 2 | `v2_approach_p2_left` | 1 | Place the target slightly left from the center reference while keeping it in a safe approach region. | Yes | Move it once before the run, about `3-5 cm` left from center. Keep it fixed during the 6 repeats. | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v2_approach_p2_left.npz --primitives 2,3,4,5,6,7 --repeats 6 --task-id 1` |
| 2 | `v2_approach_p3_right` | 1 | Place the target slightly right from the center reference while keeping it in a safe approach region. | Yes | Move it once before the run, about `3-5 cm` right from center. Keep it fixed during the 6 repeats. | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v2_approach_p3_right.npz --primitives 2,3,4,5,6,7 --repeats 6 --task-id 1` |
| 2 | `v2_approach_p4_front` | 1 | Place the target slightly forward from the center reference while keeping it reachable and safe. | Yes | Move it once before the run, about `3-5 cm` in the front direction. Keep it fixed during the 6 repeats. | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v2_approach_p4_front.npz --primitives 2,3,4,5,6,7 --repeats 6 --task-id 1` |
| 2 | `v2_approach_p5_back` | 1 | Place the target slightly backward from the center reference while keeping it reachable and safe. | Yes | Move it once before the run, about `3-5 cm` in the back direction. Keep it fixed during the 6 repeats. | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v2_approach_p5_back.npz --primitives 2,3,4,5,6,7 --repeats 6 --task-id 1` |
| 3 | `v3_verify_offsets_dense` | 0 | Use any 3 of the 5 placements that were most stable in earlier runs. | Yes | Before each batch, move the target once to the chosen placement. Do not move it during the repeats. | `uv run python scripts/collect_real_calibration.py --config configs/base.yaml --deploy-config configs/deployment.yaml --output data/real/v3_verify_offsets_dense.npz --primitives 2,0,1,2,3 --repeats 10 --task-id 0` |

## Recommended Execution Order

To minimize manual work:

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
12. Optional dense follow-up batch:
    - `v3_verify_offsets_dense`

## Suggested Dataset Size

This plan gives a more appropriate first complete dataset for both fine-tuning
and offline testing:

- L1: 5 placements x 10 repeats = 50 repeats
- L2: 5 placements x 6 repeats = 30 repeats

Total:

- 80 repeats across L1 and L2

Rough transition count:

- L1: about `250` transitions
- L2: about `180` transitions

Total:

- about `430` transitions

This is more suitable for:

- first-round fine-tuning / adaptation
- offline testing
- behavior analysis

## Recommended Data Split

If the goal is to support both fine-tuning and offline testing, do not use the
entire dataset for adaptation.

Two simple split strategies are recommended:

1. Split by placement

- `P1 / P2 / P3 / P4` for fine-tuning / adaptation
- `P5` held out for offline testing

2. Split by repeat

- earlier repeats for fine-tuning / adaptation
- last `2` repeats from each placement held out for offline testing

If you want a smaller distribution gap between train and test, the second
strategy is usually better.

## Post-Run Checklist

After each batch:

1. Verify the saved `.npz`:

```bash
uv run python -c "import numpy as np; d=np.load('data/real/<batch_name>.npz'); print(d.files); print(d['images'].shape, d['states'].shape, d['primitive_ids'].shape)"
```

2. Inspect the session directory:

- `meta.json`
- `frames/`
- `preview.mp4`

3. Confirm:

- the target remains visible
- the frames are sharp enough
- the primitive sequence looks correct
- the output file names match the intended batch

## Notes

- Preserve raw-resolution images during collection.
- Keep the structured `.npz` files as the main training / analysis asset.
- Use `frames/` and `preview.mp4` for audit and debugging.
- If environment factors change naturally during collection, just record them in
  notes; do not treat them as controlled variables in this version.
- Treat L3 as a later, separate collection plan once the real robot grasp
  pipeline is stable enough.

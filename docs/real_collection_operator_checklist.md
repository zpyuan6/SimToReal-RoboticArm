# Real Collection Operator Checklist

This is the short, execution-focused version of the real collection protocol.

Use this file when you are standing next to the robot and want to know:

- what to run
- what to look for
- what files should appear

Reference files:

- detailed protocol: [real_collection_plan_v2.md](/F:/RoboticArm/docs/real_collection_plan_v2.md)
- collection plan: [real_collection_plan_v2.yaml](/F:/RoboticArm/configs/real_collection_plan_v2.yaml)

## Before you start

Make sure all of the following are true:

1. The robot mount is fixed.
2. The camera mount is fixed.
3. The blue drop zone is fixed.
4. The target object is the expected one: `wire_ear_cork`.
5. The camera feed is live and sharp.
6. The correct serial port and camera index are available.

This collection protocol also runs the primitive executor faster than the old slow collection setup:

- collection dwell: `primitive_sleep_s = 2.0`
- older slow L3 config: `primitive_sleep_s = 4.0`

So the collection sessions should run at about double the previous per-primitive pace.

Set the project cache path first:

```powershell
$env:UV_CACHE_DIR='F:\RoboticArm\.uv-cache'
```

## Step 1: validate primitive mapping

Run:

```powershell
uv run python scripts\validate_actions.py --config configs/base.yaml --deploy-config configs/deployment_l3.yaml --task level3_pick_place --primitives 2,9,10,11,12,13
```

Expected result:

- a validation session directory is written under `data/raw/action_validation`
- the real arm visibly executes:
  - centered observation
  - pregrasp servo
  - grasp
  - lift
  - transport
  - place
- the comparison images and preview should show the simulated and real motions are qualitatively aligned

Stop and fix the primitive mapping if:

- the arm executes the wrong primitive
- pregrasp/grasp/lift/place semantics are visibly wrong
- the object would be approached from the wrong side

## Step 2: collect transition sessions

The sessions must be collected in this order:

1. `calib_l1_center`
2. `calib_l2_center`
3. `calib_l3_center`
4. `heldout_l1_offset`
5. `heldout_l2_offset`
6. `heldout_l3_offset`

### Session 1: `calib_l1_center`

Run:

```powershell
uv run python scripts\collect_real_transition_session.py --plan configs/real_collection_plan_v2.yaml --session calib_l1_center --operator "<name>"
```

Before pressing Enter for each episode:

- place the target in the center third of the `obs_center` view
- keep it outside the blue drop zone
- keep it outside direct gripper contact

Expected arm behavior:

- `obs_left` and `obs_right` visibly move the view to opposite sides
- `obs_center` re-centers the target
- `verify_target` is a brief stable hold

Keep the episode if:

- the target stays visible
- observation poses are distinct
- `obs_center` improves centering

Redo the episode if:

- there is motion blur
- the target leaves the frame
- `verify_target` causes unexpected motion

Expected files after finishing the session:

- a new directory under `data/real_v2/transitions/`
- inside it:
  - `session_dataset.npz`
  - `meta.json`
  - `frames/`
  - `preview.mp4` if preview saving is enabled

### Session 2: `calib_l2_center`

Run:

```powershell
uv run python scripts\collect_real_transition_session.py --plan configs/real_collection_plan_v2.yaml --session calib_l2_center --operator "<name>"
```

Before pressing Enter for each episode:

- use the same `center_band` placement regime as L1
- keep the object reachable but not already touched

Expected arm behavior:

- `prealign_grasp` moves to a frontal pre-grasp pose
- `approach_coarse` makes the target visibly closer
- `approach_fine` refines this without overshooting
- `retreat` clearly backs away

Keep the episode if:

- the target appears larger after approach
- fine approach does not pass the target
- retreat restores clearance

Redo the episode if:

- the arm overshoots
- the target becomes unobservable
- retreat goes the wrong way

Expected files:

- same session outputs as above

### Session 3: `calib_l3_center`

Run:

```powershell
uv run python scripts\collect_real_transition_session.py --plan configs/real_collection_plan_v2.yaml --session calib_l3_center --operator "<name>"
```

Before pressing Enter for each episode:

- keep the target in the same center-band family
- keep the blue drop zone fixed and visible
- ensure the object starts away from the drop zone

Expected arm behavior:

- `pregrasp_servo` lowers toward a plausible pre-grasp state
- `grasp_execute` closes around the object
- `lift_object` visibly raises the object
- `transport_to_dropzone` moves toward the blue zone
- `place_object` opens above the blue zone

Keep the episode if:

- the grasp looks semantically correct
- the object is visibly lifted
- the object is moved over the drop zone before release

Redo the episode if:

- grasp closes in empty space
- lift does not raise the object
- place occurs away from the drop zone

### Session 4: `heldout_l1_offset`

Run:

```powershell
uv run python scripts\collect_real_transition_session.py --plan configs/real_collection_plan_v2.yaml --session heldout_l1_offset --operator "<name>"
```

Placement rule:

- do not reuse center-band placements
- use left/right/front offsets

Expected behavior:

- observation correction should be more obvious than in calibration

### Session 5: `heldout_l2_offset`

Run:

```powershell
uv run python scripts\collect_real_transition_session.py --plan configs/real_collection_plan_v2.yaml --session heldout_l2_offset --operator "<name>"
```

Placement rule:

- keep the object reachable
- use left/right/front offsets not used in calibration

Expected behavior:

- approach chain still works, but the starting pose is visibly different from calibration

### Session 6: `heldout_l3_offset`

Run:

```powershell
uv run python scripts\collect_real_transition_session.py --plan configs/real_collection_plan_v2.yaml --session heldout_l3_offset --operator "<name>"
```

Placement rule:

- keep the drop zone fixed
- only move the object start pose
- do not reuse calibration L3 placements

Expected behavior:

- the object starts from a visibly different workspace location
- the chain still shows grasp, lift, transport, and place

## Step 3: check the collected sessions

For every session directory under `data/real_v2/transitions`, check:

1. `meta.json` exists
2. `session_dataset.npz` exists
3. `frames/` contains before/after images
4. `preview.mp4` exists if preview saving was enabled

Open the preview video and verify:

- each primitive alternates between a `before` frame and an `after` frame
- there is visible state change for motion primitives
- the target stays identifiable
- L3 clips keep the blue drop zone visible during transport/place

Reject the session and recollect if:

- the video is blurry
- before/after frames do not reflect the primitive label
- the target is repeatedly occluded or lost

## Step 4: merge sessions into explicit splits

Run:

```powershell
uv run python scripts\merge_real_transition_sessions.py --root data/real_v2/transitions --output-dir data/real_v2/merged
```

Expected result:

- `data/real_v2/merged/calibration_merged.npz`
- `data/real_v2/merged/heldout_merged.npz`
- `data/real_v2/merged/calibration_sessions.csv`
- `data/real_v2/merged/heldout_sessions.csv`
- `data/real_v2/merged/calibration_meta.json`
- `data/real_v2/merged/heldout_meta.json`

Sanity checks:

- calibration and heldout manifests should list different session directories
- transition counts should be nonzero

## Step 5: run held-out transition validation

Run:

```powershell
uv run python scripts\run_real_adaptation_suite.py --config configs/real_eval_shiftgrid_a8g05_ff.yaml --calibration-data data/real_v2/merged/calibration_merged.npz --heldout-data data/real_v2/merged/heldout_merged.npz --backbones feedforward --baselines no_adaptation static_adapter few_shot_finetuning ours --tag real_v2
```

Expected result:

- a results directory under `results/fixed_protocol/...`
- a `split_manifest.yaml` showing `split_mode: explicit`
- suite CSVs including:
  - `suite_task_metrics.csv`
  - `suite_summary_metrics.csv`
  - `suite_overall_metrics.csv`

This step validates:

- transition-level adaptation quality
- not task-level success

## Step 6: run task-success validation separately

Run:

```powershell
uv run python scripts\run_l3_policy_baselines.py --serial-port COM<port> --camera-index 0 --episodes 10 --operator "<name>"
```

Then summarize:

```powershell
uv run python scripts\summarize_l3_policy_baselines.py
```

Expected result:

- a baseline-comparison directory under `results/real_deployment_eval`
- per-run summaries
- a final `l3_policy_baseline_summary.csv`

This is the real task-success validation path.

## Minimal pass criteria for the whole protocol

The collection run is acceptable only if all of the following are true:

1. Primitive preflight looks correct.
2. All six transition sessions are collected.
3. Calibration and held-out sessions use visibly different placement regimes.
4. Preview videos look semantically correct.
5. Merge outputs are produced successfully.
6. Held-out real evaluation consumes explicit split files.
7. Real task-success validation is run separately from transition validation.

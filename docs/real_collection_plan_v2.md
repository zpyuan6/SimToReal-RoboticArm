# Real Collection Plan v2

This document is the operator-facing protocol for new real-data collection. It is meant to be sufficient on its own: after reading this file and [real_collection_plan_v2.yaml](/F:/RoboticArm/configs/real_collection_plan_v2.yaml), you should know what to collect, how to place the target, how to reject a bad episode, and what good preview videos should look like.

The protocol separates three things that used to be mixed together:

1. `calibration` transition sessions used for adapter calibration
2. `heldout` transition sessions never used for calibration
3. separate task-success deployment runs for policy-level validation

The core rule is simple: **short transition data are for calibration and held-out transition validation; task-success validation is collected and scored separately.**

For a concise operator checklist with exact commands and expected outputs, see:

- [real_collection_operator_checklist.md](/F:/RoboticArm/docs/real_collection_operator_checklist.md)

## Files that define the protocol

- collection plan: [real_collection_plan_v2.yaml](/F:/RoboticArm/configs/real_collection_plan_v2.yaml)
- transition collector: [collect_real_transition_session.py](/F:/RoboticArm/scripts/collect_real_transition_session.py)
- session merger: [merge_real_transition_sessions.py](/F:/RoboticArm/scripts/merge_real_transition_sessions.py)
- held-out real evaluation: [run_real_adaptation_suite.py](/F:/RoboticArm/scripts/run_real_adaptation_suite.py)
- task-success evaluation: [run_l3_policy_baselines.py](/F:/RoboticArm/scripts/run_l3_policy_baselines.py)
- primitive preflight validator: [validate_actions.py](/F:/RoboticArm/scripts/validate_actions.py)

## 1. Physical setup assumptions

The collection plan assumes:

- the robot base stays fixed
- the camera mount stays fixed
- the table stays fixed
- the blue drop zone stays fixed
- the object identity stays fixed within one collection run

Use the same object for both calibration and held-out transition sessions on a given day:

- `wire_ear_cork`

Use the same placement target region for all L3 transport/place sequences:

- `blue_dropzone`

If you move the camera or robot mount, treat that as a new collection day and restart the protocol.

## Execution speed

For this collection protocol, the collector overrides the real primitive dwell to:

- `primitive_sleep_s: 2.0`

This is intentionally faster than the older `deployment_l3.yaml` default of `4.0`, so transition collection runs at roughly double speed while keeping the same primitive scripts.

## 2. Placement regimes

The YAML plan uses `layout_tag` to describe how the target is placed.

### `center_band`
Use this for `calibration` sessions.

Operational meaning:

- when the arm is at `obs_center`, the target should appear in the middle third of the frame horizontally
- the target should not begin inside the drop zone
- the target should not start already in contact with the gripper
- small orientation variation is allowed across repeats

This is the easier, in-distribution placement band.

### `front_left_front_right`
Use this for `heldout` sessions.

Operational meaning:

- the target is intentionally offset from the `center_band`
- use one of:
  - one object-width left of center
  - one object-width right of center
  - one object-length closer to the arm
- keep the target reachable and fully visible after one corrective observation step

This is the held-out placement family. Do not reuse exact `center_band` placements here.

## 3. Primitive sequences that will be collected

### L1: `level1_verify`
Purpose:
- collect observation and confirmation transitions

Sequences:
- `obs_left -> obs_center -> verify_target`
- `obs_right -> obs_center -> verify_target`
- `obs_center -> verify_target`

Expected use:
- adapter calibration for observation semantics
- held-out validation for observation correction

### L2: `level2_approach`
Purpose:
- collect observation-to-pregrasp transitions

Sequences:
- `obs_center -> prealign_grasp -> approach_coarse -> approach_fine -> retreat`
- `obs_left -> obs_center -> prealign_grasp -> approach_coarse -> approach_fine -> retreat`
- `obs_right -> obs_center -> prealign_grasp -> approach_coarse -> approach_fine -> retreat`

Expected use:
- adapter calibration for approach-stage dynamics
- held-out validation for approach transitions

### L3: `level3_pick_place`
Purpose:
- collect full manipulation-stage transitions

Sequences:
- `obs_center -> pregrasp_servo -> grasp_execute -> lift_object -> transport_to_dropzone -> place_object -> retreat`
- `reobserve -> pregrasp_servo -> grasp_execute -> lift_object -> transport_to_dropzone -> place_object -> retreat`

Expected use:
- adapter calibration for manipulation dynamics
- held-out validation for full pick-place transition chains

## 4. What "successful collection" means

This protocol does **not** mark transition sessions as task-success or task-failure. Instead, collection is considered good if the recorded transitions are physically valid and semantically aligned with the named primitive.

### Global rejection conditions
Redo an episode immediately if any of the following happens:

- camera blackout or dropped frames
- severe motion blur hiding the object
- a human hand remains in frame during execution
- obvious collision or unsafe motion
- the arm executes the wrong primitive mapping
- the target leaves the frame when it should remain observable

### L1 acceptance
Keep the episode if:

- `obs_left` and `obs_right` visibly move the view to opposite sides
- `obs_center` returns the target toward the center band
- `verify_target` looks like a short stable hold, not a large extra motion

Reject if:

- observation poses are indistinguishable
- `obs_center` does not improve target centering
- `verify_target` causes unexpected movement

### L2 acceptance
Keep the episode if:

- `prealign_grasp` moves into a coarse frontal approach pose
- `approach_coarse` makes the target noticeably closer/larger
- `approach_fine` refines this rather than overshooting
- `retreat` increases clearance and backs away

Reject if:

- the arm passes the target during `approach_fine`
- `retreat` keeps moving forward instead of reversing
- the target becomes unobservable because of bad approach geometry

### L3 acceptance
Keep the episode if:

- `pregrasp_servo` lowers toward a plausible pre-grasp state
- `grasp_execute` closes on the object rather than empty space
- `lift_object` visibly raises the object
- `transport_to_dropzone` moves the grasped object toward the blue zone
- `place_object` opens over the drop zone region

Reject if:

- the gripper closes clearly away from the object
- the object is never lifted after `lift_object`
- `place_object` occurs outside the blue drop zone region
- the full clip is mechanically impossible or clearly mistimed

## 5. What the preview videos should look like

Each collected session writes:

- `session_dataset.npz`
- `meta.json`
- `frames/`
- optionally `preview.mp4`

The preview video is expected to alternate:

- `before: <primitive>`
- `after: <primitive>`

for each collected step.

### Good preview characteristics

- every primitive produces a visible state change between before and after
- the target silhouette remains identifiable
- the blue drop zone boundary remains identifiable in L3 sessions
- before/after images match the named primitive semantics
- the sequence looks physically smooth enough that the transition could be used for calibration

### Bad preview characteristics

- before/after frames are visually identical for primitives that should move
- the object is lost because of blur or occlusion
- the camera has shifted between episodes
- labels say one primitive but the arm visibly performed another
- the object starts inside or on top of the drop zone for L3 calibration clips

## 6. Recommended collection order

Collect in this order:

1. `calib_l1_center`
2. `calib_l2_center`
3. `calib_l3_center`
4. `heldout_l1_offset`
5. `heldout_l2_offset`
6. `heldout_l3_offset`

This order is deliberate:

- first confirm observation primitives
- then confirm approach primitives
- then confirm manipulation primitives
- only after calibration placements are clean, move to held-out placements

## 7. How to execute the collection plan

### Step 1: preflight primitive validation

```powershell
$env:UV_CACHE_DIR='F:\RoboticArm\.uv-cache'
uv run python scripts\validate_actions.py --config configs/base.yaml --deploy-config configs/deployment_l3.yaml --task level3_pick_place --primitives 2,9,10,11,12,13
```

Do not start collection until this looks correct.

### Step 2: run each transition session

Example:

```powershell
$env:UV_CACHE_DIR='F:\RoboticArm\.uv-cache'
uv run python scripts\collect_real_transition_session.py --plan configs/real_collection_plan_v2.yaml --session calib_l1_center --operator "<name>"
```

Run the same command with:

- `calib_l2_center`
- `calib_l3_center`
- `heldout_l1_offset`
- `heldout_l2_offset`
- `heldout_l3_offset`

During collection:

- press Enter to start each episode
- inspect the motion
- choose:
  - `keep`
  - `redo`
  - `quit`

Use `redo` aggressively. Bad episodes are worse than fewer episodes.

### Step 3: merge sessions

```powershell
$env:UV_CACHE_DIR='F:\RoboticArm\.uv-cache'
uv run python scripts\merge_real_transition_sessions.py --root data/real_v2/transitions --output-dir data/real_v2/merged
```

This writes:

- `data/real_v2/merged/calibration_merged.npz`
- `data/real_v2/merged/heldout_merged.npz`

and manifests:

- `data/real_v2/merged/calibration_sessions.csv`
- `data/real_v2/merged/heldout_sessions.csv`

## 8. How the collected data are used

### Calibration
`calibration_merged.npz` is used for:

- adapter calibration only

It should never be used for final held-out reporting.

### Held-out transition validation
`heldout_merged.npz` is used for:

- transition-level offline validation only

Run:

```powershell
$env:UV_CACHE_DIR='F:\RoboticArm\.uv-cache'
uv run python scripts\run_real_adaptation_suite.py `
  --config configs/real_eval_shiftgrid_a8g05_ff.yaml `
  --calibration-data data/real_v2/merged/calibration_merged.npz `
  --heldout-data data/real_v2/merged/heldout_merged.npz `
  --backbones feedforward `
  --baselines no_adaptation static_adapter few_shot_finetuning ours `
  --tag real_v2
```

This measures:

- `transition_mse`
- `primitive_loss`
- `primitive_match`
- `stage_loss`
- `latent_reg`

This is **proxy validation**, not task-success validation.

### Task-success validation
Run separately:

```powershell
$env:UV_CACHE_DIR='F:\RoboticArm\.uv-cache'
uv run python scripts\run_l3_policy_baselines.py --serial-port COM<port> --camera-index 0 --episodes 10 --operator "<name>"
```

Then summarize:

```powershell
$env:UV_CACHE_DIR='F:\RoboticArm\.uv-cache'
uv run python scripts\summarize_l3_policy_baselines.py
```

This is the clean task-level validation path.

## 9. What to look for in the generated files

For each collected session directory:

- check `meta.json`
  - task
  - split role
  - layout tag
  - placement guide
  - acceptance criteria
  - video expectations
- check `preview.mp4`
  - primitive semantics are visually correct
- check `session_dataset.npz`
  - transition count is nonzero
- check `frames/`
  - filenames align with episode and step indices

If any of those are inconsistent, recollect before merging.

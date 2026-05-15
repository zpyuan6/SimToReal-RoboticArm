# Continuous-Control Refactor Plan

## Goal

Replace the primitive-policy mainline with a unified continuous/chunk control
pipeline while preserving the core PLICA research question:

- sim-to-real latent calibration
- small real calibration budget
- low-cost robot arm deployment

The new host-controller comparison set is fixed to three backbone families:

1. ACT - action-chunking transformer
2. Diffusion Policy - generative continuous action-sequence policy
3. SmolVLA - lightweight VLA policy

The existing primitive system remains available only as a legacy baseline during
migration. It is no longer the target mainline architecture.

## Backbone Coverage Rationale

The new host-controller suite is organized by **backbone structure**, not by
output discretization:

- `ACT`: transformer chunking backbone
- `Diffusion Policy`: diffusion-based generative backbone
- `SmolVLA`: lightweight VLA backbone

This gives one representative model for each of the three dominant policy
families that are lightweight enough to be realistic on the current platform.

## Baseline Policy Rule

For submission-quality experiments, **all three host-controller baselines must
use official upstream implementations**:

- `ACT` from official LeRobot support
- `Diffusion Policy` from official LeRobot support
- `SmolVLA` from official LeRobot support

Local reimplementations may be used only as internal scaffolds during migration.
They must not be reported as formal baselines in the paper.

## Official Baseline Mapping

The paper baselines must map to official upstream policies as follows:

- `ACT`
  - official LeRobot `act` policy type
  - official minimal architecture configuration
- `Diffusion Policy`
  - official LeRobot `diffusion` policy type
  - official minimal architecture configuration
- `SmolVLA`
  - official LeRobot `smolvla` policy family
  - official released base checkpoint: `lerobot/smolvla_base`

This distinction is deliberate:

- `ACT` and `Diffusion Policy` are reported as official architecture baselines
  instantiated through upstream config definitions.
- `SmolVLA` is reported as an official lightweight VLA baseline initialized from
  the upstream released base model, not from a local reproduction.

For submission baselines, upstream model defaults should be preserved unless a
change is strictly required by an upstream-documented compatibility constraint.
Task-side settings may change, but architecture-defining model hyperparameters
must not be reduced just to simplify training.

## New Unified Interface

All new host controllers must share the same interface:

- input:
  - camera images
  - proprioception / robot state
  - optional task text
- output:
  - continuous joint action or action chunk
- adaptation hook:
  - a latent tensor exposed through a wrapper-specific contract

Scaffold types live under:

- [src/ttla/control/types.py](/F:/RoboticArm/src/ttla/control/types.py)
- [src/ttla/control/base.py](/F:/RoboticArm/src/ttla/control/base.py)
- [src/ttla/control/__init__.py](/F:/RoboticArm/src/ttla/control/__init__.py)

Backbone wrapper scaffolds live under:

- [src/ttla/control/backbones/act.py](/F:/RoboticArm/src/ttla/control/backbones/act.py)
- [src/ttla/control/backbones/diffusion.py](/F:/RoboticArm/src/ttla/control/backbones/diffusion.py)
- [src/ttla/control/backbones/smolvla.py](/F:/RoboticArm/src/ttla/control/backbones/smolvla.py)

These wrappers are now **official-policy wrappers**, not local baseline
implementations. They are intentionally import-safe when LeRobot is absent, but
their `forward` path must not be used for reported experiments until the
official loaders are connected.

Initial configuration templates live under:

- [configs/continuous_control_template.yaml](/F:/RoboticArm/configs/continuous_control_template.yaml)
- [configs/continuous_act_template.yaml](/F:/RoboticArm/configs/continuous_act_template.yaml)
- [configs/continuous_diffusion_template.yaml](/F:/RoboticArm/configs/continuous_diffusion_template.yaml)
- [configs/continuous_smolvla_template.yaml](/F:/RoboticArm/configs/continuous_smolvla_template.yaml)

## Action Interface

The shared action interface for the first migration pass is:

- control mode: `joint_delta`
- action dimension: `6`
- horizon:
  - `1` for step-wise control
  - `H > 1` for chunked control

This is the least risky common denominator because:

- it avoids introducing an IK dependency into the first refactor
- it can represent ACT chunks directly
- it can represent diffusion/VLA outputs after per-step slicing
- it can be mirrored on the real arm with the least control-stack change

The alternative `joint_target` mode remains available in the interface type,
but it is not the recommended first integration target.

## Data Interface

The primitive-transition dataset is not sufficient for the new mainline.
The new supervised unit is a short continuous trajectory window:

- `images[t]`
- `proprio[t]`
- `task_text`
- `actions[t:t+H]`
- `episode_id`
- `step_id`

Required changes:

1. Add a new continuous-control dataset class instead of reusing the primitive
   classification dataset.
2. Regenerate simulation data with continuous actions or scripted continuous
   traces.
3. Export the generated trajectories into a **local official LeRobot dataset**
   before formal baseline training.
4. Replace primitive-labeled real calibration sessions with short continuous
   control segments.

## Environment Changes

The current MuJoCo environment still exposes:

- `env.step(primitive_id)`

The new mainline environment must expose one of:

- `env.step(action)`
- `env.step_chunk(action_chunk)`

The current MuJoCo world, rendering, randomization, and task-success logic can
be retained. What must be removed from the mainline is the primitive execution
layer itself.

Recommended migration strategy:

1. Keep the current primitive environment intact as a legacy baseline.
2. Add a new continuous-control environment beside it.
3. Port task-success logic into the new environment before removing any legacy
   components.

## Real-Robot Deployment Changes

The current deployment stack is built around:

- primitive selection
- fixed scripted executor

The new deployment stack must be rebuilt around:

- continuous joint deltas or joint targets
- optional chunk execution with interpolation

Recommended first-pass executor:

- real robot receives short joint-delta actions
- safety clamps are applied before sending commands
- chunk policies are executed as repeated delta steps

## PLICA Migration

PLICA must be reframed from:

- primitive-level latent calibration

to:

- latent calibration for continuous/chunk visuomotor control backbones

Backbone-specific latent hooks:

- `ACT`: transformer chunk latent
- `Diffusion Policy`: diffusion conditioning latent
- `SmolVLA`: multimodal fused condition latent

The host-controller wrappers should define these targets explicitly through:

- `latent_target_name()`

## Recommended Implementation Order

### Phase 1

Scaffold and freeze interfaces:

1. unified control backbone wrappers
2. unified action specification
3. new continuous-control dataset schema
4. migration design doc

### Phase 2

Implement ACT first:

1. dataset adapter
2. sim training loop
3. pseudo-real evaluation
4. real deployment path

### Phase 3

Implement Diffusion Policy.

### Phase 4

Implement SmolVLA.

### Phase 5

Reintroduce PLICA on top of the three backbones through a shared adapter
interface.

## Official Training Workflow

Formal ACT / Diffusion Policy / SmolVLA baselines must follow this workflow:

1. Generate continuous simulation trajectories:
   - [scripts/generate_continuous_sim_data.py](/F:/RoboticArm/scripts/generate_continuous_sim_data.py)
2. Export each split to a local official LeRobot dataset:
   - [scripts/export_continuous_to_lerobot.py](/F:/RoboticArm/scripts/export_continuous_to_lerobot.py)
3. Launch the upstream LeRobot trainer through the repo wrapper:
   - [scripts/launch_official_lerobot_train.py](/F:/RoboticArm/scripts/launch_official_lerobot_train.py)
4. Only after official backbone training is in place, attach the continuous
   PLICA adapter path.

Example commands:

```powershell
$env:UV_CACHE_DIR='F:\RoboticArm\.uv-cache'
uv run python scripts\generate_continuous_sim_data.py --config configs\continuous_act_template.yaml --train 200 --val 40 --test 40
uv run python scripts\export_continuous_to_lerobot.py --config configs\continuous_act_template.yaml --input data\continuous\train.npz --output-root data\lerobot\roarm_continuous_train --repo-id roarm_continuous_train
uv run python scripts\launch_official_lerobot_train.py --config configs\continuous_act_template.yaml --run
```

For the other official baselines, change only the config path:

- [continuous_diffusion_template.yaml](/F:/RoboticArm/configs/continuous_diffusion_template.yaml)
- [continuous_smolvla_template.yaml](/F:/RoboticArm/configs/continuous_smolvla_template.yaml)

For smoke runs, the launcher also supports temporary overrides:

```powershell
uv run python scripts\launch_official_lerobot_train.py --config configs\continuous_act_template.yaml --steps 2 --batch-size 2 --output-dir outputs\train\lerobot\act_smoke --job-name roarm_act_smoke --run
```

For formal runs on this machine, prefer the staged early-stop wrapper:

- [scripts/train_continuous_with_early_stop.py](/F:/RoboticArm/scripts/train_continuous_with_early_stop.py)

This wrapper:

1. launches the official upstream trainer in stages
2. resumes from `checkpoints/last`
3. evaluates each stage checkpoint with validation action loss
4. stops when the validation loss no longer improves for the configured patience
5. after training stops, runs rollout selection over saved checkpoints and writes the final best checkpoint

Current template defaults:

- `ACT`
  - `max_steps=50000`
  - `stage_steps=5000`
- `Diffusion Policy`
  - `max_steps=50000`
  - `stage_steps=5000`
- `SmolVLA`
  - `max_steps=50000`
  - `stage_steps=2000`

Validation is evaluated on the continuous NPZ validation split, not on rollout.
This keeps stage selection cheap and deterministic. Rollout evaluation remains a
separate final checkpoint-selection step.

Example:

```powershell
$env:UV_CACHE_DIR='F:\RoboticArm\.uv-cache'
uv run python scripts\train_continuous_with_early_stop.py --config configs\continuous_act_template.yaml --policy-device cuda --run
```

This writes:

- `stage_history.csv`
- `best_checkpoint.txt`
- `best_stage_step.txt`
- `final_rollout_selection/checkpoint_ranking.csv`
- `final_rollout_selection/best_checkpoint.txt`

under:

- `outputs/train/lerobot/<job>/early_stop_selection`
- `outputs/train/lerobot/<job>/final_rollout_selection`

For long official runs on this Windows machine, prefer launching from a normal
PowerShell terminal instead of relying on detached background jobs from the
embedded Codex shell. The official training chain itself is valid, but detached
child process behavior inside the current Codex shell has been inconsistent.

## Official Evaluation Workflow

The new continuous-control mainline uses a dedicated rollout evaluator instead
of the legacy primitive-checkpoint evaluator:

- single-backbone evaluator:
  - [scripts/evaluate_continuous_backbone.py](/F:/RoboticArm/scripts/evaluate_continuous_backbone.py)
- three-backbone suite runner:
  - [scripts/run_continuous_eval_suite.py](/F:/RoboticArm/scripts/run_continuous_eval_suite.py)

The evaluator:

- loads the official backbone through the repo wrapper
- rolls out on the continuous MuJoCo environment
- writes:
  - `episodes.csv`
  - `summary.csv`

Because the official trainers do not expose a repo-native `best` checkpoint
directory in these local runs, rollout-based checkpoint selection is handled by:

- [scripts/select_continuous_checkpoint.py](/F:/RoboticArm/scripts/select_continuous_checkpoint.py)

This script scans numbered official checkpoints, evaluates each on the
continuous MuJoCo validation rollout, and writes:

- `checkpoint_ranking.csv`
- `best_checkpoint.txt`

Example single-backbone evaluations:

```powershell
$env:UV_CACHE_DIR='F:\RoboticArm\.uv-cache'
uv run python scripts\evaluate_continuous_backbone.py --config configs\continuous_act_template.yaml --policy-path outputs\train\lerobot\act_v1_smoke\checkpoints\000020\pretrained_model --policy-device cuda --episodes-per-task 8 --output-dir results\continuous_eval\act
uv run python scripts\evaluate_continuous_backbone.py --config configs\continuous_diffusion_template.yaml --policy-path outputs\train\lerobot\diffusion_v1_smoke\checkpoints\000010\pretrained_model --policy-device cuda --episodes-per-task 8 --output-dir results\continuous_eval\diffusion
uv run python scripts\evaluate_continuous_backbone.py --config configs\continuous_smolvla_template.yaml --policy-path external_models\smolvla_base --policy-device cuda --episodes-per-task 8 --output-dir results\continuous_eval\smolvla
```

Example suite evaluation:

```powershell
$env:UV_CACHE_DIR='F:\RoboticArm\.uv-cache'
$env:TRANSFORMERS_OFFLINE='1'
$env:HF_HUB_OFFLINE='1'
uv run python scripts\run_continuous_eval_suite.py --act-policy-path outputs\train\lerobot\act_v1_smoke\checkpoints\000020\pretrained_model --diffusion-policy-path outputs\train\lerobot\diffusion_v1_smoke\checkpoints\000010\pretrained_model --smolvla-policy-path external_models\smolvla_base --episodes-per-task 8 --output-root results\continuous_eval_suite\v1
```

Example checkpoint selection:

```powershell
$env:UV_CACHE_DIR='F:\RoboticArm\.uv-cache'
uv run python scripts\select_continuous_checkpoint.py --config configs\continuous_act_template.yaml --train-output-dir outputs\train\lerobot\act_v1_official --episodes-per-task 8 --policy-device cuda --output-root results\continuous_eval\act_v1_selection
```

The local prototype trainer under:

- [src/ttla/training/train_continuous.py](/F:/RoboticArm/src/ttla/training/train_continuous.py)

is retained only for internal bring-up. It must not be used for reported
baseline numbers when `control.official.enforce_official_training` is enabled.

## Required Runtime Caches on This Machine

Official backbone loading currently needs workspace-local caches to avoid
Windows permission failures:

- `HF_HOME=F:\RoboticArm\.hf-home`
- `TORCH_HOME=F:\RoboticArm\.torch-home`
- `UV_CACHE_DIR=F:\RoboticArm\.uv-cache`
- `PYTHONIOENCODING=utf-8`

The official launcher script sets these automatically during training. Manual
ad-hoc runs should set the same paths explicitly.

The launcher additionally supports explicit cache overrides:

- `--hf-home`
- `--torch-home`
- `--uv-cache-dir`
- `--offline`

This is useful when a large upstream checkpoint leaves a partially written cache
entry behind and a fresh cache root is needed for retry.

When `policy_path` points to a local directory such as
`external_models/smolvla_base`, the launcher now automatically enables:

- `TRANSFORMERS_OFFLINE=1`
- `HF_HUB_OFFLINE=1`

This avoids accidental online weight resolution during official training.

## Current Smoke Validation Status

The current continuous-control mainline has been validated as follows:

- `ACT`
  - official local LeRobot dataset export: passed
  - official `lerobot-train` smoke run: passed
- `Diffusion Policy`
  - official local LeRobot dataset export: passed
  - official `lerobot-train` smoke run: passed
- `SmolVLA`
  - official wrapper and launcher path: wired
  - official upstream weights mirrored locally under:
    - `external_models/smolvla_base`
    - `external_models/SmolVLM2-500M-Video-Instruct`
  - local offline `SmolVLAPolicy.from_pretrained(...)` load: passed
  - backbone-specific dataset export path added to:
    - [scripts/export_continuous_to_lerobot.py](/F:/RoboticArm/scripts/export_continuous_to_lerobot.py)
  - `smolvla` schema export now:
    - duplicates the single camera into
      `observation.images.camera1/2/3`
    - resizes images to `256x256`
    - reduces `observation.state` to 6-D `qpos`
  - official 1-step `lerobot-train` smoke run on the schema-adapted dataset:
    passed

The local `smolvla_base/config.json` has been patched so `vlm_model_name`
points to the mirrored local `SmolVLM2-500M-Video-Instruct` directory. This
keeps the model structure unchanged while avoiding the Windows Hugging Face
cache finalization failure seen during direct online loading.

## Near-Term Rule

Until the new continuous pipeline is fully functional:

- do not delete the validated primitive baseline
- do not modify the validated primitive scripts unless a fatal bug appears
- treat the primitive path as a reference system during migration


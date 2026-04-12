# TTLA for RoArm-M3

This repository contains a reproducible minimal research pipeline for:

- simulation-first training of short-horizon primitive policies,
- adapter-only real-world latent calibration,
- pseudo-real and real-robot evaluation on Waveshare RoArm-M3-S,
- result plotting and an IEEE Transactions on Consumer Electronics manuscript draft.

## Scope

The project targets three primitive-level desktop tasks:

- `level1_verify`: observation and verification,
- `level2_approach`: observation, pre-alignment, and approach,
- `level3_pick_place`: reobserve, pregrasp servo, grasp, lift, move, and place.

The method trains an encoder, a primitive-selection policy, and an action-conditioned transition model in simulation. At deployment, the backbone stays frozen and only a lightweight adapter is calibrated so real observations map back into the simulation latent space.

## Simulated Device

The simulator models a simplified Waveshare RoArm-M3-S desktop robot with:

- 6 controllable joints (`base`, `shoulder`, `elbow`, `wrist pitch`, `wrist roll`, `gripper`),
- a forearm-mounted monocular camera attached through an AI-kit style bracket,
- a single LED work light near the gripper,
- and a short-horizon tabletop workspace with lightweight visual domain shifts and control perturbations.

The robot geometry, joint limits, and main link dimensions are aligned with the public RoArm ROS description, while the simulator intentionally remains lighter than a full digital twin.

## Repository Layout

- `configs/`: experiment and deployment configs.
- `src/ttla/`: simulator, models, training, evaluation, adaptation, deployment interfaces.
- `scripts/`: runnable entry points.
- `data/`: generated simulation datasets and real probe logs.
- `results/`: CSV summaries, plots, and run artifacts.
- `manuscript/`: IEEE-style LaTeX paper.

## Environment Setup

Initialize the Python environment with `uv`:

```bash
uv sync
```

This creates the project virtual environment and installs the pinned dependencies from `pyproject.toml` and `uv.lock`.

If you want to verify that MuJoCo and the simulator import correctly before training, run:

```bash
uv run python scripts/play_env.py --config configs/base.yaml --mode expert --task level1_verify --disable-gui --save-dir results/replays/smoke
```

## Simulator Visualization

If you only want one command to inspect the simulator, use:

```bash
uv run python scripts/play_env_viewer.py --config configs/base.yaml --mode manual --task level1_verify
```

This is the recommended visualization entry point for this repository because it combines:

- the MuJoCo native 3D viewer,
- the TTLA task loop,
- manual primitive keyboard control,
- and the forearm camera feed.

There are three different simulator visualization modes in this repository. They look similar, but their interaction model is different.

### 1. Recommended: MuJoCo Native Viewer With TTLA Control

Use this when you want a native MuJoCo-looking window and still want to run our task and policy loop.

```bash
uv run python scripts/play_env_viewer.py --config configs/base.yaml --mode manual --task level2_approach --save-dir results/replays/viewer_demo
```

This mode uses the native MuJoCo viewer as the main window and overlays TTLA status information on top of it.

What this mode is for:

- comparing skill execution inside a near-native MuJoCo viewer,
- running manual, expert, or policy control with episode logging,
- debugging the research loop while keeping MuJoCo's 3D interaction style,
- checking the forearm camera viewpoint while you adjust its mount pose in the MJCF.

Available commands:

- manual mode:

```bash
uv run python scripts/play_env_viewer.py --config configs/base.yaml --mode manual --task level2_approach --save-dir results/replays/viewer_demo
```

- expert autoplay:

```bash
uv run python scripts/play_env_viewer.py --config configs/base.yaml --mode expert --task level3_pick_place --save-dir results/replays/viewer_expert_demo
```

- policy autoplay:

```bash
uv run python scripts/play_env_viewer.py --config configs/base.yaml --mode policy --baseline ours --checkpoint results/checkpoints/best_model.pt --task level3_pick_place --save-dir results/replays/viewer_policy_demo
```

- hide MuJoCo's left or right UI:

```bash
uv run python scripts/play_env_viewer.py --config configs/base.yaml --mode manual --task level1_verify --hide-left-ui --hide-right-ui
```

Primitive controls added by this mode:

- `a` / `d` / `c`: observe left / right / center
- `v`: verify target
- `p`: prealign grasp
- `e` / `r`: approach coarse / fine
- `t`: retreat
- `o`: reobserve
- `g`: pregrasp servo
- `f`: grasp execute
- `l`: lift object
- `m`: transport to dropzone
- `y`: place object
- `x`: abort
- `n`: save current episode and reset
- `q` or `Esc`: save and quit

MuJoCo interaction behavior in this mode:

- free camera movement still works,
- the viewer's own actuator sliders are live,
- the left and right MuJoCo UI panels still work,
- the visual style is close to native MuJoCo viewer.

Limitations of this mode:

- this is a **passive viewer**, not a standalone native `simulate` process,
- the Python script owns the task loop and stepping logic,
- some native viewer shortcuts such as full play / pause semantics are not equivalent to pure MuJoCo viewer mode,
- dragging benchmark objects is not guaranteed to work as in native simulate because the task objects are part of the scripted benchmark setup.

### 2. Pure MuJoCo Viewer

Use this when you want the closest behavior to native `simulate` / `viewer`.

```bash
uv run python -m mujoco.viewer --mjcf src/ttla/sim/mjcf/roarm_simplified.xml
```

What this mode is for:

- inspecting the robot model and world configuration,
- checking joint ranges and actuator sliders,
- using MuJoCo's own camera and UI controls,
- getting the most native viewer experience.

What works here:

- MuJoCo's built-in camera interaction,
- MuJoCo's own control panel sliders,
- MuJoCo's own play / pause / stepping controls,
- MuJoCo's own UI panels and overlays.

Important note:

- This mode does **not** run the TTLA policy, adaptation loop, episode logger, or task logic.
- This mode also does **not** automatically show the forearm camera as a separate panel. Use `scripts/play_env_viewer.py` if you want the task loop plus the forearm camera feed.

### 3. OpenCV Dashboard Player

Use this when you want the clearest task-oriented dashboard instead of the most native MuJoCo UI.

```bash
uv run python scripts/play_env.py --config configs/base.yaml --mode manual --task level2_approach --save-dir results/replays/manual_demo
```

This player renders the forearm camera and an overview camera to a custom primitive dashboard with action history and task-state metrics.

Available commands:

- manual mode:

```bash
uv run python scripts/play_env.py --config configs/base.yaml --mode manual --task level2_approach --save-dir results/replays/manual_demo
```

- expert autoplay:

```bash
uv run python scripts/play_env.py --config configs/base.yaml --mode expert --task level3_pick_place --save-dir results/replays/expert_demo
```

- policy autoplay:

```bash
uv run python scripts/play_env.py --config configs/base.yaml --mode policy --baseline ours --checkpoint results/checkpoints/best_model.pt --task level3_pick_place --save-dir results/replays/ours_demo
```

Dashboard controls:

- `a` / `d` / `c`: observe left / right / center
- `v`: verify target
- `p`: prealign grasp
- `e` / `r`: approach coarse / fine
- `t`: retreat
- `o`: reobserve
- `g`: pregrasp servo
- `f`: grasp execute
- `l`: lift object
- `m`: transport to dropzone
- `y`: place object
- `x`: abort
- `n`: save current episode and reset
- `q`: save and quit

Manual mode is step-based: each key press triggers one primitive.

### Replay

Replay a saved episode:

```bash
uv run python scripts/replay_episode.py --episode-dir results/replays/ours_demo/episode_YYYYMMDD_HHMMSS
```

Replay controls:

- `space`: pause or resume
- `a` / `d`: previous or next frame
- `q`: quit

### Which Mode Should I Use?

- Use `scripts/play_env_viewer.py` when you want the main recommended simulator entry point.
- Use `python -m mujoco.viewer --mjcf ...` when you want the most native MuJoCo interaction and UI.
- Use `scripts/play_env.py` when you want the clearest primitive-task dashboard and explicit forearm camera visualization.

### Why Some Simulate Interactions Do Not Work In Scripted Modes

This is the main distinction that caused the confusion:

- `python -m mujoco.viewer --mjcf ...` is a pure native MuJoCo viewer process.
- `scripts/play_env_viewer.py` is a Python-controlled passive viewer.
- `scripts/play_env.py` is a fully custom dashboard.

So:

- native play / pause and some simulate shortcuts only behave exactly as expected in pure MuJoCo viewer mode,
- the scripted viewer keeps MuJoCo's camera and slider interaction, but the task loop is still controlled by Python,
- benchmark objects are currently part of a fixed scripted environment, so dragging them is not equivalent to interactive sandbox editing.

For non-interactive smoke tests, append `--disable-gui` to either visualization script.

## Quick Start

```bash
uv sync
uv run python scripts/generate_sim_data.py --config configs/base.yaml
uv run python scripts/train_main.py --config configs/base.yaml
uv run python scripts/run_experiments.py --config configs/base.yaml
uv run python scripts/plot_results.py --config configs/base.yaml
```

## Primitive Task Interface

All current visualizers and training scripts assume a discrete primitive policy. The three task levels share one primitive vocabulary:

- Observation primitives:
  - `obs_left`
  - `obs_right`
  - `obs_center`
  - `verify_target`
- Approach / recovery primitives:
  - `prealign_grasp`
  - `approach_coarse`
  - `approach_fine`
  - `retreat`
  - `reobserve`
- Manipulation primitives:
  - `pregrasp_servo`
  - `grasp_execute`
  - `lift_object`
  - `transport_to_dropzone`
  - `place_object`
- Termination primitive:
  - `abort`

Level-1 uses observation and verification only. Level-2 adds alignment and approach. Level-3 adds continuous closed-loop execution inside `pregrasp_servo`, `grasp_execute`, `lift_object`, `transport_to_dropzone`, and `place_object`, while the high-level policy still selects only a discrete primitive ID.

## Controller Backbones

The current codebase supports multiple policy backbones under the same `encoder -> policy -> transition -> adapter` method structure. Switch them by editing `model.backbone_type` in [configs/base.yaml](/F:/RoboticArm/configs/base.yaml).

Supported values:

- `feedforward`
  - `z_t`: fused image-state latent for the current observation.
  - `a_t`: current primitive ID.
  - `T(z_t, a_t)`: one-step latent transition.
- `recurrent`
  - `z_t`: GRU hidden state after consuming the recent observation history.
  - `a_t`: current primitive ID.
  - `T(z_t, a_t)`: one-step latent transition from the recurrent latent.
- `chunking`
  - `z_t`: fused image-state latent for the current decision point.
  - `a_t`: first primitive in the predicted primitive chunk.
  - `T(z_t, a_t)`: one-step latent transition conditioned on the executed primitive, while the policy also predicts future primitive slots.
- `language`
  - `z_t`: fused image-state-task latent, where task identity is embedded as a lightweight language/task token.
  - `a_t`: current primitive ID.
  - `T(z_t, a_t)`: one-step latent transition.
- `diffusion`
  - `z_t`: fused image-state latent.
  - `a_t`: primitive sampled by iterative denoising over a primitive action embedding.
  - `T(z_t, a_t)`: one-step latent transition.

The adapter and transition training logic remain the same across all backbones. Only the policy backbone changes.

## Real-Robot Workflow

1. Mount the USB camera on the RoArm forearm.
2. Connect the RoArm USB serial interface.
3. Update `configs/deployment.yaml` with the correct camera index and serial port.
4. Run:

```bash
uv run python scripts/deploy_roarm.py --config configs/deployment.yaml --mode probe
```

Live monitoring for the physical RoArm:

```bash
uv run python scripts/live_monitor_roarm.py --config configs/deployment.yaml
```

The live monitor shows the USB camera, current joint targets, last serial command, and last feedback line. Controls:

- `a` / `d`: base target minus / plus
- `w` / `s`: shoulder target minus / plus
- `z` / `x`: elbow target minus / plus
- `f`: request feedback
- `r`: reset pose
- `p`: save a snapshot into the current monitor session
- `q`: quit

If you only want to check the camera feed without opening the serial port:

```bash
uv run python scripts/live_monitor_roarm.py --config configs/deployment.yaml --camera-only
```

For non-interactive smoke tests, append `--disable-gui` to either visualization script.
For the live monitor, use `--dry-run` to verify configuration and output paths without touching hardware.

## Notes

- The simulator is intentionally simplified. It is designed for method comparison, not high-fidelity digital twinning.
- When real hardware is unavailable, the evaluation scripts fall back to pseudo-real context shifts.

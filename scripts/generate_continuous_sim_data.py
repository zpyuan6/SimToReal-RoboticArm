from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ttla.config import load_config
from ttla.sim import ContinuousRoArmSimEnv, ContinuousWaypointExpert
from ttla.sim.context import full_context_vector, neutral_context
from ttla.utils.io import ensure_dir, save_npz


def _save_bundle(path: Path, payload: dict[str, np.ndarray], compression: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compression == "compressed":
        save_npz(path, **payload)
        return
    np.savez(path, **payload)


def _hold_action_from_transition(transition, control_mode: str) -> np.ndarray:
    if str(control_mode) == "joint_target":
        return np.asarray(transition.next_proprio[:6], dtype=np.float32).copy()
    return np.zeros_like(np.asarray(transition.action, dtype=np.float32))


def collect_split(
    env: ContinuousRoArmSimEnv,
    expert: ContinuousWaypointExpert,
    episodes: int,
    *,
    split_name: str,
    log_every: int,
    success_only: bool,
    max_attempts_per_episode: int,
    context_mode: str,
    l1_terminal_hold_steps: int,
) -> dict[str, np.ndarray]:
    images = []
    proprio = []
    actions = []
    tasks = []
    success = []
    contexts = []
    contexts_full = []
    target_init_pos = []
    drop_init_pos = []
    episode_ids = []
    step_ids = []
    task_text = []
    task_names = env.cfg["tasks"]
    task_targets = {task_name: episodes // len(task_names) for task_name in task_names}
    for task_name in task_names[: episodes % len(task_names)]:
        task_targets[task_name] += 1
    accepted = 0
    attempt_counts = {task_name: 0 for task_name in task_names}
    accepted_counts = {task_name: 0 for task_name in task_names}
    for task_name in task_names:
        while accepted_counts[task_name] < task_targets[task_name]:
            attempt_counts[task_name] += 1
            if attempt_counts[task_name] > task_targets[task_name] * max(1, max_attempts_per_episode):
                raise RuntimeError(
                    f"[{split_name}] expert could not generate enough successful {task_name} episodes: "
                    f"{accepted_counts[task_name]}/{task_targets[task_name]} accepted after {attempt_counts[task_name]} attempts"
                )
            context = neutral_context() if context_mode == "neutral" else None
            env.reset(task_name=task_name, context=context)
            expert.reset(task_name=task_name)
            episode_context_full = full_context_vector(env.context)
            episode_target_init = env._target_body_position().astype(np.float32).copy()
            episode_drop_init = env._dropzone_position().astype(np.float32).copy()
            done = False
            step_idx = 0
            episode_transitions = []
            episode_success = 0
            while not done:
                action = expert.act(env)
                _, _, done, info = env.step_action(action)
                episode_transitions.append(info["transition"])
                episode_success = max(episode_success, int(info["success"]))
                step_idx += 1
            if success_only and not episode_success:
                continue
            if task_name == "level1_verify" and episode_success and episode_transitions and int(l1_terminal_hold_steps) > 0:
                final_transition = episode_transitions[-1]
                hold_action = _hold_action_from_transition(final_transition, env.control_mode)
                for _ in range(int(l1_terminal_hold_steps)):
                    episode_transitions.append(
                        type(final_transition)(
                            image=final_transition.next_image.copy(),
                            proprio=final_transition.next_proprio.copy(),
                            action=hold_action.copy(),
                            next_image=final_transition.next_image.copy(),
                            next_proprio=final_transition.next_proprio.copy(),
                            task_id=final_transition.task_id,
                            success=1,
                            context=final_transition.context.copy(),
                        )
                    )
            if accepted == 0 or ((accepted + 1) % max(1, log_every) == 0) or accepted + 1 == episodes:
                print(
                    f"[{split_name}] accepted {accepted + 1}/{episodes} "
                    f"({task_name} attempt {attempt_counts[task_name]} success={episode_success})",
                    flush=True,
                )
            episode_id = accepted
            for local_step_idx, transition in enumerate(episode_transitions):
                images.append(transition.image)
                proprio.append(transition.proprio)
                actions.append(transition.action)
                tasks.append(transition.task_id)
                success.append(transition.success)
                contexts.append(transition.context)
                contexts_full.append(episode_context_full)
                target_init_pos.append(episode_target_init)
                drop_init_pos.append(episode_drop_init)
                episode_ids.append(episode_id)
                step_ids.append(local_step_idx)
                task_text.append(TASK_TEXT_BY_ID[int(transition.task_id)])
            accepted_counts[task_name] += 1
            accepted += 1
    print(
        f"[{split_name}] accepted_counts={accepted_counts} attempt_counts={attempt_counts}",
        flush=True,
    )
    return {
        "images": np.asarray(images, dtype=np.uint8),
        "proprio": np.asarray(proprio, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "tasks": np.asarray(tasks, dtype=np.int64),
        "success": np.asarray(success, dtype=np.int64),
        "contexts": np.asarray(contexts, dtype=np.float32),
        "contexts_full": np.asarray(contexts_full, dtype=np.float32),
        "target_init_pos": np.asarray(target_init_pos, dtype=np.float32),
        "drop_init_pos": np.asarray(drop_init_pos, dtype=np.float32),
        "episode_ids": np.asarray(episode_ids, dtype=np.int64),
        "step_ids": np.asarray(step_ids, dtype=np.int64),
        "task_text": np.asarray(task_text, dtype=object),
    }


TASK_TEXT_BY_ID = {
    0: "center the target object in the camera view",
    1: "move the gripper into a stable pre-grasp approach state",
    2: "pick up the object and place it in the blue drop zone",
}


def _expert_kwargs(cfg: dict) -> dict[str, float]:
    raw = cfg.get("continuous_expert") or cfg.get("teacher") or cfg.get("sim", {}).get("expert") or {}
    allowed = {"max_arm_delta", "max_gripper_delta", "servo_gain", "damping"}
    return {key: float(raw[key]) for key in allowed if key in raw}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/continuous_act_template.yaml")
    parser.add_argument("--output-root", default="data/continuous")
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    parser.add_argument("--episodes", type=int, help="Episode count for a single split run")
    parser.add_argument("--train-episodes", type=int, default=60)
    parser.add_argument("--val-episodes", type=int, default=12)
    parser.add_argument("--test-episodes", type=int, default=18)
    parser.add_argument("--compression", choices=("compressed", "raw"), default="compressed")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--context-mode", choices=("neutral", "random"), default="neutral")
    parser.add_argument("--max-attempts-per-episode", type=int, default=20)
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--l1-terminal-hold-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, help="Override config seed for this generation run.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 7))
    np.random.seed(seed)
    control_cfg = cfg["control"]
    env = ContinuousRoArmSimEnv(
        cfg["sim"],
        seed=seed,
        action_low=np.asarray(control_cfg["action"].get("clamp_low", [-0.25] * 6), dtype=np.float32),
        action_high=np.asarray(control_cfg["action"].get("clamp_high", [0.25] * 6), dtype=np.float32),
        control_mode=control_cfg["action"].get("control_mode", "joint_delta"),
    )
    expert = ContinuousWaypointExpert(**_expert_kwargs(cfg))
    output_root = ensure_dir(args.output_root)
    splits = {
        "train": int(args.train_episodes),
        "val": int(args.val_episodes),
        "test": int(args.test_episodes),
    }
    target_splits = splits.keys() if args.split == "all" else (args.split,)
    for split in target_splits:
        count = int(args.episodes) if args.episodes is not None else splits[split]
        payload = collect_split(
            env,
            expert,
            episodes=count,
            split_name=split,
            log_every=int(args.log_every),
            success_only=not bool(args.allow_failures),
            max_attempts_per_episode=int(args.max_attempts_per_episode),
            context_mode=str(args.context_mode),
            l1_terminal_hold_steps=int(args.l1_terminal_hold_steps),
        )
        _save_bundle(
            Path(output_root) / f"{split}.npz",
            payload,
            compression=args.compression,
        )
    env.close()


if __name__ == "__main__":
    main()

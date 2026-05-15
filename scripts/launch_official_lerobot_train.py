from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from ttla.config import load_config
from ttla.utils.io import ensure_dir


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _cache_env(
    root: Path,
    hf_home_override: str | None = None,
    torch_home_override: str | None = None,
    uv_cache_override: str | None = None,
    offline: bool = False,
) -> dict[str, str]:
    hf_home = ensure_dir(Path(hf_home_override) if hf_home_override else root / ".hf-home")
    torch_home = ensure_dir(Path(torch_home_override) if torch_home_override else root / ".torch-home")
    uv_cache = ensure_dir(Path(uv_cache_override) if uv_cache_override else root / ".uv-cache")
    env = os.environ.copy()
    env.setdefault("HF_HOME", str(hf_home))
    env.setdefault("TORCH_HOME", str(torch_home))
    env.setdefault("UV_CACHE_DIR", str(uv_cache))
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if offline:
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        env.setdefault("HF_HUB_OFFLINE", "1")
    return env


def _policy_path_is_local(root: Path, cfg: dict) -> bool:
    policy_path = cfg["control"]["official"].get("policy_path")
    if not policy_path:
        return False
    candidate = Path(policy_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.exists()


def resolve_resume_config_path(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    candidate = output_dir / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
    if not candidate.exists():
        raise FileNotFoundError(f"Resume config not found: {candidate}")
    return candidate


def build_lerobot_command(cfg: dict, overrides: dict[str, object] | None = None) -> list[str]:
    control_cfg = cfg["control"]
    official_cfg = control_cfg["official"]
    train_cfg = dict(cfg["official_train"])
    dataset_cfg = dict(train_cfg["dataset"])
    overrides = overrides or {}
    for key in (
        "batch_size",
        "steps",
        "output_dir",
        "job_name",
        "policy_device",
        "wandb_enable",
        "save_freq",
        "eval_freq",
    ):
        if key in overrides and overrides[key] is not None:
            train_cfg[key] = overrides[key]
    for key, dataset_key in (("dataset_repo_id", "repo_id"), ("dataset_root", "root")):
        if key in overrides and overrides[key] is not None:
            dataset_cfg[dataset_key] = overrides[key]
    resume = bool(overrides.get("resume", False))
    resume_config_path = overrides.get("resume_config_path")
    if resume and resume_config_path is None:
        resume_config_path = resolve_resume_config_path(train_cfg["output_dir"])
    cmd = [
        "uv",
        "run",
        "lerobot-train",
        f"--dataset.repo_id={dataset_cfg['repo_id']}",
        f"--dataset.root={dataset_cfg['root']}",
        f"--batch_size={train_cfg['batch_size']}",
        f"--steps={train_cfg['steps']}",
        f"--output_dir={train_cfg['output_dir']}",
        f"--job_name={train_cfg['job_name']}",
        f"--policy.device={train_cfg['policy_device']}",
        f"--wandb.enable={str(bool(train_cfg.get('wandb_enable', False))).lower()}",
        "--policy.push_to_hub=false",
    ]
    if train_cfg.get("save_freq") is not None:
        cmd.append(f"--save_freq={train_cfg['save_freq']}")
    if train_cfg.get("eval_freq") is not None:
        cmd.append(f"--eval_freq={train_cfg['eval_freq']}")
    if resume:
        cmd.append("--resume=true")
        cmd.append(f"--config_path={resume_config_path}")
    else:
        policy_path = official_cfg.get("policy_path")
        if policy_path:
            cmd.append(f"--policy.path={policy_path}")
        else:
            cmd.append(f"--policy.type={official_cfg['policy_type']}")
        for key, value in official_cfg.get("config_overrides", {}).items():
            if key == "device":
                continue
            cmd.append(f"--policy.{key}={value}")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/continuous_act_template.yaml")
    parser.add_argument("--run", action="store_true", help="Actually launch lerobot-train")
    parser.add_argument("--steps", type=int, help="Override training steps for a smoke or short run")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument("--job-name", help="Override job name")
    parser.add_argument("--policy-device", help="Override policy device")
    parser.add_argument("--dataset-root", help="Override dataset.root for this run")
    parser.add_argument("--dataset-repo-id", help="Override dataset.repo_id for this run")
    parser.add_argument("--save-freq", type=int, help="Override checkpoint save frequency")
    parser.add_argument("--eval-freq", type=int, help="Override official trainer eval frequency")
    parser.add_argument("--resume", action="store_true", help="Resume an existing official training run")
    parser.add_argument("--resume-config-path", help="Explicit path to checkpoint train_config.json for resume")
    parser.add_argument("--hf-home", help="Override HF_HOME for this run")
    parser.add_argument("--torch-home", help="Override TORCH_HOME for this run")
    parser.add_argument("--uv-cache-dir", help="Override UV_CACHE_DIR for this run")
    parser.add_argument("--offline", action="store_true", help="Force offline Hugging Face / Transformers mode")
    parser.add_argument("--detach", action="store_true", help="Spawn lerobot-train as a detached background process")
    parser.add_argument("--stdout-log", help="Stdout log file for detached runs")
    parser.add_argument("--stderr-log", help="Stderr log file for detached runs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = _project_root()
    cmd = build_lerobot_command(
        cfg,
        overrides={
            "steps": args.steps,
            "batch_size": args.batch_size,
            "output_dir": args.output_dir,
            "job_name": args.job_name,
            "policy_device": args.policy_device,
            "dataset_root": args.dataset_root,
            "dataset_repo_id": args.dataset_repo_id,
            "save_freq": args.save_freq,
            "eval_freq": args.eval_freq,
            "resume": args.resume,
            "resume_config_path": args.resume_config_path,
        },
    )
    print(" ".join(cmd))
    if not args.run:
        return
    offline = args.offline or _policy_path_is_local(root, cfg)
    env = _cache_env(
        root,
        hf_home_override=args.hf_home,
        torch_home_override=args.torch_home,
        uv_cache_override=args.uv_cache_dir,
        offline=offline,
    )
    if args.detach:
        job_name = args.job_name or cfg["official_train"]["job_name"]
        stdout_target = args.stdout_log or str(root / "outputs" / "train" / "lerobot" / f"{job_name}.out.log")
        stderr_target = args.stderr_log or str(root / "outputs" / "train" / "lerobot" / f"{job_name}.err.log")
        creationflags = 0
        if os.name == "nt":
            creationflags = (
                subprocess.DETACHED_PROCESS
                | subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.CREATE_NO_WINDOW
            )
        stdout_handle = subprocess.DEVNULL
        stderr_handle = subprocess.DEVNULL
        stdout_log_path = None
        stderr_log_path = None
        if stdout_target.upper() not in {"NUL", "DEVNULL"}:
            stdout_log_path = Path(stdout_target)
            stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
            stdout_handle = stdout_log_path.open("ab")
        if stderr_target.upper() not in {"NUL", "DEVNULL"}:
            stderr_log_path = Path(stderr_target)
            stderr_log_path.parent.mkdir(parents=True, exist_ok=True)
            stderr_handle = stderr_log_path.open("ab")
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=root,
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                creationflags=creationflags,
                close_fds=False,
            )
        finally:
            if stdout_log_path is not None:
                stdout_handle.close()
            if stderr_log_path is not None:
                stderr_handle.close()
        print(f"DETACHED_PID={proc.pid}")
        print(f"STDOUT_LOG={stdout_target}")
        print(f"STDERR_LOG={stderr_target}")
        return
    subprocess.run(cmd, check=True, env=env, cwd=root)


if __name__ == "__main__":
    main()

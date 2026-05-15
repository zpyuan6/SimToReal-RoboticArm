from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--stdout-log", required=True)
    parser.add_argument("--stderr-log", required=True)
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Environment override in KEY=VALUE form; can be repeated",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        raise SystemExit("No command provided")
    return args


def build_env(overrides: list[str]) -> dict[str, str]:
    env = os.environ.copy()
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"Invalid --env override: {item}")
        key, value = item.split("=", 1)
        env[key] = value
    return env


def main() -> None:
    args = parse_args()
    cwd = Path(args.cwd)
    stdout_log = Path(args.stdout_log)
    stderr_log = Path(args.stderr_log)
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stderr_log.parent.mkdir(parents=True, exist_ok=True)
    env = build_env(args.env)

    creationflags = 0
    if os.name == "nt":
        creationflags = (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        )

    with stdout_log.open("ab") as stdout_handle, stderr_log.open("ab") as stderr_handle:
        proc = subprocess.Popen(
            args.command,
            cwd=str(cwd),
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            creationflags=creationflags,
            close_fds=False,
        )
    print(proc.pid)


if __name__ == "__main__":
    main()

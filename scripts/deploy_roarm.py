from __future__ import annotations

import argparse

from ttla.config import load_config
from ttla.deployment import DeploymentRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", default="probe", choices=["probe"])
    args = parser.parse_args()
    cfg = load_config(args.config)
    runner = DeploymentRunner(cfg)
    try:
        if args.mode == "probe":
            path = runner.run_probe_episode()
            print(f"log_dir={path}")
    finally:
        runner.close()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import torch

from ttla.config import load_config
from ttla.deployment import DeploymentRunner
from ttla.models import TTLAModel, load_model_state

if TYPE_CHECKING:
    from ttla.training import build_model


def _load_model(cfg: dict, checkpoint_path: str, device: str) -> TTLAModel:
    from ttla.training import build_model

    payload = torch.load(checkpoint_path, map_location=device)
    model = build_model(cfg).to(device)
    load_model_state(model, payload["model_state"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--deploy-config", default="configs/deployment.yaml")
    parser.add_argument("--mode", default="probe", choices=["probe", "sequence", "policy"])
    parser.add_argument("--checkpoint", default="results/checkpoints/best_model.pt")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--primitives", default="0,2,3,14")
    args = parser.parse_args()
    cfg = load_config(args.config)
    deploy_cfg = load_config(args.deploy_config)
    model = _load_model(cfg, args.checkpoint, cfg["train"]["device"]) if args.mode == "policy" else None
    runner = DeploymentRunner(deploy_cfg, model=model, device=cfg["train"]["device"])
    try:
        if args.mode == "probe":
            path = runner.run_probe_episode()
        elif args.mode == "sequence":
            primitive_ids = [int(item.strip()) for item in args.primitives.split(",") if item.strip()]
            path = runner.run_primitive_sequence(primitive_ids)
        else:
            path = runner.run_policy_episode(task_id=args.task_id)
        print(f"log_dir={path}")
    finally:
        runner.close()


if __name__ == "__main__":
    main()

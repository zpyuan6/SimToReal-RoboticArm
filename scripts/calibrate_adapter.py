from __future__ import annotations

import argparse

from ttla.config import load_config
from ttla.training import calibrate_adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="results/checkpoints/best_model.pt")
    parser.add_argument("--real-data", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    path = calibrate_adapter(cfg, args.checkpoint, args.real_data)
    print(f"saved_adapter={path}")


if __name__ == "__main__":
    main()

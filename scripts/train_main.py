from __future__ import annotations

import argparse

from ttla.config import load_config
from ttla.data import load_split
from ttla.training import train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_path = load_split(cfg["paths"]["data_root"], "train")
    val_path = load_split(cfg["paths"]["data_root"], "val")
    checkpoint = train_model(cfg, train_path, val_path)
    print(f"saved_checkpoint={checkpoint}")


if __name__ == "__main__":
    main()

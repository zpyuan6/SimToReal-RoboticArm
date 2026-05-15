from __future__ import annotations

import argparse

from ttla.training.train_continuous import train_continuous_backbone


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/continuous_act_template.yaml")
    args = parser.parse_args()
    history = train_continuous_backbone(args.config)
    print(history)


if __name__ == "__main__":
    main()

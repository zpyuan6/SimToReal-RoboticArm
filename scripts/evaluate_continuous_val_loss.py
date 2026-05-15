from __future__ import annotations

import argparse
from pathlib import Path

from ttla.config import load_config
from ttla.evaluation.evaluate_continuous_val_loss import evaluate_continuous_validation_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate continuous backbone offline validation action loss.")
    parser.add_argument("--config", required=True, help="Continuous backbone config.")
    parser.add_argument("--policy-path", required=True, help="Official checkpoint dir or training output root.")
    parser.add_argument("--output-dir", required=True, help="Directory for validation loss CSV.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override validation batch size.")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional cap on validation batches.")
    parser.add_argument("--policy-device", default=None, help="Override inference device.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    summary_path = evaluate_continuous_validation_loss(
        cfg,
        policy_path=args.policy_path,
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        policy_device=args.policy_device,
    )
    print(f"val_loss_summary={summary_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _infer_method(run_name: str) -> str:
    for method in ("no_adaptation", "static_adapter", "few_shot_finetuning", "plica"):
        if f"_{method}_" in run_name or run_name.startswith(method):
            return method
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize repeated-trial real L3 policy-mode runs.")
    parser.add_argument("--output-root", default="results/real_deployment_eval")
    parser.add_argument("--run-prefix", default="l3_policy")
    parser.add_argument("--output-csv", default="results/real_deployment_eval/l3_policy_baseline_summary.csv")
    args = parser.parse_args()

    output_root = ROOT / args.output_root
    rows: list[dict[str, object]] = []

    for run_dir in sorted(output_root.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith(args.run_prefix):
            continue
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows.append(
            {
                "run_name": run_dir.name,
                "method": _infer_method(run_dir.name),
                "task": payload.get("task", ""),
                "episodes_total": payload.get("episodes_total", ""),
                "episodes_success": payload.get("episodes_success", ""),
                "episodes_fail": payload.get("episodes_fail", ""),
                "episodes_partial": payload.get("episodes_partial", ""),
                "episode_success_rate": payload.get("episode_success_rate", ""),
                "checkpoint": payload.get("checkpoint", ""),
                "summary_json": str(summary_path),
            }
        )

    output_csv = ROOT / args.output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_name",
                "method",
                "task",
                "episodes_total",
                "episodes_success",
                "episodes_fail",
                "episodes_partial",
                "episode_success_rate",
                "checkpoint",
                "summary_json",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()

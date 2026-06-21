from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize IEEE-style continuous sim-to-real protocol outputs with repeated-seed statistics "
            "and paired method comparisons."
        )
    )
    parser.add_argument("--input-root", default="results/paper_continuous_sim2real_protocol")
    parser.add_argument("--success-csv", default=None)
    parser.add_argument("--transition-csv", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--primary-method", default="ours_multimodel_adaptive")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260603)
    return parser.parse_args()


def _stats_frame(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict[str, object] = {col: value for col, value in zip(group_cols, keys)}
        row["n"] = int(len(group))
        for value_col in value_cols:
            values = pd.to_numeric(group[value_col], errors="coerce").dropna().to_numpy(dtype=np.float64)
            n = int(len(values))
            if n == 0:
                row[f"{value_col}_mean"] = np.nan
                row[f"{value_col}_std"] = np.nan
                row[f"{value_col}_sem"] = np.nan
                row[f"{value_col}_ci95_low"] = np.nan
                row[f"{value_col}_ci95_high"] = np.nan
                continue
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if n > 1 else 0.0
            sem = float(std / np.sqrt(n)) if n > 1 else 0.0
            ci = 1.96 * sem
            row[f"{value_col}_mean"] = mean
            row[f"{value_col}_std"] = std
            row[f"{value_col}_sem"] = sem
            row[f"{value_col}_ci95_low"] = mean - ci
            row[f"{value_col}_ci95_high"] = mean + ci
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def _add_success_rank(stats: pd.DataFrame) -> pd.DataFrame:
    if stats.empty or "success_mean" not in stats.columns:
        return stats
    rank_cols = [col for col in ["config_label", "eval_split", "eval_type", "split", "task"] if col in stats.columns]
    out = stats.copy()
    out = out.sort_values(rank_cols + ["success_mean", "steps_mean"], ascending=[True] * len(rank_cols) + [False, True])
    out["success_rank"] = out.groupby(rank_cols, dropna=False)["success_mean"].rank(method="min", ascending=False)
    return out


def _paired_success_comparisons(
    success: pd.DataFrame,
    *,
    primary_method: str,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    group_cols = ["config_label", "eval_split", "eval_type", "split", "task"]
    unit_cols = [col for col in ["profile", "seed"] if col in success.columns]
    if not unit_cols:
        raise ValueError("Success CSV must include at least one repeated-unit column such as profile or seed.")

    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, object]] = []
    for keys, group in success.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        group_meta = {col: value for col, value in zip(group_cols, keys)}
        primary = group[group["baseline"] == primary_method]
        if primary.empty:
            continue
        primary_unit = primary[unit_cols + ["success"]].rename(columns={"success": "primary_success"})
        primary_unit = primary_unit.groupby(unit_cols, as_index=False)["primary_success"].mean()
        for baseline in sorted(set(group["baseline"]) - {primary_method}):
            baseline_unit = group[group["baseline"] == baseline][unit_cols + ["success"]].rename(
                columns={"success": "baseline_success"}
            )
            baseline_unit = baseline_unit.groupby(unit_cols, as_index=False)["baseline_success"].mean()
            merged = primary_unit.merge(baseline_unit, on=unit_cols, how="inner")
            if merged.empty:
                continue
            diff = (merged["primary_success"] - merged["baseline_success"]).to_numpy(dtype=np.float64)
            n = int(len(diff))
            obs = float(diff.mean())
            if n == 1:
                boot = diff.copy()
                p_value = 1.0
            else:
                sample_indices = rng.integers(0, n, size=(int(bootstrap_samples), n))
                boot = diff[sample_indices].mean(axis=1)
                signs = rng.choice(np.asarray([-1.0, 1.0]), size=(int(bootstrap_samples), n))
                null = (diff.reshape(1, -1) * signs).mean(axis=1)
                p_value = float((np.count_nonzero(np.abs(null) >= abs(obs)) + 1) / (int(bootstrap_samples) + 1))
            ci_low = float(np.percentile(boot, 2.5))
            ci_high = float(np.percentile(boot, 97.5))
            if ci_low > 0.0:
                winner = primary_method
            elif ci_high < 0.0:
                winner = baseline
            else:
                winner = "tie_or_inconclusive"
            rows.append(
                {
                    **group_meta,
                    "primary_method": primary_method,
                    "baseline": baseline,
                    "n_pairs": n,
                    "primary_mean": float(merged["primary_success"].mean()),
                    "baseline_mean": float(merged["baseline_success"].mean()),
                    "mean_diff": obs,
                    "diff_ci95_low": ci_low,
                    "diff_ci95_high": ci_high,
                    "p_signflip": p_value,
                    "winner": winner,
                }
            )
    return pd.DataFrame.from_records(rows)


def main() -> None:
    args = _parse_args()
    input_root = Path(args.input_root)
    combined_root = input_root / "combined"
    success_path = Path(args.success_csv) if args.success_csv else combined_root / "paper_success_summary.csv"
    transition_path = Path(args.transition_csv) if args.transition_csv else combined_root / "paper_transition_metrics.csv"
    output_dir = Path(args.output_dir) if args.output_dir else combined_root
    output_dir.mkdir(parents=True, exist_ok=True)

    if not success_path.exists():
        raise FileNotFoundError(f"Missing paper success summary: {success_path}")
    success = pd.read_csv(success_path)
    success_stats = _stats_frame(
        success,
        ["config_label", "eval_split", "eval_type", "baseline", "split", "task"],
        ["success", "steps"],
    )
    success_stats = _add_success_rank(success_stats)
    success_by_profile = _stats_frame(
        success,
        ["config_label", "profile", "eval_split", "eval_type", "baseline", "split", "task"],
        ["success", "steps"],
    )
    pairwise = _paired_success_comparisons(
        success,
        primary_method=str(args.primary_method),
        bootstrap_samples=int(args.bootstrap_samples),
        seed=int(args.seed),
    )

    success_stats_out = output_dir / "paper_success_stats.csv"
    success_by_profile_out = output_dir / "paper_success_stats_by_profile.csv"
    pairwise_out = output_dir / "paper_pairwise_success_comparisons.csv"
    success_stats.to_csv(success_stats_out, index=False)
    success_by_profile.to_csv(success_by_profile_out, index=False)
    pairwise.to_csv(pairwise_out, index=False)

    transition_stats_out = output_dir / "paper_transition_stats.csv"
    if transition_path.exists():
        transition = pd.read_csv(transition_path)
        transition_stats = _stats_frame(
            transition,
            ["config_label", "eval_split", "baseline", "split", "task"],
            ["action_mse", "action_mae"],
        )
        transition_stats.to_csv(transition_stats_out, index=False)
    else:
        transition_stats_out = Path("")

    print(f"paper_success_stats_csv={success_stats_out}")
    print(f"paper_success_stats_by_profile_csv={success_by_profile_out}")
    print(f"paper_pairwise_success_comparisons_csv={pairwise_out}")
    if transition_stats_out:
        print(f"paper_transition_stats_csv={transition_stats_out}")


if __name__ == "__main__":
    main()

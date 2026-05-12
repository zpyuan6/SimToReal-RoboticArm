from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from ttla.config import load_config
from ttla.models import TTLAModel, load_model_state
from ttla.training import build_model
from ttla.utils.io import ensure_dir


ROOT = Path(__file__).resolve().parents[1]

METHODS = {
    "no_adaptation": ROOT / "results" / "fixed_protocol" / "backbone_suite" / "feedforward" / "checkpoints" / "best_model.pt",
    "static_adapter": ROOT / "results" / "fixed_protocol" / "backbone_suite_real" / "feedforward" / "checkpoints" / "static_adapter_calibrated.pt",
    "few_shot_finetuning": ROOT / "results" / "fixed_protocol" / "backbone_suite_real" / "feedforward" / "checkpoints" / "few_shot_finetuned.pt",
    "plica": ROOT / "results" / "fixed_protocol" / "backbone_suite_real" / "feedforward" / "checkpoints" / "adapter_calibrated.pt",
}


def _load_cfg_for_checkpoint(default_cfg: dict[str, Any], checkpoint_path: Path) -> dict[str, Any]:
    snapshot_path = checkpoint_path.parents[1] / "config_snapshot.yaml"
    if snapshot_path.exists():
        with snapshot_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    return default_cfg


def _load_model(cfg: dict[str, Any], checkpoint_path: Path) -> TTLAModel:
    device = torch.device(cfg["train"]["device"])
    payload = torch.load(checkpoint_path, map_location=device)
    model = build_model(cfg).to(device)
    load_model_state(model, payload["model_state"])
    model.eval()
    return model


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path)
    return {key: payload[key] for key in payload.files}


def _load_meta(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_episodes(payload: dict[str, np.ndarray], meta: dict[str, Any], label: str):
    offset = 0
    for ep_idx, episode in enumerate(meta["episodes"]):
        transitions = int(episode["transition_count"])
        yield {
            "episode_index": ep_idx,
            "attempt_idx": int(episode["attempt_idx"]),
            "label": label,
            "transition_count": transitions,
            "images": payload["images"][offset : offset + transitions],
            "states": payload["states"][offset : offset + transitions],
            "primitive_ids": payload["primitive_ids"][offset : offset + transitions],
            "tasks": payload["tasks"][offset : offset + transitions],
            "primitive_names": episode["primitive_names"],
        }
        offset += transitions


def _evaluate_episode(model: TTLAModel, episode: dict[str, Any], device: torch.device) -> dict[str, Any]:
    runtime_state = model.init_runtime_state(batch_size=1, device=device)
    step_records: list[dict[str, Any]] = []
    full_match = True
    match_count = 0
    for step_idx in range(int(episode["transition_count"])):
        image = (
            torch.from_numpy(episode["images"][step_idx])
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(device)
            / 255.0
        )
        state = torch.from_numpy(episode["states"][step_idx]).unsqueeze(0).float().to(device)
        task_ids = torch.tensor([int(episode["tasks"][step_idx])], dtype=torch.long, device=device)
        target_primitive = int(episode["primitive_ids"][step_idx])
        with torch.no_grad():
            predicted, runtime_state, _ = model.act(
                image,
                state,
                runtime_state,
                use_adapter=True,
                task_ids=task_ids,
            )
        predicted_primitive = int(predicted.item())
        matched = predicted_primitive == target_primitive
        if matched:
            match_count += 1
        else:
            full_match = False
        step_records.append(
            {
                "step": step_idx,
                "predicted_primitive": predicted_primitive,
                "target_primitive": target_primitive,
                "matched": matched,
            }
        )
    return {
        "full_match": full_match,
        "step_match_count": match_count,
        "step_match_rate": float(match_count / max(1, int(episode["transition_count"]))),
        "step_records": step_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline held-out real L3 episode-level baseline comparison.")
    parser.add_argument("--config", default="configs/fixed_protocol.yaml")
    parser.add_argument("--success-npz", default="data/real/v3_pick_place_p1_center_success.npz")
    parser.add_argument("--failure-npz", default="data/real/v3_pick_place_p1_center_failure.npz")
    parser.add_argument("--success-meta", default="data/real/v3_pick_place_p1_center_session/success/meta.json")
    parser.add_argument("--failure-meta", default="data/real/v3_pick_place_p1_center_session/failure/meta.json")
    parser.add_argument("--output-dir", default="results/fixed_protocol/real_l3_offline_episode_eval")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["train"]["device"])

    success_payload = _load_npz(ROOT / args.success_npz)
    failure_payload = _load_npz(ROOT / args.failure_npz)
    success_meta = _load_meta(ROOT / args.success_meta)
    failure_meta = _load_meta(ROOT / args.failure_meta)

    episodes = [
        *list(_iter_episodes(success_payload, success_meta, "success")),
        *list(_iter_episodes(failure_payload, failure_meta, "failure")),
    ]
    episodes = sorted(episodes, key=lambda item: item["attempt_idx"])

    output_dir = ensure_dir(ROOT / args.output_dir)
    episode_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for method, checkpoint_path in METHODS.items():
        method_cfg = _load_cfg_for_checkpoint(cfg, checkpoint_path)
        model = _load_model(method_cfg, checkpoint_path)
        success_total = sum(ep["label"] == "success" for ep in episodes)
        failure_total = sum(ep["label"] == "failure" for ep in episodes)
        matched_success = 0
        matched_failure = 0
        full_match_total = 0
        total_step_match_rate = 0.0

        for episode in episodes:
            result = _evaluate_episode(model, episode, device)
            is_success_label = episode["label"] == "success"
            if result["full_match"]:
                full_match_total += 1
                if is_success_label:
                    matched_success += 1
                else:
                    matched_failure += 1
            total_step_match_rate += result["step_match_rate"]
            episode_rows.append(
                {
                    "method": method,
                    "attempt_idx": episode["attempt_idx"],
                    "label": episode["label"],
                    "transition_count": episode["transition_count"],
                    "full_match": int(result["full_match"]),
                    "step_match_count": result["step_match_count"],
                    "step_match_rate": result["step_match_rate"],
                    "offline_success": int(result["full_match"] and is_success_label),
                }
            )

        total_episodes = len(episodes)
        summary_rows.append(
            {
                "method": method,
                "episodes_total": total_episodes,
                "success_episodes_total": success_total,
                "failure_episodes_total": failure_total,
                "full_match_episodes": full_match_total,
                "matched_success_episodes": matched_success,
                "matched_failure_episodes": matched_failure,
                "offline_success_rate": float(matched_success / total_episodes) if total_episodes else 0.0,
                "success_episode_recovery": float(matched_success / success_total) if success_total else 0.0,
                "failure_episode_match_rate": float(matched_failure / failure_total) if failure_total else 0.0,
                "mean_step_match_rate": float(total_step_match_rate / total_episodes) if total_episodes else 0.0,
            }
        )

    episode_df = pd.DataFrame(episode_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["offline_success_rate", "success_episode_recovery", "mean_step_match_rate"],
        ascending=[False, False, False],
    )

    episode_df.to_csv(output_dir / "l3_offline_episode_rows.csv", index=False)
    summary_df.to_csv(output_dir / "l3_offline_episode_summary.csv", index=False)

    print(summary_df.to_string(index=False))
    print(f"saved_episode_rows={output_dir / 'l3_offline_episode_rows.csv'}")
    print(f"saved_summary={output_dir / 'l3_offline_episode_summary.csv'}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load(path: str | Path) -> dict[str, np.ndarray]:
    npz = np.load(Path(path), allow_pickle=True)
    return {key: npz[key] for key in npz.files}


def _episode_offset(payload: dict[str, np.ndarray], offset: int) -> dict[str, np.ndarray]:
    out = dict(payload)
    out["episode_ids"] = payload["episode_ids"].astype(np.int64) + int(offset)
    return out


def mix_npz(inputs: list[str | Path], output: str | Path) -> Path:
    if not inputs:
        raise ValueError("At least one input NPZ is required.")
    payloads: list[dict[str, np.ndarray]] = []
    episode_offset = 0
    for path in inputs:
        payload = _load(path)
        payload = _episode_offset(payload, episode_offset)
        payloads.append(payload)
        episode_offset = int(payload["episode_ids"].max()) + 1

    keys = list(payloads[0].keys())
    mixed: dict[str, np.ndarray] = {}
    for key in keys:
        mixed[key] = np.concatenate([payload[key] for payload in payloads], axis=0)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **mixed)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Concatenate continuous NPZ datasets with episode id offsets.")
    parser.add_argument("--input", action="append", required=True, help="Input NPZ. Pass multiple times.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output_path = mix_npz(args.input, args.output)
    print(output_path)


if __name__ == "__main__":
    main()

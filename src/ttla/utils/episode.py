from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .io import ensure_dir, write_json


@dataclass
class EpisodeBuffer:
    frames: list[np.ndarray] = field(default_factory=list)
    states: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    contexts: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    infos: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        frame: np.ndarray,
        state: np.ndarray,
        action: int,
        context: np.ndarray,
        reward: float,
        info: dict[str, Any],
    ) -> None:
        self.frames.append(frame.copy())
        self.states.append(state.copy())
        self.actions.append(int(action))
        self.contexts.append(context.copy())
        self.rewards.append(float(reward))
        self.infos.append(info)

    def save(self, episode_dir: str | Path, metadata: dict[str, Any]) -> Path:
        target = ensure_dir(episode_dir)
        np.savez_compressed(
            target / "episode.npz",
            frames=np.asarray(self.frames, dtype=np.uint8),
            states=np.asarray(self.states, dtype=np.float32),
            actions=np.asarray(self.actions, dtype=np.int64),
            contexts=np.asarray(self.contexts, dtype=np.float32),
            rewards=np.asarray(self.rewards, dtype=np.float32),
        )
        write_json(target / "meta.json", metadata)
        self._write_preview_video(target / "preview.mp4")
        return target

    def _write_preview_video(self, path: Path) -> None:
        if not self.frames:
            return
        height, width = self.frames[0].shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (width, height))
        try:
            for frame in self.frames:
                writer.write(frame[:, :, ::-1] if frame.shape[-1] == 3 else frame)
        finally:
            writer.release()

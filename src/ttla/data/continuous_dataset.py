from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_payload(path: str | Path) -> dict[str, np.ndarray]:
    payload = np.load(Path(path), allow_pickle=True)
    return {key: payload[key] for key in payload.files}


class ContinuousTrajectoryDataset(Dataset):
    """Windowed dataset scaffold for continuous/chunk control backbones.

    Required NPZ fields:
    - images: uint8 [N, H, W, C]
    - proprio: float32 [N, P]
    - actions: float32 [N, A]
    - tasks: int64 [N]
    - episode_ids: int64 [N]
    - step_ids: int64 [N]

    Optional fields:
    - task_text: object/string [N]
    """

    def __init__(
        self,
        path: str | Path,
        history_len: int = 1,
        action_horizon: int = 1,
    ) -> None:
        payload = _load_payload(path)
        self.images = payload["images"]
        self.proprio = payload["proprio"]
        self.actions = payload["actions"]
        self.tasks = payload["tasks"].astype(np.int64)
        self.episode_ids = payload["episode_ids"].astype(np.int64)
        self.step_ids = payload["step_ids"].astype(np.int64)
        self.task_text = payload.get("task_text")
        self.history_len = int(history_len)
        self.action_horizon = int(action_horizon)
        self._episode_to_indices: dict[int, np.ndarray] = {}
        self._index_to_position = np.zeros(len(self.tasks), dtype=np.int64)
        for episode_id in np.unique(self.episode_ids):
            indices = np.flatnonzero(self.episode_ids == episode_id)
            indices = indices[np.argsort(self.step_ids[indices])]
            self._episode_to_indices[int(episode_id)] = indices
            for pos, index in enumerate(indices):
                self._index_to_position[index] = pos

    def __len__(self) -> int:
        return len(self.tasks)

    def _history_indices(self, indices: np.ndarray, end_pos: int) -> np.ndarray:
        positions = []
        for offset in range(self.history_len - 1, -1, -1):
            pos = max(0, end_pos - offset)
            positions.append(indices[pos])
        return np.asarray(positions, dtype=np.int64)

    def _action_chunk(self, indices: np.ndarray, start_pos: int) -> np.ndarray:
        chunk = []
        last = indices[min(start_pos, len(indices) - 1)]
        for offset in range(self.action_horizon):
            pos = start_pos + offset
            if pos >= len(indices):
                chunk.append(self.actions[last])
            else:
                last = indices[pos]
                chunk.append(self.actions[last])
        return np.asarray(chunk, dtype=np.float32)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        episode_indices = self._episode_to_indices[int(self.episode_ids[idx])]
        pos = int(self._index_to_position[idx])
        history_indices = self._history_indices(episode_indices, pos)
        image_history = torch.from_numpy(self.images[history_indices]).permute(0, 3, 1, 2).float() / 255.0
        proprio_history = torch.from_numpy(self.proprio[history_indices]).float()
        action_chunk = torch.from_numpy(self._action_chunk(episode_indices, pos)).float()
        item: dict[str, torch.Tensor | str] = {
            "images": image_history,
            "proprio": proprio_history,
            "actions": action_chunk,
            "task": torch.tensor(self.tasks[idx]).long(),
            "episode_id": torch.tensor(self.episode_ids[idx]).long(),
            "step_id": torch.tensor(self.step_ids[idx]).long(),
        }
        if self.task_text is not None:
            value = self.task_text[idx]
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            item["task_text"] = str(value)
        return item


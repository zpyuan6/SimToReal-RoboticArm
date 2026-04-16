from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..sim.task_defs import supervision_stage_id


def _load_payload(path: str | Path) -> dict[str, np.ndarray]:
    payload = np.load(Path(path))
    return {key: payload[key] for key in payload.files}


class TrajectoryDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        payload = _load_payload(path)
        self.images = payload["images"]
        self.states = payload["states"]
        self.primitive_ids = payload["primitive_ids"]
        self.next_images = payload["next_images"]
        self.next_states = payload["next_states"]
        self.tasks = payload["tasks"]
        self.success = payload["success"]
        self.contexts = payload["contexts"] if "contexts" in payload else np.zeros((len(self.primitive_ids), 8), dtype=np.float32)
        self.episode_ids = payload.get("episode_ids", np.arange(len(self.primitive_ids), dtype=np.int64))
        self.step_ids = payload.get("step_ids", np.zeros(len(self.primitive_ids), dtype=np.int64))
        self._start_token = int(self.primitive_ids.max()) + 1 if len(self.primitive_ids) > 0 else 0
        if "stage_ids" in payload:
            self.stage_ids = payload["stage_ids"].astype(np.int64)
        else:
            self.stage_ids = np.asarray(
                [supervision_stage_id(int(task_id), int(primitive_id)) for task_id, primitive_id in zip(self.tasks, self.primitive_ids)],
                dtype=np.int64,
            )
        self.prev_primitive_ids = np.full(len(self.primitive_ids), self._start_token, dtype=np.int64)
        self.next_stage_ids = np.copy(self.stage_ids)
        for episode_id in np.unique(self.episode_ids):
            indices = np.flatnonzero(self.episode_ids == episode_id)
            indices = indices[np.argsort(self.step_ids[indices])]
            if len(indices) <= 1:
                continue
            self.prev_primitive_ids[indices[1:]] = self.primitive_ids[indices[:-1]]
            self.next_stage_ids[indices[:-1]] = self.stage_ids[indices[1:]]

    def __len__(self) -> int:
        return len(self.primitive_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image = torch.from_numpy(self.images[idx]).permute(2, 0, 1).float() / 255.0
        next_image = torch.from_numpy(self.next_images[idx]).permute(2, 0, 1).float() / 255.0
        return {
            "image": image,
            "state": torch.from_numpy(self.states[idx]).float(),
            "primitive_id": torch.tensor(self.primitive_ids[idx]).long(),
            "prev_primitive_id": torch.tensor(self.prev_primitive_ids[idx]).long(),
            "next_prev_primitive_id": torch.tensor(self.primitive_ids[idx]).long(),
            "next_image": next_image,
            "next_state": torch.from_numpy(self.next_states[idx]).float(),
            "task": torch.tensor(self.tasks[idx]).long(),
            "success": torch.tensor(self.success[idx]).float(),
            "context": torch.from_numpy(self.contexts[idx]).float(),
            "stage_id": torch.tensor(self.stage_ids[idx]).long(),
            "next_stage_id": torch.tensor(self.next_stage_ids[idx]).long(),
            "episode_id": torch.tensor(self.episode_ids[idx]).long(),
            "step_id": torch.tensor(self.step_ids[idx]).long(),
        }


class HistoryTrajectoryDataset(TrajectoryDataset):
    def __init__(self, path: str | Path, history_len: int = 4, chunk_size: int = 3) -> None:
        super().__init__(path)
        self.history_len = history_len
        self.chunk_size = chunk_size
        self._episode_to_indices: dict[int, np.ndarray] = {}
        self._position_in_episode = np.zeros(len(self.primitive_ids), dtype=np.int64)
        for episode_id in np.unique(self.episode_ids):
            indices = np.flatnonzero(self.episode_ids == episode_id)
            indices = indices[np.argsort(self.step_ids[indices])]
            self._episode_to_indices[int(episode_id)] = indices
            for pos, index in enumerate(indices):
                self._position_in_episode[index] = pos

    def _prev_primitives_for_positions(
        self,
        indices: np.ndarray,
        positions: list[int],
    ) -> np.ndarray:
        start_token = self.primitive_ids.max() + 1
        values = np.full(len(positions), start_token, dtype=np.int64)
        for i, pos in enumerate(positions):
            if pos <= 0:
                continue
            values[i] = self.primitive_ids[indices[pos - 1]]
        return values

    def _build_window(
        self,
        indices: np.ndarray,
        end_pos: int,
        append_next: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if append_next:
            start_pos = max(0, end_pos - self.history_len + 2)
            history_positions = list(range(start_pos, end_pos + 1))
            history_indices = indices[start_pos : end_pos + 1]
            images = list(self.images[history_indices]) + [self.next_images[indices[end_pos]]]
            states = list(self.states[history_indices]) + [self.next_states[indices[end_pos]]]
            prev_primitives = list(self._prev_primitives_for_positions(indices, history_positions))
            primitive_ids = list(self.primitive_ids[history_indices])
            prev_primitives.append(self.primitive_ids[indices[end_pos]])
            primitive_ids.append(self.primitive_ids[indices[end_pos]])
        else:
            start_pos = max(0, end_pos - self.history_len + 1)
            history_positions = list(range(start_pos, end_pos + 1))
            history_indices = indices[start_pos : end_pos + 1]
            images = list(self.images[history_indices])
            states = list(self.states[history_indices])
            prev_primitives = list(self._prev_primitives_for_positions(indices, history_positions))
            primitive_ids = list(self.primitive_ids[history_indices])
        valid_len = min(len(images), self.history_len)
        pad_len = self.history_len - valid_len
        image_shape = self.images[0].shape
        state_shape = self.states[0].shape
        padded_images = [np.zeros(image_shape, dtype=np.uint8) for _ in range(pad_len)] + images[-self.history_len :]
        padded_states = [np.zeros(state_shape, dtype=np.float32) for _ in range(pad_len)] + states[-self.history_len :]
        padded_prev_primitives = [self.primitive_ids.max() + 1 for _ in range(pad_len)] + prev_primitives[-self.history_len :]
        padded_primitive_ids = [0 for _ in range(pad_len)] + primitive_ids[-self.history_len :]
        mask = np.zeros(self.history_len, dtype=np.float32)
        mask[-valid_len:] = 1.0
        return (
            np.asarray(padded_images),
            np.asarray(padded_states, dtype=np.float32),
            np.asarray(padded_prev_primitives, dtype=np.int64),
            np.asarray(padded_primitive_ids, dtype=np.int64),
            mask,
        )

    def _chunk_targets(self, indices: np.ndarray, pos: int) -> tuple[np.ndarray, np.ndarray]:
        values = np.full(self.chunk_size, self.primitive_ids[indices[pos]], dtype=np.int64)
        mask = np.zeros(self.chunk_size, dtype=np.float32)
        current_index = indices[pos]
        current_task = int(self.tasks[current_index])
        current_stage = supervision_stage_id(current_task, int(self.primitive_ids[current_index]))
        for offset in range(self.chunk_size):
            next_pos = pos + offset
            if next_pos >= len(indices):
                break
            next_index = indices[next_pos]
            if supervision_stage_id(current_task, int(self.primitive_ids[next_index])) != current_stage:
                break
            values[offset] = self.primitive_ids[next_index]
            mask[offset] = 1.0
        return values, mask

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = super().__getitem__(idx)
        episode_id = int(self.episode_ids[idx])
        indices = self._episode_to_indices[episode_id]
        pos = int(self._position_in_episode[idx])
        history_images, history_states, history_prev_primitives, history_primitive_ids, history_mask = self._build_window(indices, pos, append_next=False)
        next_history_images, next_history_states, next_history_prev_primitives, _next_history_primitive_ids, next_history_mask = self._build_window(indices, pos, append_next=True)
        chunk_ids, chunk_mask = self._chunk_targets(indices, pos)
        sample.update(
            {
                "history_images": torch.from_numpy(history_images).permute(0, 3, 1, 2).float() / 255.0,
                "history_states": torch.from_numpy(history_states).float(),
                "history_prev_primitives": torch.from_numpy(history_prev_primitives).long(),
                "history_primitive_ids": torch.from_numpy(history_primitive_ids).long(),
                "history_mask": torch.from_numpy(history_mask).float(),
                "next_history_images": torch.from_numpy(next_history_images).permute(0, 3, 1, 2).float() / 255.0,
                "next_history_states": torch.from_numpy(next_history_states).float(),
                "next_history_prev_primitives": torch.from_numpy(next_history_prev_primitives).long(),
                "next_history_mask": torch.from_numpy(next_history_mask).float(),
                "chunk_primitive_ids": torch.from_numpy(chunk_ids).long(),
                "chunk_mask": torch.from_numpy(chunk_mask).float(),
            }
        )
        return sample


class RealCalibrationDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        payload = _load_payload(path)
        self.images = payload["images"]
        self.states = payload["states"]
        self.primitive_ids = payload["primitive_ids"]
        self.next_images = payload["next_images"]
        self.next_states = payload["next_states"]
        self.episode_ids = payload.get("episode_ids", np.arange(len(self.primitive_ids), dtype=np.int64))
        self.step_ids = payload.get("step_ids", np.zeros(len(self.primitive_ids), dtype=np.int64))
        self._start_token = int(self.primitive_ids.max()) + 1 if len(self.primitive_ids) > 0 else 0
        if "tasks" in payload:
            self.tasks = payload["tasks"]
        elif self.states.shape[-1] >= 2:
            self.tasks = np.rint(self.states[:, -2]).astype(np.int64)
        else:
            self.tasks = np.zeros(len(self.primitive_ids), dtype=np.int64)
        if "stage_ids" in payload:
            self.stage_ids = payload["stage_ids"].astype(np.int64)
        else:
            self.stage_ids = np.asarray(
                [supervision_stage_id(int(task_id), int(primitive_id)) for task_id, primitive_id in zip(self.tasks, self.primitive_ids)],
                dtype=np.int64,
            )
        self.prev_primitive_ids = np.full(len(self.primitive_ids), self._start_token, dtype=np.int64)
        self.next_stage_ids = np.copy(self.stage_ids)
        for episode_id in np.unique(self.episode_ids):
            indices = np.flatnonzero(self.episode_ids == episode_id)
            indices = indices[np.argsort(self.step_ids[indices])]
            if len(indices) <= 1:
                continue
            self.prev_primitive_ids[indices[1:]] = self.primitive_ids[indices[:-1]]
            self.next_stage_ids[indices[:-1]] = self.stage_ids[indices[1:]]

    def __len__(self) -> int:
        return len(self.primitive_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image = torch.from_numpy(self.images[idx]).permute(2, 0, 1).float() / 255.0
        next_image = torch.from_numpy(self.next_images[idx]).permute(2, 0, 1).float() / 255.0
        return {
            "image": image,
            "state": torch.from_numpy(self.states[idx]).float(),
            "primitive_id": torch.tensor(self.primitive_ids[idx]).long(),
            "prev_primitive_id": torch.tensor(self.prev_primitive_ids[idx]).long(),
            "next_prev_primitive_id": torch.tensor(self.primitive_ids[idx]).long(),
            "next_image": next_image,
            "next_state": torch.from_numpy(self.next_states[idx]).float(),
            "task": torch.tensor(self.tasks[idx]).long(),
            "stage_id": torch.tensor(self.stage_ids[idx]).long(),
            "next_stage_id": torch.tensor(self.next_stage_ids[idx]).long(),
        }


def load_split(data_root: str | Path, split: str) -> Path:
    return Path(data_root) / f"{split}.npz"

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES
from PIL import Image
from ttla.config import load_config


STATE_NAMES = (
    "qpos_0",
    "qpos_1",
    "qpos_2",
    "qpos_3",
    "qpos_4",
    "qpos_5",
    "qvel_0",
    "qvel_1",
    "qvel_2",
    "qvel_3",
    "qvel_4",
    "qvel_5",
    "task_level1_verify",
    "task_level2_approach",
    "task_level3_pick_place",
    "progress",
)

GEOMETRY_STATE_NAMES = (
    "target_x",
    "target_y",
    "target_z",
    "drop_x",
    "drop_y",
    "drop_z",
)

ACTION_NAMES = (
    "joint_0",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
)

IMAGE_KEY = "observation.images.main"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
SMOLVLA_IMAGE_KEYS = (
    "observation.images.camera1",
    "observation.images.camera2",
    "observation.images.camera3",
)


def _parse_repeat_specs(values: list[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid repeat spec '{value}'. Expected task_name=repeat_count.")
        task_name, repeat_text = value.split("=", 1)
        task_name = task_name.strip()
        repeat_count = int(repeat_text)
        if repeat_count < 1:
            raise ValueError(f"Repeat count for '{task_name}' must be >= 1, got {repeat_count}.")
        mapping[task_name] = repeat_count
    return mapping


def _resize_image(image: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    height, width = (int(shape_hw[0]), int(shape_hw[1]))
    pil_image = Image.fromarray(image.astype(np.uint8))
    resized = pil_image.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _schema_from_config(cfg: dict, schema: str) -> str:
    if schema != "auto":
        return schema
    backbone_name = str(cfg.get("control", {}).get("backbone_name", "")).strip().lower()
    if backbone_name == "smolvla":
        return "smolvla"
    return "default"


def _dataset_features_default(image_shape: tuple[int, int, int], proprio_dim: int, action_dim: int) -> dict:
    if action_dim != len(ACTION_NAMES):
        raise ValueError(f"Expected {len(ACTION_NAMES)}-D action, got {action_dim}")
    state_names = _state_names(int(proprio_dim))
    features = dict(DEFAULT_FEATURES)
    features[IMAGE_KEY] = {
        "dtype": "image",
        "shape": tuple(int(v) for v in image_shape),
        "names": ["height", "width", "channels"],
    }
    features[STATE_KEY] = {
        "dtype": "float32",
        "shape": (int(proprio_dim),),
        "names": state_names,
    }
    features[ACTION_KEY] = {
        "dtype": "float32",
        "shape": (int(action_dim),),
        "names": list(ACTION_NAMES),
    }
    return features


def _state_names(proprio_dim: int) -> list[str]:
    base_names = list(STATE_NAMES)
    if proprio_dim == len(base_names):
        return base_names
    geometry_names = base_names + list(GEOMETRY_STATE_NAMES)
    if proprio_dim == len(geometry_names):
        return geometry_names
    if proprio_dim > len(base_names):
        extra = [f"extra_state_{idx}" for idx in range(proprio_dim - len(base_names))]
        return base_names + extra
    return [f"state_{idx}" for idx in range(proprio_dim)]


def _dataset_features_smolvla(image_shape: tuple[int, int, int], proprio_dim: int, action_dim: int) -> dict:
    expected_proprio_dim = 6
    if proprio_dim < expected_proprio_dim:
        raise ValueError(f"Expected at least {expected_proprio_dim}-D proprio, got {proprio_dim}")
    if action_dim != len(ACTION_NAMES):
        raise ValueError(f"Expected {len(ACTION_NAMES)}-D action, got {action_dim}")
    features = dict(DEFAULT_FEATURES)
    for image_key in SMOLVLA_IMAGE_KEYS:
        features[image_key] = {
            "dtype": "image",
            "shape": tuple(int(v) for v in image_shape),
            "names": ["height", "width", "channels"],
        }
    features[STATE_KEY] = {
        "dtype": "float32",
        "shape": (expected_proprio_dim,),
        "names": list(STATE_NAMES[:expected_proprio_dim]),
    }
    features[ACTION_KEY] = {
        "dtype": "float32",
        "shape": (int(action_dim),),
        "names": list(ACTION_NAMES),
    }
    return features


def _dataset_features(
    schema: str,
    image_shape: tuple[int, int, int],
    proprio_dim: int,
    action_dim: int,
) -> dict:
    if schema == "smolvla":
        return _dataset_features_smolvla(image_shape, proprio_dim, action_dim)
    return _dataset_features_default(image_shape, proprio_dim, action_dim)


def _frame_default(image: np.ndarray, proprio: np.ndarray, action: np.ndarray, task: str) -> dict:
    return {
        IMAGE_KEY: image,
        STATE_KEY: proprio.astype(np.float32),
        ACTION_KEY: action.astype(np.float32),
        "task": task,
    }


def _frame_smolvla(image: np.ndarray, proprio: np.ndarray, action: np.ndarray, task: str) -> dict:
    resized_image = _resize_image(image, (256, 256))
    frame = {
        image_key: resized_image.copy()
        for image_key in SMOLVLA_IMAGE_KEYS
    }
    frame[STATE_KEY] = proprio[:6].astype(np.float32)
    frame[ACTION_KEY] = action.astype(np.float32)
    frame["task"] = task
    return frame


def _frame_for_schema(schema: str, image: np.ndarray, proprio: np.ndarray, action: np.ndarray, task: str) -> dict:
    if schema == "smolvla":
        return _frame_smolvla(image, proprio, action, task)
    return _frame_default(image, proprio, action, task)


def _episode_ids_order(episode_ids: np.ndarray) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for episode_id in episode_ids.tolist():
        episode_id = int(episode_id)
        if episode_id not in seen:
            seen.add(episode_id)
            ordered.append(episode_id)
    return ordered


def export_npz_to_lerobot(
    npz_path: str | Path,
    output_root: str | Path,
    repo_id: str,
    fps: int,
    schema: str = "default",
    repeat_task: dict[str, int] | None = None,
) -> Path:
    bundle = np.load(npz_path, allow_pickle=True)
    images = bundle["images"]
    proprio = bundle["proprio"]
    actions = bundle["actions"]
    episode_ids = bundle["episode_ids"]
    step_ids = bundle["step_ids"]
    tasks = bundle["tasks"]
    task_text = bundle["task_text"]
    repeat_task = repeat_task or {}

    dataset_root = Path(output_root)
    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    image_shape = tuple(int(v) for v in images.shape[1:])
    if schema == "smolvla":
        image_shape = (256, 256, image_shape[-1])
    features = _dataset_features(
        schema=schema,
        image_shape=image_shape,
        proprio_dim=int(proprio.shape[-1]),
        action_dim=int(actions.shape[-1]),
    )
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=dataset_root,
        fps=int(fps),
        robot_type="roarm_m3",
        features=features,
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=0,
    )

    for episode_id in _episode_ids_order(episode_ids):
        indices = np.nonzero(episode_ids == episode_id)[0]
        if indices.size == 0:
            continue
        indices = indices[np.argsort(step_ids[indices])]
        task_name = str(tasks[indices[0]])
        repeat_count = int(repeat_task.get(task_name, 1))
        for _ in range(repeat_count):
            for idx in indices.tolist():
                dataset.add_frame(
                    _frame_for_schema(
                        schema=schema,
                        image=images[idx],
                        proprio=proprio[idx],
                        action=actions[idx],
                        task=str(task_text[idx]),
                    )
                )
            dataset.save_episode(parallel_encoding=False)
    return dataset_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/continuous_act_template.yaml")
    parser.add_argument("--input", required=True, help="Path to a continuous NPZ file")
    parser.add_argument("--output-root", required=True, help="Output directory for the local LeRobot dataset")
    parser.add_argument("--repo-id", required=True, help="LeRobot dataset repo_id metadata name")
    parser.add_argument(
        "--schema",
        choices=("auto", "default", "smolvla"),
        default="auto",
        help="Official dataset schema view to export",
    )
    parser.add_argument(
        "--repeat-task",
        action="append",
        default=[],
        help="Repeat spec in the form task_name=repeat_count. Repeat count includes the original copy.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    sim_cfg = cfg["sim"]
    fps = round(1.0 / (float(sim_cfg["control_dt"]) * float(sim_cfg["action_repeat"])))
    schema = _schema_from_config(cfg, args.schema)
    output_root = export_npz_to_lerobot(
        npz_path=args.input,
        output_root=args.output_root,
        repo_id=args.repo_id,
        fps=fps,
        schema=schema,
        repeat_task=_parse_repeat_specs(args.repeat_task),
    )
    print(output_root)


if __name__ == "__main__":
    main()

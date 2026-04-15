from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..models import BaseTTLAModel, load_model_state
from ..sim import RoArmSimEnv
from ..sim.task_defs import TASK_TO_ID, supervision_stage_id
from ..training import build_model
from .baselines import baseline_overrides


def _select_primitive(
    model: BaseTTLAModel,
    obs: dict,
    runtime_state,
    use_adapter: bool,
    device: torch.device,
    task_id: int,
    latent_alignment: dict[str, np.ndarray] | None = None,
) -> tuple[int, object]:
    image = torch.from_numpy(obs["image"]).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    state = torch.from_numpy(obs["state"]).unsqueeze(0).float().to(device)
    task_ids = torch.tensor([task_id], device=device, dtype=torch.long)
    primitive, runtime_state, z_used = model.act(image, state, runtime_state, use_adapter=use_adapter, task_ids=task_ids)
    if latent_alignment is not None:
        z_used = _apply_latent_alignment(z_used, task_id, latent_alignment)
        logits = model.condition_policy_logits(model.policy_logits(z_used), z=z_used, task_ids=task_ids, state=state)
        primitive = logits.argmax(dim=-1)
        if isinstance(runtime_state, dict):
            runtime_state["prev_primitive"] = primitive.detach()
    return int(primitive.item()), runtime_state


def _make_online_optimizer(model: BaseTTLAModel, cfg: dict) -> torch.optim.Optimizer | None:
    online_cfg = cfg.get("adaptation", {})
    if not bool(online_cfg.get("online_refinement", False)):
        return None
    model.freeze_backbone()
    params = [param for param in model.adapter_parameters() if param.requires_grad]
    if not params:
        return None
    return torch.optim.AdamW(
        params,
        lr=float(online_cfg.get("online_lr", online_cfg.get("lr", 1.0e-4))),
        weight_decay=float(online_cfg.get("online_weight_decay", 0.0)),
    )


def _obs_to_tensors(obs: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    image = torch.from_numpy(obs["image"]).permute(2, 0, 1).float().to(device) / 255.0
    state = torch.from_numpy(obs["state"]).float().to(device)
    return image, state


def _build_recurrent_online_batch(
    model: BaseTTLAModel,
    obs_history: list[dict],
    executed_primitives: list[int],
    current_primitive: int,
    next_obs: dict,
    task_id: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    history_len = int(getattr(model, "history_len", 4))
    start_token = int(model.action_dim)

    def _window_tensors(obs_seq: list[dict], primitive_seq: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        trimmed_obs = obs_seq[-history_len:]
        valid_len = len(trimmed_obs)
        pad_len = history_len - valid_len
        images: list[torch.Tensor] = []
        states: list[torch.Tensor] = []
        prev_primitives: list[int] = []
        for local_idx, obs in enumerate(trimmed_obs):
            global_idx = len(obs_seq) - valid_len + local_idx
            prev_primitive = start_token if global_idx == 0 else primitive_seq[global_idx - 1]
            image_t, state_t = _obs_to_tensors(obs, device)
            images.append(image_t)
            states.append(state_t)
            prev_primitives.append(prev_primitive)
        if pad_len > 0:
            image_shape = images[0].shape if images else torch.Size((3, 84, 84))
            state_shape = states[0].shape if states else torch.Size((18,))
            pad_images = [torch.zeros(image_shape, device=device) for _ in range(pad_len)]
            pad_states = [torch.zeros(state_shape, device=device) for _ in range(pad_len)]
            pad_prev = [start_token] * pad_len
            images = pad_images + images
            states = pad_states + states
            prev_primitives = pad_prev + prev_primitives
        mask = torch.zeros(history_len, device=device)
        mask[-valid_len:] = 1.0
        return (
            torch.stack(images, dim=0),
            torch.stack(states, dim=0),
            torch.tensor(prev_primitives, device=device, dtype=torch.long),
            mask,
        )

    current_images, current_states, current_prev, current_mask = _window_tensors(obs_history, executed_primitives)
    next_images, next_states, next_prev, next_mask = _window_tensors(obs_history + [next_obs], executed_primitives + [current_primitive])
    stage_id = supervision_stage_id(task_id, current_primitive)
    return {
        "history_images": current_images.unsqueeze(0),
        "history_states": current_states.unsqueeze(0),
        "history_prev_primitives": current_prev.unsqueeze(0),
        "history_mask": current_mask.unsqueeze(0),
        "next_history_images": next_images.unsqueeze(0),
        "next_history_states": next_states.unsqueeze(0),
        "next_history_prev_primitives": next_prev.unsqueeze(0),
        "next_history_mask": next_mask.unsqueeze(0),
        "image": current_images[-1:].clone(),
        "state": current_states[-1:].clone(),
        "next_image": next_images[-1:].clone(),
        "next_state": next_states[-1:].clone(),
        "primitive_id": torch.tensor([current_primitive], device=device, dtype=torch.long),
        "prev_primitive_id": torch.tensor([current_prev[-1].item()], device=device, dtype=torch.long),
        "next_prev_primitive_id": torch.tensor([current_primitive], device=device, dtype=torch.long),
        "task": torch.tensor([task_id], device=device, dtype=torch.long),
        "stage_id": torch.tensor([stage_id], device=device, dtype=torch.long),
    }


def _build_single_step_online_batch(
    obs: dict,
    next_obs: dict,
    primitive_id: int,
    task_id: int,
    device: torch.device,
    action_dim: int,
    prev_primitive: int | None = None,
) -> dict[str, torch.Tensor]:
    image, state = _obs_to_tensors(obs, device)
    next_image, next_state = _obs_to_tensors(next_obs, device)
    stage_id = supervision_stage_id(task_id, primitive_id)
    return {
        "image": image.unsqueeze(0),
        "state": state.unsqueeze(0),
        "next_image": next_image.unsqueeze(0),
        "next_state": next_state.unsqueeze(0),
        "primitive_id": torch.tensor([primitive_id], device=device, dtype=torch.long),
        "prev_primitive_id": torch.tensor([action_dim if prev_primitive is None else prev_primitive], device=device, dtype=torch.long),
        "next_prev_primitive_id": torch.tensor([primitive_id], device=device, dtype=torch.long),
        "task": torch.tensor([task_id], device=device, dtype=torch.long),
        "stage_id": torch.tensor([stage_id], device=device, dtype=torch.long),
    }


def _online_refine_adapter(
    model: BaseTTLAModel,
    optimizer: torch.optim.Optimizer | None,
    cfg: dict,
    batch: dict[str, torch.Tensor],
) -> None:
    if optimizer is None:
        return
    steps = int(cfg.get("adaptation", {}).get("online_refinement_steps", 1))
    if steps <= 0:
        return
    reg_weight = float(cfg.get("adaptation", {}).get("online_reg_weight", cfg.get("adaptation", {}).get("adapter_reg_weight", 0.1)))
    transition_weight = float(cfg.get("adaptation", {}).get("online_transition_weight", 1.0))
    stage_weight = float(cfg.get("adaptation", {}).get("online_stage_loss_weight", 0.0))
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        z, next_z = model.compute_latents(batch)
        z = z.detach()
        next_z = next_z.detach()
        z_prime, next_z_prime = model.compute_adapted_latents(batch)
        pred_next = model.predict_next(z_prime, batch["primitive_id"], batch.get("state"), task_ids=batch.get("task"))
        loss_adapt = F.mse_loss(pred_next, next_z_prime)
        loss_reg = F.mse_loss(z_prime, z) + 0.5 * F.mse_loss(next_z_prime, next_z)
        loss = transition_weight * loss_adapt + reg_weight * loss_reg
        if stage_weight > 0.0:
            loss = loss + stage_weight * model.compute_stage_loss(batch, z_prime)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.adapter_parameters()), 5.0)
        optimizer.step()
    model.eval()


def evaluate_checkpoint(
    cfg: dict,
    checkpoint_path: str | Path,
    baseline_name: str,
    output_path: str | Path,
    baseline_artifacts: dict | None = None,
) -> Path:
    device = torch.device(cfg["train"]["device"])
    payload = torch.load(checkpoint_path, map_location=device)
    model = build_model(cfg).to(device)
    load_model_state(model, payload["model_state"])
    model.eval()
    env = RoArmSimEnv(cfg["sim"], seed=cfg["seed"] + 101)
    baseline_cfg = baseline_overrides(baseline_name)
    baseline_artifacts = baseline_artifacts or {}
    latent_alignment = _load_latent_alignment(baseline_artifacts.get("latent_alignment_path"))
    initial_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    records: list[dict] = []
    for task_name in cfg["sim"]["tasks"]:
        for ep in range(cfg["evaluation"]["episodes_per_task"]):
            if baseline_cfg.get("tent", False):
                model.load_state_dict(initial_state)
                model.eval()
                tent_params = _tent_parameters(model)
                tent_optimizer = (
                    torch.optim.AdamW(
                        tent_params,
                        lr=float(cfg.get("tent", {}).get("lr", 1.0e-4)),
                        weight_decay=float(cfg.get("tent", {}).get("weight_decay", 0.0)),
                    )
                    if tent_params
                    else None
                )
            else:
                model.eval()
                tent_optimizer = None
            online_optimizer = _make_online_optimizer(model, cfg) if baseline_name == "ours" else None
            obs = env.reset(task_name=task_name)
            obs_history = [obs]
            executed_primitives: list[int] = []
            runtime_state = model.init_runtime_state(batch_size=1, device=device)
            success = 0
            info = {
                "visibility": 0.0,
                "center_error": 0.0,
                "verified": 0,
                "grasped": 0,
                "lifted": 0,
                "placed": 0,
                "ee_target_distance": float("nan"),
                "dropzone_distance": float("nan"),
            }
            ever_verified = 0
            ever_grasped = 0
            ever_lifted = 0
            ever_placed = 0
            for step in range(cfg["sim"]["episode_horizon"]):
                if baseline_cfg.get("input_norm", False):
                    obs = {**obs, "image": _normalize_image(obs["image"])}
                if baseline_cfg.get("tent", False):
                    primitive_id, runtime_state = _select_tent_primitive(
                        model,
                        obs,
                        runtime_state,
                        tent_optimizer,
                        device,
                        TASK_TO_ID[task_name],
                    )
                else:
                    with torch.no_grad():
                        primitive_id, runtime_state = _select_primitive(
                            model,
                            obs,
                            runtime_state,
                            baseline_cfg.get("use_adapter", False),
                            device,
                            TASK_TO_ID[task_name],
                            latent_alignment=latent_alignment if baseline_cfg.get("latent_alignment", False) else None,
                        )
                next_obs, _, done, info = env.step(primitive_id)
                if baseline_name == "ours" and online_optimizer is not None:
                    if model.backbone_type == "recurrent":
                        online_batch = _build_recurrent_online_batch(
                            model,
                            obs_history,
                            executed_primitives,
                            primitive_id,
                            next_obs,
                            TASK_TO_ID[task_name],
                            device,
                        )
                    else:
                        prev_primitive = executed_primitives[-1] if executed_primitives else None
                        online_batch = _build_single_step_online_batch(
                            obs,
                            next_obs,
                            primitive_id,
                            TASK_TO_ID[task_name],
                            device,
                            model.action_dim,
                            prev_primitive=prev_primitive,
                        )
                    _online_refine_adapter(model, online_optimizer, cfg, online_batch)
                obs = next_obs
                executed_primitives.append(primitive_id)
                obs_history.append(next_obs)
                success = info["success"]
                ever_verified = max(ever_verified, int(info.get("verified", 0)))
                ever_grasped = max(ever_grasped, int(info.get("grasped", 0)))
                ever_lifted = max(ever_lifted, int(info.get("lifted", 0)))
                ever_placed = max(ever_placed, int(info.get("placed", 0)))
                if done:
                    break
            records.append(
                {
                    "baseline": baseline_name,
                    "task": task_name,
                    "episode": ep,
                    "success": success,
                    "steps": step + 1,
                    "visibility": info["visibility"],
                    "center_error": info["center_error"],
                    "verified": ever_verified,
                    "grasped": ever_grasped,
                    "lifted": ever_lifted,
                    "placed": ever_placed,
                    "final_ee_target_distance": info["ee_target_distance"],
                    "final_dropzone_distance": info["dropzone_distance"],
                }
            )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(output, index=False)
    env.close()
    return output


def _normalize_image(image: np.ndarray) -> np.ndarray:
    x = image.astype(np.float32)
    x = (x - x.mean()) / max(x.std(), 1.0)
    x = np.clip(x * 48 + 127, 0, 255)
    return x.astype(np.uint8)


def _load_latent_alignment(path: str | Path | None) -> dict[str, np.ndarray] | None:
    if path is None:
        return None
    payload = np.load(Path(path))
    return {key: payload[key] for key in payload.files}


def _apply_latent_alignment(z: torch.Tensor, task_id: int, stats: dict[str, np.ndarray]) -> torch.Tensor:
    source_mean = torch.from_numpy(stats["source_mean"][task_id]).to(device=z.device, dtype=z.dtype)
    source_std = torch.from_numpy(stats["source_std"][task_id]).to(device=z.device, dtype=z.dtype)
    target_mean = torch.from_numpy(stats["target_mean"][task_id]).to(device=z.device, dtype=z.dtype)
    target_std = torch.from_numpy(stats["target_std"][task_id]).to(device=z.device, dtype=z.dtype)
    return ((z - target_mean) / target_std.clamp_min(1.0e-6)) * source_std + source_mean


def _tent_parameters(model: BaseTTLAModel) -> list[torch.nn.Parameter]:
    preferred = {
        "feedforward": ["fusion", "policy_head", "stage_head"],
        "recurrent": ["feature_fusion", "gru", "post_gru", "policy_head", "stage_head"],
        "chunking": ["feature_fusion", "gru", "post_gru", "chunk_head", "stage_head"],
        "language": ["fusion", "policy_head", "stage_head"],
        "diffusion": ["direct_head", "denoiser", "stage_head"],
    }.get(model.backbone_type, ["policy_head", "stage_head"])
    params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(name.startswith(prefix + ".") for prefix in preferred):
            params.append(param)
    return params


def _select_tent_primitive(
    model: BaseTTLAModel,
    obs: dict,
    runtime_state,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    task_id: int,
) -> tuple[int, object]:
    image = torch.from_numpy(obs["image"]).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    state = torch.from_numpy(obs["state"]).unsqueeze(0).float().to(device)
    task_ids = torch.tensor([task_id], device=device, dtype=torch.long)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        z, proposed_runtime = model.encode_step(image, state, runtime_state, task_ids=task_ids)
        logits = model.condition_policy_logits(model.policy_logits(z), z=z, task_ids=task_ids, state=state)
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        entropy.backward()
        optimizer.step()
        runtime_state = proposed_runtime
    with torch.no_grad():
        primitive, runtime_state, _ = model.act(image, state, runtime_state, use_adapter=False, task_ids=task_ids)
    return int(primitive.item()), runtime_state

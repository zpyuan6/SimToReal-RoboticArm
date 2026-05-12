from .networks import (
    BaseTTLAModel,
    ChunkingTTLAModel,
    DiffusionPrimitiveTTLAModel,
    FeedForwardTTLAModel,
    ImageEncoder,
    LanguageConditionedTTLAModel,
    RecurrentTTLAModel,
    TTLAModel,
    build_backbone_model,
)


def load_model_state(model, state_dict):
    state = dict(state_dict)
    model_state = model.state_dict()
    allowed_missing_prefixes = (
        "feature_adapter.",
        "feature_adapter_task_embedding.",
        "feature_adapter_progress_embedding.",
        "feature_adapter_prev_action_embedding.",
        "feature_adapter_condition.",
        "feature_adapter_gate.",
        "direct_head.",
        "condition_adapter.",
        "condition_adapter_task_embedding.",
        "condition_adapter_stage_embedding.",
        "condition_adapter_condition.",
        "condition_adapter_gate.",
        "adapter_task_embedding.",
        "adapter_stage_embedding.",
        "adapter_prev_action_embedding.",
        "adapter_condition.",
        "adapter_gate.",
        "adapter_non_observation.",
        "adapter_non_observation_condition.",
        "adapter_non_observation_gate.",
        "transition_action_adapter_embedding.",
        "transition_residual_adapter.",
        "transition_non_observation_residual_adapter.",
        "policy_residual_adapter.",
        "inverse_head.",
        "latent_align_source_mean",
        "latent_align_source_std",
        "latent_align_target_mean",
        "latent_align_target_std",
        "adapter_stage_scale_vector",
    )
    dropped_keys = []
    for key, value in list(state.items()):
        if key in model_state and model_state[key].shape != value.shape:
            if key.startswith(allowed_missing_prefixes):
                state.pop(key)
                dropped_keys.append(key)
            else:
                raise RuntimeError(
                    f"Incompatible shape for required parameter '{key}': checkpoint {tuple(value.shape)} vs model {tuple(model_state[key].shape)}"
                )
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = [
        key
        for key in incompatible.unexpected_keys
        if not key.startswith("direct_head.")
        and not key.startswith(allowed_missing_prefixes)
    ]
    missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith(allowed_missing_prefixes)
    ]
    if unexpected or missing:
        raise RuntimeError(
            f"Incompatible model state. Missing={missing}, Unexpected={unexpected}"
        )
    return incompatible

__all__ = [
    "BaseTTLAModel",
    "ChunkingTTLAModel",
    "DiffusionPrimitiveTTLAModel",
    "FeedForwardTTLAModel",
    "ImageEncoder",
    "LanguageConditionedTTLAModel",
    "RecurrentTTLAModel",
    "TTLAModel",
    "build_backbone_model",
    "load_model_state",
]

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
    incompatible = model.load_state_dict(state, strict=False)
    allowed_missing_prefixes = (
        "feature_adapter.",
        "feature_adapter_task_embedding.",
        "feature_adapter_progress_embedding.",
        "feature_adapter_condition.",
        "direct_head.",
        "condition_adapter.",
        "condition_adapter_task_embedding.",
        "condition_adapter_stage_embedding.",
        "condition_adapter_condition.",
        "adapter_task_embedding.",
        "adapter_stage_embedding.",
        "adapter_condition.",
    )
    unexpected = [
        key
        for key in incompatible.unexpected_keys
        if not key.startswith("direct_head.")
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

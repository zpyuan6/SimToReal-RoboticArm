from __future__ import annotations


def baseline_overrides(name: str) -> dict:
    if name == "no_adaptation":
        return {"use_adapter": False, "input_norm": False}
    if name == "domain_randomization_only":
        return {"use_adapter": False, "input_norm": False}
    if name == "input_normalization":
        return {"use_adapter": False, "input_norm": True}
    if name == "probe_feature_alignment":
        return {"use_adapter": False, "input_norm": False, "latent_alignment": True}
    if name == "static_adapter":
        return {"use_adapter": True, "input_norm": False}
    if name == "few_shot_finetuning":
        return {"use_adapter": False, "input_norm": False}
    if name == "tent_style":
        return {"use_adapter": False, "input_norm": False, "tent": True}
    if name == "ours":
        return {"use_adapter": True, "input_norm": False}
    raise KeyError(name)

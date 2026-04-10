from __future__ import annotations


def baseline_overrides(name: str) -> dict:
    if name == "no_adaptation":
        return {"adapt": False, "randomization": False, "fewshot": False}
    if name == "domain_randomization_only":
        return {"adapt": False, "randomization": True, "fewshot": False}
    if name == "input_normalization":
        return {"adapt": False, "input_norm": True, "randomization": True, "fewshot": False}
    if name == "few_shot_finetuning":
        return {"adapt": False, "randomization": True, "fewshot": True}
    if name == "ours":
        return {"adapt": True, "randomization": True, "fewshot": False}
    raise KeyError(name)

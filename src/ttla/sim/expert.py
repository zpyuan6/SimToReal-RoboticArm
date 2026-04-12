from __future__ import annotations

from .oracle import PrimitiveOracle


class ScriptedExpert(PrimitiveOracle):
    """Backward-compatible alias for the rule-based primitive oracle."""


__all__ = ["ScriptedExpert"]

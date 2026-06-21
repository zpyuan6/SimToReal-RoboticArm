from .primitives import PrimitiveExecutor, PrimitiveResult

__all__ = ["DeploymentRunner", "PrimitiveExecutor", "PrimitiveResult"]


def __getattr__(name: str):
    if name == "DeploymentRunner":
        from .runner import DeploymentRunner

        return DeploymentRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

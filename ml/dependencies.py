"""Dependency detection used by :mod:`ml` with actionable error messages."""

from __future__ import annotations

from typing import Any


class MissingOptionalDependency(RuntimeError):
    """Raised when a machine-learning dependency is unavailable."""

    def __init__(self, dependency: str, feature: str | None = None) -> None:
        self.dependency = dependency
        self.feature = feature
        suffix = f" para {feature}" if feature else ""
        super().__init__(
            f"A dependência opcional '{dependency}' é necessária{suffix}. "
            f"Instale-a no ambiente do projeto (por exemplo: pip install {dependency})."
        )


try:  # pragma: no cover - branch depends on the test environment
    import torch as torch  # type: ignore

    TORCH_AVAILABLE = True
except (ImportError, OSError):  # pragma: no cover - exercised without torch
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


try:  # pragma: no cover - optional package
    import torch_geometric as torch_geometric  # type: ignore

    PYG_AVAILABLE = True
except (ImportError, OSError):  # pragma: no cover
    torch_geometric = None  # type: ignore[assignment]
    PYG_AVAILABLE = False


def torch_available() -> bool:
    """Return whether PyTorch can be imported in the current environment."""

    return TORCH_AVAILABLE

def require_torch(feature: str = "o modelo PointNet++") -> Any:
    """Return the imported torch module or raise a clear installation error."""

    if not TORCH_AVAILABLE:
        raise MissingOptionalDependency("torch", feature)
    return torch

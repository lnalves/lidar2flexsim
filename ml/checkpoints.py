"""Persistência de pesos e metadados do PointNet++."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .config import PointNet2Config
from .dependencies import require_torch


FORMAT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    model: Any,
    *,
    optimizer: Any | None = None,
    epoch: int = 0,
    config: PointNet2Config | dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Salva ``state_dict`` e configuração em um arquivo ``.pt``."""

    torch = require_torch("salvar checkpoint")
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    model_config = config
    if model_config is None:
        model_config = getattr(model, "config", None)
    if isinstance(model_config, PointNet2Config):
        model_config = model_config.to_dict()
    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "state_dict": model.state_dict(),
        "epoch": int(epoch),
        "config": dict(model_config or {}),
        "metrics": dict(metrics or {}),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        payload["extra"] = dict(extra)
    torch.save(payload, target)
    return target


def load_checkpoint(
    path: str | Path,
    model: Any | None = None,
    *,
    optimizer: Any | None = None,
    map_location: str | Any = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Carrega um checkpoint e opcionalmente restaura modelo/otimizador.

    Checkpoints antigos que são apenas um ``state_dict`` também são aceitos.
    O payload retornado sempre possui ``state_dict`` e ``config``.
    """

    torch = require_torch("carregar checkpoint")
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(source)
    # Checkpoints are untrusted input at the application boundary. The
    # weights-only unpickler accepts tensors and primitive container metadata,
    # but refuses arbitrary Python globals that could execute code. The ML
    # requirements pin PyTorch >= 2.2, where this argument is available.
    try:
        payload = torch.load(source, map_location=map_location, weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "Carregamento seguro de checkpoint requer PyTorch >= 2.2."
        ) from exc
    if isinstance(payload, dict) and "state_dict" in payload:
        result = dict(payload)
    elif isinstance(payload, dict):
        result = {"format_version": 0, "state_dict": payload, "config": {}}
    else:
        raise ValueError(f"Checkpoint inválido: {source}")
    if not isinstance(result.get("state_dict"), Mapping):
        raise ValueError(f"Checkpoint inválido: state_dict ausente ou não é mapping: {source}")
    config = result.get("config", {})
    metrics = result.get("metrics", {})
    if config is None:
        config = {}
    if metrics is None:
        metrics = {}
    if not isinstance(config, Mapping):
        raise ValueError(f"Checkpoint inválido: config não é mapping: {source}")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"Checkpoint inválido: metrics não é mapping: {source}")
    result["config"] = dict(config)
    result["metrics"] = dict(metrics)
    if model is not None:
        incompatible = model.load_state_dict(result["state_dict"], strict=strict)
        if not strict:
            result["missing_keys"] = list(getattr(incompatible, "missing_keys", []))
            result["unexpected_keys"] = list(getattr(incompatible, "unexpected_keys", []))
    if optimizer is not None and result.get("optimizer_state_dict"):
        optimizer.load_state_dict(result["optimizer_state_dict"])
    return result

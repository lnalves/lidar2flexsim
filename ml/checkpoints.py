"""Persistência de pesos e metadados do PointNet++."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .config import PointNet2Config
from .dependencies import require_torch


FORMAT_VERSION = 2


def _normalizar_config_checkpoint(
    value: Mapping[str, Any],
    source: Path,
    *,
    allow_empty: bool,
) -> dict[str, Any]:
    """Valida e normaliza a configuração arquitetural do checkpoint."""

    if not value:
        if allow_empty:
            return {}
        raise ValueError(f"Checkpoint incompatível: config vazia: {source}")
    try:
        return PointNet2Config.from_mapping(value).to_dict()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Checkpoint incompatível: config inválida: {source}") from exc


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
    elif model_config is not None:
        if not isinstance(model_config, Mapping):
            raise ValueError("config do checkpoint deve ser um mapping")
        model_config = _normalizar_config_checkpoint(model_config, target, allow_empty=False)
    else:
        raise ValueError(
            "Não é possível salvar checkpoint sem uma configuração PointNet2Config."
        )
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
    # Replace atomically so an interrupted epoch cannot leave a half-written
    # checkpoint that looks valid to the next run.
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
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

    Checkpoints antigos que são apenas um ``state_dict`` também são aceitos
    quando o modelo é fornecido pelo chamador. Checkpoints de formato 1 têm
    a configuração normalizada (incluindo o pré-processamento padrão) e
    recebem ``migrated_from_format_version`` no payload retornado.
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
    version = result.get("format_version", 0)
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        raise ValueError(f"Checkpoint inválido: format_version não é inteiro: {source}")
    if version > FORMAT_VERSION:
        raise ValueError(
            f"Checkpoint incompatível: versão {version} > {FORMAT_VERSION}: {source}"
        )
    if not isinstance(result.get("state_dict"), Mapping):
        raise ValueError(f"Checkpoint inválido: state_dict ausente ou não é mapping: {source}")
    if any(not isinstance(key, str) for key in result["state_dict"]):
        raise ValueError(f"Checkpoint inválido: state_dict contém chave não textual: {source}")
    tensor_type = getattr(torch, "Tensor", None)
    if tensor_type is not None and any(
        not isinstance(value, tensor_type) for value in result["state_dict"].values()
    ):
        raise ValueError(f"Checkpoint inválido: state_dict contém valor não tensor: {source}")
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
    result["config"] = _normalizar_config_checkpoint(
        config,
        source,
        allow_empty=version == 0,
    )
    result["metrics"] = dict(metrics)
    if version == 1:
        # Version 1 did not serialize preprocessing. The current default is
        # raw points, so migration is deterministic and explicit in metadata.
        result["migrated_from_format_version"] = version
    if model is not None:
        incompatible = model.load_state_dict(result["state_dict"], strict=strict)
        if not strict:
            result["missing_keys"] = list(getattr(incompatible, "missing_keys", []))
            result["unexpected_keys"] = list(getattr(incompatible, "unexpected_keys", []))
    if optimizer is not None and result.get("optimizer_state_dict"):
        optimizer.load_state_dict(result["optimizer_state_dict"])
    return result

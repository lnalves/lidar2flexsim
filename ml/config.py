"""Configuração serializável do PointNet++ e do treinamento."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .preprocessing import PointPreprocessingConfig


DEFAULT_CLASS_NAMES = (
    "background",
    "Box",
    "ELFplusplus",
    "CargoBike",
    "FTS",
    "ForkLift",
)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() not in {"", "0", "false", "no", "nao", "não", "off"}
    return bool(value)


def _as_tuple(value: Any, fallback: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return fallback
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(str(item) for item in value)


@dataclass(frozen=True)
class PointNet2Config:
    """Arquitetura e pré-processamento do segmentador.

    O modelo implementado usa abstração hierárquica (amostragem + vizinhos)
    e propagação de características, seguindo a ideia do PointNet++ sem
    exigir ``torch-geometric``. ``sa*_points`` são reduzidos automaticamente
    quando um scan possui menos pontos.
    """

    num_classes: int = 6
    in_channels: int = 4
    input_points: int = 4096
    sa1_points: int = 256
    sa2_points: int = 64
    neighbors: int = 16
    hidden_channels: int = 64
    dropout: float = 0.10
    class_names: tuple[str, ...] = field(default_factory=lambda: DEFAULT_CLASS_NAMES)
    preprocessing: PointPreprocessingConfig = field(default_factory=PointPreprocessingConfig)

    def __post_init__(self) -> None:
        for name in (
            "num_classes",
            "in_channels",
            "input_points",
            "sa1_points",
            "sa2_points",
            "neighbors",
            "hidden_channels",
        ):
            value = int(getattr(self, name))
            if value < 1:
                raise ValueError(f"{name} deve ser maior que zero")
            object.__setattr__(self, name, value)
        dropout = float(self.dropout)
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout deve estar no intervalo [0, 1)")
        object.__setattr__(self, "dropout", dropout)
        classes = _as_tuple(self.class_names, DEFAULT_CLASS_NAMES)
        if len(classes) != self.num_classes:
            raise ValueError("class_names deve conter exatamente num_classes itens")
        object.__setattr__(self, "class_names", classes)
        preprocessing = self.preprocessing
        if not isinstance(preprocessing, PointPreprocessingConfig):
            preprocessing = PointPreprocessingConfig.from_mapping(preprocessing)
        object.__setattr__(self, "preprocessing", preprocessing)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None = None, **overrides: Any) -> "PointNet2Config":
        data = dict(value or {})
        if isinstance(data.get("model"), Mapping):
            data = dict(data["model"])
        elif isinstance(data.get("pointnet2"), Mapping):
            data = dict(data["pointnet2"])
        data.update(overrides)
        aliases = {
            "npoints": "input_points",
            "num_points": "input_points",
            "npoint1": "sa1_points",
            "npoint2": "sa2_points",
            "k_neighbors": "neighbors",
            "feature_channels": "hidden_channels",
            "classes": "class_names",
        }
        for old, new in aliases.items():
            if old in data and new not in data:
                data[new] = data.pop(old)
        if "class_names" in data and "num_classes" not in data:
            data["num_classes"] = len(_as_tuple(data["class_names"], DEFAULT_CLASS_NAMES))
        fields = {
            "num_classes", "in_channels", "input_points", "sa1_points",
            "sa2_points", "neighbors", "hidden_channels", "dropout", "class_names",
            "preprocessing",
        }
        unknown = sorted(set(data) - fields)
        if unknown:
            raise ValueError("Opções desconhecidas do modelo: " + ", ".join(unknown))
        if "class_names" in data:
            data["class_names"] = _as_tuple(data["class_names"], DEFAULT_CLASS_NAMES)
        if "preprocessing" in data:
            data["preprocessing"] = PointPreprocessingConfig.from_mapping(data["preprocessing"])
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["class_names"] = list(self.class_names)
        data["preprocessing"] = self.preprocessing.to_dict()
        return data


@dataclass(frozen=True)
class TrainingConfig:
    """Opções do laço de treinamento e do split temporal do dataset."""

    epochs: int = 20
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_fraction: float = 0.2
    seed: int = 42
    device: str = "auto"
    score_threshold: float = 0.50
    checkpoint_every: int = 1

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "checkpoint_every"):
            value = int(getattr(self, name))
            if value < 1:
                raise ValueError(f"{name} deve ser maior que zero")
            object.__setattr__(self, name, value)
        for name in ("learning_rate", "weight_decay"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} deve ser não negativo")
            object.__setattr__(self, name, value)
        fraction = float(self.validation_fraction)
        if not 0 < fraction < 1:
            raise ValueError("validation_fraction deve estar entre 0 e 1")
        object.__setattr__(self, "validation_fraction", fraction)
        threshold = float(self.score_threshold)
        if not 0 <= threshold <= 1:
            raise ValueError("score_threshold deve estar entre 0 e 1")
        object.__setattr__(self, "score_threshold", threshold)
        object.__setattr__(self, "device", str(self.device))
        object.__setattr__(self, "seed", int(self.seed))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None = None, **overrides: Any) -> "TrainingConfig":
        data = dict(value or {})
        if isinstance(data.get("training"), Mapping):
            data = dict(data["training"])
        data.update(overrides)
        unknown = sorted(set(data) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError("Opções desconhecidas do treinamento: " + ", ".join(unknown))
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Fallback pequeno para YAML plano quando PyYAML não está instalado.

    O arquivo de configuração distribuído é deliberadamente plano. Esse
    parser suporta strings, números, booleanos e listas JSON, o suficiente
    para editar o experimento sem tornar PyYAML obrigatório.
    """

    result: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, raw = line.split(":", 1)
        key, raw = key.strip(), raw.strip()
        if not raw:
            result[key] = None
            continue
        try:
            result[key] = json.loads(raw)
        except json.JSONDecodeError:
            lowered = raw.casefold()
            if lowered in {"true", "false"}:
                result[key] = lowered == "true"
            else:
                result[key] = raw.strip("'\"")
    return result


def load_config(path: str | Path) -> tuple[PointNet2Config, TrainingConfig]:
    """Carrega um arquivo JSON/YAML com seções ``model`` e ``training``."""

    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    text = file_path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(text) or {}
        except ImportError:
            data = _parse_simple_yaml(text)
    if not isinstance(data, Mapping):
        raise ValueError("A configuração deve ser um mapping")
    # Accept a flat model config too, which is useful for quick experiments.
    model_data = data.get("model", data.get("pointnet2", data))
    training_data = data.get("training", {})
    if not isinstance(model_data, Mapping) or not isinstance(training_data, Mapping):
        raise ValueError("As seções model e training devem ser mappings")
    return PointNet2Config.from_mapping(model_data), TrainingConfig.from_mapping(training_data)


def save_config(path: str | Path, model: PointNet2Config, training: TrainingConfig | None = None) -> Path:
    """Salva configuração em JSON (inclusive se a extensão for ``.yaml``)."""

    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {"model": model.to_dict()}
    if training is not None:
        data["training"] = training.to_dict()
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target

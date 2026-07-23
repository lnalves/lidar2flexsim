"""Calibração determinística das caixas produzidas pelo PointNet++."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PredictionCalibrationConfig:
    """Filtros pós-segmentação registrados junto ao experimento."""

    min_score: float = 0.0
    min_points: int = 1
    max_iou: float = 0.85
    per_class_min_score: Mapping[str, float] = field(default_factory=dict)
    per_class_min_points: Mapping[str, int] = field(default_factory=dict)
    per_class_min_dimensions: Mapping[str, Sequence[float]] = field(default_factory=dict)
    per_class_max_dimensions: Mapping[str, Sequence[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        score = float(self.min_score)
        if not 0 <= score <= 1:
            raise ValueError("min_score deve estar entre 0 e 1")
        object.__setattr__(self, "min_score", score)
        points = int(self.min_points)
        if points < 1:
            raise ValueError("min_points deve ser maior que zero")
        object.__setattr__(self, "min_points", points)
        iou = float(self.max_iou)
        if not 0 < iou <= 1:
            raise ValueError("max_iou deve estar entre 0 e 1")
        object.__setattr__(self, "max_iou", iou)

    @classmethod
    def from_mapping(cls, value: "PredictionCalibrationConfig | Mapping[str, Any] | None" = None) -> "PredictionCalibrationConfig":
        if isinstance(value, cls):
            return value
        data = dict(value or {})
        aliases = {
            "score_threshold": "min_score",
            "min_cluster_points": "min_points",
            "nms_iou": "max_iou",
        }
        for old, new in aliases.items():
            if old in data and new not in data:
                data[new] = data.pop(old)
        allowed = {
            "min_score", "min_points", "max_iou", "per_class_min_score",
            "per_class_min_points", "per_class_min_dimensions", "per_class_max_dimensions",
        }
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError("Opções desconhecidas da calibração: " + ", ".join(unknown))
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_score": self.min_score,
            "min_points": self.min_points,
            "max_iou": self.max_iou,
            "per_class_min_score": dict(self.per_class_min_score),
            "per_class_min_points": {str(key): int(value) for key, value in self.per_class_min_points.items()},
            "per_class_min_dimensions": {str(key): list(value) for key, value in self.per_class_min_dimensions.items()},
            "per_class_max_dimensions": {str(key): list(value) for key, value in self.per_class_max_dimensions.items()},
        }


def _aabb_iou(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    a_center = np.asarray(first.get("centro", [0, 0, 0]), dtype=float)
    b_center = np.asarray(second.get("centro", [0, 0, 0]), dtype=float)
    a_size = np.asarray(first.get("dimensoes", [0, 0, 0]), dtype=float)
    b_size = np.asarray(second.get("dimensoes", [0, 0, 0]), dtype=float)
    a_min, a_max = a_center - a_size / 2, a_center + a_size / 2
    b_min, b_max = b_center - b_size / 2, b_center + b_size / 2
    intersection = np.maximum(0.0, np.minimum(a_max, b_max) - np.maximum(a_min, b_min))
    volume = float(np.prod(intersection))
    union = float(np.prod(np.maximum(a_size, 0)) + np.prod(np.maximum(b_size, 0)) - volume)
    return volume / union if union > 0 else 0.0


def _dimensions_valid(value: Any, minimum: Sequence[float] | None, maximum: Sequence[float] | None) -> bool:
    dimensions = np.asarray(value, dtype=float)
    if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
        return False
    if minimum is not None and np.any(dimensions < np.asarray(minimum, dtype=float)):
        return False
    if maximum is not None and np.any(dimensions > np.asarray(maximum, dtype=float)):
        return False
    return True


def calibrate_predictions(
    predictions: Sequence[Mapping[str, Any]],
    clusters: Sequence[Any] | None = None,
    config: PredictionCalibrationConfig | Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[Any], dict[str, Any]]:
    """Filtra caixas e suprime duplicatas de forma determinística."""

    settings = PredictionCalibrationConfig.from_mapping(config)
    candidates: list[tuple[dict[str, Any], Any]] = []
    removed = {"score": 0, "points": 0, "dimensions": 0, "duplicate": 0}
    source_clusters = list(clusters or [])
    for index, value in enumerate(predictions):
        item = dict(value)
        class_name = str(item.get("classe", item.get("class_name", "")))
        score = float(item.get("score", 0.0))
        min_score = max(settings.min_score, float(settings.per_class_min_score.get(class_name, 0.0)))
        if score < min_score:
            removed["score"] += 1
            continue
        point_count = int(item.get("n_pontos", item.get("num_pontos", 0)) or 0)
        if point_count < max(settings.min_points, int(settings.per_class_min_points.get(class_name, 1))):
            removed["points"] += 1
            continue
        if not _dimensions_valid(
            item.get("dimensoes"),
            settings.per_class_min_dimensions.get(class_name),
            settings.per_class_max_dimensions.get(class_name),
        ):
            removed["dimensions"] += 1
            continue
        cluster = source_clusters[index] if index < len(source_clusters) else None
        candidates.append((item, cluster))

    candidates.sort(key=lambda pair: (-float(pair[0].get("score", 0.0)), str(pair[0].get("classe", ""))))
    kept: list[tuple[dict[str, Any], Any]] = []
    for candidate, cluster in candidates:
        if any(
            str(candidate.get("classe")) == str(existing.get("classe"))
            and _aabb_iou(candidate, existing) >= settings.max_iou
            for existing, _ in kept
        ):
            removed["duplicate"] += 1
            continue
        kept.append((candidate, cluster))
    return (
        [item for item, _ in kept],
        [cluster for _, cluster in kept if cluster is not None],
        {"input_predictions": len(predictions), "output_predictions": len(kept), "removed": removed, "calibration": settings.to_dict()},
    )


__all__ = ["PredictionCalibrationConfig", "calibrate_predictions"]

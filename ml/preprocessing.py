"""Pré-processamento compartilhado entre treino, validação e inferência.

O módulo não depende de PyTorch nem Open3D. Isso permite que a preparação dos
dados seja exatamente a mesma nos datasets, na inferência e nos testes leves.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() not in {
            "", "0", "false", "no", "nao", "não", "off",
        }
    return bool(value)


@dataclass(frozen=True)
class PointPreprocessingConfig:
    """Configuração serializável da transformação espacial dos pontos.

    Por padrão não removemos o chão implicitamente. O treino e a inferência
    começam da mesma nuvem e qualquer filtragem precisa estar registrada no
    checkpoint ou ser passada explicitamente pela API.
    """

    voxel: float = 0.0
    remove_ground: bool = False
    plane_distance: float = 0.05
    max_ground_tilt_deg: float = 25.0
    ground_quantile: float = 0.30
    remove_outliers: bool = False
    outlier_neighbors: int = 12
    outlier_std_ratio: float = 2.5

    def __post_init__(self) -> None:
        voxel = float(self.voxel)
        if not math.isfinite(voxel) or voxel < 0:
            raise ValueError("voxel deve ser zero ou maior")
        object.__setattr__(self, "voxel", voxel)

        distance = float(self.plane_distance)
        if not math.isfinite(distance) or distance <= 0:
            raise ValueError("plane_distance deve ser maior que zero")
        object.__setattr__(self, "plane_distance", distance)

        tilt = float(self.max_ground_tilt_deg)
        if not math.isfinite(tilt) or not 0 <= tilt < 90:
            raise ValueError("max_ground_tilt_deg deve estar entre 0 e 90")
        object.__setattr__(self, "max_ground_tilt_deg", tilt)

        quantile = float(self.ground_quantile)
        if not math.isfinite(quantile) or not 0.01 <= quantile <= 0.9:
            raise ValueError("ground_quantile deve estar entre 0,01 e 0,90")
        object.__setattr__(self, "ground_quantile", quantile)

        neighbors = int(self.outlier_neighbors)
        if neighbors < 0:
            raise ValueError("outlier_neighbors não pode ser negativo")
        object.__setattr__(self, "outlier_neighbors", neighbors)

        ratio = float(self.outlier_std_ratio)
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError("outlier_std_ratio deve ser maior que zero")
        object.__setattr__(self, "outlier_std_ratio", ratio)
        object.__setattr__(self, "remove_ground", _as_bool(self.remove_ground))
        object.__setattr__(self, "remove_outliers", _as_bool(self.remove_outliers))

    @classmethod
    def from_mapping(
        cls,
        value: "PointPreprocessingConfig | Mapping[str, Any] | None" = None,
        **overrides: Any,
    ) -> "PointPreprocessingConfig":
        if isinstance(value, cls):
            data = value.to_dict()
        else:
            data = dict(value or {})
        if isinstance(data.get("preprocessing"), Mapping):
            data = dict(data["preprocessing"])
        data.update(overrides)
        aliases = {
            "plane_dist": "plane_distance",
            "max_ground_tilt": "max_ground_tilt_deg",
            "ground_q": "ground_quantile",
            "outlier_k": "outlier_neighbors",
            "outlier_std": "outlier_std_ratio",
        }
        for old, new in aliases.items():
            if old in data and new not in data:
                data[new] = data.pop(old)
        fields = {
            "voxel", "remove_ground", "plane_distance", "max_ground_tilt_deg",
            "ground_quantile", "remove_outliers", "outlier_neighbors",
            "outlier_std_ratio",
        }
        unknown = sorted(set(data) - fields)
        if unknown:
            raise ValueError("Opções desconhecidas do pré-processamento: " + ", ".join(unknown))
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "voxel": float(self.voxel),
            "remove_ground": bool(self.remove_ground),
            "plane_distance": float(self.plane_distance),
            "max_ground_tilt_deg": float(self.max_ground_tilt_deg),
            "ground_quantile": float(self.ground_quantile),
            "remove_outliers": bool(self.remove_outliers),
            "outlier_neighbors": int(self.outlier_neighbors),
            "outlier_std_ratio": float(self.outlier_std_ratio),
        }


def _coerce_points(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError("points deve ter formato [N, >=3]")
    if not np.isfinite(array[:, :3]).all():
        raise ValueError("points contém coordenadas não finitas")
    return np.ascontiguousarray(array)


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    if len(points) == 0 or voxel <= 0:
        return points
    cells = np.floor(points[:, :3] / float(voxel)).astype(np.int64)
    _, keep = np.unique(cells, axis=0, return_index=True)
    return np.ascontiguousarray(points[np.sort(keep)])


def estimate_ground_mask(
    points: np.ndarray,
    quantile: float,
    max_tilt_deg: float,
    distance: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Retorna máscara de pontos mantidos após remover um piso horizontal."""

    if len(points) < 3:
        return np.ones(len(points), dtype=bool), {"detected": False}
    z = points[:, 2].astype(np.float64)
    cutoff = float(np.quantile(z, np.clip(float(quantile), 0.01, 0.9)))
    candidates = z <= cutoff
    if candidates.sum() < 2:
        # Quantiles on very small synthetic scans may select fewer than two
        # exact floor returns; use the lowest returns as a stable fallback.
        candidates = np.zeros(len(points), dtype=bool)
        candidates[np.argsort(z)[: min(3, len(points))]] = True
    low = z[candidates]
    plane_z = float(np.median(low))
    inliers = np.abs(z - plane_z) <= max(float(distance), 1e-3)
    diagnostics = {
        "detected": bool(inliers.sum() >= min(2, len(points))),
        "normal": [0.0, 0.0, 1.0],
        "height": plane_z,
        "offset": -plane_z,
        "tilt_deg": 0.0,
        "inliers": int(inliers.sum()),
        "candidate_points": int(candidates.sum()),
        "max_tilt_deg": float(max_tilt_deg),
    }
    return ~inliers, diagnostics


def remove_sparse_outliers(
    points: np.ndarray,
    neighbors: int,
    std_ratio: float,
) -> np.ndarray:
    if len(points) < 4 or int(neighbors) <= 0:
        return points
    cells = np.floor(points[:, :3] / 0.10).astype(np.int64)
    _, inverse, counts = np.unique(
        cells, axis=0, return_inverse=True, return_counts=True
    )
    expected = max(1.0, float(np.median(counts)))
    threshold = max(1.0, expected / max(float(std_ratio), 0.1))
    keep = counts[inverse] >= threshold
    return points[keep] if keep.sum() >= max(3, min(len(points), int(neighbors))) else points


def preprocess_points(
    points: np.ndarray,
    config: PointPreprocessingConfig | Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Aplica a transformação espacial e retorna pontos + diagnóstico."""

    current = _coerce_points(points)
    settings = PointPreprocessingConfig.from_mapping(config)
    raw_count = len(current)
    current = voxel_downsample(current, settings.voxel)
    voxel_count = len(current)
    ground: dict[str, Any] = {
        "detected": False,
        "inliers": 0,
        "candidate_points": 0,
        "height": None,
        "offset": None,
        "tilt_deg": None,
    }
    removed_ground = 0
    if settings.remove_ground:
        keep, ground = estimate_ground_mask(
            current,
            settings.ground_quantile,
            settings.max_ground_tilt_deg,
            settings.plane_distance,
        )
        removed_ground = int((~keep).sum())
        current = current[keep]
    if settings.remove_outliers:
        current = remove_sparse_outliers(
            current, settings.outlier_neighbors, settings.outlier_std_ratio
        )
    # A scene entirely classified as floor must remain representable by the
    # fixed-size sampler. Keeping the original voxelized points is safer than
    # silently manufacturing an all-zero sample.
    fallback_no_points = False
    if len(current) == 0 and voxel_count:
        current = voxel_downsample(_coerce_points(points), settings.voxel)
        fallback_no_points = True
    diagnostics = {
        "input_points": int(raw_count),
        "voxel_points": int(voxel_count),
        "processed_points": int(len(current)),
        "removed_ground_points": removed_ground,
        "fallback_no_points": fallback_no_points,
        "ground": ground,
        "preprocessing": settings.to_dict(),
    }
    return np.ascontiguousarray(current), diagnostics


__all__ = [
    "PointPreprocessingConfig",
    "estimate_ground_mask",
    "preprocess_points",
    "remove_sparse_outliers",
    "voxel_downsample",
]

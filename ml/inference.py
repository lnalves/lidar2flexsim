"""Inferência PointNet++ e conversão de labels por ponto em caixas 3D."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .checkpoints import load_checkpoint
from .calibration import PredictionCalibrationConfig, calibrate_predictions
from .config import DEFAULT_CLASS_NAMES, PointNet2Config
from .data import load_bin, prepare_point_sample
from .dependencies import require_torch
from .models.pointnet2_seg import PointNet2Segmentation
from .preprocessing import (
    PointPreprocessingConfig,
    preprocess_points,
)


def _normalizar_device(torch: Any, device: str) -> Any:
    name = str(device).strip().lower()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        name = "cpu"
    if name == "mps":
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is None or not bool(mps.is_available()):
            name = "cpu"
    return torch.device(name)


def _coerce_points(scan: str | Path | np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    if isinstance(scan, (str, Path)):
        # The shared dataset adapter preserves all four VLP-16 features.
        return load_bin(scan, num_features=4, return_features=4)
    points = np.asarray(scan, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("scan deve ser caminho .bin ou array [N, >=3]")
    if points.shape[1] == 3:
        points = np.column_stack([points, np.zeros(len(points), dtype=np.float32)])
    return np.ascontiguousarray(points[:, :4], dtype=np.float32)

def _prepare_points(points: np.ndarray, count: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        values = np.zeros((count, 4), dtype=np.float32)
        return values, np.zeros(count, dtype=np.int64)
    sample = prepare_point_sample(
        points,
        num_points=count,
        rng=np.random.default_rng(seed),
        preserve_foreground=False,
    )
    return sample.points, sample.indices


_NEIGHBOR_OFFSETS = np.array(
    [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
    dtype=np.int64,
)


def _cluster_indices(points: np.ndarray, eps: float, min_points: int) -> list[np.ndarray]:
    """Cluster a class mask with a voxel-bucketed Euclidean BFS.

    Each expansion tests every candidate of the 27 neighbouring cells in a
    single vectorized comparison. Doing it point by point made the clustering
    the slowest stage of inference on dense scans.
    """

    if len(points) == 0:
        return []
    radius = max(float(eps), 1e-4)
    coordinates = np.ascontiguousarray(points[:, :3], dtype=np.float64)
    cells = np.floor(coordinates / radius).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, cell in enumerate(cells):
        buckets.setdefault((int(cell[0]), int(cell[1]), int(cell[2])), []).append(index)
    bucket_arrays = {
        key: np.asarray(value, dtype=np.int64) for key, value in buckets.items()
    }
    squared_radius = radius * radius
    visited = np.zeros(len(coordinates), dtype=bool)
    clusters: list[np.ndarray] = []
    for root in range(len(coordinates)):
        if visited[root]:
            continue
        visited[root] = True
        queue = [root]
        component: list[int] = []
        while queue:
            current = queue.pop()
            component.append(current)
            neighborhood = cells[current] + _NEIGHBOR_OFFSETS
            groups = [
                bucket_arrays[key]
                for key in map(tuple, neighborhood.tolist())
                if key in bucket_arrays
            ]
            if not groups:
                continue
            candidates = np.concatenate(groups)
            candidates = candidates[~visited[candidates]]
            if not len(candidates):
                continue
            deltas = coordinates[candidates] - coordinates[current]
            near = candidates[np.einsum("ij,ij->i", deltas, deltas) <= squared_radius]
            if not len(near):
                continue
            visited[near] = True
            queue.extend(int(item) for item in near)
        if len(component) >= max(1, int(min_points)):
            clusters.append(np.asarray(component, dtype=np.int64))
    return clusters


def _oriented_box(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    xyz = points[:, :3].astype(np.float64)
    if len(xyz) == 0:
        return (
            np.zeros(3, dtype=np.float32),
            np.full(3, 0.05, dtype=np.float32),
            0.0,
        )
    center_xy = xyz[:, :2].mean(axis=0)
    if len(xyz) >= 2:
        centered_xy = xyz[:, :2] - center_xy
        covariance = centered_xy.T @ centered_xy
        _, vectors = np.linalg.eigh(covariance)
        axis = vectors[:, -1]
        # PCA axes have arbitrary sign; canonicalize yaw for reproducible
        # exports without changing the represented box.
        if axis[0] < 0 or (abs(axis[0]) <= 1e-12 and axis[1] < 0):
            axis = -axis
        yaw = float(math.atan2(axis[1], axis[0]))
        cosine, sine = math.cos(yaw), math.sin(yaw)
        local_x = cosine * centered_xy[:, 0] + sine * centered_xy[:, 1]
        local_y = -sine * centered_xy[:, 0] + cosine * centered_xy[:, 1]
        local_center = np.array(
            [0.5 * (local_x.min() + local_x.max()),
             0.5 * (local_y.min() + local_y.max())],
            dtype=np.float64,
        )
        center = np.array(
            [
                center_xy[0] + cosine * local_center[0] - sine * local_center[1],
                center_xy[1] + sine * local_center[0] + cosine * local_center[1],
                0.5 * (xyz[:, 2].min() + xyz[:, 2].max()),
            ],
            dtype=np.float64,
        )
        dimensions = np.array(
            [np.ptp(local_x), np.ptp(local_y), np.ptp(xyz[:, 2])], dtype=np.float64
        )
    else:
        yaw = 0.0
        center = xyz[0].copy()
        dimensions = np.zeros(3, dtype=np.float64)
    dimensions = np.maximum(dimensions, 0.05)
    return center.astype(np.float32), dimensions.astype(np.float32), yaw

def load_segmentation_model(
    checkpoint: str | Path,
    *,
    device: str = "cpu",
) -> tuple[Any, PointNet2Config]:
    """Instancia o segmentador de um checkpoint e devolve modelo e configuração.

    O streaming em tempo real reaproveita o modelo entre quadros; carregá-lo
    aqui, e não dentro de :func:`predict_points`, é o que permite pagar o
    custo do checkpoint uma única vez.
    """

    torch = require_torch("carregar o segmentador PointNet++")
    device_obj = _normalizar_device(torch, device)
    payload = load_checkpoint(checkpoint, map_location=device_obj)
    model_config = PointNet2Config.from_mapping(payload.get("config") or {})
    model = PointNet2Segmentation(model_config)
    load_checkpoint(checkpoint, model=model, map_location=device_obj)
    model.to(device_obj)
    model.eval()
    return model, model_config


def predict_points(
    scan: str | Path | np.ndarray | Sequence[Sequence[float]],
    *,
    checkpoint: str | Path | None = None,
    model: Any | None = None,
    device: str = "cpu",
    num_points: int | None = None,
    sampling_seed: int = 0,
    debug_diagnostics: bool = False,
    preprocessing: PointPreprocessingConfig | Mapping[str, Any] | None = None,
    voxel: float | None = None,
    remove_ground: bool | None = None,
    plane_distance: float | None = None,
    max_ground_tilt_deg: float | None = None,
    ground_quantile: float | None = None,
    remove_outliers: bool | None = None,
    outlier_neighbors: int | None = None,
    outlier_std_ratio: float | None = None,
) -> dict[str, Any]:
    """Executa somente a segmentação e retorna labels por ponto + diagnósticos."""

    torch = require_torch("executar inferência PointNet++")
    raw = _coerce_points(scan)
    device_obj = _normalizar_device(torch, device)
    # Model configuration determines the number of points when caller omits it.
    if model is None and checkpoint is not None:
        model, model_config = load_segmentation_model(checkpoint, device=device)
    elif model is not None:
        model_config = getattr(model, "config", PointNet2Config())
    else:
        raise RuntimeError("Informe checkpoint=... ou forneça model=... para inferência PointNet++.")
    if not isinstance(model_config, PointNet2Config):
        model_config = PointNet2Config.from_mapping(model_config)
    settings = model_config.preprocessing
    if preprocessing is not None:
        settings = PointPreprocessingConfig.from_mapping(preprocessing)
    overrides = {
        name: value for name, value in {
            "voxel": voxel,
            "remove_ground": remove_ground,
            "plane_distance": plane_distance,
            "max_ground_tilt_deg": max_ground_tilt_deg,
            "ground_quantile": ground_quantile,
            "remove_outliers": remove_outliers,
            "outlier_neighbors": outlier_neighbors,
            "outlier_std_ratio": outlier_std_ratio,
        }.items() if value is not None
    }
    if overrides:
        settings = PointPreprocessingConfig.from_mapping(settings, **overrides)
    processed, preprocessing_diagnostics = preprocess_points(raw, settings)
    requested = int(num_points or 0)
    count = requested if requested > 0 else model_config.input_points
    sampling_seed = int(sampling_seed)
    model_points, selected_indices = _prepare_points(processed, count, seed=sampling_seed)
    tensor = torch.from_numpy(model_points).unsqueeze(0).to(device=device_obj, dtype=torch.float32)
    model.to(device_obj)
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probabilities = logits.softmax(dim=-1)[0]
        confidence, labels = probabilities.max(dim=-1)
    sampling_method = (
        "zero_fill" if len(processed) == 0
        else "without_replacement" if len(processed) >= count
        else "with_replacement"
    )
    diagnostics: dict[str, Any] = {
        "input_points": int(len(raw)),
        "voxel_points": int(preprocessing_diagnostics["voxel_points"]),
        "processed_points": int(preprocessing_diagnostics["processed_points"]),
        "model_points": int(len(model_points)),
        "ground": preprocessing_diagnostics["ground"],
        "preprocessing": settings.to_dict(),
        "sampling": {
            "method": sampling_method,
            "seed": sampling_seed,
            "source_points": int(len(processed)),
            "requested_points": int(count),
            "selected_points": int(len(model_points)),
            "replacement": bool(sampling_method == "with_replacement"),
        },
        "device": str(device_obj),
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
    }
    if debug_diagnostics:
        diagnostics["selected_indices"] = selected_indices.tolist()
    return {
        "points": model_points,
        "labels": labels.cpu().numpy().astype(np.int64),
        "confidence": confidence.cpu().numpy().astype(np.float32),
        "class_names": model_config.class_names,
        "model_calibration": dict(model_config.calibration),
        "diagnostics": diagnostics,
    }


def inferir_scan(
    scan: str | Path | np.ndarray | Sequence[Sequence[float]],
    checkpoint: str | Path | None = None,
    *,
    model: Any | None = None,
    device: str = "cpu",
    num_points: int | None = None,
    sampling_seed: int = 0,
    debug_diagnostics: bool = False,
    score_threshold: float = 0.50,
    cluster_eps: float = 0.35,
    min_cluster_points: int = 5,
    class_names: Sequence[str] | None = None,
    calibration: PredictionCalibrationConfig | Mapping[str, Any] | None = None,
    preprocessing: PointPreprocessingConfig | Mapping[str, Any] | None = None,
    voxel: float | None = None,
    remove_ground: bool | None = None,
    plane_distance: float | None = None,
    max_ground_tilt_deg: float | None = None,
    ground_quantile: float | None = None,
    remove_outliers: bool | None = None,
    outlier_neighbors: int | None = None,
    outlier_std_ratio: float | None = None,
) -> dict[str, Any]:
    """Detecta objetos e retorna ``predictions`` e ``diagnostics``.

    A segmentação PointNet++ é ponto a ponto; esta função agrupa os pontos de
    cada classe em componentes espaciais e ajusta uma caixa orientada. O
    formato das previsões contém ``classe``, ``centro``, ``dimensoes`` e
    ``rotacao`` em radianos.
    """

    threshold = float(score_threshold)
    if not 0 <= threshold <= 1:
        raise ValueError("score_threshold deve estar entre 0 e 1")
    if float(cluster_eps) <= 0:
        raise ValueError("cluster_eps deve ser maior que zero")
    if int(min_cluster_points) < 1:
        raise ValueError("min_cluster_points deve ser maior que zero")
    result = predict_points(
        scan,
        checkpoint=checkpoint,
        model=model,
        device=device,
        num_points=num_points,
        sampling_seed=sampling_seed,
        debug_diagnostics=debug_diagnostics,
        preprocessing=preprocessing,
        voxel=voxel,
        remove_ground=remove_ground,
        plane_distance=plane_distance,
        max_ground_tilt_deg=max_ground_tilt_deg,
        ground_quantile=ground_quantile,
        remove_outliers=remove_outliers,
        outlier_neighbors=outlier_neighbors,
        outlier_std_ratio=outlier_std_ratio,
    )
    points = np.asarray(result["points"])
    labels = np.asarray(result["labels"])
    confidence = np.asarray(result["confidence"])
    names = tuple(class_names or result["class_names"] or DEFAULT_CLASS_NAMES)
    # Keep point clusters alongside predictions for calibration and analysis.
    records: list[tuple[dict[str, Any], np.ndarray]] = []
    class_counts: dict[str, int] = {}
    for class_id in range(1, len(names)):
        class_name = str(names[class_id])
        mask = (labels == class_id) & (confidence >= threshold)
        class_counts[class_name] = int(mask.sum())
        if not mask.any():
            continue
        candidate_points = points[mask]
        candidate_scores = confidence[mask]
        for cluster in _cluster_indices(candidate_points, float(cluster_eps), int(min_cluster_points)):
            cluster_points = candidate_points[cluster]
            center, dimensions, yaw = _oriented_box(cluster_points)
            score = float(candidate_scores[cluster].mean())
            records.append((
                {
                    "classe": class_name,
                    "class_id": int(class_id),
                    "score": score,
                    "centro": center.tolist(),
                    "dimensoes": dimensions.tolist(),
                    "rotacao": float(yaw),
                    "num_pontos": int(len(cluster_points)),
                },
                np.ascontiguousarray(cluster_points, dtype=np.float32),
            ))
    records.sort(key=lambda item: (-float(item[0]["score"]), str(item[0]["classe"])))
    predictions = [item[0] for item in records]
    clusters = [item[1] for item in records]
    configured_calibration = calibration
    if configured_calibration is None:
        configured_calibration = result["model_calibration"]
    predictions, clusters, calibration_diagnostics = calibrate_predictions(
        predictions, clusters, configured_calibration
    )
    diagnostics = dict(result["diagnostics"])
    diagnostics.update(
        {
            "predictions": int(len(predictions)),
            "object_points": int(((labels > 0) & (confidence >= threshold)).sum()),
            "score_threshold": threshold,
            "cluster_eps": float(cluster_eps),
            "min_cluster_points": int(min_cluster_points),
            "class_counts": class_counts,
            "calibration": calibration_diagnostics,
        }
    )
    return {
        "predictions": predictions,
        "clusters": clusters,
        "diagnostics": diagnostics,
    }

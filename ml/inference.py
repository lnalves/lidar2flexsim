"""Inferência PointNet++ e conversão de labels em caixas para o FlexSim."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .checkpoints import load_checkpoint
from .config import DEFAULT_CLASS_NAMES, PointNet2Config
from .data import load_bin
from .dependencies import require_torch
from .models.pointnet2_seg import PointNet2Segmentation


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


def _voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    if len(points) == 0 or voxel <= 0:
        return points
    cells = np.floor(points[:, :3] / float(voxel)).astype(np.int64)
    _, keep = np.unique(cells, axis=0, return_index=True)
    return points[np.sort(keep)]


def _estimate_ground(points: np.ndarray, quantile: float, max_tilt_deg: float, distance: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Estima um piso horizontal robusto usando a faixa inferior de altura."""

    if len(points) < 3:
        return np.ones(len(points), dtype=bool), {"detected": False}
    z = points[:, 2].astype(np.float64)
    cutoff = float(np.quantile(z, np.clip(float(quantile), 0.01, 0.9)))
    candidates = z <= cutoff
    if candidates.sum() < 3:
        return np.ones(len(points), dtype=bool), {"detected": False}
    low = z[candidates]
    # A robust center of the floor band is more stable than the single lowest
    # return, which can be a stray point.
    plane_z = float(np.median(low))
    inliers = np.abs(z - plane_z) <= max(float(distance), 1e-3)
    normal = [0.0, 0.0, 1.0]
    diagnostics = {
        "detected": bool(inliers.sum() >= 3),
        "normal": normal,
        "height": plane_z,
        "offset": -plane_z,
        "tilt_deg": 0.0,
        "inliers": int(inliers.sum()),
        "candidate_points": int(candidates.sum()),
        "max_tilt_deg": float(max_tilt_deg),
    }
    return ~inliers, diagnostics


def _remove_sparse_outliers(points: np.ndarray, neighbors: int, std_ratio: float) -> np.ndarray:
    if len(points) < 4 or int(neighbors) <= 0:
        return points
    # Voxel occupancy is a bounded approximation of statistical outlier
    # removal. It avoids the O(N^2) distance matrix for large VLP-16 scans.
    cells = np.floor(points[:, :3] / 0.10).astype(np.int64)
    unique, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    expected = max(1.0, float(np.median(counts)))
    threshold = max(1.0, expected / max(float(std_ratio), 0.1))
    keep = counts[inverse] >= threshold
    # Never erase an entire scan due to an unusually sparse scene.
    return points[keep] if keep.sum() >= max(3, min(len(points), int(neighbors))) else points


def _prepare_points(points: np.ndarray, count: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        values = np.zeros((count, 4), dtype=np.float32)
        return values, np.zeros(count, dtype=np.int64)
    rng = np.random.default_rng(seed)
    if len(points) >= count:
        # Deterministic sampling keeps UI re-runs reproducible.
        indices = np.linspace(0, len(points) - 1, count, dtype=np.int64)
    else:
        indices = np.arange(count, dtype=np.int64) % len(points)
    return points[indices], indices


def _cluster_indices(points: np.ndarray, eps: float, min_points: int) -> list[np.ndarray]:
    """Cluster a class mask with a voxel-bucketed Euclidean BFS."""

    if len(points) == 0:
        return []
    radius = max(float(eps), 1e-4)
    cells = np.floor(points[:, :3] / radius).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, cell in enumerate(cells):
        buckets.setdefault(tuple(int(item) for item in cell), []).append(index)
    visited = np.zeros(len(points), dtype=bool)
    clusters: list[np.ndarray] = []
    for root in range(len(points)):
        if visited[root]:
            continue
        visited[root] = True
        queue = [root]
        component: list[int] = []
        while queue:
            current = queue.pop()
            component.append(current)
            cell = cells[current]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for candidate in buckets.get((int(cell[0] + dx), int(cell[1] + dy), int(cell[2] + dz)), ()): 
                            if visited[candidate]:
                                continue
                            if np.linalg.norm(points[candidate, :3] - points[current, :3]) <= radius:
                                visited[candidate] = True
                                queue.append(candidate)
        if len(component) >= max(1, int(min_points)):
            clusters.append(np.asarray(component, dtype=np.int64))
    return clusters


def _oriented_box(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    xyz = points[:, :3].astype(np.float64)
    center = xyz.mean(axis=0)
    if len(xyz) >= 2:
        centered_xy = xyz[:, :2] - center[:2]
        covariance = centered_xy.T @ centered_xy
        _, vectors = np.linalg.eigh(covariance)
        axis = vectors[:, -1]
        yaw = float(math.atan2(axis[1], axis[0]))
        cosine, sine = math.cos(yaw), math.sin(yaw)
        local_x = cosine * centered_xy[:, 0] + sine * centered_xy[:, 1]
        local_y = -sine * centered_xy[:, 0] + cosine * centered_xy[:, 1]
        dimensions = np.array(
            [np.ptp(local_x), np.ptp(local_y), np.ptp(xyz[:, 2])], dtype=np.float64
        )
    else:
        yaw = 0.0
        dimensions = np.zeros(3, dtype=np.float64)
    dimensions = np.maximum(dimensions, 0.05)
    return center.astype(np.float32), dimensions.astype(np.float32), yaw


def _load_model(
    checkpoint: str | Path | None,
    model: Any | None,
    config: PointNet2Config | Mapping[str, Any] | None,
    device: Any,
) -> tuple[Any, PointNet2Config, dict[str, Any]]:
    if model is not None:
        model_config = getattr(model, "config", None)
        if model_config is None:
            model_config = config
        model_config = model_config if isinstance(model_config, PointNet2Config) else PointNet2Config.from_mapping(model_config)
        model.to(device)
        model.eval()
        return model, model_config, {}
    if checkpoint is None:
        raise RuntimeError("Informe checkpoint=... ou forneça model=... para inferência PointNet++.")
    # Read config before constructing the model, then restore its weights.
    payload = load_checkpoint(checkpoint, map_location=device)
    checkpoint_config = payload.get("config") or (config.to_dict() if isinstance(config, PointNet2Config) else config) or {}
    model_config = checkpoint_config if isinstance(checkpoint_config, PointNet2Config) else PointNet2Config.from_mapping(checkpoint_config)
    model = PointNet2Segmentation(model_config)
    load_checkpoint(checkpoint, model=model, map_location=device)
    model.to(device)
    model.eval()
    return model, model_config, payload


def predict_points(
    scan: str | Path | np.ndarray | Sequence[Sequence[float]],
    *,
    checkpoint: str | Path | None = None,
    model: Any | None = None,
    device: str = "cpu",
    num_points: int | None = None,
    voxel: float = 0.0,
    plane_distance: float = 0.05,
    max_ground_tilt_deg: float = 25.0,
    ground_quantile: float = 0.30,
    remove_outliers: bool = False,
    outlier_neighbors: int = 12,
    outlier_std_ratio: float = 2.5,
) -> dict[str, Any]:
    """Executa somente a segmentação e retorna labels por ponto + diagnósticos."""

    torch = require_torch("executar inferência PointNet++")
    raw = _coerce_points(scan)
    processed = _voxel_downsample(raw, float(voxel)) if float(voxel) > 0 else raw
    keep_mask, ground = _estimate_ground(
        processed,
        ground_quantile,
        max_ground_tilt_deg,
        plane_distance,
    )
    processed = processed[keep_mask]
    if remove_outliers:
        processed = _remove_sparse_outliers(processed, outlier_neighbors, outlier_std_ratio)
    requested = int(num_points or 0)
    device_obj = _normalizar_device(torch, device)
    # Model configuration determines the number of points when caller omits it.
    if model is None and checkpoint is not None:
        payload = load_checkpoint(checkpoint, map_location=device_obj)
        model_config = PointNet2Config.from_mapping(payload.get("config") or {})
        model = PointNet2Segmentation(model_config)
        load_checkpoint(checkpoint, model=model, map_location=device_obj)
    elif model is not None:
        model_config = getattr(model, "config", PointNet2Config())
    else:
        raise RuntimeError("Informe checkpoint=... ou forneça model=... para inferência PointNet++.")
    if not isinstance(model_config, PointNet2Config):
        model_config = PointNet2Config.from_mapping(model_config)
    count = requested if requested > 0 else model_config.input_points
    model_points, selected_indices = _prepare_points(processed, count)
    tensor = torch.from_numpy(model_points).unsqueeze(0).to(device=device_obj, dtype=torch.float32)
    model.to(device_obj)
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probabilities = logits.softmax(dim=-1)[0]
        confidence, labels = probabilities.max(dim=-1)
    return {
        "points": model_points,
        "labels": labels.cpu().numpy().astype(np.int64),
        "confidence": confidence.cpu().numpy().astype(np.float32),
        "class_names": model_config.class_names,
        "diagnostics": {
            "input_points": int(len(raw)),
            "voxel_points": int(len(processed) + int((~keep_mask).sum())),
            "processed_points": int(len(processed)),
            "model_points": int(len(model_points)),
            "selected_indices": selected_indices.tolist(),
            "ground": ground,
            "device": str(device_obj),
            "checkpoint": str(checkpoint) if checkpoint is not None else None,
        },
    }


def inferir_scan(
    scan: str | Path | np.ndarray | Sequence[Sequence[float]],
    checkpoint: str | Path | None = None,
    *,
    model: Any | None = None,
    device: str = "cpu",
    num_points: int | None = None,
    score_threshold: float = 0.50,
    cluster_eps: float = 0.35,
    min_cluster_points: int = 5,
    class_names: Sequence[str] | None = None,
    voxel: float = 0.0,
    plane_distance: float = 0.05,
    max_ground_tilt_deg: float = 25.0,
    ground_quantile: float = 0.30,
    remove_outliers: bool = False,
    outlier_neighbors: int = 12,
    outlier_std_ratio: float = 2.5,
) -> dict[str, Any]:
    """Detecta objetos e retorna ``predictions`` e ``diagnostics``.

    A segmentação PointNet++ é ponto a ponto; esta função agrupa os pontos de
    cada classe em componentes espaciais e ajusta uma caixa orientada. O
    formato das previsões usa nomes em português já consumidos pelo exportador
    existente: ``classe``, ``centro``, ``dimensoes`` e ``rotacao``.
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
        voxel=voxel,
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
    # Keep the point clusters alongside each prediction.  The service layer
    # uses them to generate STL files; the layout-only API can simply ignore
    # this field.
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
                    # ``rotacao_z``/``n_pontos`` are the canonical keys used
                    # by lidar2flexsim.py; the radian/count aliases make the
                    # neural API convenient for notebooks as well.
                    "rotacao_z": math.degrees(float(yaw)),
                    "rotacao": float(yaw),
                    "n_pontos": int(len(cluster_points)),
                    "num_pontos": int(len(cluster_points)),
                },
                np.ascontiguousarray(cluster_points, dtype=np.float32),
            ))
    records.sort(key=lambda item: (-float(item[0]["score"]), str(item[0]["classe"])))
    predictions = [item[0] for item in records]
    clusters = [item[1] for item in records]
    diagnostics = dict(result["diagnostics"])
    diagnostics.update(
        {
            "predictions": int(len(predictions)),
            "object_points": int(((labels > 0) & (confidence >= threshold)).sum()),
            "score_threshold": threshold,
            "cluster_eps": float(cluster_eps),
            "min_cluster_points": int(min_cluster_points),
            "class_counts": class_counts,
        }
    )
    ground = result["diagnostics"].get("ground", {})
    ground_inliers = int(ground.get("inliers", 0) or 0) if isinstance(ground, Mapping) else 0
    diagnostics.update(
        {
            # Canonical names shared with the geometric service/UI.
            "pontos_brutos": int(result["diagnostics"].get("input_points", 0)),
            "pontos_preprocessados": int(result["diagnostics"].get("voxel_points", 0)),
            "pontos_chao": ground_inliers,
            "pontos_objetos": int(result["diagnostics"].get("processed_points", 0)),
            "metodo": "pointnet2",
            "z_chao": ground.get("height") if isinstance(ground, Mapping) else None,
            "inclinacao_deg": ground.get("tilt_deg") if isinstance(ground, Mapping) else None,
        }
    )
    return {
        "predictions": predictions,
        "predicoes": predictions,
        "clusters": clusters,
        "diagnostics": diagnostics,
    }

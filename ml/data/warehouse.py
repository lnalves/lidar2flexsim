"""Adaptador compatível com a API original do dataset Warehouse.

O núcleo novo usado pelo PointNet++ fica em :mod:`ml.data.pointnet`; este
módulo preserva nomes anteriores (`WarehouseBox`, `load_label_boxes`, etc.)
para scripts de treinamento e notebooks que já estavam em uso.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..config import DEFAULT_CLASS_NAMES
from ..geometry import points_in_oriented_box, OrientedBox
from ..preprocessing import PointPreprocessingConfig, preprocess_points


@dataclass(frozen=True)
class WarehouseBox:
    class_name: str
    center: tuple[float, float, float]
    dimensions: tuple[float, float, float]
    yaw: float

    @classmethod
    def from_line(cls, line: str) -> "WarehouseBox":
        fields = line.split()
        if len(fields) != 8:
            raise ValueError("Cada anotação deve ter classe + 7 valores numéricos")
        try:
            values = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"Valores inválidos na anotação: {line!r}") from exc
        if not np.isfinite(values).all():
            raise ValueError(f"Anotação contém NaN/inf: {line!r}")
        dims = tuple(abs(value) for value in values[3:6])
        if any(value <= 0 for value in dims):
            raise ValueError(f"Dimensões inválidas na anotação: {line!r}")
        return cls(fields[0], tuple(values[:3]), dims, values[6])  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        return {
            "classe": self.class_name,
            "centro": list(self.center),
            "dimensoes": list(self.dimensions),
            "rotacao": self.yaw,
        }


def load_bin_points(path: str | Path, *, with_intensity: bool = True) -> np.ndarray:
    """Carrega BIN como matriz ``[x,y,z,(intensidade)]``."""

    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(source)
    values = np.fromfile(source, dtype=np.float32)
    if values.size == 0:
        return np.empty((0, 4 if with_intensity else 3), dtype=np.float32)
    # Warehouse uses four values. For an XYZ-only file, callers can request
    # ``with_intensity=False`` to avoid the unavoidable 3/4 divisibility
    # ambiguity for tiny synthetic files.
    if with_intensity:
        if values.size % 4:
            if values.size % 3:
                raise ValueError(f"Arquivo .bin não é divisível por 3/4: {source}")
            points = values.reshape(-1, 3)
            points = np.column_stack((points, np.zeros(len(points), dtype=np.float32)))
        else:
            points = values.reshape(-1, 4)
    else:
        if values.size % 3:
            raise ValueError(f"Arquivo .bin não é divisível por 3: {source}")
        points = values.reshape(-1, 3)
    points = np.asarray(points, dtype=np.float32)
    if not np.isfinite(points[:, :3]).all():
        raise ValueError(f"Arquivo .bin contém coordenadas não finitas: {source}")
    return np.ascontiguousarray(points)


def load_label_boxes(path: str | Path) -> list[WarehouseBox]:
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(source)
    boxes: list[WarehouseBox] = []
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            boxes.append(WarehouseBox.from_line(line))
        except ValueError as exc:
            raise ValueError(f"{source}:{line_number}: {exc}") from exc
    return boxes


def _as_oriented_box(box: WarehouseBox) -> OrientedBox:
    return OrientedBox(box.class_name, box.center, box.dimensions, box.yaw)


def points_to_segmentation_labels(
    points: np.ndarray,
    boxes: Sequence[WarehouseBox],
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
) -> np.ndarray:
    """Converte caixas em labels ponto a ponto (zero representa background)."""

    array = np.asarray(points)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError("points deve ter formato [N, >=3]")
    names = {str(name): index for index, name in enumerate(class_names)}
    labels = np.zeros(len(array), dtype=np.int64)
    for box in boxes:
        class_id = names.get(box.class_name)
        if class_id is None or class_id == 0:
            continue
        mask = (labels == 0) & points_in_oriented_box(array, _as_oriented_box(box))
        labels[mask] = class_id
    return labels


def temporal_split(
    scans: Sequence[str | Path], validation_fraction: float = 0.2,
) -> tuple[list[Path], list[Path]]:
    """Divide scans consecutivos em blocos de treino e validação."""

    fraction = float(validation_fraction)
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction deve estar entre 0 e 1")
    ordered = sorted((Path(scan) for scan in scans), key=lambda item: item.stem)
    if len(ordered) < 2:
        return ordered, []
    count = min(len(ordered) - 1, max(1, int(round(len(ordered) * fraction))))
    return ordered[:-count], ordered[-count:]


class WarehouseSegmentationDataset:
    """Adaptador legado que retorna tensores PyTorch quando indexado.

    O import de torch ocorre em ``__getitem__``; assim o módulo continua
    importável em ambientes de processamento geométrico sem PyTorch.
    """

    def __init__(
        self,
        scans: Sequence[str | Path],
        *,
        label_dir: str | Path | None = None,
        class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
        num_points: int = 4096,
        seed: int = 42,
        augment: bool = False,
        preprocessing: PointPreprocessingConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self.scans = [Path(scan).expanduser() for scan in scans]
        self.label_dir = Path(label_dir).expanduser() if label_dir is not None else None
        self.class_names = tuple(class_names)
        self.num_points = int(num_points)
        if self.num_points < 1:
            raise ValueError("num_points deve ser maior que zero")
        self.seed = int(seed)
        self.augment = bool(augment)
        self.preprocessing = PointPreprocessingConfig.from_mapping(preprocessing)

    def __len__(self) -> int:
        return len(self.scans)

    def _label_path(self, scan: Path) -> Path:
        if self.label_dir is not None:
            return self.label_dir / f"{scan.stem}.txt"
        return scan.parent.parent / "label" / f"{scan.stem}.txt"

    def __getitem__(self, index: int) -> dict[str, Any]:
        try:
            import torch  # type: ignore
        except (ImportError, OSError) as exc:
            raise ImportError("WarehouseSegmentationDataset requer PyTorch") from exc
        from .pointnet import prepare_point_sample

        scan = self.scans[index]
        points = load_bin_points(scan, with_intensity=True)
        boxes = load_label_boxes(self._label_path(scan))
        points, _ = preprocess_points(points, self.preprocessing)
        labels = points_to_segmentation_labels(points, boxes, self.class_names)
        rng = np.random.default_rng(self.seed + int(index))
        sample = prepare_point_sample(points, labels, num_points=self.num_points, rng=rng)
        sampled_points = sample.points
        if self.augment and len(sampled_points):
            angle = float(rng.uniform(-np.pi, np.pi))
            cosine, sine = np.cos(angle), np.sin(angle)
            xy = sampled_points[:, :2].copy()
            sampled_points[:, 0] = cosine * xy[:, 0] - sine * xy[:, 1]
            sampled_points[:, 1] = sine * xy[:, 0] + cosine * xy[:, 1]
            sampled_points[:, :3] += rng.normal(0, 0.005, size=sampled_points[:, :3].shape).astype(np.float32)
        return {
            "points": torch.from_numpy(np.ascontiguousarray(sampled_points)),
            "labels": torch.from_numpy(np.ascontiguousarray(sample.labels)),
            "scan": str(scan),
        }


__all__ = [
    "WarehouseBox",
    "WarehouseSegmentationDataset",
    "load_bin_points",
    "load_label_boxes",
    "points_to_segmentation_labels",
    "temporal_split",
]

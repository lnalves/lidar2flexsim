"""Preparação de scans Warehouse para o segmentador PointNet++.

Este módulo é o ponto de integração entre os labels em caixas orientadas e a
rede de segmentação ponto a ponto. Ele usa apenas NumPy no caminho padrão;
PyTorch só é importado se o dataset for solicitado com ``return_tensors``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from ..classes import (
    WAREHOUSE_CLASS_TO_INDEX,
    class_name_to_index,
    normalize_class_name,
)
from ..geometry import OrientedBox, points_in_oriented_box
from ..preprocessing import PointPreprocessingConfig, preprocess_points


DEFAULT_BIN_FEATURES = 4
DEFAULT_NUM_POINTS = 8192


def load_bin(
    path: str | Path,
    *,
    num_features: int = DEFAULT_BIN_FEATURES,
    return_features: int | None = None,
) -> np.ndarray:
    """Lê um ``.bin`` Warehouse como matriz ``float32`` ``[x,y,z,intensity]``.

    ``num_features`` é explícito para evitar a ambiguidade de arquivos XYZ
    cujo número total de valores também pode ser divisível por quatro.
    """

    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Arquivo BIN não encontrado: {source}")
    if not source.is_file():
        raise ValueError(f"Caminho BIN não é um arquivo: {source}")
    try:
        features = int(num_features)
    except (TypeError, ValueError) as exc:
        raise ValueError("num_features deve ser inteiro.") from exc
    if features < 3:
        raise ValueError("num_features deve ser pelo menos 3.")
    raw = np.fromfile(source, dtype=np.float32)
    if raw.size == 0:
        raise ValueError(f"Arquivo BIN vazio: {source}")
    if raw.size % features:
        raise ValueError(
            f"{source} possui {raw.size} valores; esperado múltiplo de {features}."
        )
    records = raw.reshape((-1, features))
    if not np.isfinite(records[:, :3]).all():
        raise ValueError(f"{source} contém coordenadas não finitas.")
    if return_features is None:
        requested = features
    else:
        requested = int(return_features)
        if requested < 3 or requested > features:
            raise ValueError(
                f"return_features deve estar entre 3 e {features} para este arquivo."
            )
    return np.ascontiguousarray(records[:, :requested], dtype=np.float32)


def load_warehouse_bin(path: str | Path, **kwargs: Any) -> np.ndarray:
    return load_bin(path, **kwargs)


def carregar_bin(path: str | Path, **kwargs: Any) -> np.ndarray:
    return load_bin(path, **kwargs)


def _parse_label_line(
    fields: Sequence[str], *, path: Path, line_number: int,
    strict_classes: bool,
) -> OrientedBox:
    if len(fields) != 8:
        raise ValueError(
            f"{path}:{line_number}: esperado 8 campos, encontrado {len(fields)}"
        )
    class_name = normalize_class_name(fields[0], strict=strict_classes)
    try:
        values = [float(value) for value in fields[1:]]
    except ValueError as exc:
        raise ValueError(f"{path}:{line_number}: valores numéricos inválidos") from exc
    return OrientedBox(class_name, values[:3], values[3:6], values[6])


def read_label_file(
    path: str | Path,
    *,
    strict_classes: bool = True,
) -> list[OrientedBox]:
    """Lê labels ``classe cx cy cz dx dy dz yaw`` do dataset."""

    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Arquivo de label não encontrado: {source}")
    boxes: list[OrientedBox] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            boxes.append(_parse_label_line(
                text.split(), path=source, line_number=line_number,
                strict_classes=strict_classes,
            ))
    return boxes


def load_labels(path: str | Path, **kwargs: Any) -> list[OrientedBox]:
    return read_label_file(path, **kwargs)


def ler_labels(path: str | Path, **kwargs: Any) -> list[OrientedBox]:
    return read_label_file(path, **kwargs)


@dataclass(frozen=True)
class WarehouseScan:
    """Pontos e anotações de um scan."""

    scan_id: str
    points: np.ndarray
    boxes: tuple[OrientedBox, ...] = ()

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError("points deve ter formato (N, >=3).")
        if not np.isfinite(points[:, :3]).all():
            raise ValueError("points contém coordenadas não finitas.")
        object.__setattr__(self, "points", np.ascontiguousarray(points))
        object.__setattr__(self, "scan_id", str(self.scan_id))
        object.__setattr__(self, "boxes", tuple(self.boxes))


def load_scan(
    bin_path: str | Path,
    label_path: str | Path | None = None,
    *,
    num_features: int = DEFAULT_BIN_FEATURES,
    strict_classes: bool = True,
) -> WarehouseScan:
    source = Path(bin_path).expanduser()
    points = load_bin(source, num_features=num_features)
    boxes = read_label_file(label_path, strict_classes=strict_classes) if label_path is not None else []
    return WarehouseScan(source.stem, points, tuple(boxes))


def assign_point_labels(
    points: np.ndarray | Sequence[Sequence[float]],
    boxes: Iterable[OrientedBox | Mapping[str, object] | Sequence[object]],
    *,
    class_mapping: Mapping[str, int] | None = None,
    background_index: int = 0,
    tolerance: float = 0.0,
    strict_classes: bool = True,
) -> np.ndarray:
    """Rasteriza caixas orientadas em um vetor de labels ponto a ponto."""

    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError("points deve ter formato (N, >=3).")
    background = int(background_index)
    output = np.full(len(array), background, dtype=np.int64)
    mapping = class_mapping or WAREHOUSE_CLASS_TO_INDEX
    converted: list[tuple[OrientedBox, int]] = []
    for item in boxes:
        if isinstance(item, OrientedBox):
            box = item
        elif isinstance(item, Mapping):
            box = OrientedBox.from_mapping(item)
        else:
            values = list(item)
            if len(values) != 8:
                raise ValueError("Caixa sequencial deve conter classe + 7 valores.")
            box = OrientedBox(str(values[0]), values[1:4], values[4:7], float(values[7]))  # type: ignore[arg-type]
        class_id = class_name_to_index(
            box.class_name, mapping=mapping, strict=strict_classes,
        )
        converted.append((box, class_id))
    # Smaller boxes win deterministic overlaps.
    converted.sort(key=lambda pair: pair[0].volume)
    for box, class_id in converted:
        mask = points_in_oriented_box(array, box, tolerance=tolerance)
        output[mask & (output == background)] = class_id
    return output


def label_points(*args: Any, **kwargs: Any) -> np.ndarray:
    return assign_point_labels(*args, **kwargs)


def rotular_pontos(*args: Any, **kwargs: Any) -> np.ndarray:
    return assign_point_labels(*args, **kwargs)


@dataclass(frozen=True)
class PointSample:
    """Amostra de tamanho fixo e máscara para padding."""

    points: np.ndarray
    labels: np.ndarray | None
    indices: np.ndarray
    valid_mask: np.ndarray

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float32)
        indices = np.asarray(self.indices, dtype=np.int64)
        valid = np.asarray(self.valid_mask, dtype=bool)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError("points deve ter formato (N, >=3).")
        if len(indices) != len(points) or len(valid) != len(points):
            raise ValueError("indices/valid_mask devem acompanhar points.")
        labels = None if self.labels is None else np.asarray(self.labels, dtype=np.int64)
        if labels is not None and (labels.ndim != 1 or len(labels) != len(points)):
            raise ValueError("labels deve ter formato (N,).")
        object.__setattr__(self, "points", np.ascontiguousarray(points))
        object.__setattr__(self, "labels", None if labels is None else np.ascontiguousarray(labels))
        object.__setattr__(self, "indices", np.ascontiguousarray(indices))
        object.__setattr__(self, "valid_mask", np.ascontiguousarray(valid))

    def __iter__(self) -> Iterator[np.ndarray | None]:
        yield self.points
        yield self.labels


def _generator(value: np.random.Generator | int | None) -> np.random.Generator:
    return value if isinstance(value, np.random.Generator) else np.random.default_rng(value)


def _foreground_indices(
    labels: np.ndarray,
    target: int,
    max_ratio: float,
    random: np.random.Generator,
) -> list[int]:
    """Índices de objeto que precisam entrar na amostra.

    Os objetos ocupam poucos por cento de cada scan, então descartá-los na
    amostragem apaga justamente o sinal que o segmentador precisa aprender.
    Todos os pontos de foreground são mantidos; só quando eles passariam de
    ``max_ratio`` da amostra é que cada classe é subamostrada proporcionalmente.
    """

    foreground = np.flatnonzero(labels > 0)
    if not len(foreground):
        return []
    budget = min(len(foreground), int(target), max(1, int(target * float(max_ratio))))
    if budget >= len(foreground):
        return [int(item) for item in foreground]
    classes = [
        np.flatnonzero(labels == int(class_id))
        for class_id in np.unique(labels[foreground])
    ]
    chosen: list[int] = []
    remaining = budget
    for position, choices in enumerate(sorted(classes, key=len)):
        # Classes menores escolhem primeiro e a cota é recalculada a cada passo,
        # de modo que uma classe rara nunca é engolida por uma abundante.
        share = max(1, remaining // (len(classes) - position))
        count = min(len(choices), share, remaining)
        chosen.extend(
            int(item) for item in random.choice(choices, size=count, replace=False)
        )
        remaining -= count
        if remaining <= 0:
            break
    return chosen


def prepare_point_sample(
    points: np.ndarray | Sequence[Sequence[float]],
    labels: np.ndarray | Sequence[int] | None = None,
    *,
    num_points: int = DEFAULT_NUM_POINTS,
    rng: np.random.Generator | int | None = None,
    preserve_foreground: bool = True,
    max_foreground_ratio: float = 0.5,
) -> PointSample:
    """Amostra sem reposição ou repete pontos até ``num_points``."""

    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError("points deve ter formato (N, >=3).")
    target = int(num_points)
    if target < 1:
        raise ValueError("num_points deve ser maior que zero.")
    if len(array) == 0:
        raise ValueError("Não é possível amostrar nuvem vazia.")
    target_labels = None if labels is None else np.asarray(labels, dtype=np.int64)
    if target_labels is not None and (target_labels.ndim != 1 or len(target_labels) != len(array)):
        raise ValueError("labels deve ter formato (N,).")
    if not 0 < float(max_foreground_ratio) <= 1:
        raise ValueError("max_foreground_ratio deve estar entre 0 e 1")
    random = _generator(rng)
    if len(array) >= target:
        required: list[int] = []
        if target_labels is not None and preserve_foreground:
            required = _foreground_indices(
                target_labels, target, max_foreground_ratio, random
            )
        available = np.ones(len(array), dtype=bool)
        if required:
            available[np.asarray(required, dtype=np.int64)] = False
        remaining = np.flatnonzero(available).astype(np.int64)
        count = target - len(required)
        chosen = np.concatenate((
            np.asarray(required, dtype=np.int64),
            random.choice(remaining, size=count, replace=False).astype(np.int64),
        )) if count else np.asarray(required, dtype=np.int64)
        random.shuffle(chosen)
        valid = np.ones(target, dtype=bool)
    else:
        chosen = random.choice(len(array), size=target, replace=True).astype(np.int64)
        chosen[:len(array)] = np.arange(len(array), dtype=np.int64)
        valid = np.zeros(target, dtype=bool)
        valid[:len(array)] = True
    return PointSample(
        array[chosen],
        None if target_labels is None else target_labels[chosen],
        chosen,
        valid,
    )


def sample_fixed_points(*args: Any, **kwargs: Any) -> PointSample:
    return prepare_point_sample(*args, **kwargs)


def prepare_sample(*args: Any, **kwargs: Any) -> PointSample:
    return prepare_point_sample(*args, **kwargs)


def _resolve_dirs(root: Path) -> tuple[Path, Path | None]:
    if root.is_file():
        return root.parent, None
    bin_dir = next((root / name for name in ("bin", "bins") if (root / name).is_dir()), root)
    label_dir = next((root / name for name in ("label", "labels") if (root / name).is_dir()), None)
    return bin_dir, label_dir


def _require_torch() -> Any:
    try:
        import torch  # type: ignore
    except (ImportError, OSError) as exc:
        raise ImportError(
            "WarehousePointDataset(return_tensors=True) requer PyTorch."
        ) from exc
    return torch


class WarehousePointDataset:
    """Dataset protocol compatível com DataLoader, sem herança obrigatória."""

    def __init__(
        self,
        root: str | Path,
        *,
        label_dir: str | Path | None = None,
        num_points: int = DEFAULT_NUM_POINTS,
        num_features: int = DEFAULT_BIN_FEATURES,
        scan_ids: Iterable[str] | None = None,
        random_seed: int | None = None,
        preserve_foreground: bool = True,
        return_tensors: bool = False,
        as_torch: bool | None = None,
        include_boxes: bool = False,
        strict_classes: bool = True,
        class_names: Sequence[str] | None = None,
        augment: bool = False,
        preprocessing: PointPreprocessingConfig | Mapping[str, Any] | None = None,
    ) -> None:
        source = Path(root).expanduser()
        if not source.exists():
            raise FileNotFoundError(source)
        self.bin_dir, inferred_labels = _resolve_dirs(source)
        self.label_dir = Path(label_dir).expanduser() if label_dir is not None else inferred_labels
        self.num_points = int(num_points)
        self.num_features = int(num_features)
        if self.num_points < 1 or self.num_features < 3:
            raise ValueError("num_points deve ser positivo e num_features >= 3.")
        self.random_seed = random_seed
        self.preserve_foreground = bool(preserve_foreground)
        self.return_tensors = bool(return_tensors if as_torch is None else as_torch)
        self.include_boxes = bool(include_boxes)
        self.strict_classes = bool(strict_classes)
        self.class_names = tuple(class_names or WAREHOUSE_CLASS_TO_INDEX)
        self.augment = bool(augment)
        self.preprocessing = PointPreprocessingConfig.from_mapping(preprocessing)
        self._epoch = 0
        files = sorted(self.bin_dir.glob("*.bin"))
        if scan_ids is not None:
            wanted = {Path(str(item)).stem for item in scan_ids}
            files = [item for item in files if item.stem in wanted]
        if not files:
            raise ValueError(f"Nenhum arquivo .bin encontrado em {self.bin_dir}.")
        self._files = tuple(files)

    @property
    def scan_ids(self) -> tuple[str, ...]:
        return tuple(path.stem for path in self._files)

    def set_epoch(self, epoch: int) -> None:
        """Troca o stream de aumento de dados mantendo a reprodutibilidade.

        Sem isso a seed do augment depende apenas do índice da amostra, e a
        mesma rotação é aplicada em todas as épocas — o que equivale a uma
        transformação fixa do dataset, não a aumento de dados.
        """

        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, np.integer):
            index = int(index)
        path = self._files[index]
        label_path = self.label_dir / f"{path.stem}.txt" if self.label_dir else None
        if label_path is not None and not label_path.is_file():
            label_path = None
        scan = load_scan(path, label_path, num_features=self.num_features, strict_classes=self.strict_classes)
        points, _ = preprocess_points(scan.points, self.preprocessing)
        class_mapping = {
            normalize_class_name(name, strict=False): class_id
            for class_id, name in enumerate(self.class_names)
        }
        labels = (
            assign_point_labels(
                points,
                scan.boxes,
                class_mapping=class_mapping,
                strict_classes=self.strict_classes,
            )
            if scan.boxes
            else np.zeros(len(points), dtype=np.int64)
        )
        seed = None if self.random_seed is None else int(self.random_seed) + int(index)
        sample = prepare_point_sample(
            points,
            labels,
            num_points=self.num_points,
            rng=seed,
            preserve_foreground=self.preserve_foreground,
        )
        sampled_points = sample.points.copy()
        if self.augment and len(sampled_points):
            # Stream independente do da amostragem e variável por época.
            random = np.random.default_rng(
                np.random.SeedSequence(
                    [int(self.random_seed or 0), int(index), int(self._epoch), 1]
                )
            )
            angle = float(random.uniform(-np.pi, np.pi))
            cosine, sine = np.cos(angle), np.sin(angle)
            xy = sampled_points[:, :2].copy()
            sampled_points[:, 0] = cosine * xy[:, 0] - sine * xy[:, 1]
            sampled_points[:, 1] = sine * xy[:, 0] + cosine * xy[:, 1]
            sampled_points[:, :3] += random.normal(
                0, 0.005, size=sampled_points[:, :3].shape
            ).astype(np.float32)
        result: dict[str, Any] = {
            "points": sampled_points,
            "labels": sample.labels,
            "valid_mask": sample.valid_mask,
            "indices": sample.indices,
            "scan_id": scan.scan_id,
        }
        if self.include_boxes:
            result["boxes"] = scan.boxes
        if self.return_tensors:
            torch = _require_torch()
            result["points"] = torch.from_numpy(sampled_points)
            result["labels"] = torch.from_numpy(
                sample.labels
                if sample.labels is not None
                else np.zeros(len(sampled_points), dtype=np.int64)
            )
            result["valid_mask"] = torch.from_numpy(sample.valid_mask)
            result["indices"] = torch.from_numpy(sample.indices)
        return result


def select_scan_subset(
    scans: Sequence[str | Path], maximum: int | None,
) -> list[Path]:
    """Seleciona scans uniformemente ao longo da sequência temporal.

    Recortar os primeiros ``maximum`` scans concentraria a amostra em um único
    trecho da gravação, onde só algumas classes aparecem. O espaçamento uniforme
    mantém a cobertura de toda a sequência, e por isso treino e benchmark usam
    exatamente esta função.
    """

    ordered = [Path(scan) for scan in scans]
    if maximum is None or int(maximum) >= len(ordered):
        return ordered
    count = int(maximum)
    if count < 2:
        raise ValueError("max_scans deve ser pelo menos 2")
    positions = [
        round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)
    ]
    return [ordered[position] for position in positions]


def temporal_split(
    scans: Sequence[str | Path], validation_fraction: float = 0.2,
) -> tuple[list[Path], list[Path]]:
    """Divide scans ordenados em blocos temporais de treino e validação."""

    fraction = float(validation_fraction)
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction deve estar entre 0 e 1")
    ordered = sorted((Path(scan) for scan in scans), key=lambda item: item.stem)
    if len(ordered) < 2:
        return ordered, []
    count = min(len(ordered) - 1, max(1, int(round(len(ordered) * fraction))))
    return ordered[:-count], ordered[-count:]


def temporal_three_way_split(
    scans: Sequence[str | Path],
    validation_fraction: float = 0.2,
    test_fraction: float = 0.1,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Divide scans ordenados em treino, validação e teste temporais.

    O bloco de teste fica no fim da sequência e nunca é usado pelo treino, o que
    permite ao benchmark reproduzir exatamente o split de uma execução.
    """

    validation = float(validation_fraction)
    test = float(test_fraction)
    if not 0 < validation < 1:
        raise ValueError("validation_fraction deve estar entre 0 e 1")
    if not 0 <= test < 1:
        raise ValueError("test_fraction deve estar entre 0 e 1")
    if validation + test >= 1:
        raise ValueError("validation_fraction + test_fraction deve ser menor que 1")
    ordered = sorted((Path(scan) for scan in scans), key=lambda item: item.stem)
    if len(ordered) < 3:
        train, validation_scans = temporal_split(ordered, validation)
        return train, validation_scans, []
    validation_count = max(1, int(round(len(ordered) * validation)))
    test_count = max(1, int(round(len(ordered) * test))) if test else 0
    if validation_count + test_count >= len(ordered):
        validation_count = 1
        test_count = 1 if test else 0
    train_end = len(ordered) - validation_count - test_count
    return (
        ordered[:train_end],
        ordered[train_end:train_end + validation_count],
        ordered[train_end + validation_count:],
    )


WarehouseDataset = WarehousePointDataset


__all__ = [
    "DEFAULT_BIN_FEATURES",
    "DEFAULT_NUM_POINTS",
    "PointSample",
    "WarehouseDataset",
    "WarehousePointDataset",
    "WarehouseScan",
    "assign_point_labels",
    "carregar_bin",
    "label_points",
    "ler_labels",
    "load_bin",
    "load_labels",
    "load_scan",
    "load_warehouse_bin",
    "prepare_point_sample",
    "prepare_sample",
    "read_label_file",
    "rotular_pontos",
    "sample_fixed_points",
    "select_scan_subset",
    "temporal_split",
    "temporal_three_way_split",
]

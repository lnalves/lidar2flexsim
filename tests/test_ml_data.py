"""Testes leves da preparação Warehouse para PointNet++."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml import (
    OrientedBox,
    WarehousePointDataset,
    assign_point_labels,
    class_name_to_index,
    load_bin,
    prepare_point_sample,
    read_label_file,
    points_in_oriented_box,
    select_scan_subset,
    temporal_three_way_split,
)
from ml.preprocessing import PointPreprocessingConfig, preprocess_points


def _write_scan(root: Path) -> tuple[Path, Path]:
    bin_dir = root / "bin"
    label_dir = root / "label"
    bin_dir.mkdir()
    label_dir.mkdir()
    points = np.array([
        [0.0, 0.0, 0.0, 0.2],
        [0.5, 0.0, 0.0, 0.3],
        [0.0, 0.5, 0.0, 0.4],
        [3.0, 3.0, 0.0, 0.5],
    ], dtype=np.float32)
    (bin_dir / "000000.bin").write_bytes(points.tobytes())
    (label_dir / "000000.txt").write_text(
        "# classe cx cy cz dx dy dz yaw\n"
        "Box 0 0 0 2 2 2 0\n",
        encoding="utf-8",
    )
    return bin_dir / "000000.bin", label_dir / "000000.txt"


def test_load_bin_and_labels_preserve_warehouse_contract(tmp_path: Path) -> None:
    bin_path, label_path = _write_scan(tmp_path)

    points = load_bin(bin_path)
    boxes = read_label_file(label_path)

    assert points.dtype == np.float32
    assert points.shape == (4, 4)
    assert boxes == [OrientedBox("Box", (0, 0, 0), (2, 2, 2), 0)]
    assert class_name_to_index("forklift") == 5


def test_oriented_box_membership_and_point_labels() -> None:
    angle = np.pi / 4
    box = OrientedBox("CargoBike", (1, 1, 0), (2, 1, 1), angle)
    inside = np.array([[1.0, 1.0, 0.0], [1.5, 1.0, 0.0]])
    outside = np.array([[3.0, 3.0, 0.0]])
    points = np.vstack((inside, outside))

    mask = points_in_oriented_box(points, box)
    labels = assign_point_labels(points, [box])

    assert mask.tolist() == [True, True, False]
    assert labels.tolist() == [3, 3, 0]


def test_prepare_point_sample_pads_and_preserves_foreground() -> None:
    points = np.arange(15, dtype=np.float32).reshape(5, 3)
    labels = np.array([0, 0, 1, 0, 2], dtype=np.int64)

    sample = prepare_point_sample(points, labels, num_points=8, rng=7)

    assert sample.points.shape == (8, 3)
    assert sample.labels is not None
    assert sample.labels.shape == (8,)
    assert sample.valid_mask.tolist() == [True] * 5 + [False] * 3
    assert {1, 2}.issubset(set(sample.labels.tolist()))
    np.testing.assert_array_equal(sample.points[:5], points)


def test_prepare_point_sample_keeps_every_foreground_point_that_fits() -> None:
    # Um scan real tem poucos por cento de pontos de objeto; se a amostragem
    # descartar parte deles, o segmentador perde justamente o que deve aprender.
    points = np.arange(300, dtype=np.float32).reshape(100, 3)
    labels = np.zeros(100, dtype=np.int64)
    labels[10:14] = 1
    labels[70:73] = 4

    sample = prepare_point_sample(points, labels, num_points=40, rng=3)

    assert sample.points.shape == (40, 3)
    kept = sorted(sample.indices.tolist())
    assert set(range(10, 14)).issubset(kept)
    assert set(range(70, 73)).issubset(kept)
    assert int((sample.labels > 0).sum()) == 7


def test_prepare_point_sample_caps_foreground_and_keeps_rare_classes() -> None:
    points = np.arange(300, dtype=np.float32).reshape(100, 3)
    labels = np.zeros(100, dtype=np.int64)
    labels[:60] = 1  # classe abundante
    labels[60:62] = 5  # classe rara

    sample = prepare_point_sample(
        points, labels, num_points=20, rng=5, max_foreground_ratio=0.5
    )

    assert sample.points.shape == (20, 3)
    # Sem o teto, os 62 pontos de objeto ocupariam a amostra inteira e o modelo
    # nunca veria o contexto de fundo em volta deles.
    assert int((sample.labels == 0).sum()) > 0
    # A cota é distribuída da classe rara para a abundante, nunca o contrário.
    assert int((sample.labels == 5).sum()) == 2
    assert int((sample.labels == 1).sum()) >= 1


def test_prepare_point_sample_downsamples_to_exact_size() -> None:
    points = np.arange(30, dtype=np.float32).reshape(10, 3)
    sample = prepare_point_sample(points, num_points=4, rng=0)

    assert sample.points.shape == (4, 3)
    assert sample.labels is None
    assert sample.valid_mask.all()
    assert len(set(sample.indices.tolist())) == 4


def test_dataset_returns_numpy_without_importing_torch(tmp_path: Path) -> None:
    _write_scan(tmp_path)
    dataset = WarehousePointDataset(tmp_path, num_points=6, random_seed=1)

    item = dataset[0]

    assert len(dataset) == 1
    assert item["scan_id"] == "000000"
    assert item["points"].shape == (6, 4)
    assert item["labels"].dtype == np.int64


def test_dataset_can_return_torch_tensors_when_installed(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    _write_scan(tmp_path)
    dataset = WarehousePointDataset(tmp_path, num_points=6, return_tensors=True)

    item = dataset[0]

    assert isinstance(item["points"], torch.Tensor)
    assert isinstance(item["labels"], torch.Tensor)
    assert tuple(item["points"].shape) == (6, 4)


def test_shared_preprocessing_is_applied_before_label_sampling(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    label_dir = tmp_path / "label"
    bin_dir.mkdir()
    label_dir.mkdir()
    points = np.asarray([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
    ], dtype=np.float32)
    (bin_dir / "000000.bin").write_bytes(points.tobytes())
    (label_dir / "000000.txt").write_text(
        "Box 0.5 0 1 1 1 1 0\n", encoding="utf-8"
    )

    config = PointPreprocessingConfig(remove_ground=True, plane_distance=0.01)
    dataset = WarehousePointDataset(
        tmp_path,
        num_points=4,
        random_seed=0,
        preprocessing=config,
    )
    item = dataset[0]

    assert item["points"].shape == (4, 4)
    assert np.all(item["points"][:, 2] > 0.0)
    assert 1 in set(item["labels"].tolist())


def test_shared_preprocessing_reports_ground_and_keeps_shape() -> None:
    points = np.asarray([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
    ], dtype=np.float32)

    processed, diagnostics = preprocess_points(
        points,
        PointPreprocessingConfig(remove_ground=True, plane_distance=0.01),
    )

    assert processed.shape[1] == 4
    assert diagnostics["ground"]["detected"] is True
    assert diagnostics["processed_points"] == len(processed)


def test_training_dataset_uses_the_same_preprocessing_contract(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    bin_dir = tmp_path / "bin"
    label_dir = tmp_path / "label"
    bin_dir.mkdir()
    label_dir.mkdir()
    points = np.asarray([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
    ], dtype=np.float32)
    (bin_dir / "000000.bin").write_bytes(points.tobytes())
    (label_dir / "000000.txt").write_text(
        "Box 0.5 0 1 1 1 1 0\n", encoding="utf-8"
    )

    dataset = WarehousePointDataset(
        tmp_path,
        label_dir=label_dir,
        num_points=4,
        return_tensors=True,
        preprocessing=PointPreprocessingConfig(remove_ground=True, plane_distance=0.01),
    )
    item = dataset[0]

    assert np.all(item["points"].numpy()[:, 2] > 0.0)
    assert 1 in set(item["labels"].numpy().tolist())


def test_select_scan_subset_covers_the_whole_sequence() -> None:
    scans = [Path(f"{index:06d}.bin") for index in range(3287)]

    chosen = select_scan_subset(scans, 30)

    assert len(chosen) == 30
    assert chosen[0] == scans[0]
    # O recorte contíguo (os 30 primeiros) cobria segundos de gravação e deixava
    # a maioria das classes sem nenhuma ocorrência para medir.
    assert chosen[-1] == scans[-1]
    assert select_scan_subset(scans, None) == scans
    assert select_scan_subset(scans, 9999) == scans


def test_temporal_three_way_split_is_disjoint_and_ordered() -> None:
    scans = [Path(f"{index:06d}.bin") for index in range(100)]

    train, validation, test = temporal_three_way_split(scans, 0.2, 0.1)

    assert len(train) == 70 and len(validation) == 20 and len(test) == 10
    assert train[-1].stem < validation[0].stem < test[0].stem
    assert not {item.stem for item in train} & {item.stem for item in test}
    assert train + validation + test == scans


def test_augmentation_varies_between_epochs_and_repeats_within_one(
    tmp_path: Path,
) -> None:
    _write_scan(tmp_path)
    dataset = WarehousePointDataset(tmp_path, num_points=4, random_seed=1, augment=True)

    first = dataset[0]["points"].copy()
    repeated = dataset[0]["points"].copy()
    dataset.set_epoch(1)
    second = dataset[0]["points"].copy()

    np.testing.assert_array_equal(first, repeated)
    # Uma seed presa ao índice aplicaria a mesma rotação em todas as épocas,
    # o que equivale a um dataset fixo e não a aumento de dados.
    assert not np.allclose(first, second)

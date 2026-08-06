"""Amostragem de pontos e contrato do dataset de treino."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml.data.pointnet import WarehousePointDataset, prepare_point_sample


def test_downsampling_never_repeats_a_point_and_marks_everything_valid() -> None:
    points = np.arange(60, dtype=np.float32).reshape(20, 3)

    sample = prepare_point_sample(points, num_points=8, rng=0)

    assert sample.points.shape == (8, 3)
    assert len(set(sample.indices.tolist())) == 8
    assert sample.valid_mask.all()


def test_upsampling_keeps_every_original_point_and_flags_the_padding() -> None:
    points = np.arange(9, dtype=np.float32).reshape(3, 3)

    sample = prepare_point_sample(points, num_points=8, rng=0)

    assert sample.points.shape == (8, 3)
    # Os originais vêm primeiro e intactos; o resto é repetição sinalizada.
    assert sample.indices[:3].tolist() == [0, 1, 2]
    assert sample.valid_mask[:3].all()
    assert not sample.valid_mask[3:].any()


def test_same_seed_reproduces_the_same_sample() -> None:
    points = np.arange(150, dtype=np.float32).reshape(50, 3)

    first = prepare_point_sample(points, num_points=16, rng=7)
    second = prepare_point_sample(points, num_points=16, rng=7)
    other = prepare_point_sample(points, num_points=16, rng=8)

    assert first.indices.tolist() == second.indices.tolist()
    assert first.indices.tolist() != other.indices.tolist()


def test_foreground_points_survive_an_aggressive_downsample() -> None:
    # Um scan do Warehouse tem menos de 7% de pontos de objeto: sem a reserva
    # de foreground uma amostragem uniforme apaga a classe a ser aprendida.
    points = np.random.default_rng(3).normal(size=(1000, 3)).astype(np.float32)
    labels = np.zeros(1000, dtype=np.int64)
    labels[:5] = 2

    sample = prepare_point_sample(
        points, labels, num_points=32, rng=1, preserve_foreground=True
    )

    assert int((sample.labels == 2).sum()) == 5


def test_foreground_reservation_is_capped_by_the_ratio() -> None:
    points = np.random.default_rng(3).normal(size=(200, 3)).astype(np.float32)
    labels = np.full(200, 2, dtype=np.int64)

    sample = prepare_point_sample(
        points,
        labels,
        num_points=20,
        rng=1,
        preserve_foreground=True,
        max_foreground_ratio=0.5,
    )

    # Uma nuvem inteiramente de objeto não pode monopolizar a amostra a ponto
    # de o modelo nunca ver background.
    assert sample.points.shape == (20, 3)


def test_disabling_foreground_preservation_still_returns_the_target_size() -> None:
    points = np.random.default_rng(3).normal(size=(100, 3)).astype(np.float32)
    labels = np.zeros(100, dtype=np.int64)
    labels[:4] = 1

    sample = prepare_point_sample(
        points, labels, num_points=16, rng=1, preserve_foreground=False
    )

    assert sample.points.shape == (16, 3)
    assert sample.labels.shape == (16,)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_points": 0}, "num_points"),
        ({"num_points": -3}, "num_points"),
        ({"num_points": 8, "max_foreground_ratio": 0.0}, "max_foreground_ratio"),
        ({"num_points": 8, "max_foreground_ratio": 1.5}, "max_foreground_ratio"),
    ],
)
def test_prepare_point_sample_rejects_invalid_arguments(
    kwargs: dict[str, object], message: str
) -> None:
    points = np.zeros((10, 3), dtype=np.float32)

    with pytest.raises(ValueError, match=message):
        prepare_point_sample(points, **kwargs)  # type: ignore[arg-type]


def test_prepare_point_sample_rejects_a_malformed_cloud() -> None:
    with pytest.raises(ValueError, match="points"):
        prepare_point_sample(np.zeros((10, 2), dtype=np.float32), num_points=4)
    with pytest.raises(ValueError, match="vazia"):
        prepare_point_sample(np.zeros((0, 3), dtype=np.float32), num_points=4)


def test_prepare_point_sample_rejects_labels_that_do_not_match_the_cloud() -> None:
    points = np.zeros((10, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="labels"):
        prepare_point_sample(points, np.zeros(4, dtype=np.int64), num_points=4)


def test_dataset_item_has_the_shapes_the_training_loop_expects(
    warehouse_dataset: Path,
) -> None:
    dataset = WarehousePointDataset(
        warehouse_dataset,
        label_dir=warehouse_dataset / "label",
        num_points=12,
        random_seed=0,
    )

    item = dataset[0]

    assert len(dataset) == 6
    assert dataset.scan_ids[0] == "000000"
    assert item["points"].shape == (12, 4)
    assert item["labels"].shape == (12,)
    assert item["scan_id"] == "000000"
    assert set(np.unique(item["labels"]).tolist()) <= set(range(len(dataset.class_names)))


def test_dataset_item_is_deterministic_for_a_fixed_seed(
    warehouse_dataset: Path,
) -> None:
    def build() -> WarehousePointDataset:
        return WarehousePointDataset(
            warehouse_dataset,
            label_dir=warehouse_dataset / "label",
            num_points=12,
            random_seed=42,
        )

    assert np.array_equal(build()[1]["points"], build()[1]["points"])


def test_augmentation_changes_between_epochs_but_stays_reproducible(
    warehouse_dataset: Path,
) -> None:
    def build() -> WarehousePointDataset:
        return WarehousePointDataset(
            warehouse_dataset,
            label_dir=warehouse_dataset / "label",
            num_points=12,
            random_seed=5,
            augment=True,
        )

    first_epoch = build()
    second_epoch = build()
    second_epoch.set_epoch(1)
    repeat = build()

    # Sem set_epoch a mesma rotação valeria para todas as épocas, virando uma
    # transformação fixa do dataset em vez de aumento de dados.
    assert not np.array_equal(first_epoch[0]["points"], second_epoch[0]["points"])
    assert np.array_equal(first_epoch[0]["points"], repeat[0]["points"])


def test_dataset_can_be_restricted_to_an_explicit_split(
    warehouse_dataset: Path,
) -> None:
    dataset = WarehousePointDataset(
        warehouse_dataset,
        label_dir=warehouse_dataset / "label",
        num_points=8,
        scan_ids=["000001", "000003"],
    )

    assert dataset.scan_ids == ("000001", "000003")
    assert len(dataset) == 2


def test_dataset_rejects_a_root_without_scans(tmp_path: Path) -> None:
    empty = tmp_path / "vazio"
    (empty / "bin").mkdir(parents=True)

    with pytest.raises(ValueError, match="Nenhum arquivo"):
        WarehousePointDataset(empty, num_points=8)


def test_dataset_rejects_a_missing_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        WarehousePointDataset(tmp_path / "inexistente", num_points=8)


def test_dataset_returns_tensors_when_asked(warehouse_dataset: Path) -> None:
    torch = pytest.importorskip("torch")

    dataset = WarehousePointDataset(
        warehouse_dataset,
        label_dir=warehouse_dataset / "label",
        num_points=8,
        return_tensors=True,
    )

    item = dataset[0]

    assert isinstance(item["points"], torch.Tensor)
    assert isinstance(item["labels"], torch.Tensor)
    assert item["labels"].dtype == torch.int64

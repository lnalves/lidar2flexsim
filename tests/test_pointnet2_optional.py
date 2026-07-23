"""Smoke tests do PointNet++ sem tornar PyTorch obrigatório."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pytest

import ml
from ml.config import PointNet2Config, TrainingConfig, load_config
from ml.data import load_label_boxes, points_to_segmentation_labels
from ml.preprocessing import PointPreprocessingConfig


def test_ml_import_is_safe_without_torch() -> None:
    assert isinstance(ml.torch_available(), bool)
    assert ml.PointNet2Config().num_classes == 6


def test_config_and_box_rasterization() -> None:
    model, training = load_config("ml/configs/pointnet2_seg.yaml")
    assert model.class_names[-1] == "ForkLift"
    assert training.epochs >= 1
    box = load_label_boxes("dados/warehouse/label/000000.txt")[0]
    points = np.asarray([box.center, (100.0, 100.0, 100.0)], dtype=np.float32)
    labels = points_to_segmentation_labels(points, [box])
    assert labels.shape == (2,)
    assert labels[0] == model.class_names.index("ForkLift")
    assert labels[1] == 0


def test_preprocessing_config_round_trips_and_does_not_remove_ground_by_default() -> None:
    config = PointNet2Config()

    assert config.preprocessing.remove_ground is False
    assert config.preprocessing.voxel == 0.0
    restored = PointNet2Config.from_mapping(config.to_dict())

    assert restored.preprocessing == config.preprocessing
    assert restored.calibration == config.calibration


def test_prediction_uses_checkpoint_preprocessing_contract() -> None:
    torch = pytest.importorskip("torch")
    from ml.inference import predict_points

    class BackgroundModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = PointNet2Config(
                input_points=4,
                sa1_points=2,
                sa2_points=1,
                neighbors=2,
                hidden_channels=8,
                preprocessing=PointPreprocessingConfig(remove_ground=False),
            )

        def forward(self, points: object) -> object:
            logits = torch.zeros((points.shape[0], points.shape[1], self.config.num_classes))
            logits[..., 0] = 10.0
            return logits

    points = np.asarray([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [2.0, 0.0, 2.0, 0.0],
        [3.0, 0.0, 3.0, 0.0],
    ], dtype=np.float32)
    result = predict_points(points, model=BackgroundModel(), num_points=4)

    assert result["diagnostics"]["input_points"] == 4
    assert result["diagnostics"]["processed_points"] == 4
    assert result["diagnostics"]["ground"]["detected"] is False
    assert "selected_indices" not in result["diagnostics"]
    assert result["diagnostics"]["sampling"]["method"] == "without_replacement"

    debug_result = predict_points(
        points,
        model=BackgroundModel(),
        num_points=4,
        debug_diagnostics=True,
    )
    assert len(debug_result["diagnostics"]["selected_indices"]) == 4


def test_inference_uses_model_calibration_when_no_override_is_given() -> None:
    torch = pytest.importorskip("torch")
    from ml.inference import inferir_scan

    class BoxModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = PointNet2Config(
                input_points=4,
                sa1_points=2,
                sa2_points=1,
                neighbors=2,
                hidden_channels=8,
                calibration={"min_points": 5},
            )

        def forward(self, points: object) -> object:
            logits = torch.zeros((points.shape[0], points.shape[1], self.config.num_classes))
            logits[..., 1] = 10.0
            return logits

    points = np.asarray([
        [0.0, 0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0, 0.0],
        [0.0, 0.1, 0.0, 0.0],
        [0.1, 0.1, 0.1, 0.0],
    ], dtype=np.float32)

    result = inferir_scan(
        points,
        model=BoxModel(),
        num_points=4,
        cluster_eps=0.5,
        min_cluster_points=1,
    )

    assert result["predictions"] == []
    assert result["diagnostics"]["calibration"]["removed"]["points"] == 1
    assert result["diagnostics"]["calibration"]["calibration"]["min_points"] == 5


def test_checkpoint_loader_does_not_execute_arbitrary_pickle(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from ml.checkpoints import load_checkpoint

    marker = tmp_path / "executed.txt"

    class Exploit:
        def __reduce__(self) -> object:
            return os.system, (f"touch {marker}",)

    checkpoint = tmp_path / "malicious.pt"
    torch.save({"state_dict": {}, "extra": Exploit()}, checkpoint)

    with pytest.raises((pickle.UnpicklingError, RuntimeError)):
        load_checkpoint(checkpoint)
    assert not marker.exists()


def test_checkpoint_is_versioned_and_legacy_config_is_migrated(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from ml.checkpoints import FORMAT_VERSION, load_checkpoint, save_checkpoint
    from ml.models.pointnet2_seg import PointNet2Segmentation

    config = PointNet2Config(
        input_points=8,
        sa1_points=4,
        sa2_points=2,
        neighbors=2,
        hidden_channels=8,
        preprocessing=PointPreprocessingConfig(voxel=0.1),
    )
    model = PointNet2Segmentation(config)
    current = tmp_path / "current.pt"
    save_checkpoint(current, model, config=config)
    loaded = load_checkpoint(current)

    assert FORMAT_VERSION == 2
    assert loaded["format_version"] == FORMAT_VERSION
    assert loaded["config"]["preprocessing"]["voxel"] == pytest.approx(0.1)

    legacy_config = config.to_dict()
    legacy_config.pop("preprocessing")
    legacy = tmp_path / "legacy.pt"
    torch.save({
        "format_version": 1,
        "state_dict": model.state_dict(),
        "config": legacy_config,
        "metrics": {},
    }, legacy)
    migrated = load_checkpoint(legacy)

    assert migrated["migrated_from_format_version"] == 1
    assert migrated["config"]["preprocessing"]["remove_ground"] is False

    invalid = tmp_path / "invalid.pt"
    torch.save({
        "format_version": FORMAT_VERSION,
        "state_dict": model.state_dict(),
        "config": {"num_classes": 0},
        "metrics": {},
    }, invalid)
    with pytest.raises(ValueError, match="config"):
        load_checkpoint(invalid)


def test_oriented_prediction_box_contains_asymmetric_cluster() -> None:
    from ml.geometry import OrientedBox, points_in_oriented_box
    from ml.inference import _oriented_box

    points = np.asarray([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 0.0, 0.1],
    ], dtype=np.float32)
    center, dimensions, yaw = _oriented_box(points)
    box = OrientedBox("Box", tuple(center.tolist()), tuple(dimensions.tolist()), yaw)

    assert points_in_oriented_box(points, box, tolerance=1e-5).all()


def test_model_forward_cpu_when_torch_is_available() -> None:
    torch = pytest.importorskip("torch")
    from ml.models.pointnet2_seg import PointNet2Segmentation

    config = PointNet2Config(
        input_points=16,
        sa1_points=8,
        sa2_points=4,
        neighbors=4,
        hidden_channels=8,
    )
    model = PointNet2Segmentation(config).eval()
    output = model(torch.randn(1, 16, 4))
    assert tuple(output.shape) == (1, 16, config.num_classes)


def test_training_metrics_are_global_and_report_missing_classes() -> None:
    torch = pytest.importorskip("torch")
    from ml.training import _metrics_from_confusion

    metrics = _metrics_from_confusion(
        torch.tensor([[3, 1, 0], [1, 2, 0], [0, 0, 0]]),
        3,
        ("background", "Box", "ForkLift"),
    )

    assert metrics["miou"] == pytest.approx(0.5)
    assert metrics["miou_macro"] == pytest.approx(0.55)
    assert metrics["iou_per_class"]["ForkLift"] is None
    assert metrics["support_per_class"]["Box"] == 3
    assert metrics["present_classes"] == ["background", "Box"]


def test_training_run_isolated_and_resumable(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from ml.models.pointnet2_seg import PointNet2Segmentation
    from ml.training import train_model

    model_config = PointNet2Config(
        input_points=8,
        sa1_points=4,
        sa2_points=2,
        neighbors=2,
        hidden_channels=8,
    )
    loader = [{
        "points": torch.randn(1, 8, 4),
        "labels": torch.randint(0, model_config.num_classes, (1, 8)),
    }]
    run_dir = tmp_path / "runs" / "experiment"
    first = train_model(
        loader,
        PointNet2Segmentation(model_config),
        config=TrainingConfig(epochs=1, batch_size=1, device="cpu"),
        run_dir=run_dir,
    )
    assert Path(first["last_checkpoint"]).is_file()
    assert Path(first["best_checkpoint"]).is_file()
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metadata.json").is_file()
    assert (run_dir / "history.jsonl").read_text(encoding="utf-8").count("\n") == 1

    second = train_model(
        loader,
        PointNet2Segmentation(model_config),
        config=TrainingConfig(epochs=2, batch_size=1, device="cpu"),
        run_dir=run_dir,
        resume=run_dir,
    )
    assert [int(item["epoch"]) for item in second["history"]] == [1, 2]


def test_benchmark_manifest_is_deterministic_and_disjoint(tmp_path: Path) -> None:
    from ml.benchmark import build_benchmark_manifest, load_benchmark_manifest, save_benchmark_manifest

    scans = [tmp_path / f"{index:06d}.bin" for index in range(12)]
    manifest = build_benchmark_manifest(scans, seed=7, max_scans=9)
    path = save_benchmark_manifest(tmp_path / "manifest.json", manifest)
    restored = load_benchmark_manifest(path)

    assert restored == manifest
    train = set(manifest["train_scan_ids"])
    validation = set(manifest["validation_scan_ids"])
    test = set(manifest["test_scan_ids"])
    assert train and validation and test
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)


def test_prediction_calibration_filters_and_suppresses_duplicates() -> None:
    from ml.calibration import calibrate_predictions

    predictions = [
        {"classe": "Box", "score": 0.95, "n_pontos": 20, "centro": [0, 0, 0], "dimensoes": [2, 2, 2]},
        {"classe": "Box", "score": 0.80, "n_pontos": 20, "centro": [0.1, 0, 0], "dimensoes": [2, 2, 2]},
        {"classe": "Box", "score": 0.99, "n_pontos": 2, "centro": [5, 0, 0], "dimensoes": [1, 1, 1]},
    ]
    kept, clusters, diagnostics = calibrate_predictions(
        predictions,
        ["a", "b", "c"],
        {"min_points": 5, "max_iou": 0.5},
    )

    assert len(kept) == 1
    assert clusters == ["a"]
    assert diagnostics["removed"]["duplicate"] == 1
    assert diagnostics["removed"]["points"] == 1


def test_smoke_train_is_optional() -> None:
    pytest.importorskip("torch")
    from ml.training import smoke_train

    result = smoke_train(steps=1, num_points=16)
    assert result["history"]

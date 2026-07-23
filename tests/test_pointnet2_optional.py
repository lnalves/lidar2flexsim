"""Smoke tests do PointNet++ sem tornar PyTorch obrigatório."""

from __future__ import annotations

import os
import pickle

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


def test_smoke_train_is_optional() -> None:
    pytest.importorskip("torch")
    from ml.training import smoke_train

    result = smoke_train(steps=1, num_points=16)
    assert result["history"]

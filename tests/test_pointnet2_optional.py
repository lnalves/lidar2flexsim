"""Smoke tests do PointNet++ sem tornar PyTorch obrigatório."""

from __future__ import annotations

import numpy as np
import pytest

import ml
from ml.config import PointNet2Config, TrainingConfig, load_config
from ml.data import load_label_boxes, points_to_segmentation_labels


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

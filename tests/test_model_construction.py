"""Construção do PointNet2Segmentation a partir das várias formas de config."""

from __future__ import annotations

from typing import Any

import pytest

from ml.config import PointNet2Config

pytest.importorskip("torch")


def _tiny(**overrides: Any) -> PointNet2Config:
    base = {
        "input_points": 16,
        "sa1_points": 8,
        "sa2_points": 4,
        "neighbors": 4,
        "hidden_channels": 8,
    }
    base.update(overrides)
    return PointNet2Config(**base)


def test_model_defaults_to_the_warehouse_configuration() -> None:
    from ml.models.pointnet2_seg import PointNet2Segmentation

    model = PointNet2Segmentation()

    assert model.config == PointNet2Config()
    assert model.config.num_classes == 6


def test_model_accepts_a_config_object_unchanged(tiny_model_config: Any) -> None:
    from ml.models.pointnet2_seg import PointNet2Segmentation

    model = PointNet2Segmentation(tiny_model_config)

    assert model.config is tiny_model_config


def test_model_accepts_a_mapping(tiny_model_config: Any) -> None:
    from ml.models.pointnet2_seg import PointNet2Segmentation

    model = PointNet2Segmentation(tiny_model_config.to_dict())

    assert model.config == tiny_model_config


def test_model_accepts_bare_keyword_arguments() -> None:
    from ml.models.pointnet2_seg import PointNet2Segmentation

    model = PointNet2Segmentation(hidden_channels=8, neighbors=4)

    assert model.config.hidden_channels == 8
    assert model.config.neighbors == 4


def test_keyword_arguments_override_the_given_config(tiny_model_config: Any) -> None:
    from ml.models.pointnet2_seg import PointNet2Segmentation

    model = PointNet2Segmentation(tiny_model_config, hidden_channels=16)

    assert model.config.hidden_channels == 16
    # O restante da config precisa sobreviver ao override.
    assert model.config.sa1_points == tiny_model_config.sa1_points
    assert model.config is not tiny_model_config


def test_layer_widths_follow_the_hidden_channels() -> None:
    from ml.models.pointnet2_seg import PointNet2Segmentation

    model = PointNet2Segmentation(_tiny(hidden_channels=8))

    assert model.fp2[0].in_channels == 8 * 4 + 8 * 2
    assert model.fp2[0].out_channels == 8 * 2
    assert model.fp1[0].in_channels == 8 * 2 + model.config.in_channels
    assert model.head[-1].out_channels == model.config.num_classes


def test_extra_input_channels_reach_the_first_abstraction_block() -> None:
    from ml.models.pointnet2_seg import PointNet2Segmentation

    model = PointNet2Segmentation(_tiny(in_channels=5))

    assert model.config.in_channels == 5
    assert model.fp1[0].in_channels == 8 * 2 + 5


@pytest.mark.parametrize("layout", ["bnc", "bcn"])
def test_forward_accepts_both_channel_layouts(layout: str) -> None:
    import torch

    from ml.models.pointnet2_seg import PointNet2Segmentation

    config = _tiny()
    model = PointNet2Segmentation(config).eval()
    points = torch.randn(2, 16, config.in_channels)
    if layout == "bcn":
        points = points.transpose(1, 2)

    output = model(points)

    assert tuple(output.shape) == (2, 16, config.num_classes)


def test_forward_rejects_a_cloud_with_the_wrong_channel_count() -> None:
    import torch

    from ml.models.pointnet2_seg import PointNet2Segmentation

    model = PointNet2Segmentation(_tiny()).eval()

    with pytest.raises(ValueError, match="canais"):
        model(torch.randn(1, 16, 7))
    with pytest.raises(ValueError, match="formato"):
        model(torch.randn(16, 4))


def test_predict_returns_labels_and_confidence_without_leaving_train_mode() -> None:
    import torch

    from ml.models.pointnet2_seg import PointNet2Segmentation

    config = _tiny()
    model = PointNet2Segmentation(config).train()

    labels, confidence = model.predict(torch.randn(1, 16, config.in_channels))

    assert tuple(labels.shape) == (1, 16)
    assert tuple(confidence.shape) == (1, 16)
    assert int(labels.max()) < config.num_classes
    assert float(confidence.min()) >= 0.0
    assert float(confidence.max()) <= 1.0
    # predict() alterna para eval internamente e precisa devolver o modelo ao
    # modo em que o encontrou, senão o treino seguinte roda sem dropout.
    assert model.training is True

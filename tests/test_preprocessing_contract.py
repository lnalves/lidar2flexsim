"""Contratos de validação e de estimativa de chão do pré-processamento."""

from __future__ import annotations

import numpy as np
import pytest

from ml.preprocessing import PointPreprocessingConfig, estimate_ground_mask


def test_default_config_is_a_no_op_over_the_cloud() -> None:
    config = PointPreprocessingConfig()

    # Treino e inferência precisam partir da mesma nuvem: nenhuma filtragem
    # pode estar ligada por padrão.
    assert config.voxel == 0.0
    assert config.remove_ground is False
    assert config.remove_outliers is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("voxel", -0.1, "voxel"),
        ("voxel", float("nan"), "voxel"),
        ("plane_distance", 0.0, "plane_distance"),
        ("plane_distance", -1.0, "plane_distance"),
        ("max_ground_tilt_deg", 90.0, "max_ground_tilt_deg"),
        ("max_ground_tilt_deg", -1.0, "max_ground_tilt_deg"),
        ("ground_quantile", 0.0, "ground_quantile"),
        ("ground_quantile", 0.95, "ground_quantile"),
        ("outlier_neighbors", -1, "outlier_neighbors"),
        ("outlier_std_ratio", 0.0, "outlier_std_ratio"),
    ],
)
def test_post_init_rejects_values_outside_the_valid_range(
    field: str, value: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PointPreprocessingConfig(**{field: value})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("nao", False),
        ("não", False),
        ("false", False),
        ("off", False),
        ("0", False),
        ("", False),
        ("sim", True),
        ("true", True),
        (1, True),
        (0, False),
    ],
)
def test_post_init_normalizes_the_boolean_flags_from_config_files(
    raw: object, expected: bool
) -> None:
    config = PointPreprocessingConfig(remove_ground=raw, remove_outliers=raw)

    # JSON/YAML entregam strings; sem a normalização "false" viraria True.
    assert config.remove_ground is expected
    assert config.remove_outliers is expected


def test_post_init_coerces_numeric_strings_to_floats() -> None:
    config = PointPreprocessingConfig(voxel="0.25", outlier_neighbors="8")

    assert isinstance(config.voxel, float)
    assert config.voxel == pytest.approx(0.25)
    assert config.outlier_neighbors == 8


def test_estimate_ground_mask_drops_the_floor_and_keeps_the_object() -> None:
    floor = np.column_stack(
        [
            np.linspace(-2.0, 2.0, 40),
            np.linspace(-2.0, 2.0, 40),
            np.zeros(40),
        ]
    ).astype(np.float32)
    obj = np.asarray([[0.0, 0.0, 1.0], [0.1, 0.1, 1.2]], dtype=np.float32)
    points = np.vstack([floor, obj])

    mask, diagnostics = estimate_ground_mask(
        points, quantile=0.30, max_tilt_deg=25.0, distance=0.05
    )

    assert diagnostics["detected"] is True
    assert diagnostics["height"] == pytest.approx(0.0, abs=1e-6)
    assert diagnostics["inliers"] == 40
    assert mask[-2:].all()
    assert not mask[:40].any()


def test_estimate_ground_mask_keeps_everything_on_a_degenerate_cloud() -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)

    mask, diagnostics = estimate_ground_mask(
        points, quantile=0.3, max_tilt_deg=25.0, distance=0.05
    )

    # Com menos de três pontos não há plano a estimar; descartar qualquer um
    # deles perderia sinal sem evidência.
    assert mask.all()
    assert diagnostics == {"detected": False}


def test_estimate_ground_mask_falls_back_to_the_lowest_returns() -> None:
    # O quantil sobre uma nuvem sintética minúscula seleciona menos de dois
    # pontos; o fallback pelos retornos mais baixos evita divisão por zero.
    points = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 5.0], [2.0, 0.0, 5.0], [3.0, 0.0, 5.0]],
        dtype=np.float32,
    )

    mask, diagnostics = estimate_ground_mask(
        points, quantile=0.01, max_tilt_deg=25.0, distance=0.05
    )

    assert diagnostics["candidate_points"] >= 2
    assert mask.shape == (4,)

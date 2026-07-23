"""Testes do contrato leve compartilhado entre a UI e os serviços.

Este arquivo deliberadamente não importa Open3D: os controles da interface e
as validações de parâmetros devem continuar testáveis em ambientes mínimos.
"""

from __future__ import annotations

import pytest

from core.models import ParameterError, PipelineParameters, Progress


def test_parametros_padrao_sao_os_do_warehouse() -> None:
    parametros = PipelineParameters()

    assert parametros.to_dict() == {
        "voxel": 0.05,
        "eps": 0.25,
        "min_points": 20,
        "plane_distance": 0.05,
        "oriented_box": False,
        "max_ground_tilt_deg": 25.0,
        "ground_quantile": 0.30,
        "remove_outliers": True,
        "outlier_neighbors": 12,
        "outlier_std_ratio": 2.5,
        "cluster_mode": "3d",
        "detector_backend": "heuristic",
        "model_checkpoint": None,
        "device": "auto",
        "score_threshold": 0.5,
        "num_points": 4096,
    }


@pytest.mark.parametrize("campo", ["voxel", "eps", "plane_distance"])
def test_parametros_rejeitam_valores_nao_positivos(campo: str) -> None:
    with pytest.raises(ParameterError):
        PipelineParameters.from_value({campo: 0})


def test_parametros_aceitam_aliases_da_interface() -> None:
    parametros = PipelineParameters.from_value(
        {"plane_dist": 0.08, "min_cluster_points": 12, "oriented": True}
    )

    assert parametros.plane_distance == pytest.approx(0.08)
    assert parametros.min_points == 12
    assert parametros.oriented_box is True


def test_parametros_de_piso_e_clusterizacao_sao_normalizados() -> None:
    parametros = PipelineParameters.from_value({
        "max_ground_tilt": 20,
        "ground_q": 0.25,
        "remove_outliers": False,
        "outlier_k": 8,
        "outlier_std": 3.0,
        "cluster_space": "XY",
    })

    assert parametros.max_ground_tilt_deg == 20.0
    assert parametros.ground_quantile == pytest.approx(0.25)
    assert parametros.remove_outliers is False
    assert parametros.outlier_neighbors == 8
    assert parametros.cluster_mode == "bev"


def test_parametros_pointnet2_validam_backend_e_checkpoint() -> None:
    parametros = PipelineParameters.from_value({
        "detector_backend": "pointnet2",
        "model_checkpoint": "modelos/pointnet2.pt",
        "device": "cpu",
        "score_threshold": 0.65,
        "num_points": 2048,
    })

    assert parametros.detector_backend == "pointnet2"
    assert parametros.model_checkpoint == "modelos/pointnet2.pt"
    assert parametros.device == "cpu"
    assert parametros.score_threshold == pytest.approx(0.65)
    assert parametros.num_points == 2048


def test_parametros_rejeitam_chave_desconhecida() -> None:
    with pytest.raises(ParameterError, match="desconhecidos"):
        PipelineParameters.from_value({"dbscan_eps": 0.2})


def test_progress_normaliza_percentual_e_serializa() -> None:
    evento = Progress(current=2, total=4, stage="clusterizando", scan_id="000002")

    assert evento.percent == pytest.approx(50.0)
    assert evento.to_dict() == {
        "current": 2,
        "total": 4,
        "percent": pytest.approx(50.0),
        "message": "",
        "stage": "clusterizando",
        "scan_id": "000002",
        "cancelled": False,
    }

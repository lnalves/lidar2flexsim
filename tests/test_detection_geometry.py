"""Testes geométricos da segmentação robusta do Warehouse LiDAR."""

from __future__ import annotations

import numpy as np
import pytest

o3d = pytest.importorskip("open3d")

from lidar2flexsim import clusterizar, remover_chao  # noqa: E402


def _pcd(points: np.ndarray):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(float))
    return cloud


def test_remover_chao_rejeita_plano_vertical_e_retorna_diagnostico() -> None:
    # Piso horizontal esparso e uma parede muito populosa: escolher apenas o
    # maior plano poderia remover a parede como acontecia no protótipo.
    x, y = np.meshgrid(np.linspace(-3, 3, 31), np.linspace(-3, 3, 31))
    piso = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    y_wall, z_wall = np.meshgrid(np.linspace(-5, 5, 61), np.linspace(0, 3, 31))
    parede = np.column_stack((np.full(y_wall.size, 2.0), y_wall.ravel(), z_wall.ravel()))
    objeto = np.array([[0.2, 0.2, altura] for altura in np.linspace(0.2, 1.8, 12)])
    objetos, chao, z_chao, diagnostico = remover_chao(
        _pcd(np.vstack((piso, parede, objeto))),
        dist=0.04,
        max_ground_tilt_deg=20.0,
        ground_quantile=0.30,
        return_diagnostics=True,
    )

    assert diagnostico["metodo"] == "ransac_horizontal"
    assert diagnostico["inclinacao_deg"] < 20.0
    assert z_chao == pytest.approx(0.0, abs=0.04)
    assert len(chao.points) > 500
    # A parede não deve desaparecer por ser o maior plano da cena.
    assert len(objetos.points) > 500


def test_clusterizar_bev_unifica_camadas_verticais_do_mesmo_objeto() -> None:
    xy = np.array([
        [0.00, 0.00], [0.05, 0.00], [0.00, 0.05], [0.05, 0.05],
        [0.10, 0.00], [0.00, 0.10],
    ])
    camada_baixa = np.column_stack((xy, np.zeros(len(xy))))
    camada_alta = np.column_stack((xy, np.full(len(xy), 0.8)))
    cloud = _pcd(np.vstack((camada_baixa, camada_alta)))

    clusters_3d = clusterizar(cloud, eps=0.15, min_points=2, cluster_mode="3d")
    clusters_bev = clusterizar(cloud, eps=0.15, min_points=2, cluster_mode="bev")

    assert len(clusters_3d) == 2
    assert len(clusters_bev) == 1
    assert len(clusters_bev[0].points) == 12

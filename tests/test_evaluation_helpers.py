"""Casos de borda do avaliador que não requerem Open3D."""

from __future__ import annotations

from pathlib import Path

import pytest

from avaliar_deteccoes import (
    carregar_predicoes,
    erro_geometrico,
    ler_labels,
    predicao_para_caixa,
    resumir_scans,
)


def test_ler_labels_ignora_comentarios_e_linhas_vazias(tmp_path: Path) -> None:
    arquivo = tmp_path / "000001.txt"
    arquivo.write_text(
        "# cabeçalho\n\nForkLift 1 2 3 4 5 6 0.5\n", encoding="utf-8"
    )

    labels = ler_labels(arquivo)

    assert labels == [
        {
            "classe": "ForkLift",
            "centro": [1.0, 2.0, 3.0],
            "dimensoes": [4.0, 5.0, 6.0],
            "yaw_rad": 0.5,
        }
    ]


def test_ler_labels_rejeita_numero_de_campos_incorreto(tmp_path: Path) -> None:
    arquivo = tmp_path / "ruim.txt"
    arquivo.write_text("Box 0 0 0 1 1 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="esperado 8 campos"):
        ler_labels(arquivo)


def test_carregar_predicoes_aceita_layout_de_um_scan(tmp_path: Path) -> None:
    arquivo = tmp_path / "layout.json"
    arquivo.write_text(
        '[{"classe":"Box","centro":[0,0,1],'
        '"dimensoes":[2,1,2],"rotacao_z":90}]',
        encoding="utf-8",
    )

    scans = carregar_predicoes(arquivo, "000001")

    assert scans["000001"][0]["classe"] == "Box"
    assert scans["000001"][0]["yaw_rad"] == pytest.approx(1.57079632679)


def test_resumir_scans_calcula_tp_fp_fn_e_erros(tmp_path: Path) -> None:
    label_dir = tmp_path / "label"
    label_dir.mkdir()
    (label_dir / "000001.txt").write_text(
        "Box 0 0 1 2 1 2 0\n", encoding="utf-8"
    )
    predicoes = {
        "000001": [
            predicao_para_caixa(
                {
                    "classe": "Box",
                    "centro": [0, 0, 1],
                    "dimensoes": [2, 1, 2],
                    "rotacao_z": 0,
                }
            )
        ]
    }

    resultado = resumir_scans(predicoes, label_dir, [0.25, 0.5], None)

    for threshold in ("0.25", "0.5"):
        metricas = resultado["por_threshold"][threshold]
        assert metricas["scans_avaliados"] == 1
        assert (metricas["tp"], metricas["fp"], metricas["fn"]) == (1, 0, 0)
        assert metricas["precisao"] == pytest.approx(1.0)
        assert metricas["recall"] == pytest.approx(1.0)
        assert metricas["f1"] == pytest.approx(1.0)
        assert metricas["erro_centro_medio_m"] == pytest.approx(0.0)


def test_erro_geometrico_considera_simetria_de_pi() -> None:
    predicao = {
        "yaw_rad": 0.0,
        "centro": [0.0, 0.0, 0.0],
        "dimensoes": [1.0, 2.0, 1.0],
    }
    verdade = {
        "yaw_rad": 3.141592653589793,
        "centro": [0.0, 0.0, 0.0],
        "dimensoes": [1.0, 2.0, 1.0],
    }

    assert erro_geometrico(predicao, verdade)["erro_yaw_rad"] == pytest.approx(0.0)

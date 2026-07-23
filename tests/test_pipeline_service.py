"""Testes do serviço de aplicação que permanecem headless.

Os casos de cancelamento terminam antes de importar Open3D; assim eles
verificam o contrato de progresso/UI em qualquer ambiente de CI.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

from core.models import PipelineParameters, PipelineServiceError, Progress
from core.pipeline_service import avaliar_predicoes, processar_dataset, processar_scan


def test_processar_scan_respeita_cancelamento_antes_do_pipeline(tmp_path: Path) -> None:
    scan = tmp_path / "000001.bin"
    scan.write_bytes(b"scan")
    cancelado = threading.Event()
    cancelado.set()
    eventos: list[Progress] = []

    resultado = processar_scan(
        scan,
        parametros=PipelineParameters(),
        cancelar_evento=cancelado,
        callback_progresso=eventos.append,
    )

    assert resultado["cancelado"] is True
    assert resultado["cancelled"] is True
    assert resultado["predicoes"] == []
    assert eventos and eventos[-1].cancelled is True
    assert eventos[-1].stage == "cancelado"


def test_processar_dataset_cancelado_grava_documento_sem_open3d(tmp_path: Path) -> None:
    raiz = tmp_path / "warehouse"
    bin_dir = raiz / "bin"
    bin_dir.mkdir(parents=True)
    for scan_id in ("000001", "000002"):
        (bin_dir / f"{scan_id}.bin").write_bytes(b"scan")
    destino = tmp_path / "resultado.json"
    cancelado = threading.Event()
    cancelado.set()
    eventos: list[Progress] = []

    resultado = processar_dataset(
        raiz,
        cancelar_evento=cancelado,
        callback_progresso=eventos.append,
        saida=destino,
    )

    assert resultado["cancelado"] is True
    assert resultado["processados"] == 0
    assert resultado["scans"] == {}
    assert destino.exists()
    assert json.loads(destino.read_text(encoding="utf-8"))["cancelled"] is True
    assert any(evento.stage == "cancelado" for evento in eventos)


def test_processar_scan_pointnet2_exige_checkpoint(tmp_path: Path) -> None:
    scan = tmp_path / "000001.bin"
    scan.write_bytes(b"scan")

    with pytest.raises(PipelineServiceError, match="model_checkpoint"):
        processar_scan(
            scan,
            parametros=PipelineParameters(detector_backend="pointnet2"),
        )


def test_processar_scan_pointnet2_normaliza_predicoes_e_parametros(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scan = tmp_path / "000001.bin"
    scan.write_bytes(b"scan")
    recebido: dict[str, object] = {}

    def fake_inferir_scan(*args: object, **kwargs: object) -> dict[str, object]:
        recebido.update(kwargs)
        return {
            "predictions": [{
                "classe": "ForkLift",
                "centro": [1, 2, 3],
                "dimensoes": [2, 1, 1],
                "rotacao_z": 15,
                "n_pontos": 8,
            }],
            "clusters": [],
            "diagnostics": {"predictions": 1},
        }

    import ml.inference

    monkeypatch.setattr(ml.inference, "inferir_scan", fake_inferir_scan)
    resultado = processar_scan(
        scan,
        parametros=PipelineParameters(
            detector_backend="pointnet2",
            model_checkpoint="modelo.pt",
            eps=0.31,
            min_points=7,
        ),
    )

    assert resultado["predicoes"][0]["classe"] == "ForkLift"
    assert recebido["cluster_eps"] == pytest.approx(0.31)
    assert recebido["min_cluster_points"] == 7
    assert resultado["diagnostico"]["backend"] == "pointnet2"


def test_processar_scan_pointnet2_exporta_stl_e_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scan = tmp_path / "000001.bin"
    scan.write_bytes(b"scan")

    import ml.inference

    monkeypatch.setattr(
        ml.inference,
        "inferir_scan",
        lambda *args, **kwargs: {
            "predictions": [{
                "classe": "Box",
                "centro": [0, 0, 1],
                "dimensoes": [1, 1, 1],
                "rotacao_z": 0,
                "n_pontos": 4,
            }],
            "clusters": [np.asarray([
                [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
            ], dtype=np.float32)],
            "diagnostics": {},
        },
    )

    class FakePipeline:
        MAPA_FLEXSIM = {"Box": "VisualTool"}

        @staticmethod
        def exportar_stl(cluster: object, path: str) -> bool:
            assert hasattr(cluster, "points")
            Path(path).write_text("solid fake", encoding="utf-8")
            return True

        @staticmethod
        def exportar_layout(objects: list[dict[str, object]], folder: str, mapa: object) -> None:
            assert objects[0]["rotacao_z"] == 0
            Path(folder, "layout.json").write_text("[]", encoding="utf-8")
            Path(folder, "layout.csv").write_text("", encoding="utf-8")

        @staticmethod
        def gerar_flexscript(objects: list[dict[str, object]], path: str, mapa: object) -> None:
            Path(path).write_text("// fake", encoding="utf-8")

    monkeypatch.setattr("core.pipeline_service._carregar_pipeline", lambda: FakePipeline())
    resultado = processar_scan(
        scan,
        parametros=PipelineParameters(
            detector_backend="pointnet2", model_checkpoint="modelo.pt"
        ),
        saida=tmp_path / "saida",
        exportar=True,
    )

    assert resultado["predicoes"][0]["arquivo_stl"] == "Box_1.stl"
    assert (tmp_path / "saida" / "modelos_3d" / "Box_1.stl").exists()
    assert (tmp_path / "saida" / "layout.json").exists()


def test_avaliar_predicoes_do_servico_escreve_metricas(tmp_path: Path) -> None:
    label_dir = tmp_path / "label"
    label_dir.mkdir()
    (label_dir / "000001.txt").write_text(
        "Box 0 0 1 2 1 2 0\n", encoding="utf-8"
    )
    predicoes = {
        "scans": {
            "000001": [
                {
                    "classe": "Box",
                    "centro": [0, 0, 1],
                    "dimensoes": [2, 1, 2],
                    "rotacao_z": 0,
                }
            ]
        }
    }
    destino = tmp_path / "metricas.json"

    resultado = avaliar_predicoes(predicoes, label_dir, thresholds=(0.5,), saida=destino)

    assert resultado["ok"] is True
    assert resultado["por_threshold"]["0.5"]["f1"] == 1.0
    assert json.loads(destino.read_text(encoding="utf-8"))["thresholds"] == [0.5]

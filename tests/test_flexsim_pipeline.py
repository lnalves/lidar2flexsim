"""Laço em tempo real ponta a ponta: fonte → modelo → tracking → publicação."""

from __future__ import annotations

import json
from typing import Any
from urllib.request import urlopen

import numpy as np
import pytest

pytest.importorskip("torch")

from ml.config import PointNet2Config
from ml.flexsim import SCENE_FORMAT, LivePointSource, ReplayPointSource, SceneServer
from ml.flexsim.pipeline import StreamConfig, StreamPipeline


@pytest.fixture
def tiny_model() -> Any:
    from ml.models.pointnet2_seg import PointNet2Segmentation

    config = PointNet2Config(
        input_points=64, sa1_points=16, sa2_points=8, neighbors=4, hidden_channels=8
    )
    model = PointNet2Segmentation(config)
    model.eval()
    return model


@pytest.fixture
def scan_dataset(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    rng = np.random.default_rng(7)
    for index in range(4):
        points = rng.normal(scale=2.0, size=(200, 4)).astype(np.float32)
        points.tofile(bin_dir / f"{index:06d}.bin")
    return tmp_path


def _pipeline(model: Any, **kwargs: Any) -> StreamPipeline:
    return StreamPipeline(
        "modelo-em-memoria.pt",
        config=StreamConfig(num_points=64, score_threshold=0.0),
        model=model,
        **kwargs,
    )


def test_replay_completo_publica_uma_cena_por_quadro(tiny_model, scan_dataset, tmp_path):
    saida = tmp_path / "flexsim"
    pipeline = _pipeline(tiny_model, output_dir=saida)
    source = ReplayPointSource(scan_dataset, rate_hz=1000.0, realtime=False)

    resumo = pipeline.run(source)

    assert resumo["frames"] == 4
    assert resumo["event"] == "summary"
    assert (saida / "scene.json").is_file()
    assert (saida / "scene.csv").is_file()
    assert (saida / "lidar_bridge.txt").is_file()
    assert json.loads((saida / "scene.json").read_text(encoding="utf-8"))["frame"] == 3


def test_cena_publicada_identifica_a_fonte_e_o_scan(tiny_model, scan_dataset):
    pipeline = _pipeline(tiny_model)
    pipeline.run(ReplayPointSource(scan_dataset, rate_hz=1000.0, realtime=False))

    scene = pipeline.last_scene
    assert scene is not None
    assert scene["format"] == SCENE_FORMAT
    assert scene["source"] == "replay"
    assert scene["scan_id"] == "000003"
    assert scene["diagnostics"]["input_points"] == 200


def test_max_frames_interrompe_antes_do_fim_da_fonte(tiny_model, scan_dataset):
    pipeline = _pipeline(tiny_model)
    resumo = pipeline.run(
        ReplayPointSource(scan_dataset, rate_hz=1000.0, realtime=False), max_frames=2
    )
    assert resumo["frames"] == 2


def test_stop_encerra_o_laco_no_proximo_quadro(tiny_model, scan_dataset):
    pipeline = _pipeline(tiny_model)
    source = ReplayPointSource(scan_dataset, rate_hz=1000.0, loop=True, realtime=False)

    def parar(report: Any) -> None:
        if report.frame >= 1:
            pipeline.stop()

    resumo = pipeline.run(source, on_frame=parar)
    assert resumo["frames"] == 2
    assert not pipeline.running


def test_ids_persistem_entre_quadros_do_mesmo_objeto(tiny_model, tmp_path):
    # Um bloco denso e imóvel, repetido em vários scans: qualquer classe que
    # o modelo atribua tem de continuar sendo o mesmo objeto quadro a quadro.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    rng = np.random.default_rng(3)
    bloco = rng.normal(scale=0.2, size=(300, 4)).astype(np.float32)
    for index in range(6):
        bloco.tofile(bin_dir / f"{index:06d}.bin")

    pipeline = _pipeline(tiny_model)
    vistos: list[set[int]] = []
    pipeline.run(
        ReplayPointSource(tmp_path, rate_hz=1000.0, realtime=False),
        on_frame=lambda _: vistos.append(
            {item["track_id"] for item in (pipeline.last_scene or {}).get("objects", [])}
        ),
    )

    publicados = [item for item in vistos if item]
    assert len(publicados) >= 4, f"nenhum objeto estável foi publicado: {vistos}"
    assert set.intersection(*publicados), "os IDs mudaram apesar da cena imóvel"


def test_fonte_ao_vivo_entrega_o_quadro_mais_recente(tiny_model):
    """Uma fonte ao vivo descarta atraso: o pipeline vê a leitura mais nova."""

    pipeline = _pipeline(tiny_model)
    source = LivePointSource()
    rng = np.random.default_rng(11)
    for _ in range(4):
        source.push(rng.normal(scale=0.5, size=(150, 4)).astype(np.float32))
    source.close()

    resumo = pipeline.run(source)

    assert resumo["frames"] == 1
    assert source.dropped == 3
    assert (pipeline.last_scene or {})["frame"] == 3


def test_servidor_reflete_a_ultima_cena_do_pipeline(tiny_model, scan_dataset):
    server = SceneServer(port=0).start()
    try:
        pipeline = _pipeline(tiny_model, server=server)
        pipeline.run(ReplayPointSource(scan_dataset, rate_hz=1000.0, realtime=False))
        payload = json.loads(urlopen(f"{server.url}/scene.json", timeout=5).read())
        assert payload["frame"] == 3
        assert payload["scan_id"] == "000003"
        health = json.loads(urlopen(f"{server.url}/health", timeout=5).read())
        assert health["revision"] == 4
    finally:
        server.stop()


def test_relatorio_por_quadro_traz_o_orcamento_de_tempo(tiny_model, scan_dataset):
    pipeline = _pipeline(tiny_model)
    relatorios: list[Any] = []
    pipeline.run(
        ReplayPointSource(scan_dataset, rate_hz=1000.0, realtime=False),
        on_frame=relatorios.append,
    )
    assert len(relatorios) == 4
    for report in relatorios:
        evento = report.to_event()
        assert evento["event"] == "scene"
        assert evento["inference_ms"] > 0
        assert evento["total_ms"] >= evento["inference_ms"]


def test_fonte_e_fechada_mesmo_com_erro_no_processamento(tiny_model, scan_dataset):
    pipeline = _pipeline(tiny_model)
    source = ReplayPointSource(scan_dataset, rate_hz=1000.0, realtime=False)

    def explodir(_: Any) -> None:
        raise RuntimeError("falha no consumidor")

    with pytest.raises(RuntimeError, match="falha no consumidor"):
        pipeline.run(source, on_frame=explodir)
    assert not pipeline.running
    assert list(source.frames()) == []

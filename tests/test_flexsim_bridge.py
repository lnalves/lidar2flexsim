"""Contrato da ponte LiDAR → FlexSim: transformação, tracking, cena e transporte."""

from __future__ import annotations

import json
import math
import threading
import time
from urllib.request import urlopen

import numpy as np
import pytest

from ml.flexsim import (
    CSV_COLUMNS,
    DEFAULT_FLEXSIM_OBJECTS,
    SCENE_FORMAT,
    LivePointSource,
    ObjectTracker,
    ReplayPointSource,
    SceneServer,
    SceneStore,
    SensorPlacement,
    TrackingConfig,
    build_flexscript_bridge,
    build_scene,
    empty_scene,
    flexsim_corner,
    load_object_map,
    object_name,
    scene_to_csv,
    write_box_stl,
    write_scene_files,
)


def _prediction(
    classe: str = "ForkLift",
    centro: tuple[float, float, float] = (1.0, 2.0, 0.5),
    *,
    dimensoes: tuple[float, float, float] = (2.0, 1.0, 1.6),
    rotacao: float = 0.0,
    score: float = 0.9,
    class_id: int = 5,
    num_pontos: int = 40,
) -> dict[str, object]:
    return {
        "classe": classe,
        "class_id": class_id,
        "score": score,
        "centro": list(centro),
        "dimensoes": list(dimensoes),
        "rotacao": rotacao,
        "num_pontos": num_pontos,
    }


class TestSensorPlacement:
    def test_identidade_preserva_geometria(self) -> None:
        placement = SensorPlacement()
        assert placement.transform_center((1.0, 2.0, 3.0)) == (1.0, 2.0, 3.0)
        assert placement.transform_dimensions((2.0, 1.0, 0.5)) == (2.0, 1.0, 0.5)
        assert placement.transform_yaw_deg(0.0) == 0.0

    def test_rotacao_e_translacao_compoem_na_ordem_documentada(self) -> None:
        placement = SensorPlacement(translation=(10.0, 0.0, 0.0), yaw_deg=90.0)
        x, y, z = placement.transform_center((1.0, 0.0, 2.0))
        assert x == pytest.approx(10.0)
        assert y == pytest.approx(1.0)
        assert z == pytest.approx(2.0)

    def test_escala_afeta_posicao_dimensoes_e_velocidade(self) -> None:
        placement = SensorPlacement(scale=2.0)
        assert placement.transform_center((1.0, 1.0, 1.0)) == (2.0, 2.0, 2.0)
        assert placement.transform_dimensions((1.0, 1.0, 1.0)) == (2.0, 2.0, 2.0)
        assert placement.transform_velocity((1.0, 0.0, 0.0)) == (2.0, 0.0, 0.0)

    def test_yaw_sai_em_graus_normalizados(self) -> None:
        placement = SensorPlacement(yaw_deg=350.0)
        assert placement.transform_yaw_deg(math.radians(20.0)) == pytest.approx(10.0)

    def test_velocidade_ignora_translacao(self) -> None:
        placement = SensorPlacement(translation=(5.0, 5.0, 5.0))
        assert placement.transform_velocity((1.0, 0.0, 0.0)) == (1.0, 0.0, 0.0)

    def test_configuracao_rejeita_chave_desconhecida(self) -> None:
        with pytest.raises(ValueError, match="desconhecidas"):
            SensorPlacement.from_mapping({"rotation": 10})

    def test_escala_nao_positiva_e_recusada(self) -> None:
        with pytest.raises(ValueError, match="scale"):
            SensorPlacement(scale=0.0)


class TestFlexsimCorner:
    def test_canto_reproduz_o_centro_sem_rotacao(self) -> None:
        corner = flexsim_corner((0.0, 0.0, 0.0), (2.0, 4.0, 6.0), 0.0)
        assert corner == pytest.approx((-1.0, 2.0, -3.0))

    def test_canto_gira_junto_com_o_objeto(self) -> None:
        corner = flexsim_corner((0.0, 0.0, 0.0), (2.0, 4.0, 6.0), 90.0)
        assert corner == pytest.approx((-2.0, -1.0, -3.0))

    def test_conversao_e_reversivel_pela_definicao_do_flexsim(self) -> None:
        center, size, rotation = (3.0, -2.0, 1.0), (2.0, 4.0, 6.0), 30.0
        cx, cy, cz = flexsim_corner(center, size, rotation)
        angle = math.radians(rotation)
        local = (0.5 * size[0], -0.5 * size[1])
        recovered = (
            cx + math.cos(angle) * local[0] - math.sin(angle) * local[1],
            cy + math.sin(angle) * local[0] + math.cos(angle) * local[1],
            cz + 0.5 * size[2],
        )
        assert recovered == pytest.approx(center)


class TestObjectTracker:
    def test_faixa_so_e_publicada_apos_min_hits(self) -> None:
        tracker = ObjectTracker(TrackingConfig(min_hits=2))
        first = tracker.update([_prediction()], timestamp=0.0)
        assert first.tracks == ()
        second = tracker.update([_prediction()], timestamp=0.1)
        assert len(second.tracks) == 1
        assert second.tracks[0].state == "new"

    def test_id_persiste_com_o_objeto_em_movimento(self) -> None:
        tracker = ObjectTracker(TrackingConfig(min_hits=1))
        ids = []
        for step in range(5):
            result = tracker.update(
                [_prediction(centro=(step * 0.2, 0.0, 0.5))], timestamp=step * 0.1
            )
            ids.append(result.tracks[0].track_id)
        assert len(set(ids)) == 1

    def test_salto_maior_que_o_gate_cria_faixa_nova(self) -> None:
        tracker = ObjectTracker(TrackingConfig(min_hits=1, max_distance=0.5))
        first = tracker.update([_prediction(centro=(0.0, 0.0, 0.5))], timestamp=0.0)
        second = tracker.update([_prediction(centro=(9.0, 0.0, 0.5))], timestamp=0.1)
        # A faixa antiga continua viva em coasting; o que importa é que a
        # detecção distante não foi absorvida por ela.
        novas = {item.track_id for item in second.tracks} - {
            item.track_id for item in first.tracks
        }
        assert len(novas) == 1

    def test_classes_diferentes_nao_sao_associadas(self) -> None:
        tracker = ObjectTracker(TrackingConfig(min_hits=1))
        first = tracker.update([_prediction("ForkLift")], timestamp=0.0)
        second = tracker.update([_prediction("Box", class_id=1)], timestamp=0.1)
        novas = {item.track_id for item in second.tracks} - {
            item.track_id for item in first.tracks
        }
        assert len(novas) == 1
        assert {item.classe for item in second.tracks} == {"ForkLift", "Box"}

    def test_objeto_ausente_sobrevive_ate_max_age_e_some_depois(self) -> None:
        tracker = ObjectTracker(TrackingConfig(min_hits=1, max_age=2))
        tracker.update([_prediction()], timestamp=0.0)
        estados = []
        for step in range(1, 5):
            result = tracker.update([], timestamp=step * 0.1)
            estados.append((len(result.tracks), len(result.removed)))
        assert estados[0] == (1, 0)
        assert estados[-1] == (0, 0)
        assert any(removed for _, removed in estados)

    def test_faixa_removida_carrega_a_classe_para_o_nome(self) -> None:
        tracker = ObjectTracker(TrackingConfig(min_hits=1, max_age=0))
        tracker.update([_prediction("CargoBike", class_id=3)], timestamp=0.0)
        result = tracker.update([], timestamp=0.1)
        assert [item.classe for item in result.removed] == ["CargoBike"]

    def test_velocidade_acompanha_o_deslocamento(self) -> None:
        tracker = ObjectTracker(TrackingConfig(min_hits=1, smoothing=1.0))
        for step in range(6):
            result = tracker.update(
                [_prediction(centro=(step * 0.1, 0.0, 0.5))], timestamp=step * 0.1
            )
        assert result.tracks[0].velocity[0] == pytest.approx(1.0, rel=0.2)

    def test_yaw_nao_oscila_com_a_ambiguidade_do_pca(self) -> None:
        tracker = ObjectTracker(TrackingConfig(min_hits=1, smoothing=0.5))
        tracker.update([_prediction(rotacao=0.0)], timestamp=0.0)
        result = tracker.update([_prediction(rotacao=math.pi)], timestamp=0.1)
        # Sem o alinhamento, a média de 0 e π daria π/2 — uma orientação que
        # nenhuma das duas observações jamais viu.
        assert math.sin(result.tracks[0].yaw_rad) == pytest.approx(0.0, abs=1e-9)

    def test_ids_nao_sao_reaproveitados_apos_reset(self) -> None:
        tracker = ObjectTracker(TrackingConfig(min_hits=1))
        first = tracker.update([_prediction()], timestamp=0.0)
        tracker.reset()
        second = tracker.update([_prediction()], timestamp=1.0)
        assert second.tracks[0].track_id > first.tracks[0].track_id

    def test_smoothing_fora_do_intervalo_e_recusado(self) -> None:
        with pytest.raises(ValueError, match="smoothing"):
            TrackingConfig(smoothing=0.0)


class TestScene:
    def _cena(self, **kwargs: object) -> dict:
        tracker = ObjectTracker(TrackingConfig(min_hits=1))
        result = tracker.update([_prediction()], timestamp=0.0)
        return build_scene(result.tracks, result.removed, frame=7, timestamp=1.5, **kwargs)

    def test_payload_declara_formato_unidades_e_ancora(self) -> None:
        scene = self._cena()
        assert scene["format"] == SCENE_FORMAT
        assert scene["units"] == {"length": "m", "rotation": "deg", "velocity": "m/s"}
        assert scene["anchor"] == "corner"
        assert scene["frame"] == 7

    def test_objeto_traz_nome_estavel_e_tipo_flexsim(self) -> None:
        scene = self._cena()
        objeto = scene["objects"][0]
        assert objeto["name"] == object_name(objeto["track_id"], "ForkLift")
        assert objeto["flexsim_class"] == DEFAULT_FLEXSIM_OBJECTS["ForkLift"]

    def test_location_e_center_descrevem_a_mesma_caixa(self) -> None:
        scene = self._cena()
        objeto = scene["objects"][0]
        esperado = flexsim_corner(objeto["center"], objeto["size"], objeto["rotation"][2])
        assert objeto["location"] == pytest.approx(esperado, abs=1e-3)

    def test_placement_e_aplicado_ao_publicar(self) -> None:
        scene = self._cena(placement={"translation": [100.0, 0.0, 0.0]})
        assert scene["objects"][0]["center"][0] == pytest.approx(101.0)
        assert scene["sensor"]["translation"] == [100.0, 0.0, 0.0]

    def test_cena_vazia_e_valida(self) -> None:
        scene = empty_scene()
        assert scene["objects"] == []
        assert scene["removed"] == []
        assert scene["format"] == SCENE_FORMAT

    def test_json_da_cena_nao_contem_nan(self) -> None:
        json.dumps(self._cena(), allow_nan=False)


class TestSceneCsv:
    def _csv(self) -> list[str]:
        tracker = ObjectTracker(TrackingConfig(min_hits=1, max_age=0))
        tracker.update([_prediction()], timestamp=0.0)
        result = tracker.update([], timestamp=0.1)
        scene = build_scene(result.tracks, result.removed, frame=3, timestamp=2.0)
        return scene_to_csv(scene).splitlines()

    def test_cabecalho_declara_formato_frame_e_contagem(self) -> None:
        linhas = self._csv()
        formato, frame, _, total = linhas[0].split(",")
        assert formato == SCENE_FORMAT
        assert int(frame) == 3
        assert int(total) == len(linhas) - 2

    def test_segunda_linha_lista_as_colunas(self) -> None:
        assert self._csv()[1].split(",") == list(CSV_COLUMNS)

    def test_remocao_aparece_como_linha_com_estado_removed(self) -> None:
        linhas = self._csv()
        estados = [linha.split(",")[4] for linha in linhas[2:]]
        assert "removed" in estados

    def test_nome_removido_bate_com_o_nome_criado(self) -> None:
        tracker = ObjectTracker(TrackingConfig(min_hits=1, max_age=0))
        criado = tracker.update([_prediction()], timestamp=0.0)
        removido = tracker.update([], timestamp=0.1)
        nome_criado = build_scene(criado.tracks, criado.removed)["objects"][0]["name"]
        nome_removido = build_scene(removido.tracks, removido.removed)["removed"][0]["name"]
        assert nome_criado == nome_removido

    def test_campos_nunca_introduzem_virgula_extra(self) -> None:
        tracker = ObjectTracker(TrackingConfig(min_hits=1))
        result = tracker.update([_prediction(classe="Classe, com virgula")], timestamp=0.0)
        scene = build_scene(result.tracks, result.removed)
        linha = scene_to_csv(scene).splitlines()[2]
        assert len(linha.split(",")) == len(CSV_COLUMNS)


class TestObjectMap:
    def test_padrao_cobre_todas_as_classes_do_dataset(self) -> None:
        mapping = load_object_map()
        for classe in ("Box", "ELFplusplus", "CargoBike", "FTS", "ForkLift"):
            assert classe in mapping

    def test_sobrescrita_parcial_mantem_o_restante(self) -> None:
        mapping = load_object_map({"flexsim_objects": {"Box": "Rack"}})
        assert mapping["Box"] == "Rack"
        assert mapping["ForkLift"] == DEFAULT_FLEXSIM_OBJECTS["ForkLift"]

    def test_dicionario_direto_tambem_e_aceito(self) -> None:
        assert load_object_map({"Box": "Queue"})["Box"] == "Queue"

    def test_arquivo_ausente_falha_explicitamente(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_object_map(tmp_path / "inexistente.json")


class TestExportArtifacts:
    def test_escrita_de_cena_gera_json_e_csv_coerentes(self, tmp_path) -> None:
        scene = empty_scene(frame=2)
        paths = write_scene_files(tmp_path, scene)
        assert json.loads(paths["json"].read_text(encoding="utf-8"))["frame"] == 2
        assert paths["csv"].read_text(encoding="utf-8").startswith(SCENE_FORMAT)

    def test_escrita_nao_deixa_arquivos_temporarios(self, tmp_path) -> None:
        write_scene_files(tmp_path, empty_scene())
        assert not list(tmp_path.glob(".*.tmp"))

    def test_reescrita_substitui_o_conteudo(self, tmp_path) -> None:
        write_scene_files(tmp_path, empty_scene(frame=1))
        paths = write_scene_files(tmp_path, empty_scene(frame=2))
        assert json.loads(paths["json"].read_text(encoding="utf-8"))["frame"] == 2

    def test_flexscript_cria_atualiza_e_destroi(self, tmp_path) -> None:
        script = build_flexscript_bridge(csv_path=tmp_path / "scene.csv", container="Cena")
        assert 'Model.find("Cena")' in script
        assert "destroyobject" in script
        assert "setLocation" in script
        assert SCENE_FORMAT in script

    def test_flexscript_le_as_colunas_nas_posicoes_certas(self, tmp_path) -> None:
        script = build_flexscript_bridge(csv_path=tmp_path / "scene.csv")
        # FlexScript indexa arrays a partir de 1.
        assert f"campo[{CSV_COLUMNS.index('state') + 1}]" in script
        assert f"campo[{CSV_COLUMNS.index('loc_x') + 1}]" in script
        assert f"campo[{CSV_COLUMNS.index('size_x') + 1}]" in script
        assert f"campo[{CSV_COLUMNS.index('rot_z') + 1}]" in script

    def test_stl_de_caixa_tem_doze_triangulos(self, tmp_path) -> None:
        caminho = write_box_stl(tmp_path / "caixa.stl")
        dados = caminho.read_bytes()
        assert int.from_bytes(dados[80:84], "little") == 12
        assert len(dados) == 84 + 12 * 50

    def test_stl_rejeita_dimensao_invalida(self, tmp_path) -> None:
        with pytest.raises(ValueError):
            write_box_stl(tmp_path / "caixa.stl", (0.0, 1.0, 1.0))


class TestSceneServer:
    def test_publica_json_csv_e_health(self) -> None:
        store = SceneStore()
        server = SceneServer(store, port=0).start()
        try:
            store.publish(empty_scene(frame=11))
            payload = json.loads(urlopen(f"{server.url}/scene.json", timeout=5).read())
            assert payload["frame"] == 11
            csv_text = urlopen(f"{server.url}/scene.csv", timeout=5).read().decode()
            assert csv_text.startswith(SCENE_FORMAT)
            health = json.loads(urlopen(f"{server.url}/health", timeout=5).read())
            assert health["status"] == "ok"
            assert health["frame"] == 11
        finally:
            server.stop()

    def test_rota_desconhecida_responde_404(self) -> None:
        from urllib.error import HTTPError

        server = SceneServer(port=0).start()
        try:
            with pytest.raises(HTTPError) as info:
                urlopen(f"{server.url}/nao-existe", timeout=5)
            assert info.value.code == 404
        finally:
            server.stop()

    def test_serve_antes_do_primeiro_quadro(self) -> None:
        server = SceneServer(port=0).start()
        try:
            payload = json.loads(urlopen(f"{server.url}/scene", timeout=5).read())
            assert payload["objects"] == []
        finally:
            server.stop()

    def test_revisao_avanca_a_cada_publicacao(self) -> None:
        store = SceneStore()
        store.publish(empty_scene(frame=1))
        store.publish(empty_scene(frame=2))
        assert store.revision == 2


class TestSources:
    def test_replay_respeita_max_frames(self, tmp_path) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        for index in range(3):
            points = np.full((8, 4), float(index), dtype=np.float32)
            points.tofile(bin_dir / f"{index:06d}.bin")
        source = ReplayPointSource(tmp_path, rate_hz=1000.0, max_frames=2, realtime=False)
        frames = list(source.frames())
        assert [frame.index for frame in frames] == [0, 1]
        assert frames[0].scan_id == "000000"

    def test_replay_em_loop_reinicia_o_dataset(self, tmp_path) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        np.zeros((8, 4), dtype=np.float32).tofile(bin_dir / "000000.bin")
        source = ReplayPointSource(
            tmp_path, rate_hz=1000.0, loop=True, max_frames=3, realtime=False
        )
        assert len(list(source.frames())) == 3

    def test_replay_sem_scans_falha_cedo(self, tmp_path) -> None:
        (tmp_path / "bin").mkdir()
        with pytest.raises(FileNotFoundError):
            ReplayPointSource(tmp_path)

    def test_replay_respeita_o_intervalo_do_sensor(self, tmp_path) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        for index in range(3):
            np.zeros((8, 4), dtype=np.float32).tofile(bin_dir / f"{index:06d}.bin")
        source = ReplayPointSource(tmp_path, rate_hz=50.0, realtime=True)
        started = time.monotonic()
        list(source.frames())
        assert time.monotonic() - started >= 2 / 50.0

    def test_close_interrompe_o_replay(self, tmp_path) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        for index in range(50):
            np.zeros((8, 4), dtype=np.float32).tofile(bin_dir / f"{index:06d}.bin")
        source = ReplayPointSource(tmp_path, rate_hz=1000.0, realtime=False)
        received = []
        for frame in source.frames():
            received.append(frame)
            if len(received) == 3:
                source.close()
        assert len(received) == 3

    def test_fonte_ao_vivo_mantem_apenas_o_quadro_mais_novo(self) -> None:
        source = LivePointSource()
        for value in range(5):
            source.push(np.full((4, 4), float(value), dtype=np.float32))
        source.close()
        frames = list(source.frames())
        assert len(frames) == 1
        assert frames[0].points[0, 0] == pytest.approx(4.0)
        assert source.dropped == 4

    def test_fonte_ao_vivo_entrega_em_ordem_para_consumidor_atento(self) -> None:
        source = LivePointSource()
        recebidos: list[int] = []

        def consumir() -> None:
            for frame in source.frames():
                recebidos.append(frame.index)

        worker = threading.Thread(target=consumir)
        worker.start()
        for _ in range(3):
            source.push(np.zeros((4, 4), dtype=np.float32))
            time.sleep(0.02)
        source.close()
        worker.join(timeout=5)
        assert recebidos == sorted(recebidos)

    def test_push_apos_close_e_ignorado(self) -> None:
        source = LivePointSource()
        source.close()
        assert source.push(np.zeros((4, 4), dtype=np.float32)) is None

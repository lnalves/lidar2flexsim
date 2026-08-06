"""Fluxo ponta a ponta do CLI de avaliação de detecções 3D."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.evaluation import main


def _labels(root: Path) -> Path:
    labels = root / "label"
    labels.mkdir(parents=True)
    (labels / "000000.txt").write_text("Box 0 0 0 2 2 2 0\n", encoding="utf-8")
    return labels


def _predictions(root: Path, **overrides: object) -> Path:
    box = {
        "classe": "Box",
        "centro": [0.0, 0.0, 0.0],
        "dimensoes": [2.0, 2.0, 2.0],
        "yaw_rad": 0.0,
    }
    box.update(overrides)
    path = root / "predicoes.json"
    path.write_text(json.dumps({"scans": {"000000": [box]}}), encoding="utf-8")
    return path


def _run(monkeypatch: pytest.MonkeyPatch, *argv: str) -> None:
    monkeypatch.setattr("sys.argv", ["ml.evaluation", *argv])
    main()


def test_main_scores_a_perfect_match_and_writes_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    labels = _labels(tmp_path)
    predictions = _predictions(tmp_path)
    output = tmp_path / "saida" / "metricas.json"

    _run(
        monkeypatch,
        "--predicoes", str(predictions),
        "--labels", str(labels),
        "--saida", str(output),
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["formato"] == "pointnet2-metricas-v1"
    assert report["class_aware"] is False
    for threshold in ("0.25", "0.5"):
        assert report["por_threshold"][threshold]["scans_avaliados"] == 1
        assert report["por_threshold"][threshold]["f1"] == pytest.approx(1.0)
    assert "Métricas gravadas em" in capsys.readouterr().out


def test_main_scores_a_disjoint_prediction_as_a_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = _labels(tmp_path)
    predictions = _predictions(tmp_path, centro=[50.0, 50.0, 0.0])
    output = tmp_path / "metricas.json"

    _run(
        monkeypatch,
        "--predicoes", str(predictions),
        "--labels", str(labels),
        "--saida", str(output),
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["por_threshold"]["0.5"]["f1"] == pytest.approx(0.0)
    # O scan foi avaliado; apenas nenhuma caixa casou.
    assert report["por_threshold"]["0.5"]["scans_avaliados"] == 1


def test_main_accepts_a_single_scan_label_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = _labels(tmp_path)
    predictions = _predictions(tmp_path)
    output = tmp_path / "metricas.json"

    # Apontar para o .txt precisa inferir o scan-id do nome do arquivo.
    _run(
        monkeypatch,
        "--predicoes", str(predictions),
        "--labels", str(labels / "000000.txt"),
        "--saida", str(output),
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["por_threshold"]["0.25"]["scans_avaliados"] == 1


def test_main_honours_custom_iou_thresholds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = _labels(tmp_path)
    predictions = _predictions(tmp_path)
    output = tmp_path / "metricas.json"

    _run(
        monkeypatch,
        "--predicoes", str(predictions),
        "--labels", str(labels),
        "--saida", str(output),
        "--iou-thresholds", "0.1", "0.9",
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert sorted(report["por_threshold"]) == ["0.1", "0.9"]


def test_class_aware_matching_records_the_supplied_class_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = _labels(tmp_path)
    predictions = _predictions(tmp_path, classe="caixa")
    class_map = tmp_path / "mapa.json"
    class_map.write_text(json.dumps({"caixa": "Box"}), encoding="utf-8")
    output = tmp_path / "metricas.json"

    _run(
        monkeypatch,
        "--predicoes", str(predictions),
        "--labels", str(labels),
        "--saida", str(output),
        "--class-aware",
        "--class-map", str(class_map),
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["class_aware"] is True
    assert report["class_map"] == {"caixa": "Box"}
    assert report["por_threshold"]["0.5"]["f1"] == pytest.approx(1.0)


def test_main_fails_loudly_when_no_prediction_has_a_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = _labels(tmp_path)
    path = tmp_path / "predicoes.json"
    path.write_text(
        json.dumps({
            "scans": {
                "999999": [{
                    "classe": "Box",
                    "centro": [0.0, 0.0, 0.0],
                    "dimensoes": [1.0, 1.0, 1.0],
                }]
            }
        }),
        encoding="utf-8",
    )

    # Um relatório vazio passaria por "modelo sem detecções"; o erro explícito
    # aponta o descasamento de scan_id.
    with pytest.raises(FileNotFoundError, match="label correspondente"):
        _run(
            monkeypatch,
            "--predicoes", str(path),
            "--labels", str(labels),
            "--saida", str(tmp_path / "metricas.json"),
        )


def test_main_rejects_a_labels_directory_that_does_not_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predictions = _predictions(tmp_path)

    with pytest.raises(FileNotFoundError, match="Pasta de labels"):
        _run(
            monkeypatch,
            "--predicoes", str(predictions),
            "--labels", str(tmp_path / "inexistente"),
            "--saida", str(tmp_path / "metricas.json"),
        )

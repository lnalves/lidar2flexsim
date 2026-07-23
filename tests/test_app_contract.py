"""Testes headless dos contratos usados pela interface NiceGUI.

Importar ``app`` não deve abrir a janela nem exigir Open3D/NiceGUI. A criação
da página só ocorre quando ``main``/``run_app`` é chamado.
"""

from __future__ import annotations

from pathlib import Path

from app import (
    GuiController,
    PARAMETER_PRESETS,
    ProcessingConfig,
    ProgressState,
    list_scan_files,
    validate_dataset,
)


def test_validate_dataset_habilita_bin_e_reporta_partes_opcionais(tmp_path: Path) -> None:
    raiz = tmp_path / "warehouse"
    (raiz / "BIN").mkdir(parents=True)
    (raiz / "BIN" / "000010.bin").write_bytes(b"scan")

    info = validate_dataset(raiz)

    assert info.ready is True
    assert info.complete is False
    assert info.bin_count == 1
    assert info.bin_dir == raiz / "BIN"
    assert info.label_dir is None
    assert info.vis_dir is None
    assert not info.errors
    assert info.warnings


def test_validate_dataset_inexistente_nao_lanca_excecao(tmp_path: Path) -> None:
    info = validate_dataset(tmp_path / "missing")

    assert info.ready is False
    assert info.complete is False
    assert info.errors


def test_list_scan_files_ordena_ids_numericos(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for nome in ("10.bin", "2.bin", "01.bin", "ruido.txt"):
        (bin_dir / nome).write_bytes(b"x")

    assert [item.name for item in list_scan_files(bin_dir)] == [
        "01.bin",
        "2.bin",
        "10.bin",
    ]


def test_progress_state_atualiza_com_limites() -> None:
    estado = ProgressState()
    estado.update(current=2, total=4, message="Lendo")
    assert estado.snapshot() == {
        "current": 2,
        "total": 4,
        "fraction": 0.5,
        "message": "Lendo",
        "running": False,
        "cancelled": False,
        "error": None,
    }

    estado.update(current=-2, fraction=4.0)
    assert estado.snapshot()["current"] == 0
    assert estado.snapshot()["fraction"] == 1.0


def test_processing_config_eh_serializavel() -> None:
    config = ProcessingConfig(
        dataset_dir=Path("dados/warehouse"),
        bin_dir=Path("dados/warehouse/bin"),
        label_dir=Path("dados/warehouse/label"),
        vis_dir=None,
        output_dir=Path("saida/teste"),
        scan_paths=[Path("dados/warehouse/bin/000001.bin")],
        params=PARAMETER_PRESETS["equilibrado"],
    )

    payload = config.as_dict()

    assert payload["dataset_dir"] == "dados/warehouse"
    assert payload["scan_paths"] == ["dados/warehouse/bin/000001.bin"]
    assert payload["voxel"] == 0.05
    assert payload["params"]["oriented_box"] is True


def test_format_diagnostics_resume_segmentacao() -> None:
    texto = GuiController._format_diagnostics({
        "000001": {
            "clusters": 4,
            "pontos_chao": 100,
            "pontos_objetos": 900,
            "inclinacao_deg": 1.5,
            "z_chao": -0.6,
            "metodo": "ransac_horizontal",
        }
    })

    assert "4 clusters" in texto
    assert "inclinação média" in texto
    assert "-0.600 m" in texto

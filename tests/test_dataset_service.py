"""Testes da descoberta/validação do Warehouse Dataset.

O serviço só examina nomes e metadados dos arquivos; por isso estes testes não
precisam instalar Open3D nem carregar scans de centenas de megabytes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.dataset_service import listar_scans, localizar_diretorios, validar_dataset


def _criar_dataset(
    raiz: Path,
    *,
    scans: tuple[str, ...] = ("000002", "000001"),
    incluir_labels: bool = True,
    incluir_vis: bool = True,
) -> Path:
    bin_dir = raiz / "bin"
    bin_dir.mkdir(parents=True)
    for scan_id in scans:
        # O validador não decodifica o conteúdo. Um byte já representa um
        # arquivo de scan para o teste estrutural.
        (bin_dir / f"{scan_id}.bin").write_bytes(b"scan")

    if incluir_labels:
        label_dir = raiz / "label"
        label_dir.mkdir()
        (label_dir / f"{scans[-1]}.txt").write_text(
            "Box 0 0 0 1 1 1 0\n", encoding="utf-8"
        )
        # Este arquivo extra não deve ser confundido com um scan.
        (label_dir / "README.md").write_text("fixture", encoding="utf-8")

    if incluir_vis:
        vis_dir = raiz / "vis"
        vis_dir.mkdir()
        # A validação verifica extensão/presença, não decodifica a imagem.
        (vis_dir / f"{scans[-1]}.png").write_bytes(b"png")

    return raiz


def test_validar_dataset_completo(tmp_path: Path) -> None:
    raiz = _criar_dataset(tmp_path / "warehouse")

    resultado = validar_dataset(raiz)

    assert resultado["valido"] is True
    assert resultado["valid"] is True
    assert resultado["completo"] is False  # o segundo scan não tem pares
    assert resultado["contagens"] == {"bin": 2, "label": 1, "vis": 1}
    assert resultado["scan_ids"] == ["000001", "000002"]
    assert resultado["labels_faltantes"] == ["000002"]
    assert resultado["vis_faltantes"] == ["000002"]
    assert resultado["erros"] == []
    assert resultado["avisos"]


def test_validar_dataset_aceita_diretamente_a_pasta_bin(tmp_path: Path) -> None:
    raiz = _criar_dataset(tmp_path / "warehouse", scans=("000001",))
    bin_dir = raiz / "bin"

    diretorios = localizar_diretorios(bin_dir)
    resultado = validar_dataset(bin_dir)

    assert diretorios["root"] == raiz
    assert resultado["valido"] is True
    assert Path(resultado["bin_dir"]) == bin_dir


def test_validar_dataset_inexistente_tem_diagnostico_util(tmp_path: Path) -> None:
    resultado = validar_dataset(tmp_path / "nao-existe")

    assert resultado["valido"] is False
    assert resultado["erros"]
    assert any("não encontrado" in erro for erro in resultado["erros"])


def test_listar_scans_ordena_por_nome_e_rejeita_pasta_sem_bin(tmp_path: Path) -> None:
    raiz = _criar_dataset(tmp_path / "warehouse", scans=("10", "2", "01"))

    assert [scan.name for scan in listar_scans(raiz)] == [
        "01.bin",
        "10.bin",
        "2.bin",
    ]

    with pytest.raises(FileNotFoundError, match="Pasta bin"):
        listar_scans(tmp_path / "vazia")

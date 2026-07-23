"""Serviços de descoberta e validação do Warehouse LiDAR Dataset.

O serviço aceita tanto a raiz do dataset (contendo ``bin/``, ``label/`` e
``vis/``) quanto diretamente a pasta ``bin/``. Nenhum arquivo é copiado ou
carregado para a memória durante a validação; apenas nomes e metadados básicos
são inspecionados, o que mantém a operação rápida mesmo para milhares de
scans.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


BIN_EXTENSIONS = {".bin"}
LABEL_EXTENSIONS = {".txt"}
VIS_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _as_path(pasta: str | Path) -> Path:
    if isinstance(pasta, Path):
        return pasta.expanduser()
    if isinstance(pasta, str):
        if not pasta.strip():
            raise ValueError("A pasta do dataset não pode ser vazia.")
        return Path(pasta).expanduser()
    raise TypeError("pasta deve ser str ou pathlib.Path.")


def _diretorio_por_nome(raiz: Path, nomes: Iterable[str]) -> Path | None:
    """Encontra um subdiretório sem diferenciar maiúsculas/minúsculas."""
    candidatos = {nome.casefold() for nome in nomes}
    try:
        for item in raiz.iterdir():
            if item.is_dir() and item.name.casefold() in candidatos:
                return item
    except OSError:
        return None
    return None


def localizar_diretorios(pasta: str | Path) -> dict[str, Path | None]:
    """Resolve raiz e diretórios ``bin``, ``label`` e ``vis``.

    ``vis 2`` é aceito como fallback, pois algumas cópias do dataset possuem
    esse nome após uma segunda extração dos arquivos de visualização.
    """
    caminho = _as_path(pasta)
    if caminho.name.casefold() == "bin" and caminho.is_dir():
        raiz = caminho.parent
    else:
        raiz = caminho

    bin_dir = _diretorio_por_nome(raiz, ("bin",))
    label_dir = _diretorio_por_nome(raiz, ("label", "labels"))
    vis_dir = _diretorio_por_nome(raiz, ("vis", "visualization", "visualizations"))
    if vis_dir is None:
        vis_dir = _diretorio_por_nome(raiz, ("vis 2", "vis2"))

    # Quando a pasta passada não é a raiz, mas contém os arquivos diretamente,
    # permitir validação/processamento sem obrigar o usuário a reorganizá-la.
    if bin_dir is None and caminho.is_dir():
        try:
            if any(item.is_file() and item.suffix.casefold() == ".bin" for item in caminho.iterdir()):
                bin_dir = caminho
                raiz = caminho.parent
        except OSError:
            pass

    return {"root": raiz, "bin": bin_dir, "label": label_dir, "vis": vis_dir}


def _arquivos(diretorio: Path | None, extensoes: set[str]) -> list[Path]:
    if diretorio is None or not diretorio.is_dir():
        return []
    try:
        return sorted(
            (item for item in diretorio.iterdir()
             if item.is_file() and item.suffix.casefold() in extensoes),
            key=lambda item: item.name.casefold(),
        )
    except OSError:
        return []


def listar_scans(pasta: str | Path) -> list[Path]:
    """Lista scans ``.bin`` ordenados pelo nome.

    A função é deliberadamente pequena e independente do pipeline para ser
    usada pela tela de seleção de scans.
    """
    diretorios = localizar_diretorios(pasta)
    bin_dir = diretorios["bin"]
    if bin_dir is None:
        raise FileNotFoundError(f"Pasta bin não encontrada em: {pasta}")
    scans = _arquivos(bin_dir, BIN_EXTENSIONS)
    if not scans:
        raise FileNotFoundError(f"Nenhum arquivo .bin encontrado em: {bin_dir}")
    return scans


def _mensagem_erro(caminho: Path, detalhe: str) -> str:
    return f"{detalhe}: {caminho}"


def validar_dataset(pasta: str | Path) -> dict[str, Any]:
    """Valida a estrutura local do dataset e retorna um diagnóstico.

    O campo ``valido`` (e seu alias em inglês ``valid``) indica se existe uma
    pasta ``bin`` com ao menos um scan, requisito mínimo para processar. Os
    diretórios ``label`` e ``vis`` são opcionais para processamento, mas sua
    ausência é reportada em ``avisos``/``warnings``. ``completo`` indica a
    presença dos três conjuntos.

    O retorno contém somente strings, números, listas e dicionários, portanto
    pode ser enviado diretamente para a interface ou serializado em JSON.
    """
    try:
        caminho = _as_path(pasta)
    except (TypeError, ValueError) as exc:
        return {
            "valido": False,
            "valid": False,
            "completo": False,
            "pasta": str(pasta),
            "root": str(pasta),
            "bin_dir": None,
            "label_dir": None,
            "vis_dir": None,
            "bin": None,
            "label": None,
            "vis": None,
            "diretorios": {"bin": None, "label": None, "vis": None},
            "contagens": {"bin": 0, "label": 0, "vis": 0},
            "counts": {"bin": 0, "label": 0, "vis": 0},
            "scan_ids": [],
            "num_scans": 0,
            "erros": [str(exc)],
            "errors": [str(exc)],
            "avisos": [],
            "warnings": [],
        }

    erros: list[str] = []
    avisos: list[str] = []
    if not caminho.exists():
        erros.append(_mensagem_erro(caminho, "Caminho não encontrado"))
        diretorios: dict[str, Path | None] = {
            "root": caminho, "bin": None, "label": None, "vis": None
        }
    elif not caminho.is_dir():
        erros.append(_mensagem_erro(caminho, "O caminho não é uma pasta"))
        diretorios = {"root": caminho.parent, "bin": None, "label": None, "vis": None}
    else:
        diretorios = localizar_diretorios(caminho)

    bin_files = _arquivos(diretorios.get("bin"), BIN_EXTENSIONS)
    label_files = _arquivos(diretorios.get("label"), LABEL_EXTENSIONS)
    vis_files = _arquivos(diretorios.get("vis"), VIS_EXTENSIONS)

    if diretorios.get("bin") is None:
        erros.append(_mensagem_erro(caminho, "Pasta bin não encontrada"))
    elif not bin_files:
        erros.append(_mensagem_erro(diretorios["bin"], "Nenhum scan .bin encontrado"))

    if diretorios.get("label") is None:
        avisos.append(_mensagem_erro(caminho, "Pasta label não encontrada"))
    elif not label_files:
        avisos.append(_mensagem_erro(diretorios["label"], "Nenhum label .txt encontrado"))

    if diretorios.get("vis") is None:
        avisos.append(_mensagem_erro(caminho, "Pasta vis não encontrada"))
    elif not vis_files:
        avisos.append(_mensagem_erro(diretorios["vis"], "Nenhuma imagem de visualização encontrada"))

    bin_ids = {arquivo.stem for arquivo in bin_files}
    label_ids = {arquivo.stem for arquivo in label_files}
    vis_ids = {arquivo.stem for arquivo in vis_files}
    labels_faltantes = sorted(bin_ids - label_ids)
    vis_faltantes = sorted(bin_ids - vis_ids)
    if labels_faltantes and label_files:
        avisos.append(f"{len(labels_faltantes)} scan(s) sem label correspondente")
    if vis_faltantes and vis_files:
        avisos.append(f"{len(vis_faltantes)} scan(s) sem visualização correspondente")

    caminhos = {
        chave: (str(valor) if valor is not None else None)
        for chave, valor in diretorios.items()
        if chave != "root"
    }
    root = diretorios.get("root") or caminho
    contagens = {"bin": len(bin_files), "label": len(label_files), "vis": len(vis_files)}
    scan_info = {
        "total": len(bin_files),
        "com_label": len(bin_ids & label_ids),
        "sem_label": len(bin_ids - label_ids) if label_files else len(bin_files),
        "com_vis": len(bin_ids & vis_ids),
        "sem_vis": len(bin_ids - vis_ids) if vis_files else len(bin_files),
    }
    valido = bool(bin_files) and not any("Pasta bin" in erro for erro in erros)
    completo = valido and bool(label_files) and bool(vis_files) and not labels_faltantes and not vis_faltantes
    resultado: dict[str, Any] = {
        "valido": valido,
        "valid": valido,
        "completo": completo,
        "pasta": str(caminho),
        "root": str(root),
        "bin_dir": caminhos.get("bin"),
        "label_dir": caminhos.get("label"),
        "vis_dir": caminhos.get("vis"),
        "bin": caminhos.get("bin"),
        "label": caminhos.get("label"),
        "vis": caminhos.get("vis"),
        "diretorios": caminhos,
        "contagens": contagens,
        "counts": contagens.copy(),
        "scans": scan_info,
        "num_scans": len(bin_files),
        "scan_ids": sorted(bin_ids),
        "labels_faltantes": labels_faltantes,
        "vis_faltantes": vis_faltantes,
        "erros": erros,
        "errors": erros.copy(),
        "avisos": avisos,
        "warnings": avisos.copy(),
    }
    return resultado


validate_dataset = validar_dataset


__all__ = ["BIN_EXTENSIONS", "LABEL_EXTENSIONS", "VIS_EXTENSIONS",
           "listar_scans", "localizar_diretorios", "validar_dataset",
           "validate_dataset"]

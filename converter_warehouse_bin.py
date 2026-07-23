"""Converte scans ``.bin`` do Warehouse LiDAR para ``.pcd``.

O pipeline principal já lê ``.bin`` diretamente. Este utilitário é útil para
inspecionar os scans no CloudCompare, MeshLab ou em ferramentas que não leem o
formato binário do dataset.

Exemplos:
    python converter_warehouse_bin.py dados/warehouse/bin/000000.bin
    python converter_warehouse_bin.py dados/warehouse/bin --saida dados/pcd
"""

from __future__ import annotations

import argparse
from pathlib import Path

import open3d as o3d

from lidar2flexsim import carregar_bin


def arquivos_entrada(caminho: Path) -> list[Path]:
    if caminho.is_file():
        if caminho.suffix.lower() != ".bin":
            raise ValueError("A entrada deve ser um arquivo .bin ou uma pasta com arquivos .bin.")
        return [caminho]
    if caminho.is_dir():
        return sorted(caminho.glob("*.bin"))
    raise FileNotFoundError(f"Entrada não encontrada: {caminho}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Converte Warehouse LiDAR .bin para .pcd")
    parser.add_argument("entrada", type=Path, help="Arquivo .bin ou pasta contendo scans")
    parser.add_argument("--saida", type=Path,
                        help="Arquivo .pcd (entrada única) ou pasta de saída")
    args = parser.parse_args()

    entradas = arquivos_entrada(args.entrada)
    if not entradas:
        raise FileNotFoundError(f"Nenhum arquivo .bin encontrado em {args.entrada}")

    if args.saida is None:
        saida = (args.entrada.with_suffix(".pcd") if args.entrada.is_file()
                 else args.entrada.parent / f"{args.entrada.name}_pcd")
    else:
        saida = args.saida

    saida_is_arquivo = args.entrada.is_file() and saida.suffix.lower() == ".pcd"
    if saida_is_arquivo:
        saida.parent.mkdir(parents=True, exist_ok=True)
    else:
        saida.mkdir(parents=True, exist_ok=True)

    for arquivo in entradas:
        pcd = carregar_bin(str(arquivo))
        destino = saida if saida_is_arquivo else saida / f"{arquivo.stem}.pcd"
        if not o3d.io.write_point_cloud(str(destino), pcd):
            raise RuntimeError(f"Falha ao escrever {destino}")
        print(f"Convertido: {arquivo.name} -> {destino}")


if __name__ == "__main__":
    main()

"""Processa vários scans do Warehouse LiDAR e grava predições em JSON.

As caixas previstas são mantidas no sistema de coordenadas sensor-cêntrico do
dataset. O arquivo produzido pode ser avaliado por ``avaliar_deteccoes.py``.
Não exportamos STL por padrão: isso evita gerar milhares de malhas durante a
calibração dos parâmetros.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lidar2flexsim import detectar_objetos


def listar_scans(pasta: Path, inicio: int = 0, limite: int | None = None) -> list[Path]:
    arquivos = sorted(pasta.glob("*.bin"))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo .bin encontrado em {pasta}")
    selecionados = arquivos[inicio:]
    return selecionados if limite is None else selecionados[:limite]


def main() -> None:
    parser = argparse.ArgumentParser(description="Processa scans Warehouse LiDAR")
    parser.add_argument("bin_dir", type=Path, help="Pasta bin/ do dataset")
    parser.add_argument("--saida", type=Path,
                        default=Path("saida/predicoes_warehouse.json"),
                        help="JSON de predições")
    parser.add_argument("--inicio", type=int, default=0,
                        help="Índice do primeiro scan após a ordenação")
    parser.add_argument("--limite", type=int,
                        help="Quantidade máxima de scans a processar")
    parser.add_argument("--voxel", type=float, default=0.05,
                        help="Voxel do downsampling (m)")
    parser.add_argument("--eps", type=float, default=0.25,
                        help="Raio do DBSCAN (m)")
    parser.add_argument("--min-points", type=int, default=20,
                        help="Mínimo de pontos por cluster")
    parser.add_argument("--plane-distance", type=float, default=0.05,
                        help="Distância máxima do RANSAC ao plano (m)")
    parser.add_argument("--max-ground-tilt", type=float, default=25.0,
                        help="Inclinação máxima aceita para o plano do piso (graus)")
    parser.add_argument("--ground-quantile", type=float, default=0.30,
                        help="Quantil inferior usado como candidatos ao piso")
    parser.add_argument("--no-outlier-filter", action="store_true",
                        help="Desativa a remoção estatística de outliers")
    parser.add_argument("--outlier-neighbors", type=int, default=12,
                        help="Vizinhos usados no filtro estatístico")
    parser.add_argument("--outlier-std-ratio", type=float, default=2.5,
                        help="Tolerância do filtro estatístico")
    parser.add_argument("--cluster-mode", choices=("3d", "bev"), default="3d",
                        help="Espaço de clusterização: 3d ou projeção BEV")
    parser.add_argument("--oriented-box", action="store_true",
                        help="Usa bounding boxes orientadas")
    args = parser.parse_args()

    scans = listar_scans(args.bin_dir, args.inicio, args.limite)
    predicoes = {}
    diagnosticos = {}
    for indice, scan in enumerate(scans, start=args.inicio):
        scan_id = scan.stem
        print(f"\n[{indice + 1}/{args.inicio + len(scans)}] {scan.name}")
        deteccao = detectar_objetos(
            str(scan), voxel=args.voxel, eps=args.eps,
            min_points=args.min_points, plane_distance=args.plane_distance,
            oriented=args.oriented_box,
            max_ground_tilt_deg=args.max_ground_tilt,
            ground_quantile=args.ground_quantile,
            remove_outliers=not args.no_outlier_filter,
            outlier_neighbors=args.outlier_neighbors,
            outlier_std_ratio=args.outlier_std_ratio,
            cluster_mode=args.cluster_mode,
            return_diagnostics=True)
        if len(deteccao) == 5:
            resultados, _, _, _, diagnostico = deteccao
            diagnosticos[scan_id] = diagnostico
        else:
            resultados, _, _, _ = deteccao
        for resultado in resultados:
            resultado["scan_id"] = scan_id
        predicoes[scan_id] = resultados

    documento = {
        "formato": "lidar2flexsim-predicoes-v1",
        "parametros": {
            "voxel": args.voxel,
            "eps": args.eps,
            "min_points": args.min_points,
            "plane_distance": args.plane_distance,
            "oriented_box": args.oriented_box,
            "max_ground_tilt_deg": args.max_ground_tilt,
            "ground_quantile": args.ground_quantile,
            "remove_outliers": not args.no_outlier_filter,
            "outlier_neighbors": args.outlier_neighbors,
            "outlier_std_ratio": args.outlier_std_ratio,
            "cluster_mode": args.cluster_mode,
        },
        "scans": predicoes,
        "diagnosticos": diagnosticos,
    }
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    with args.saida.open("w", encoding="utf-8") as arquivo:
        json.dump(documento, arquivo, indent=2, ensure_ascii=False)
    print(f"\nPredições gravadas em {args.saida} ({len(predicoes)} scans)")


if __name__ == "__main__":
    main()

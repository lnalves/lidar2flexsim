"""
Pipeline:
  1. Carregamento da nuvem de pontos (.bin, .ply, .pcd, .xyz)
  2. Pré-processamento (downsampling + remoção de ruído)
  3. Remoção do chão via RANSAC horizontal (com fallback robusto em z)
  4. Clusterização dos objetos via DBSCAN 3D ou projeção BEV
  5. Classificação heurística por geometria (dimensões da bounding box)
  6. Exportação:
     - um arquivo .stl por objeto detectado (FlexSim importa .stl)
     - layout.json / layout.csv com posição, dimensões e classe
     - build_flexsim.txt: FlexScript que auto-constrói o modelo no FlexSim

Uso:
  python lidar2flexsim.py entrada.bin --saida ./saida
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import open3d as o3d


EXTENSOES_NUVEM = {".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb"}
NUM_FEATURES_BIN = 4


def carregar_bin(caminho: str, num_features: int = NUM_FEATURES_BIN) -> o3d.geometry.PointCloud:
    """Carrega um scan binário no formato Warehouse LiDAR.

    Cada registro contém ``x, y, z, intensidade`` como ``float32``. A
    intensidade é preservada apenas durante a leitura; o pipeline geométrico
    utiliza as três primeiras colunas.
    """
    caminho_path = Path(caminho)
    dados = np.fromfile(caminho_path, dtype=np.float32)
    if dados.size == 0:
        raise ValueError(f"Nenhum ponto encontrado em '{caminho}'.")
    if dados.size % num_features != 0:
        raise ValueError(
            f"'{caminho}' possui {dados.size} valores; esperado múltiplo de "
            f"{num_features} (x, y, z, intensidade)."
        )

    registros = dados.reshape(-1, num_features)
    if num_features < 3:
        raise ValueError("num_features deve ser pelo menos 3.")
    pontos = registros[:, :3]
    if not np.isfinite(pontos).all():
        raise ValueError(f"'{caminho}' contém coordenadas não finitas.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pontos.astype(np.float64))
    print(f"[1] Nuvem BIN carregada: {len(pcd.points):,} pontos "
          f"(x, y, z, intensidade)")
    return pcd


def carregar_nuvem(caminho: str) -> o3d.geometry.PointCloud:
    """Carrega .bin do Warehouse LiDAR ou formatos suportados pelo Open3D."""
    extensao = Path(caminho).suffix.lower()
    if extensao == ".bin":
        return carregar_bin(caminho)
    if extensao not in EXTENSOES_NUVEM:
        print(f"Aviso: extensão '{extensao or '<sem extensão>'}' não validada; "
              "tentando Open3D.")
    pcd = o3d.io.read_point_cloud(caminho)
    if len(pcd.points) == 0:
        raise ValueError(f"Nenhum ponto encontrado em '{caminho}'. "
                         "Formatos suportados: .bin, .ply, .pcd e .xyz")
    print(f"[1] Nuvem carregada: {len(pcd.points):,} pontos")
    return pcd


def preprocessar(pcd: o3d.geometry.PointCloud,
                 voxel: float = 0.03,
                 remove_outliers: bool = True,
                 outlier_neighbors: int = 12,
                 outlier_std_ratio: float = 2.5) -> o3d.geometry.PointCloud:
    """Reduz a nuvem e remove ruído sem apagar superfícies esparsas.

    Scans de um VLP-16 têm regiões válidas com poucos vizinhos. Por isso a
    remoção estatística é configurável e usa parâmetros menos agressivos que
    os valores originais do protótipo. ``outlier_neighbors=0`` é um atalho
    explícito para desativá-la.
    """
    if voxel <= 0:
        raise ValueError("voxel deve ser maior que zero.")
    pcd = pcd.voxel_down_sample(voxel_size=voxel)
    if (remove_outliers and outlier_neighbors > 0 and
            outlier_std_ratio > 0 and len(pcd.points) > outlier_neighbors):
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=max(1, int(outlier_neighbors)),
            std_ratio=float(outlier_std_ratio),
        )
    print(f"[2] Após pré-processamento: {len(pcd.points):,} pontos "
          f"(voxel={voxel} m, outliers={'sim' if remove_outliers and outlier_neighbors > 0 else 'não'})")
    return pcd


def _estimativa_z_chao(pontos: np.ndarray, quantile: float) -> float:
    """Estima a altura do piso pelo pico mais populoso da faixa inferior."""
    if pontos.size == 0:
        return 0.0
    z = np.asarray(pontos[:, 2], dtype=float)
    limite = float(np.quantile(z, quantile))
    baixos = z[z <= limite]
    if baixos.size < 3:
        baixos = z
    if baixos.size < 3:
        return float(np.median(baixos))

    # O histograma evita que poucos retornos abaixo do piso arrastem a
    # mediana. O número de bins cresce lentamente com a quantidade de pontos.
    minimo, maximo = np.percentile(baixos, [1.0, 99.0])
    if not np.isfinite(minimo) or not np.isfinite(maximo) or maximo <= minimo:
        return float(np.median(baixos))
    bins = max(12, min(96, int(np.sqrt(baixos.size))))
    hist, bordas = np.histogram(baixos, bins=bins, range=(minimo, maximo))
    indice = int(np.argmax(hist))
    faixa = baixos[(baixos >= bordas[indice]) &
                   (baixos <= bordas[indice + 1])]
    return float(np.median(faixa if faixa.size else baixos))


def _normalizar_plano(modelo: np.ndarray) -> tuple[np.ndarray, float]:
    vetor = np.asarray(modelo[:3], dtype=float)
    norma = float(np.linalg.norm(vetor))
    if norma <= 1e-12:
        raise ValueError("RANSAC retornou um plano degenerado.")
    coeficientes = np.asarray(modelo, dtype=float) / norma
    if coeficientes[2] < 0:
        coeficientes *= -1.0
    return coeficientes[:3], float(coeficientes[3])


def remover_chao(pcd: o3d.geometry.PointCloud,
                 dist: float = 0.03,
                 max_ground_tilt_deg: float = 25.0,
                 ground_quantile: float = 0.30,
                 return_diagnostics: bool = False):
    """Remove somente um plano compatível com o piso.

    O comportamento antigo escolhia o maior plano da cena. Em um armazém,
    uma parede pode ter mais retornos que o piso e era removida por engano.
    Primeiro restringimos o RANSAC aos pontos da faixa inferior de ``z`` e
    depois rejeitamos planos cuja normal esteja inclinada além de
    ``max_ground_tilt_deg``. Se o RANSAC ainda falhar, usamos um plano
    horizontal na moda robusta de ``z`` em vez de remover uma parede.

    A tupla histórica ``(objetos, chao, z_chao)`` continua sendo retornada.
    Com ``return_diagnostics=True`` um quarto item contém métricas para a UI.
    """
    if dist <= 0:
        raise ValueError("dist deve ser maior que zero.")
    if not 0.01 <= ground_quantile <= 0.9:
        raise ValueError("ground_quantile deve estar entre 0,01 e 0,90.")
    if not 0 < max_ground_tilt_deg < 90:
        raise ValueError("max_ground_tilt_deg deve estar entre 0 e 90 graus.")

    pontos = np.asarray(pcd.points, dtype=float)
    total = len(pontos)
    if total == 0:
        diagnostico = {
            "metodo": "sem_pontos", "plano": None, "normal": None,
            "inclinacao_deg": None, "z_chao": None,
            "pontos_entrada": 0, "pontos_candidatos": 0,
            "pontos_chao": 0, "pontos_objetos": 0,
        }
        vazio = o3d.geometry.PointCloud()
        retorno = (vazio, vazio, 0.0)
        return (*retorno, diagnostico) if return_diagnostics else retorno

    limite = float(np.quantile(pontos[:, 2], ground_quantile))
    indices_candidatos = np.flatnonzero(pontos[:, 2] <= limite)
    candidato = pcd.select_by_index(indices_candidatos.tolist())
    cos_tilt = math.cos(math.radians(max_ground_tilt_deg))
    modelo: np.ndarray | None = None
    metodo = "ransac_horizontal"
    normal: np.ndarray | None = None
    d = 0.0
    inliers: np.ndarray

    if len(candidato.points) >= 3:
        try:
            # Mantém a calibração reproduzível entre execuções e scans.
            o3d.utility.random.seed(0)
            modelo_ransac, inliers_candidato = candidato.segment_plane(
                distance_threshold=dist,
                ransac_n=3,
                num_iterations=1000,
            )
            normal_ransac, d_ransac = _normalizar_plano(np.asarray(modelo_ransac))
            if abs(float(normal_ransac[2])) >= cos_tilt:
                modelo = np.asarray([*normal_ransac, d_ransac], dtype=float)
                normal, d = normal_ransac, d_ransac
        except (RuntimeError, ValueError):
            modelo = None

    if modelo is None:
        metodo = "fallback_z_horizontal"
        z_fallback = _estimativa_z_chao(pontos, ground_quantile)
        normal = np.asarray([0.0, 0.0, 1.0], dtype=float)
        d = -z_fallback
        modelo = np.asarray([0.0, 0.0, 1.0, d], dtype=float)

    distancias = np.abs(pontos @ normal + d)
    inliers = np.flatnonzero(distancias <= dist)
    if inliers.size == 0:
        # Mantém uma saída válida mesmo em uma nuvem sem piso observável.
        chao = o3d.geometry.PointCloud()
        objetos = pcd
        z_chao = float(np.median(pontos[:, 2]))
    else:
        chao = pcd.select_by_index(inliers.tolist())
        objetos = pcd.select_by_index(inliers.tolist(), invert=True)
        z_chao = float(np.median(pontos[inliers, 2]))

    inclinacao = math.degrees(math.acos(min(1.0, abs(float(normal[2])))))
    diagnostico = {
        "metodo": metodo,
        "plano": [round(float(valor), 6) for valor in modelo],
        "normal": [round(float(valor), 6) for valor in normal],
        "inclinacao_deg": round(float(inclinacao), 3),
        "z_chao": round(z_chao, 4),
        "pontos_entrada": total,
        "pontos_candidatos": int(indices_candidatos.size),
        "pontos_chao": int(inliers.size),
        "pontos_objetos": int(total - inliers.size),
        "ground_quantile": float(ground_quantile),
        "max_ground_tilt_deg": float(max_ground_tilt_deg),
    }
    a, b, c, d_modelo = modelo
    print(f"[3] Chão removido ({metodo}): "
          f"{a:.2f}x+{b:.2f}y+{c:.2f}z+{d_modelo:.2f}=0, "
          f"{len(inliers):,} pontos, z_chao={z_chao:.3f} m, "
          f"inclinação={inclinacao:.1f}°")
    retorno = (objetos, chao, z_chao)
    return (*retorno, diagnostico) if return_diagnostics else retorno


def clusterizar(pcd: o3d.geometry.PointCloud,
                eps: float = 0.15,
                min_points: int = 40,
                cluster_mode: str = "3d"):
    """Agrupa retornos por DBSCAN em 3D ou em projeção BEV.

    A projeção ``bev`` conecta pontos de diferentes alturas que pertencem ao
    mesmo objeto (por exemplo, rodas e cabine de uma empilhadeira), reduzindo
    a fragmentação típica de um VLP-16. O modo 3D permanece disponível para
    preservar o comportamento do protótipo em cenas sintéticas.
    """
    if eps <= 0 or min_points < 1:
        raise ValueError("eps e min_points devem ser positivos.")
    modo = str(cluster_mode).strip().casefold()
    if modo in {"xy", "2d"}:
        modo = "bev"
    if modo not in {"3d", "bev"}:
        raise ValueError("cluster_mode deve ser '3d' ou 'bev'.")

    pontos = np.asarray(pcd.points, dtype=float)
    if modo == "bev" and len(pontos):
        projetada = np.column_stack((pontos[:, 0], pontos[:, 1],
                                     np.zeros(len(pontos), dtype=float)))
        nuvem_cluster = o3d.geometry.PointCloud()
        nuvem_cluster.points = o3d.utility.Vector3dVector(projetada)
    else:
        nuvem_cluster = pcd
    labels = np.asarray(nuvem_cluster.cluster_dbscan(
        eps=float(eps), min_points=int(min_points), print_progress=False
    ))
    n = int(labels.max() + 1) if labels.size and labels.max() >= 0 else 0
    print(f"[4] DBSCAN {modo.upper()}: {n} clusters encontrados "
          f"(eps={eps}, min_points={min_points})")
    clusters = []
    for i in range(n):
        idx = np.flatnonzero(labels == i)
        clusters.append(pcd.select_by_index(idx.tolist()))
    return clusters

def classificar(cluster: o3d.geometry.PointCloud, z_chao: float,
                oriented: bool = False) -> dict:

    # A OBB reduz o erro de dimensões para objetos diagonais. Nuvens muito
    # pequenas/coplanares podem fazer o Qhull falhar; nesse caso usamos AABB
    # para não descartar o scan inteiro por causa de um fragmento.
    obb_oriented = bool(oriented)
    if oriented:
        try:
            obb = cluster.get_oriented_bounding_box(robust=True)
        except RuntimeError:
            obb = cluster.get_axis_aligned_bounding_box()
            obb_oriented = False
    else:
        obb = cluster.get_axis_aligned_bounding_box()
    # AABB expõe get_extent(); OBB expõe extent como propriedade.
    ext = (np.asarray(obb.extent) if obb_oriented else obb.get_extent())
    centro = obb.get_center()
    dx, dy, dz = float(ext[0]), float(ext[1]), float(ext[2])
    comprimento = max(dx, dy)
    largura = min(dx, dy)
    altura_topo = float(obb.get_max_bound()[2] - z_chao)
    razao = comprimento / max(largura, 1e-6)
    if obb_oriented:
        # A rotação da OBB é uma matriz em coordenadas do sensor.
        rotacao_z = math.degrees(math.atan2(obb.R[1, 0], obb.R[0, 0]))
    else:
        rotacao_z = 0.0 if dx >= dy else 90.0

    if 1.3 <= altura_topo <= 2.1 and comprimento < 1.0 and razao < 2.0:
        classe = "operador"
    elif comprimento >= 2.0 and razao >= 2.5 and 0.4 <= altura_topo <= 1.5:
        classe = "esteira"
    elif 0.6 <= altura_topo <= 1.3 and 0.8 <= comprimento <= 2.5 \
            and razao < 2.5:
        classe = "bancada"
    else:
        classe = "desconhecido"

    return {
        "classe": classe,
        "centro": [round(float(c), 3) for c in centro],
        "dimensoes": [round(dx, 3), round(dy, 3), round(dz, 3)],
        "altura_topo": round(altura_topo, 3),
        "rotacao_z": round(float(rotacao_z), 3),
        "n_pontos": len(cluster.points),
        "box_oriented": obb_oriented,
    }

def exportar_stl(cluster: o3d.geometry.PointCloud,
                 caminho: str) -> bool:
    """Grava uma malha STL, com fallback para a caixa do cluster.

    Retornos VLP-16 podem ser coplanares (por exemplo, uma face de uma
    empilhadeira). Nessa situação tanto alpha-shape quanto o fecho convexo
    podem ser degenerados; uma caixa mínima ainda é uma representação útil
    para importar o objeto no FlexSim.
    """
    try:
        malha = o3d.geometry.TriangleMesh \
            .create_from_point_cloud_alpha_shape(cluster, alpha=0.3)
        if len(malha.triangles) == 0:
            raise RuntimeError("alpha shape vazia")
    except Exception:
        try:
            malha, _ = cluster.compute_convex_hull()
        except Exception:
            bounds = cluster.get_axis_aligned_bounding_box()
            minimo = bounds.get_min_bound()
            maximo = bounds.get_max_bound()
            tamanho = np.maximum(np.asarray(maximo) - np.asarray(minimo), 0.05)
            malha = o3d.geometry.TriangleMesh.create_box(
                width=float(tamanho[0]),
                height=float(tamanho[1]),
                depth=float(tamanho[2]),
            )
            malha.translate(minimo)
    malha.compute_vertex_normals()
    return o3d.io.write_triangle_mesh(caminho, malha)



MAPA_FLEXSIM = {
    "esteira": "Conveyor",
    "bancada": "Processor",
    "operador": "Operator",
    "desconhecido": "VisualTool",
}


def gerar_flexscript(objetos: list, caminho: str, mapa_flexsim: dict | None = None):

    mapa_flexsim = mapa_flexsim or MAPA_FLEXSIM

    linhas = [
        "// Gerado automaticamente por lidar2flexsim.py",
        "// Cole no Script Console do FlexSim e execute.",
        "",
    ]
    for i, o in enumerate(objetos):
        tipo = mapa_flexsim.get(o["classe"], mapa_flexsim.get("desconhecido", "VisualTool"))
        x, y, _ = o["centro"]
        dx, dy, dz = o["dimensoes"]
        nome = f"{o['classe']}_{i+1}"
        linhas += [
            f'Object obj{i} = Object.create("{tipo}");',
            f'obj{i}.name = "{nome}";',
            f'obj{i}.location = Vec3({x}, {y}, 0);',
            f'obj{i}.size = Vec3({dx}, {dy}, {dz});',
            f'obj{i}.rotation = Vec3(0, 0, {o["rotacao_z"]});',
            "",
        ]
    with open(caminho, "w") as f:
        f.write("\n".join(linhas))


def exportar_layout(objetos: list, pasta: str, mapa_flexsim: dict | None = None):
    mapa_flexsim = mapa_flexsim or MAPA_FLEXSIM
    with open(os.path.join(pasta, "layout.json"), "w") as f:
        json.dump(objetos, f, indent=2, ensure_ascii=False)
    with open(os.path.join(pasta, "layout.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "classe", "objeto_flexsim", "x", "y", "z",
                    "dim_x", "dim_y", "dim_z", "rotacao_z", "n_pontos"])
        for i, o in enumerate(objetos):
            w.writerow([i + 1, o["classe"],
                        mapa_flexsim.get(o["classe"], mapa_flexsim.get("desconhecido", "VisualTool")),
                        *o["centro"], *o["dimensoes"],
                        o["rotacao_z"], o["n_pontos"]])


def carregar_config(caminho: str | None) -> dict:
    """Carrega configuração opcional de mapeamento para objetos FlexSim."""
    if caminho is None:
        return {"flexsim_objects": dict(MAPA_FLEXSIM)}
    with open(caminho, encoding="utf-8") as f:
        config = json.load(f)
    mapa = config.get("flexsim_objects")
    if not isinstance(mapa, dict):
        raise ValueError("A configuração deve conter 'flexsim_objects' como objeto JSON.")
    resultado = dict(MAPA_FLEXSIM)
    resultado.update({str(k): str(v) for k, v in mapa.items()})
    return {**config, "flexsim_objects": resultado}


def detectar_objetos(entrada: str, voxel: float = 0.03,
                     eps: float = 0.15, min_points: int = 40,
                     plane_distance: float = 0.03,
                     oriented: bool = False,
                     max_ground_tilt_deg: float = 25.0,
                     ground_quantile: float = 0.30,
                     remove_outliers: bool = True,
                     outlier_neighbors: int = 12,
                     outlier_std_ratio: float = 2.5,
                     cluster_mode: str = "3d",
                     return_diagnostics: bool = False):
    """Executa percepção e retorna resultados, nuvens e clusters.

    A forma histórica retorna quatro itens. Com ``return_diagnostics=True``
    retorna um quinto item com contagens e a qualidade do plano estimado.
    """
    pcd = carregar_nuvem(entrada)
    pontos_brutos = len(pcd.points)
    pcd = preprocessar(
        pcd,
        voxel=voxel,
        remove_outliers=remove_outliers,
        outlier_neighbors=outlier_neighbors,
        outlier_std_ratio=outlier_std_ratio,
    )
    pontos_preprocessados = len(pcd.points)
    objetos_pcd, chao, z_chao, diagnostico = remover_chao(
        pcd,
        dist=plane_distance,
        max_ground_tilt_deg=max_ground_tilt_deg,
        ground_quantile=ground_quantile,
        return_diagnostics=True,
    )
    clusters = clusterizar(
        objetos_pcd,
        eps=eps,
        min_points=min_points,
        cluster_mode=cluster_mode,
    )
    resultados = []
    for c in clusters:
        resultados.append(classificar(c, z_chao, oriented=oriented))
    diagnostico.update({
        "pontos_brutos": pontos_brutos,
        "pontos_preprocessados": pontos_preprocessados,
        "clusters": len(clusters),
        "cluster_mode": str(cluster_mode),
        "eps": float(eps),
        "min_points": int(min_points),
        "outlier_neighbors": int(outlier_neighbors),
        "outlier_std_ratio": float(outlier_std_ratio),
        "remove_outliers": bool(remove_outliers),
    })
    resultado = (resultados, objetos_pcd, chao, clusters)
    return (*resultado, diagnostico) if return_diagnostics else resultado

def main():
    ap = argparse.ArgumentParser(description="LiDAR → FlexSim")
    ap.add_argument("entrada", help="Nuvem de pontos (.bin, .ply, .pcd, .xyz)")
    ap.add_argument("--saida", default="./saida", help="Pasta de saída")
    ap.add_argument("--voxel", type=float, default=0.03,
                    help="Tamanho do voxel para downsampling (m)")
    ap.add_argument("--eps", type=float, default=0.15,
                    help="Raio de vizinhança do DBSCAN (m)")
    ap.add_argument("--min-points", type=int, default=40,
                    help="Mínimo de pontos por cluster")
    ap.add_argument("--plane-distance", type=float, default=0.03,
                    help="Distância máxima do RANSAC ao plano do chão (m)")
    ap.add_argument("--max-ground-tilt", type=float, default=25.0,
                    help="Inclinação máxima aceita para o plano do chão (graus)")
    ap.add_argument("--ground-quantile", type=float, default=0.30,
                    help="Quantil inferior usado como candidatos ao piso")
    ap.add_argument("--no-outlier-filter", action="store_true",
                    help="Desativa a remoção estatística de outliers")
    ap.add_argument("--outlier-neighbors", type=int, default=12,
                    help="Vizinhos usados no filtro estatístico")
    ap.add_argument("--outlier-std-ratio", type=float, default=2.5,
                    help="Tolerância do filtro estatístico")
    ap.add_argument("--cluster-mode", choices=("3d", "bev"), default="3d",
                    help="Espaço de clusterização: 3d ou projeção BEV")
    ap.add_argument("--oriented-box", action="store_true",
                    help="Usa bounding boxes orientadas para dimensões e yaw")
    ap.add_argument("--backend", choices=("heuristic", "pointnet2"),
                    default="heuristic",
                    help="Backend de detecção: heurístico ou PointNet++")
    ap.add_argument("--checkpoint", "--model-checkpoint",
                    dest="model_checkpoint",
                    help="Checkpoint .pt/.pth do backend PointNet++")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"),
                    default="auto",
                    help="Dispositivo para inferência PointNet++")
    ap.add_argument("--score-threshold", type=float, default=0.50,
                    help="Confiança mínima por ponto no PointNet++")
    ap.add_argument("--num-points", type=int, default=4096,
                    help="Quantidade fixa de pontos por amostra PointNet++")
    ap.add_argument("--config", help="JSON opcional com mapeamento FlexSim")
    args = ap.parse_args()

    if args.backend == "pointnet2":
        # Keep the historical CLI untouched for the heuristic path, while
        # routing the neural path through the same service used by the UI.
        from core.models import PipelineParameters
        from core.pipeline_service import processar_scan

        mapa_flexsim = None
        if args.config:
            mapa_flexsim = carregar_config(args.config).get("flexsim_objects")
        parametros = PipelineParameters(
            voxel=args.voxel,
            eps=args.eps,
            min_points=args.min_points,
            plane_distance=args.plane_distance,
            oriented_box=args.oriented_box,
            max_ground_tilt_deg=args.max_ground_tilt,
            ground_quantile=args.ground_quantile,
            remove_outliers=not args.no_outlier_filter,
            outlier_neighbors=args.outlier_neighbors,
            outlier_std_ratio=args.outlier_std_ratio,
            cluster_mode=args.cluster_mode,
            detector_backend="pointnet2",
            model_checkpoint=args.model_checkpoint,
            device=args.device,
            score_threshold=args.score_threshold,
            num_points=args.num_points,
        )
        resultado = processar_scan(
            args.entrada,
            parametros=parametros,
            saida=args.saida,
            exportar=True,
            mapa_flexsim=mapa_flexsim,
        )
        print(f"\nConcluído: {resultado['objetos_detectados']} objeto(s) detectado(s)")
        print(f"Saída em: {os.path.abspath(args.saida)}")
        return

    os.makedirs(args.saida, exist_ok=True)
    pasta_stl = os.path.join(args.saida, "modelos_3d")
    os.makedirs(pasta_stl, exist_ok=True)

    config = carregar_config(args.config)
    mapa_flexsim = config["flexsim_objects"]
    resultados, objetos_pcd, chao, clusters = detectar_objetos(
        args.entrada, voxel=args.voxel, eps=args.eps,
        min_points=args.min_points, plane_distance=args.plane_distance,
        oriented=args.oriented_box,
        max_ground_tilt_deg=args.max_ground_tilt,
        ground_quantile=args.ground_quantile,
        remove_outliers=not args.no_outlier_filter,
        outlier_neighbors=args.outlier_neighbors,
        outlier_std_ratio=args.outlier_std_ratio,
        cluster_mode=args.cluster_mode)

    print("\n[5] Classificação e exportação:")
    resultados_export = []
    for i, c in enumerate(clusters):
        info = resultados[i]
        stl = os.path.join(pasta_stl, f"{info['classe']}_{i+1}.stl")
        exportar_stl(c, stl)
        info["arquivo_stl"] = os.path.basename(stl)
        resultados_export.append(info)
        print(f"    #{i+1}: {info['classe']:<12} "
              f"dim={info['dimensoes']} m  centro={info['centro']}  "
              f"→ {info['arquivo_stl']}")

    exportar_layout(resultados_export, args.saida, mapa_flexsim)
    gerar_flexscript(resultados_export, os.path.join(args.saida,
                                                      "build_flexsim.txt"),
                     mapa_flexsim)

    cena = chao + objetos_pcd
    try:
        malha_cena, _ = cena.compute_convex_hull()
    except Exception:
        malha_cena = None

    resumo = {c: sum(1 for r in resultados_export if r["classe"] == c)
              for c in set(r["classe"] for r in resultados_export)}
    print(f"\n[6] Concluído. Resumo: {resumo}")
    print(f"    Saída em: {os.path.abspath(args.saida)}")
    print("    → Importe os .stl no FlexSim (Propriedades > Visuals > Shape)")
    print("    → Ou cole build_flexsim.txt no Script Console do FlexSim")


if __name__ == "__main__":
    main()

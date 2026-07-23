"""Avalia caixas 3D previstas contra os labels do Warehouse LiDAR.

Uso com o processamento em lote:

    python avaliar_deteccoes.py \
        --predicoes saida/predicoes_warehouse.json \
        --labels dados/warehouse/label \
        --saida saida/metricas_warehouse.json

Por padrão, a avaliação é geométrica (independente da classe), pois as
heurísticas atuais do projeto produzem ``operador/esteira/bancada`` e o
dataset usa outra ontologia. Use ``--class-aware`` após configurar um mapa de
classes compatível.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable


def ler_labels(caminho: Path) -> list[dict]:
    """Lê um arquivo do dataset: classe x y z dx dy dz yaw(rad)."""
    caixas = []
    with caminho.open(encoding="utf-8") as arquivo:
        for numero_linha, linha in enumerate(arquivo, start=1):
            linha = linha.strip()
            if not linha or linha.startswith("#"):
                continue
            campos = linha.split()
            if len(campos) != 8:
                raise ValueError(
                    f"{caminho}:{numero_linha}: esperado 8 campos, encontrado {len(campos)}"
                )
            classe = campos[0]
            valores = [float(valor) for valor in campos[1:]]
            caixas.append({
                "classe": classe,
                "centro": valores[:3],
                "dimensoes": valores[3:6],
                "yaw_rad": valores[6],
            })
    return caixas


def predicao_para_caixa(predicao: dict) -> dict:
    dimensoes = predicao.get("dimensoes")
    centro = predicao.get("centro")
    if not isinstance(dimensoes, list) or len(dimensoes) != 3:
        raise ValueError("Predição sem 'dimensoes' [dx, dy, dz].")
    if not isinstance(centro, list) or len(centro) != 3:
        raise ValueError("Predição sem 'centro' [x, y, z].")
    # O exportador do pipeline usa graus para compatibilidade com o FlexSim.
    yaw_graus = float(predicao.get("rotacao_z", 0.0))
    return {
        "classe": str(predicao.get("classe", "desconhecido")),
        "centro": [float(v) for v in centro],
        "dimensoes": [float(v) for v in dimensoes],
        "yaw_rad": math.radians(yaw_graus),
    }


def carregar_predicoes(caminho: Path, scan_id: str | None) -> dict[str, list[dict]]:
    with caminho.open(encoding="utf-8") as arquivo:
        documento = json.load(arquivo)

    if isinstance(documento, list):
        if scan_id is None:
            raise ValueError("Use --scan-id ao avaliar um layout.json sem campo 'scans'.")
        return {scan_id: [predicao_para_caixa(item) for item in documento]}

    if not isinstance(documento, dict):
        raise ValueError("JSON de predições deve ser uma lista ou um objeto com 'scans'.")
    scans = documento.get("scans", documento)
    if not isinstance(scans, dict):
        raise ValueError("Campo 'scans' deve ser um objeto scan_id -> predições.")
    if scan_id is not None:
        if scan_id not in scans:
            raise KeyError(f"Scan {scan_id} não encontrado nas predições.")
        scans = {scan_id: scans[scan_id]}
    return {
        str(chave): [predicao_para_caixa(item) for item in valores]
        for chave, valores in scans.items()
    }


def vertices_retangulo(caixa: dict) -> list[tuple[float, float]]:
    cx, cy = caixa["centro"][:2]
    dx, dy = caixa["dimensoes"][:2]
    yaw = caixa["yaw_rad"]
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    vertices = []
    for local_x, local_y in ((-dx / 2, -dy / 2), (dx / 2, -dy / 2),
                             (dx / 2, dy / 2), (-dx / 2, dy / 2)):
        vertices.append((
            cx + cos_yaw * local_x - sin_yaw * local_y,
            cy + sin_yaw * local_x + cos_yaw * local_y,
        ))
    return vertices


def produto_vetorial(a: tuple[float, float], b: tuple[float, float],
                     p: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def intersecao_segmentos(s: tuple[float, float], e: tuple[float, float],
                         a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    ds = (e[0] - s[0], e[1] - s[1])
    dc = (b[0] - a[0], b[1] - a[1])
    denominador = ds[0] * dc[1] - ds[1] * dc[0]
    if abs(denominador) < 1e-12:
        return e
    numerador = (a[0] - s[0]) * dc[1] - (a[1] - s[1]) * dc[0]
    t = numerador / denominador
    return (s[0] + t * ds[0], s[1] + t * ds[1])


def recortar_poligono(subject: list[tuple[float, float]],
                      clip: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Sutherland-Hodgman para dois polígonos convexos anti-horários."""
    resultado = subject
    for indice in range(len(clip)):
        if not resultado:
            break
        a = clip[indice - 1]
        b = clip[indice]
        entrada = resultado
        resultado = []
        anterior = entrada[-1]
        anterior_dentro = produto_vetorial(a, b, anterior) >= -1e-9
        for atual in entrada:
            atual_dentro = produto_vetorial(a, b, atual) >= -1e-9
            if atual_dentro:
                if not anterior_dentro:
                    resultado.append(intersecao_segmentos(anterior, atual, a, b))
                resultado.append(atual)
            elif anterior_dentro:
                resultado.append(intersecao_segmentos(anterior, atual, a, b))
            anterior, anterior_dentro = atual, atual_dentro
    return resultado


def area_poligono(vertices: Iterable[tuple[float, float]]) -> float:
    pontos = list(vertices)
    if len(pontos) < 3:
        return 0.0
    return abs(sum(
        pontos[i][0] * pontos[(i + 1) % len(pontos)][1]
        - pontos[(i + 1) % len(pontos)][0] * pontos[i][1]
        for i in range(len(pontos))
    )) / 2.0


def iou_3d(a: dict, b: dict) -> float:
    inter_xy = area_poligono(recortar_poligono(
        vertices_retangulo(a), vertices_retangulo(b)))
    if inter_xy <= 0:
        return 0.0
    az = a["centro"][2]
    bz = b["centro"][2]
    ah = a["dimensoes"][2]
    bh = b["dimensoes"][2]
    inter_z = max(0.0, min(az + ah / 2, bz + bh / 2)
                  - max(az - ah / 2, bz - bh / 2))
    inter = inter_xy * inter_z
    volume_a = abs(a["dimensoes"][0] * a["dimensoes"][1] * ah)
    volume_b = abs(b["dimensoes"][0] * b["dimensoes"][1] * bh)
    uniao = volume_a + volume_b - inter
    return inter / uniao if uniao > 0 else 0.0


def erro_geometrico(predicao: dict, verdade: dict) -> dict:
    centro_erro = [abs(a - b) for a, b in zip(predicao["centro"], verdade["centro"])]
    dimensao_erro = [abs(a - b) for a, b in zip(predicao["dimensoes"], verdade["dimensoes"])]
    diferenca_yaw = abs((predicao["yaw_rad"] - verdade["yaw_rad"] + math.pi)
                        % (2 * math.pi) - math.pi)
    return {
        "erro_centro_m": centro_erro,
        "erro_dimensao_m": dimensao_erro,
        "erro_yaw_rad": diferenca_yaw,
    }


def classe_compativel(predicao: dict, verdade: dict,
                      mapa_classes: dict | None) -> bool:
    if mapa_classes is None:
        return True
    classe_prevista = mapa_classes.get(predicao["classe"], predicao["classe"])
    return classe_prevista == verdade["classe"]


def casar(predicoes: list[dict], verdades: list[dict], threshold: float,
          mapa_classes: dict | None) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    candidatos = []
    for i, predicao in enumerate(predicoes):
        for j, verdade in enumerate(verdades):
            if not classe_compativel(predicao, verdade, mapa_classes):
                continue
            valor = iou_3d(predicao, verdade)
            if valor >= threshold:
                candidatos.append((valor, i, j))
    candidatos.sort(reverse=True)
    usados_pred, usados_verdade = set(), set()
    pares = []
    for valor, i, j in candidatos:
        if i in usados_pred or j in usados_verdade:
            continue
        usados_pred.add(i)
        usados_verdade.add(j)
        pares.append((i, j, valor))
    return pares, set(range(len(predicoes))) - usados_pred, set(range(len(verdades))) - usados_verdade


def resumir_scans(predicoes: dict[str, list[dict]], labels_dir: Path,
                  thresholds: list[float], mapa_classes: dict | None) -> dict:
    por_threshold = {}
    por_scan = {}
    for threshold in thresholds:
        tp = fp = fn = 0
        ious, erros_centro, erros_dimensao, erros_yaw = [], [], [], []
        scans_avaliados = 0
        for scan_id, predicoes_scan in sorted(predicoes.items()):
            label_path = labels_dir / f"{scan_id}.txt"
            if not label_path.exists():
                continue
            verdades = ler_labels(label_path)
            pares, falsos_positivos, falsos_negativos = casar(
                predicoes_scan, verdades, threshold, mapa_classes)
            scans_avaliados += 1
            tp += len(pares)
            fp += len(falsos_positivos)
            fn += len(falsos_negativos)
            ious.extend(valor for _, _, valor in pares)
            for i, j, _ in pares:
                erros = erro_geometrico(predicoes_scan[i], verdades[j])
                erros_centro.extend(erros["erro_centro_m"])
                erros_dimensao.extend(erros["erro_dimensao_m"])
                erros_yaw.append(erros["erro_yaw_rad"])
            por_scan.setdefault(scan_id, {})[str(threshold)] = {
                "tp": len(pares), "fp": len(falsos_positivos),
                "fn": len(falsos_negativos), "ious": [round(v, 6) for _, _, v in pares],
            }
        precisao = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precisao * recall / (precisao + recall) if precisao + recall else 0.0
        por_threshold[str(threshold)] = {
            "scans_avaliados": scans_avaliados,
            "tp": tp, "fp": fp, "fn": fn,
            "precisao": precisao, "recall": recall, "f1": f1,
            "iou_medio_deteccoes_corretas": sum(ious) / len(ious) if ious else 0.0,
            "erro_centro_medio_m": sum(erros_centro) / len(erros_centro) if erros_centro else 0.0,
            "erro_dimensao_medio_m": sum(erros_dimensao) / len(erros_dimensao) if erros_dimensao else 0.0,
            "erro_yaw_medio_rad": sum(erros_yaw) / len(erros_yaw) if erros_yaw else 0.0,
        }
    return {"por_threshold": por_threshold, "por_scan": por_scan}


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia detecções 3D do pipeline")
    parser.add_argument("--predicoes", type=Path, required=True,
                        help="layout.json ou JSON produzido por processar_warehouse.py")
    parser.add_argument("--labels", type=Path, required=True,
                        help="Pasta label/ ou arquivo .txt de um scan")
    parser.add_argument("--scan-id",
                        help="ID quando --predicoes é um layout.json de um único scan")
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.25, 0.5],
                        help="Limiares de IoU 3D a calcular")
    parser.add_argument("--class-aware", action="store_true",
                        help="Exige compatibilidade de classe durante o casamento")
    parser.add_argument("--class-map", type=Path,
                        help="JSON opcional: classe prevista -> classe do dataset")
    parser.add_argument("--saida", type=Path,
                        default=Path("saida/metricas_warehouse.json"),
                        help="Arquivo JSON de métricas")
    args = parser.parse_args()

    labels_dir = args.labels
    if labels_dir.is_file():
        if args.scan_id is None:
            args.scan_id = labels_dir.stem
        labels_dir = labels_dir.parent

    mapa_classes = None
    if args.class_aware:
        mapa_classes = {}
        if args.class_map:
            with args.class_map.open(encoding="utf-8") as arquivo:
                mapa_classes = json.load(arquivo)

    predicoes = carregar_predicoes(args.predicoes, args.scan_id)
    if not labels_dir.exists() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Pasta de labels não encontrada: {labels_dir}")
    resultado = {
        "formato": "lidar2flexsim-metricas-v1",
        "class_aware": args.class_aware,
        "class_map": mapa_classes,
        **resumir_scans(predicoes, labels_dir, args.iou_thresholds, mapa_classes),
    }
    if not any(item["scans_avaliados"] > 0
               for item in resultado["por_threshold"].values()):
        raise FileNotFoundError(
            "Nenhum scan das predições possui label correspondente. "
            "Verifique --labels e os nomes dos scan_id."
        )
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    with args.saida.open("w", encoding="utf-8") as arquivo:
        json.dump(resultado, arquivo, indent=2, ensure_ascii=False)
    print(json.dumps(resultado["por_threshold"], indent=2, ensure_ascii=False))
    print(f"Métricas gravadas em {args.saida}")


if __name__ == "__main__":
    main()

"""API de aplicação para processamento e avaliação do pipeline LiDAR.

As funções deste módulo são síncronas por desenho: a interface pode executá-
las em uma thread de trabalho, enquanto scripts científicos continuam podendo
chamá-las diretamente. A CLI histórica permanece disponível, com um caminho
PointNet++ que reutiliza estes serviços.

O Open3D é importado somente no momento do processamento. Isso torna possível
abrir a interface, validar datasets e editar parâmetros antes de instalar o
runtime nativo.
"""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .dataset_service import listar_scans, localizar_diretorios
from .models import (
    ParameterError,
    PipelineParameters,
    PipelineServiceError,
    Progress,
    ProgressCallback,
)


def _carregar_pipeline() -> Any:
    """Importa o módulo legado e produz erro orientado à interface."""
    try:
        import lidar2flexsim  # type: ignore
    except ModuleNotFoundError as exc:
        if exc.name in {"open3d", "numpy"}:
            raise PipelineServiceError(
                "Dependência de processamento ausente. Instale Open3D e NumPy "
                "no ambiente usado para executar o aplicativo."
            ) from exc
        raise
    except ImportError as exc:
        raise PipelineServiceError(
            f"Não foi possível carregar o pipeline LiDAR: {exc}"
        ) from exc
    return lidar2flexsim


def _carregar_avaliador() -> Any:
    """Importa o avaliador legado sem executar sua CLI."""
    try:
        import avaliar_deteccoes  # type: ignore
    except ImportError as exc:
        raise PipelineServiceError(
            f"Não foi possível carregar o avaliador de detecções: {exc}"
        ) from exc
    return avaliar_deteccoes


def _normalizar_parametros(
    parametros: PipelineParameters | Mapping[str, Any] | None = None,
    **overrides: Any,
) -> PipelineParameters:
    """Normaliza parâmetros e filtra aliases usados por controles da UI."""
    if parametros is None and "params" in overrides:
        parametros = overrides.get("params")
    if parametros is None and "parameters" in overrides:
        parametros = overrides.get("parameters")
    uteis: dict[str, Any] = {}
    for nome in (
        "voxel",
        "eps",
        "min_points",
        "plane_distance",
        "oriented_box",
        "max_ground_tilt_deg",
        "ground_quantile",
        "remove_outliers",
        "outlier_neighbors",
        "outlier_std_ratio",
        "cluster_mode",
        "detector_backend",
        "model_checkpoint",
        "device",
        "score_threshold",
        "num_points",
        "oriented",
        "plane_dist",
        "min_cluster_points",
    ):
        if nome in overrides and overrides[nome] is not None:
            uteis[nome] = overrides[nome]
    return PipelineParameters.from_value(parametros, **uteis)


def _cancelamento_solicitado(cancelar: Any) -> bool:
    """Aceita ``threading.Event``, callables e tokens de bibliotecas de UI."""
    if cancelar is None:
        return False
    if isinstance(cancelar, bool):
        return cancelar
    for nome in ("is_set", "is_cancelled", "cancelled"):
        atributo = getattr(cancelar, nome, None)
        if atributo is None:
            continue
        try:
            return bool(atributo() if callable(atributo) else atributo)
        except TypeError:
            continue
    if callable(cancelar):
        try:
            return bool(cancelar())
        except TypeError:
            return False
    return False


def _emitir(callback: ProgressCallback | Callable[[Any], None] | None,
            evento: Progress) -> None:
    """Entrega evento ao callback sem deixar erro de UI quebrar o pipeline.

    O callback é uma extensão de observabilidade; falhas nele são ignoradas
    deliberadamente para preservar o resultado científico. ``Progress`` expõe
    ``to_dict`` e acesso por atributos, permitindo que a UI escolha o formato.
    """
    if callback is None:
        return
    try:
        callback(evento)
    except Exception:
        # Um callback de interface não deve abortar uma execução longa. O
        # usuário ainda receberá o resultado ou a exceção real do pipeline.
        return


def _evento(
    callback: ProgressCallback | Callable[[Any], None] | None,
    *,
    current: int,
    total: int,
    message: str,
    stage: str,
    scan_id: str | None = None,
    percent: float | None = None,
    result: dict[str, Any] | None = None,
    error: str | None = None,
    cancelled: bool = False,
    **extra: Any,
) -> None:
    if percent is None:
        percent = (current / total * 100.0) if total else 0.0
    _emitir(
        callback,
        Progress(
            current=current,
            total=total,
            message=message,
            stage=stage,
            scan_id=scan_id,
            percent=percent,
            result=result,
            error=error,
            cancelled=cancelled,
            extra=extra,
        ),
    )


def _serializar_resultados(resultados: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Copia previsões e garante tipos nativos para JSON/UI."""
    serializados: list[dict[str, Any]] = []
    for item in resultados:
        copia: dict[str, Any] = {}
        for chave, valor in item.items():
            if isinstance(valor, (str, int, float, bool)) or valor is None:
                copia[chave] = valor
            elif isinstance(valor, (list, tuple)):
                copia[chave] = [
                    float(v) if isinstance(v, (int, float)) else v for v in valor
                ]
            elif isinstance(valor, Mapping):
                copia[chave] = dict(valor)
            else:
                copia[chave] = str(valor)
        serializados.append(copia)
    return serializados


def _executar_pointnet2(
    caminho: Path,
    params: PipelineParameters,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[Any]]:
    """Executa o backend neural sem tornar PyTorch obrigatório para a UI.

    O pacote ``ml`` é opcional: o aplicativo continua abrindo e o backend
    heurístico continua disponível em ambientes sem torch. Erros de instalação
    ou de checkpoint são convertidos em mensagens orientadas ao usuário.
    """
    if not params.model_checkpoint:
        raise PipelineServiceError(
            "O backend PointNet++ exige um arquivo model_checkpoint (.pt/.pth)."
        )
    try:
        from ml.inference import inferir_scan  # type: ignore
    except ModuleNotFoundError as exc:
        if exc.name in {"torch", "torch_geometric", "ml"}:
            raise PipelineServiceError(
                "Dependências do PointNet++ ausentes. Instale requirements-ml.txt "
                "em um ambiente separado do aplicativo principal."
            ) from exc
        raise
    except ImportError as exc:
        raise PipelineServiceError(f"Não foi possível carregar o PointNet++: {exc}") from exc

    try:
        retorno = inferir_scan(
            str(caminho),
            checkpoint=params.model_checkpoint,
            device=params.device,
            num_points=params.num_points,
            score_threshold=params.score_threshold,
            cluster_eps=params.eps,
            min_cluster_points=params.min_points,
        )
    except PipelineServiceError:
        raise
    except Exception as exc:
        raise PipelineServiceError(f"Falha na inferência PointNet++: {exc}") from exc

    if isinstance(retorno, Mapping):
        resultados = retorno.get("predictions", retorno.get("predicoes", []))
        diagnostico = retorno.get("diagnostics", retorno.get("diagnostico", {}))
        clusters = retorno.get("clusters", [])
    elif isinstance(retorno, tuple) and len(retorno) >= 2:
        resultados, diagnostico = retorno[:2]
        clusters = retorno[2] if len(retorno) > 2 else []
    else:
        raise PipelineServiceError(
            "O backend PointNet++ retornou um formato desconhecido."
        )
    if not isinstance(resultados, Sequence) or isinstance(resultados, (str, bytes)):
        raise PipelineServiceError("O PointNet++ não retornou uma lista de previsões.")
    diagnostico = dict(diagnostico) if isinstance(diagnostico, Mapping) else {}
    diagnostico.setdefault("backend", "pointnet2")
    return [dict(item) for item in resultados if isinstance(item, Mapping)], diagnostico, list(clusters or [])


def _exportar_scan(
    pipeline: Any,
    clusters: Sequence[Any],
    resultados: Sequence[Mapping[str, Any]],
    pasta_saida: Path,
    mapa_flexsim: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Exporta STL/layout/FlexScript de um único scan."""
    pasta_saida.mkdir(parents=True, exist_ok=True)
    pasta_stl = pasta_saida / "modelos_3d"
    pasta_stl.mkdir(parents=True, exist_ok=True)
    exportados = _serializar_resultados(resultados)
    for indice, (cluster, info) in enumerate(zip(clusters, exportados), start=1):
        classe = str(info.get("classe", "desconhecido"))
        stl = pasta_stl / f"{classe}_{indice}.stl"
        # The heuristic backend already returns Open3D PointCloud objects.
        # PointNet++ returns NumPy arrays to keep the ML package independent
        # of Open3D, so adapt those arrays only at the export boundary.
        objeto_exportavel = cluster
        if not hasattr(cluster, "compute_convex_hull"):
            try:
                import numpy as np  # type: ignore
                import open3d as o3d  # type: ignore

                pontos = np.asarray(cluster, dtype=np.float64)
                if pontos.ndim != 2 or pontos.shape[1] < 3 or len(pontos) < 3:
                    raise ValueError("cluster PointNet++ sem pontos suficientes para STL")
                objeto_exportavel = o3d.geometry.PointCloud()
                objeto_exportavel.points = o3d.utility.Vector3dVector(pontos[:, :3])
            except Exception as exc:
                raise PipelineServiceError(
                    f"Falha ao preparar STL do objeto {indice}: {exc}"
                ) from exc
        try:
            escrito = bool(pipeline.exportar_stl(objeto_exportavel, str(stl)))
        except Exception as exc:
            raise PipelineServiceError(
                f"Falha ao exportar STL do objeto {indice}: {exc}"
            ) from exc
        if not escrito:
            raise PipelineServiceError(f"Open3D não gravou o STL: {stl}")
        info["arquivo_stl"] = stl.name

    mapa = dict(mapa_flexsim or getattr(pipeline, "MAPA_FLEXSIM", {}))
    try:
        pipeline.exportar_layout(exportados, str(pasta_saida), mapa)
        pipeline.gerar_flexscript(exportados, str(pasta_saida / "build_flexsim.txt"), mapa)
    except Exception as exc:
        raise PipelineServiceError(
            f"Falha ao gerar arquivos para o FlexSim: {exc}"
        ) from exc
    return exportados


def processar_scan(
    scan: str | Path,
    parametros: PipelineParameters | Mapping[str, Any] | None = None,
    callback_progresso: ProgressCallback | Callable[[Any], None] | None = None,
    cancelar_evento: Any = None,
    *,
    cancelamento: Any = None,
    saida: str | Path | None = None,
    output_dir: str | Path | None = None,
    exportar: bool = False,
    **opcoes: Any,
) -> dict[str, Any]:
    """Processa um scan e retorna previsões serializáveis.

    Args:
        scan: Caminho para ``.bin`` ou qualquer formato aceito pelo pipeline.
        parametros: ``PipelineParameters`` ou mapping com parâmetros.
        callback_progresso: callback que recebe :class:`Progress`.
        cancelar_evento: ``threading.Event``/callable consultado antes das
            etapas; ``cancelamento`` é alias aceito pela interface.
        saida/output_dir: pasta usada quando ``exportar=True``.
        exportar: quando verdadeiro, cria STL, layout e FlexScript.

    Raises:
        FileNotFoundError: se o arquivo não existir.
        ParameterError: se os parâmetros forem inválidos.
        PipelineServiceError: se Open3D ou a exportação falhar.
    """
    if callback_progresso is None:
        callback_progresso = (
            opcoes.pop("progress_callback", None)
            or opcoes.pop("on_progress", None)
            or opcoes.pop("callback", None)
        )
    if cancelar_evento is None:
        cancelar_evento = (
            opcoes.pop("cancel_event", None)
            or opcoes.pop("cancel_evento", None)
            or opcoes.pop("cancelar", None)
        )
    mapa_flexsim = (
        opcoes.pop("mapa_flexsim", None)
        or opcoes.pop("flexsim_map", None)
    )
    caminho = Path(scan).expanduser()
    if not caminho.exists():
        raise FileNotFoundError(f"Scan não encontrado: {caminho}")
    if not caminho.is_file():
        raise ValueError(f"O caminho do scan não é um arquivo: {caminho}")

    token_cancelamento = cancelar_evento if cancelar_evento is not None else cancelamento
    params = _normalizar_parametros(parametros, **opcoes)
    scan_id = caminho.stem
    if _cancelamento_solicitado(token_cancelamento):
        _evento(
            callback_progresso,
            current=0,
            total=1,
            message="Processamento cancelado",
            stage="cancelado",
            scan_id=scan_id,
            cancelled=True,
        )
        return {
            "ok": False,
            "cancelado": True,
            "cancelled": True,
            "scan_id": scan_id,
            "arquivo": str(caminho),
            "predicoes": [],
            "resultados": [],
            "parametros": params.to_dict(),
        }

    _evento(
        callback_progresso,
        current=0,
        total=1,
        message=f"Processando {caminho.name}",
        stage="carregando",
        scan_id=scan_id,
    )
    pipeline = None
    try:
        if params.detector_backend == "pointnet2":
            resultados, diagnostico, clusters = _executar_pointnet2(caminho, params)
            _objetos_pcd = _chao = None
            diagnostico.setdefault("clusters", len(clusters))
        else:
            pipeline = _carregar_pipeline()
            deteccao = pipeline.detectar_objetos(
                str(caminho),
                voxel=params.voxel,
                eps=params.eps,
                min_points=params.min_points,
                plane_distance=params.plane_distance,
                oriented=params.oriented_box,
                max_ground_tilt_deg=params.max_ground_tilt_deg,
                ground_quantile=params.ground_quantile,
                remove_outliers=params.remove_outliers,
                outlier_neighbors=params.outlier_neighbors,
                outlier_std_ratio=params.outlier_std_ratio,
                cluster_mode=params.cluster_mode,
                return_diagnostics=True,
            )
            if len(deteccao) == 5:
                resultados, _objetos_pcd, _chao, clusters, diagnostico = deteccao
            else:  # compatibilidade com uma implementação externa antiga
                resultados, _objetos_pcd, _chao, clusters = deteccao
                diagnostico = {"clusters": len(clusters)}
    except (FileNotFoundError, ValueError, ParameterError, PipelineServiceError):
        raise
    except Exception as exc:
        raise PipelineServiceError(
            f"Falha ao processar o scan {caminho.name}: {exc}"
        ) from exc

    predicoes = _serializar_resultados(resultados)
    for previsao in predicoes:
        previsao["scan_id"] = scan_id

    _evento(
        callback_progresso,
        current=1,
        total=1,
        message=f"{len(predicoes)} objeto(s) detectado(s)",
        stage="concluido",
        scan_id=scan_id,
        result={"objetos": len(predicoes)},
    )

    cancelado = _cancelamento_solicitado(token_cancelamento)
    exportados: list[dict[str, Any]] | None = None
    destino = output_dir if output_dir is not None else saida
    if exportar and not cancelado:
        if destino is None:
            raise ValueError("Informe saida/output_dir quando exportar=True.")
        destino_path = Path(destino).expanduser()
        # ``saida`` pode ser um arquivo JSON usado pela função em lote; para
        # um scan, um diretório é a interpretação mais segura.
        if destino_path.suffix.lower() in {".json", ".csv", ".txt"}:
            destino_path = destino_path.parent / destino_path.stem
        _evento(
            callback_progresso,
            current=1,
            total=1,
            message="Gerando arquivos do FlexSim",
            stage="exportando",
            scan_id=scan_id,
        )
        if pipeline is None:
            pipeline = _carregar_pipeline()
        exportados = _exportar_scan(
            pipeline, clusters, predicoes, destino_path, mapa_flexsim
        )
        predicoes = exportados

    resultado = {
        "ok": not cancelado,
        "cancelado": cancelado,
        "cancelled": cancelado,
        "scan_id": scan_id,
        "arquivo": str(caminho),
        "predicoes": predicoes,
        # Alias para telas que chamam o conteúdo de resultados.
        "resultados": predicoes,
        "objetos_detectados": len(predicoes),
        "parametros": params.to_dict(),
        "diagnostico": dict(diagnostico),
    }
    if destino is not None:
        resultado["saida"] = str(Path(destino).expanduser())
    return resultado


def _selecionar_scans(
    pasta: str | Path,
    inicio: int = 0,
    limite: int | None = None,
    scan_ids: Iterable[str] | None = None,
) -> list[Path]:
    scans = listar_scans(pasta)
    if inicio < 0:
        raise ValueError("inicio não pode ser negativo.")
    if limite is not None and limite < 0:
        raise ValueError("limite não pode ser negativo.")
    if scan_ids is not None:
        desejados = {Path(str(item)).stem for item in scan_ids}
        scans = [scan for scan in scans if scan.stem in desejados]
    else:
        scans = scans[inicio:]
        if limite is not None:
            scans = scans[:limite]
    if not scans:
        raise FileNotFoundError("Nenhum scan selecionado para processamento.")
    return scans


def _destino_json(saida: str | Path | None) -> Path | None:
    if saida is None:
        return None
    caminho = Path(saida).expanduser()
    if caminho.exists() and caminho.is_dir():
        return caminho / "predicoes_warehouse.json"
    if caminho.suffix.lower() != ".json":
        return caminho / "predicoes_warehouse.json"
    return caminho


def processar_dataset(
    pasta: str | Path,
    parametros: PipelineParameters | Mapping[str, Any] | None = None,
    callback_progresso: ProgressCallback | Callable[[Any], None] | None = None,
    cancelar_evento: Any = None,
    *,
    cancelamento: Any = None,
    inicio: int = 0,
    limite: int | None = None,
    scan_ids: Iterable[str] | None = None,
    scans: Iterable[str | Path] | None = None,
    saida: str | Path | None = None,
    output_file: str | Path | None = None,
    continuar_em_erro: bool = True,
    **opcoes: Any,
) -> dict[str, Any]:
    """Processa um ou vários scans, emitindo progresso e aceitando cancelamento.

    ``pasta`` pode ser a raiz do dataset ou diretamente ``bin/``. Por padrão,
    erros de um scan são registrados em ``erros`` e o processamento continua;
    use ``continuar_em_erro=False`` para interromper na primeira falha.
    """
    if callback_progresso is None:
        callback_progresso = (
            opcoes.pop("progress_callback", None)
            or opcoes.pop("on_progress", None)
            or opcoes.pop("callback", None)
        )
    if cancelar_evento is None:
        cancelar_evento = (
            opcoes.pop("cancel_event", None)
            or opcoes.pop("cancel_evento", None)
            or opcoes.pop("cancelar", None)
        )
    token_cancelamento = cancelar_evento if cancelar_evento is not None else cancelamento
    params = _normalizar_parametros(parametros, **opcoes)
    if scans is not None:
        selecionados = [Path(item).expanduser() for item in scans]
        if not selecionados:
            raise FileNotFoundError("Nenhum scan selecionado para processamento.")
    else:
        selecionados = _selecionar_scans(pasta, inicio, limite, scan_ids)

    total = len(selecionados)
    predicoes_por_scan: dict[str, list[dict[str, Any]]] = {}
    diagnosticos_por_scan: dict[str, dict[str, Any]] = {}
    erros: list[dict[str, Any]] = []
    processados = 0
    cancelado = False
    _evento(
        callback_progresso,
        current=0,
        total=total,
        message=f"Iniciando processamento de {total} scan(s)",
        stage="iniciando",
    )

    for indice, caminho in enumerate(selecionados, start=1):
        if _cancelamento_solicitado(token_cancelamento):
            cancelado = True
            _evento(
                callback_progresso,
                current=processados,
                total=total,
                message="Processamento cancelado pelo usuário",
                stage="cancelado",
                cancelled=True,
            )
            break

        def relatar(evento: Progress, *, _indice: int = indice,
                    _scan_id: str = caminho.stem) -> None:
            # A etapa interna ocupa a fração do scan atual, para a barra da UI
            # permanecer monotônica e refletir o dataset todo.
            interno = min(100.0, max(0.0, float(evento.percent)))
            percentual = ((_indice - 1) + interno / 100.0) / total * 100.0
            _evento(
                callback_progresso,
                current=_indice - 1 if interno < 100 else _indice,
                total=total,
                message=evento.message,
                stage=evento.stage,
                scan_id=_scan_id,
                percent=percentual,
                cancelled=evento.cancelled,
            )

        try:
            resultado_scan = processar_scan(
                caminho,
                params,
                callback_progresso=relatar,
                cancelar_evento=token_cancelamento,
                **opcoes,
            )
            if resultado_scan.get("cancelado"):
                cancelado = True
                break
            predicoes_por_scan[caminho.stem] = resultado_scan["predicoes"]
            diagnosticos_por_scan[caminho.stem] = dict(
                resultado_scan.get("diagnostico", {})
            )
            processados += 1
            _evento(
                callback_progresso,
                current=processados,
                total=total,
                message=f"Scan {caminho.name} concluído",
                stage="concluido",
                scan_id=caminho.stem,
                percent=processados / total * 100.0,
                result={"objetos": len(resultado_scan["predicoes"])},
            )
        except Exception as exc:
            erro = {
                "scan_id": caminho.stem,
                "arquivo": str(caminho),
                "tipo": type(exc).__name__,
                "mensagem": str(exc),
            }
            erros.append(erro)
            _evento(
                callback_progresso,
                current=indice,
                total=total,
                message=f"Erro em {caminho.name}: {exc}",
                stage="erro",
                scan_id=caminho.stem,
                percent=indice / total * 100.0,
                error=str(exc),
            )
            if not continuar_em_erro:
                raise

    documento: dict[str, Any] = {
        "ok": not erros and not cancelado,
        "formato": "lidar2flexsim-predicoes-v1",
        "parametros": params.to_dict(),
        "scans": predicoes_por_scan,
        "diagnosticos": diagnosticos_por_scan,
        "erros": erros,
        "cancelado": cancelado,
        "cancelled": cancelado,
        "total_scans": total,
        "processados": processados,
        "inicio": inicio,
        "limite": limite,
    }
    destino = output_file if output_file is not None else saida
    destino_path = _destino_json(destino)
    if destino_path is not None:
        try:
            destino_path.parent.mkdir(parents=True, exist_ok=True)
            destino_path.write_text(
                json.dumps(documento, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            documento["saida"] = str(destino_path)
        except OSError as exc:
            raise PipelineServiceError(
                f"Não foi possível gravar predições em {destino_path}: {exc}"
            ) from exc

    _evento(
        callback_progresso,
        current=processados,
        total=total,
        message=("Processamento cancelado" if cancelado else "Processamento concluído"),
        stage=("cancelado" if cancelado else "concluido"),
        percent=processados / total * 100.0 if total else 100.0,
        cancelled=cancelado,
        result={"scans": processados, "erros": len(erros)},
    )
    return documento


def _predicao_para_caixa(avaliador: Any, item: Mapping[str, Any]) -> dict[str, Any]:
    """Aceita previsão no formato do pipeline ou caixa já normalizada."""
    if "yaw_rad" in item and "rotacao_z" not in item:
        copia = dict(item)
        copia["rotacao_z"] = math.degrees(float(item["yaw_rad"]))
        return avaliador.predicao_para_caixa(copia)
    return avaliador.predicao_para_caixa(dict(item))


def _normalizar_predicoes(avaliador: Any, predicoes: Any,
                          scan_id: str | None = None) -> dict[str, list[dict[str, Any]]]:
    if isinstance(predicoes, (str, Path)):
        return avaliador.carregar_predicoes(Path(predicoes), scan_id)

    documento = predicoes
    if isinstance(documento, Mapping) and "predicoes" in documento and "scans" not in documento:
        inferido = scan_id or documento.get("scan_id")
        if inferido is None:
            raise ValueError("Resultado de um scan requer scan_id para avaliação.")
        documento = {str(inferido): documento["predicoes"]}

    if isinstance(documento, list):
        inferido = scan_id
        if inferido is None:
            ids = {str(item["scan_id"]) for item in documento
                   if isinstance(item, Mapping) and item.get("scan_id")}
            if len(ids) == 1:
                inferido = ids.pop()
        if inferido is None:
            raise ValueError("Predições em lista requerem scan_id.")
        documento = {str(inferido): documento}

    if not isinstance(documento, Mapping):
        raise ValueError("predicoes deve ser caminho, lista ou objeto com scans.")
    scans = documento.get("scans", documento)
    if not isinstance(scans, Mapping):
        raise ValueError("Campo scans deve ser um mapping scan_id -> predições.")
    if scan_id is not None:
        if scan_id not in scans:
            raise KeyError(f"Scan {scan_id} não encontrado nas predições.")
        scans = {scan_id: scans[scan_id]}
    normalizadas: dict[str, list[dict[str, Any]]] = {}
    for chave, valores in scans.items():
        if not isinstance(valores, Sequence) or isinstance(valores, (str, bytes)):
            raise ValueError(f"Predições do scan {chave} devem ser uma lista.")
        normalizadas[str(chave)] = [
            _predicao_para_caixa(avaliador, item) for item in valores
        ]
    return normalizadas


def avaliar_predicoes(
    predicoes: Any,
    labels: str | Path,
    thresholds: Iterable[float] = (0.25, 0.5),
    *,
    iou_thresholds: Iterable[float] | None = None,
    class_aware: bool = False,
    class_map: Mapping[str, str] | str | Path | None = None,
    scan_id: str | None = None,
    saida: str | Path | None = None,
    callback_progresso: ProgressCallback | Callable[[Any], None] | None = None,
    cancelar_evento: Any = None,
) -> dict[str, Any]:
    """Avalia predições contra labels e retorna métricas JSON-serializáveis.

    ``predicoes`` pode ser o caminho do JSON gerado por
    :func:`processar_dataset`, um dicionário desse formato ou a lista de um
    único scan. ``labels`` aceita a pasta ``label/`` ou um arquivo ``.txt``.
    """
    avaliador = _carregar_avaliador()
    labels_path = Path(labels).expanduser()
    if labels_path.is_file():
        if scan_id is None:
            scan_id = labels_path.stem
        labels_dir = labels_path.parent
    else:
        diretorios = localizar_diretorios(labels_path)
        labels_dir = diretorios.get("label") or labels_path
    if not labels_dir.exists() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Pasta de labels não encontrada: {labels_dir}")

    if isinstance(class_map, (str, Path)):
        try:
            class_map = json.loads(Path(class_map).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Mapa de classes inválido: {class_map}") from exc
    mapa: dict[str, str] | None = None
    if class_aware:
        mapa = {str(k): str(v) for k, v in dict(class_map or {}).items()}

    predicoes_map = _normalizar_predicoes(avaliador, predicoes, scan_id)
    thresholds_final = list(iou_thresholds if iou_thresholds is not None else thresholds)
    if not thresholds_final:
        raise ValueError("Informe ao menos um threshold de IoU.")
    try:
        thresholds_final = [float(valor) for valor in thresholds_final]
    except (TypeError, ValueError) as exc:
        raise ValueError("Thresholds de IoU devem ser numéricos.") from exc
    if any(valor < 0.0 or valor > 1.0 for valor in thresholds_final):
        raise ValueError("Thresholds de IoU devem estar entre 0 e 1.")

    _evento(
        callback_progresso,
        current=0,
        total=len(predicoes_map),
        message="Avaliando detecções",
        stage="avaliando",
    )
    resultado_metricas = avaliador.resumir_scans(
        predicoes_map, labels_dir, thresholds_final, mapa
    )
    if not any(
        valores.get("scans_avaliados", 0) > 0
        for valores in resultado_metricas["por_threshold"].values()
    ):
        raise FileNotFoundError(
            "Nenhum scan das predições possui label correspondente. "
            "Verifique os nomes dos scan_id e a pasta de labels."
        )

    cancelado = _cancelamento_solicitado(cancelar_evento)
    documento: dict[str, Any] = {
        "ok": not cancelado,
        "formato": "lidar2flexsim-metricas-v1",
        "class_aware": class_aware,
        "class_map": mapa,
        "thresholds": thresholds_final,
        **resultado_metricas,
        "cancelado": cancelado,
        "cancelled": cancelado,
    }
    if saida is not None:
        destino = Path(saida).expanduser()
        if destino.suffix.lower() != ".json":
            destino = destino / "metricas_warehouse.json"
        try:
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_text(
                json.dumps(documento, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            documento["saida"] = str(destino)
        except OSError as exc:
            raise PipelineServiceError(
                f"Não foi possível gravar métricas em {destino}: {exc}"
            ) from exc
    _evento(
        callback_progresso,
        current=len(predicoes_map),
        total=len(predicoes_map),
        message="Avaliação concluída",
        stage="concluido",
        percent=100.0,
        result=documento.get("por_threshold"),
        cancelled=cancelado,
    )
    return documento


def exportar_flexsim(
    scan: str | Path,
    saida: str | Path,
    parametros: PipelineParameters | Mapping[str, Any] | None = None,
    **opcoes: Any,
) -> dict[str, Any]:
    """Processa e exporta um scan para STL/layout/FlexScript.

    É um atalho para a ação "Exportar para FlexSim" da interface.
    """
    return processar_scan(
        scan,
        parametros=parametros,
        saida=saida,
        exportar=True,
        **opcoes,
    )


def export_flexsim(
    scan: str | Path | None = None,
    saida: str | Path | None = None,
    parametros: PipelineParameters | Mapping[str, Any] | None = None,
    *,
    result: Mapping[str, Any] | None = None,
    predictions: Any = None,
    predicoes: Any = None,
    output_dir: str | Path | None = None,
    **opcoes: Any,
) -> dict[str, Any]:
    """Exporta resultados para arquivos consumíveis pelo FlexSim.

    A forma principal recebe ``scan`` e o reprocessa para gerar STL individuais
    com :func:`exportar_flexsim`. A forma usada pelo botão da interface pode
    receber ``result``/``predictions`` já calculados; nesse caso gera
    ``layout.json``, ``layout.csv`` e ``build_flexsim.txt`` a partir das caixas
    disponíveis (sem repetir a detecção).
    """
    destino = output_dir if output_dir is not None else saida
    if scan is not None:
        return exportar_flexsim(
            scan,
            destino or "saida",
            parametros=parametros,
            **opcoes,
        )

    if destino is None:
        raise ValueError("Informe saida/output_dir para exportação FlexSim.")
    dados = predictions if predictions is not None else predicoes
    if dados is None and result is not None:
        dados = result.get("predictions", result.get("predicoes", result.get("scans", {})))
    if dados is None:
        dados = {}

    objetos: list[dict[str, Any]] = []
    if isinstance(dados, Mapping):
        for id_scan, valores in dados.items():
            if not isinstance(valores, Sequence) or isinstance(valores, (str, bytes)):
                continue
            for valor in valores:
                if isinstance(valor, Mapping):
                    item = dict(valor)
                    item.setdefault("scan_id", str(id_scan))
                    objetos.append(item)
    elif isinstance(dados, Sequence) and not isinstance(dados, (str, bytes)):
        objetos = [dict(valor) for valor in dados if isinstance(valor, Mapping)]

    destino_path = Path(destino).expanduser()
    destino_path.mkdir(parents=True, exist_ok=True)
    pipeline = _carregar_pipeline()
    mapa = dict(getattr(pipeline, "MAPA_FLEXSIM", {}))
    try:
        pipeline.exportar_layout(objetos, str(destino_path), mapa)
        script_path = destino_path / "build_flexsim.txt"
        pipeline.gerar_flexscript(objetos, str(script_path), mapa)
    except Exception as exc:
        raise PipelineServiceError(
            f"Falha ao exportar resultados para o FlexSim: {exc}"
        ) from exc
    return {
        "ok": True,
        "output_dir": str(destino_path),
        "saida": str(destino_path),
        "arquivos": [
            str(destino_path / "layout.json"),
            str(destino_path / "layout.csv"),
            str(script_path),
        ],
        "files": [
            str(destino_path / "layout.json"),
            str(destino_path / "layout.csv"),
            str(script_path),
        ],
        "objetos": len(objetos),
    }


# Aliases em inglês facilitam integração com componentes que não usam nomes
# portugueses, sem duplicar a implementação ou quebrar a API planejada.
process_scan = processar_scan
process_dataset = processar_dataset
evaluate_predictions = avaliar_predicoes
evaluate_dataset = avaliar_predicoes


__all__ = [
    "PipelineParameters",
    "Progress",
    "avaliar_predicoes",
    "evaluate_predictions",
    "evaluate_dataset",
    "export_flexsim",
    "exportar_flexsim",
    "processar_dataset",
    "processar_scan",
    "process_dataset",
    "process_scan",
]

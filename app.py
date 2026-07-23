"""Interface desktop (NiceGUI) para o pipeline LiDAR → FlexSim.

Este módulo fornece o MVP da interface gráfica do projeto. A interface
seleciona uma pasta local do dataset, valida ``bin/``, ``label/`` e ``vis/``,
executa o processamento em uma thread de trabalho e apresenta os resultados
sem chamar a CLI por ``subprocess``.

Instalação e execução::

    python -m pip install nicegui open3d numpy
    python app.py

Quando a dependência NiceGUI não está instalada, importar este módulo
continua sendo seguro (por exemplo, para executar testes do núcleo). A
execução pelo ``__main__`` termina com uma mensagem orientando a instalação.
O modo nativo é solicitado quando suportado pela versão do NiceGUI; se o
backend nativo não estiver disponível, a aplicação é aberta no navegador.

As APIs do núcleo são descobertas em tempo de execução. O contrato preferido
é ``core.pipeline_service``, ``core.evaluation_service`` e
``core.export_service`` com funções ``process_dataset``, ``evaluate_dataset``
e ``export_flexsim`` (ou métodos equivalentes). Enquanto o núcleo é
refatorado, há um adaptador de compatibilidade que chama as funções Python já
existentes em ``lidar2flexsim.py``/``avaliar_deteccoes.py``; a CLI nunca é
invocada pela interface.
"""

from __future__ import annotations

import inspect
import json
import os
import platform
import subprocess
import threading
import webbrowser
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


try:  # NiceGUI é opcional para permitir importação e testes headless.
    from nicegui import ui
except ImportError:  # pragma: no cover - depende do ambiente de execução
    ui = None  # type: ignore[assignment]


ProgressCallback = Callable[..., None]


PARAMETER_PRESETS: dict[str, dict[str, Any]] = {
    "rápido": {
        "voxel": 0.08,
        "eps": 0.35,
        "min_points": 10,
        "plane_distance": 0.05,
        "oriented_box": False,
        "max_ground_tilt_deg": 30.0,
        "ground_quantile": 0.35,
        "remove_outliers": True,
        "outlier_neighbors": 8,
        "outlier_std_ratio": 3.0,
        "cluster_mode": "3d",
        "detector_backend": "heuristic",
        "model_checkpoint": None,
        "device": "auto",
        "score_threshold": 0.50,
        "num_points": 4096,
    },
    "equilibrado": {
        "voxel": 0.05,
        "eps": 0.25,
        "min_points": 20,
        "plane_distance": 0.05,
        "oriented_box": True,
        "max_ground_tilt_deg": 25.0,
        "ground_quantile": 0.30,
        "remove_outliers": True,
        "outlier_neighbors": 12,
        "outlier_std_ratio": 2.5,
        "cluster_mode": "3d",
        "detector_backend": "heuristic",
        "model_checkpoint": None,
        "device": "auto",
        "score_threshold": 0.50,
        "num_points": 4096,
    },
    "detalhado": {
        "voxel": 0.03,
        "eps": 0.15,
        "min_points": 40,
        "plane_distance": 0.03,
        "oriented_box": True,
        "max_ground_tilt_deg": 20.0,
        "ground_quantile": 0.25,
        "remove_outliers": True,
        "outlier_neighbors": 12,
        "outlier_std_ratio": 2.5,
        "cluster_mode": "3d",
        "detector_backend": "heuristic",
        "model_checkpoint": None,
        "device": "auto",
        "score_threshold": 0.50,
        "num_points": 4096,
    },
}


@dataclass(frozen=True)
class DatasetInfo:
    """Diagnóstico de uma pasta do Warehouse LiDAR.

    ``ready`` exige somente uma pasta ``bin`` com scans. ``complete`` indica
    que as três pastas esperadas foram encontradas; labels e visualizações são
    úteis para avaliação/preview, mas não impedem uma execução de detecção.
    """

    root: Path
    bin_dir: Path | None
    label_dir: Path | None
    vis_dir: Path | None
    bin_count: int = 0
    label_count: int = 0
    vis_count: int = 0
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return self.bin_dir is not None and self.bin_count > 0

    @property
    def complete(self) -> bool:
        return self.ready and self.label_dir is not None and self.vis_dir is not None


@dataclass
class ProgressState:
    """Estado compartilhado entre a thread do pipeline e a UI."""

    current: int = 0
    total: int = 0
    fraction: float = 0.0
    message: str = "Aguardando processamento"
    running: bool = False
    cancelled: bool = False
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, current: int | None = None, total: int | None = None,
               fraction: float | None = None, message: str | None = None) -> None:
        with self.lock:
            if current is not None:
                self.current = max(0, int(current))
            if total is not None:
                self.total = max(0, int(total))
            if fraction is not None:
                self.fraction = min(1.0, max(0.0, float(fraction)))
            elif self.total:
                self.fraction = min(1.0, max(0.0, self.current / self.total))
            if message is not None:
                self.message = str(message)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "current": self.current,
                "total": self.total,
                "fraction": self.fraction,
                "message": self.message,
                "running": self.running,
                "cancelled": self.cancelled,
                "error": self.error,
            }


@dataclass
class ProcessingConfig:
    """Configuração serializável enviada ao serviço do núcleo."""

    dataset_dir: Path
    bin_dir: Path
    label_dir: Path | None
    vis_dir: Path | None
    output_dir: Path
    scan_paths: list[Path]
    params: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_dir": str(self.dataset_dir),
            "bin_dir": str(self.bin_dir),
            "label_dir": str(self.label_dir) if self.label_dir else None,
            "vis_dir": str(self.vis_dir) if self.vis_dir else None,
            "output_dir": str(self.output_dir),
            "scan_paths": [str(path) for path in self.scan_paths],
            "params": dict(self.params),
            # Nomes planos mantêm compatibilidade com serviços simples.
            **dict(self.params),
        }


def _normalizar_raiz_dataset(path: str | os.PathLike[str]) -> Path:
    """Aceita tanto a raiz do dataset quanto a própria pasta ``bin``."""

    root = Path(path).expanduser().resolve()
    if root.name.lower() == "bin" and root.parent.is_dir():
        return root.parent
    return root


def _contar_arquivos(path: Path | None, extensoes: Iterable[str]) -> int:
    if path is None or not path.is_dir():
        return 0
    wanted = {ext.lower() for ext in extensoes}
    try:
        return sum(1 for item in path.iterdir()
                   if item.is_file() and item.suffix.lower() in wanted)
    except OSError:
        return 0


def validate_dataset(path: str | os.PathLike[str]) -> DatasetInfo:
    """Valida a estrutura local ``bin/``, ``label/`` e ``vis/``.

    A função não cria diretórios nem toca no conteúdo dos scans. Uma pasta
    ``bin`` existente com pelo menos um ``.bin`` é suficiente para habilitar a
    detecção; ausência de labels/visualizações é reportada como aviso.
    """

    try:
        root = _normalizar_raiz_dataset(path)
    except (TypeError, ValueError) as exc:
        # Mantém o contrato headless: entradas inválidas viram diagnóstico,
        # não uma exceção que derruba a página.
        return DatasetInfo(
            root=Path("."), bin_dir=None, label_dir=None, vis_dir=None,
            errors=(f"Caminho inválido: {exc}",),
        )
    errors: list[str] = []
    warnings: list[str] = []
    if not root.exists():
        errors.append(f"Pasta não encontrada: {root}")
        return DatasetInfo(root, None, None, None, errors=tuple(errors))
    if not root.is_dir():
        errors.append(f"O caminho não é uma pasta: {root}")
        return DatasetInfo(root, None, None, None, errors=tuple(errors))

    bin_dir = root / "bin"
    label_dir = root / "label"
    vis_dir = root / "vis"
    # Também aceitamos nomes em maiúsculas em cópias do dataset.
    try:
        children = {item.name.lower(): item for item in root.iterdir() if item.is_dir()}
    except OSError as exc:
        errors.append(f"Não foi possível ler a pasta: {exc}")
        return DatasetInfo(root, None, None, None, errors=tuple(errors))
    bin_dir = children.get("bin")
    label_dir = children.get("label")
    vis_dir = children.get("vis") or children.get("vis 2") or children.get("vis2")

    if bin_dir is None:
        errors.append("A pasta bin/ não foi encontrada.")
    bin_count = _contar_arquivos(bin_dir, (".bin",))
    if bin_dir is not None and bin_count == 0:
        errors.append("A pasta bin/ não contém arquivos .bin.")
    label_count = _contar_arquivos(label_dir, (".txt",))
    vis_count = _contar_arquivos(vis_dir, (".png", ".jpg", ".jpeg", ".webp", ".bmp"))
    if label_dir is None:
        warnings.append("A pasta label/ não foi encontrada; métricas não estarão disponíveis.")
    elif label_count == 0:
        warnings.append("A pasta label/ não contém arquivos .txt.")
    if vis_dir is None:
        warnings.append("A pasta vis/ não foi encontrada; preview não estará disponível.")
    elif vis_count == 0:
        warnings.append("A pasta vis/ não contém imagens reconhecidas.")

    return DatasetInfo(
        root=root,
        bin_dir=bin_dir,
        label_dir=label_dir,
        vis_dir=vis_dir,
        bin_count=bin_count,
        label_count=label_count,
        vis_count=vis_count,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _scan_sort_key(path: Path) -> tuple[int, str]:
    try:
        return (0, f"{int(path.stem):012d}")
    except ValueError:
        return (1, path.stem.lower())


def list_scan_files(bin_dir: Path) -> list[Path]:
    """Lista scans ``.bin`` em ordem numérica, com fallback alfabético."""

    try:
        return sorted((item for item in Path(bin_dir).iterdir()
                       if item.is_file() and item.suffix.lower() == ".bin"),
                      key=_scan_sort_key)
    except OSError:
        return []


def _invoke_tolerant(function: Callable[..., Any], context: Mapping[str, Any]) -> Any:
    """Invoca uma API do núcleo ignorando argumentos opcionais desconhecidos."""

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(**dict(context))
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    kwargs = {name: value for name, value in context.items() if name in signature.parameters}
    # APIs da camada core usam ``**opcoes`` para aliases históricos. Nesses
    # casos encaminhamos também o contexto completo; as funções oficiais
    # filtram os nomes de parâmetros geométricos antes de validá-los. Para
    # funções sem ``**kwargs`` permanecemos estritos e não enviamos metadados
    # da UI que elas não conhecem.
    if accepts_kwargs:
        kwargs.update({name: value for name, value in context.items() if name not in kwargs})
    # Alguns serviços usam config/settings em vez de vários parâmetros.
    if "config" in signature.parameters and "config" not in kwargs:
        kwargs["config"] = context.get("config", context)
    return function(**kwargs)


def _call_progress(callback: ProgressCallback | None, *args: Any, **kwargs: Any) -> None:
    if callback is None:
        return
    try:
        callback(*args, **kwargs)
    except TypeError:
        # Serviços simples podem aceitar apenas um dicionário.
        payload = kwargs or (args[0] if len(args) == 1 else args)
        try:
            callback(payload)
        except TypeError:
            callback()


class CoreAdapter:
    """Ponte entre a UI e diferentes versões da camada de serviços.

    O adaptador primeiro procura funções em ``core``. Em repositórios onde a
    refatoração ainda não foi aplicada, usa diretamente as funções Python do
    pipeline legado. Não executa scripts de terminal.
    """

    def __init__(self) -> None:
        self._modules: dict[str, Any] = {}
        for name in ("pipeline_service", "dataset_service", "evaluation_service", "export_service"):
            try:
                self._modules[name] = __import__(f"core.{name}", fromlist=["*"])
            except (ImportError, ModuleNotFoundError):
                continue

    @staticmethod
    def _find_function(modules: Iterable[Any], names: Iterable[str]) -> Callable[..., Any] | None:
        for module in modules:
            for name in names:
                candidate = getattr(module, name, None)
                if callable(candidate):
                    return candidate
        return None

    def process(self, config: ProcessingConfig, progress: ProgressCallback | None,
                cancel_event: threading.Event) -> dict[str, Any]:
        pipeline_modules = [self._modules[name] for name in ("pipeline_service", "dataset_service")
                            if name in self._modules]
        function = self._find_function(
            pipeline_modules,
            ("process_dataset", "process_scans", "run_dataset", "run_pipeline"),
        )
        if function is not None:
            context = {
                "config": config.as_dict(),
                "settings": config.as_dict(),
                # Nomes do serviço oficial (em português).
                "pasta": config.dataset_dir,
                "dataset_dir": config.dataset_dir,
                "bin_dir": config.bin_dir,
                "label_dir": config.label_dir,
                "vis_dir": config.vis_dir,
                "output_dir": config.output_dir,
                "scan_paths": config.scan_paths,
                "scans": config.scan_paths,
                "params": config.params,
                "parameters": config.params,
                "parametros": config.params,
                "progress_callback": progress,
                "on_progress": progress,
                "callback_progresso": progress,
                "cancel_event": cancel_event,
                "cancelamento": cancel_event,
                "cancelar_evento": cancel_event,
                # O serviço em lote aceita arquivo ou diretório e grava o
                # JSON de previsões. A UI mantém a pasta como destino visual.
                "saida": config.output_dir / "predicoes_warehouse.json",
                "output_file": config.output_dir / "predicoes_warehouse.json",
            }
            result = _invoke_tolerant(function, context)
            return self._normalise_result(result, config)
        return self._legacy_process(config, progress, cancel_event)

    @staticmethod
    def _normalise_result(result: Any, config: ProcessingConfig) -> dict[str, Any]:
        if result is None:
            return {"output_dir": str(config.output_dir), "scans": {}, "predictions": {}}
        if isinstance(result, Mapping):
            data = dict(result)
            data.setdefault("output_dir", str(config.output_dir))
            if "predictions" not in data and "predicoes" in data:
                data["predictions"] = data["predicoes"]
            if "scans" not in data and isinstance(data.get("predictions"), Mapping):
                data["scans"] = data["predictions"]
            return data
        return {"output_dir": str(config.output_dir), "result": result}

    def _legacy_process(self, config: ProcessingConfig, progress: ProgressCallback | None,
                        cancel_event: threading.Event) -> dict[str, Any]:
        try:
            from lidar2flexsim import detectar_objetos
        except ImportError as exc:
            raise RuntimeError(
                "A camada core não está disponível e o pipeline legado não pôde ser importado."
            ) from exc
        output_dir = config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions: dict[str, list[dict[str, Any]]] = {}
        diagnostics: dict[str, dict[str, Any]] = {}
        total = len(config.scan_paths)
        params = config.params
        for index, scan in enumerate(config.scan_paths, start=1):
            if cancel_event.is_set():
                break
            _call_progress(progress, index - 1, total, f"Processando {scan.name}")
            detection = detectar_objetos(
                str(scan),
                voxel=float(params["voxel"]),
                eps=float(params["eps"]),
                min_points=int(params["min_points"]),
                plane_distance=float(params["plane_distance"]),
                oriented=bool(params["oriented_box"]),
                max_ground_tilt_deg=float(params.get("max_ground_tilt_deg", 25.0)),
                ground_quantile=float(params.get("ground_quantile", 0.30)),
                remove_outliers=bool(params.get("remove_outliers", True)),
                outlier_neighbors=int(params.get("outlier_neighbors", 12)),
                outlier_std_ratio=float(params.get("outlier_std_ratio", 2.5)),
                cluster_mode=str(params.get("cluster_mode", "3d")),
                return_diagnostics=True,
            )
            if len(detection) == 5:
                objects, _, _, _, diagnostic = detection
                diagnostics[scan.stem] = dict(diagnostic)
            else:
                objects, _, _, _ = detection
            for obj in objects:
                obj["scan_id"] = scan.stem
            predictions[scan.stem] = objects
            _call_progress(progress, index, total, f"Concluído {scan.name}")
        document = {
            "formato": "lidar2flexsim-predicoes-v1",
            "parametros": dict(params),
            "scans": predictions,
            "diagnosticos": diagnostics,
        }
        predictions_path = output_dir / "predicoes_warehouse.json"
        predictions_path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
        return {
            "output_dir": str(output_dir),
            "predictions": predictions,
            "scans": predictions,
            "diagnosticos": diagnostics,
            "files": [str(predictions_path)],
            "cancelled": cancel_event.is_set(),
        }

    def evaluate(self, result: Mapping[str, Any], config: ProcessingConfig) -> dict[str, Any] | None:
        if config.label_dir is None or not config.label_dir.is_dir():
            return None
        function = self._find_function(
            [self._modules[name] for name in ("evaluation_service", "pipeline_service", "dataset_service")
             if name in self._modules],
            ("evaluate_dataset", "evaluate_predictions", "evaluate"),
        )
        predictions = result.get("predictions", result.get("scans", {}))
        if function is not None:
            context = {
                "config": config.as_dict(),
                "predictions": predictions,
                "predicoes": predictions,
                "predictions_path": result.get("predictions_path"),
                "labels_dir": config.label_dir,
                "label_dir": config.label_dir,
                "labels": config.label_dir,
                "thresholds": [0.25, 0.5],
                "iou_thresholds": [0.25, 0.5],
                "saida": config.output_dir / "metricas_warehouse.json",
            }
            value = _invoke_tolerant(function, context)
            return dict(value) if isinstance(value, Mapping) else {"result": value}
        try:
            from avaliar_deteccoes import resumir_scans
        except ImportError:
            return None
        if not isinstance(predictions, Mapping):
            return None
        return resumir_scans(dict(predictions), config.label_dir, [0.25, 0.5], None)

    def export(self, result: Mapping[str, Any], config: ProcessingConfig) -> dict[str, Any]:
        function = self._find_function(
            [self._modules[name] for name in ("export_service", "pipeline_service")
             if name in self._modules],
            # The per-scan service also exports STL geometry.  Prefer it over
            # the layout-only helper so PointNet++ predictions reach FlexSim
            # with the same artifacts as the heuristic backend.
            ("exportar_flexsim", "export_flexsim", "export_results", "export_dataset"),
        )
        if function is not None:
            # O serviço atual expõe ``exportar_flexsim(scan, saida, ...)``
            # para um scan. Quando não houver um exportador de dataset,
            # executamos esse atalho por scan em subpastas independentes para
            # não sobrescrever STL/layouts de frames diferentes.
            if getattr(function, "__name__", "") in {"exportar_flexsim", "export_scan"}:
                destinos: list[dict[str, Any]] = []
                multiplos = len(config.scan_paths) > 1
                for scan in config.scan_paths:
                    destino = config.output_dir / scan.stem if multiplos else config.output_dir
                    contexto_scan = {
                        "scan": scan,
                        "entrada": scan,
                        "saida": destino,
                        "output_dir": destino,
                        "parametros": config.params,
                        "parameters": config.params,
                        "config": config.as_dict(),
                    }
                    value = _invoke_tolerant(function, contexto_scan)
                    destinos.append(dict(value) if isinstance(value, Mapping) else {"result": value})
                return {"output_dir": str(config.output_dir), "exports": destinos,
                        "files": [str(config.output_dir)]}
            context = {
                "config": config.as_dict(),
                "result": result,
                "predictions": result.get("predictions", result.get("scans", {})),
                "output_dir": config.output_dir,
                "saida": config.output_dir,
            }
            value = _invoke_tolerant(function, context)
            return dict(value) if isinstance(value, Mapping) else {"result": value}
        return self._legacy_export(result, config)

    @staticmethod
    def _legacy_export(result: Mapping[str, Any], config: ProcessingConfig) -> dict[str, Any]:
        try:
            from lidar2flexsim import exportar_layout, gerar_flexscript
        except ImportError as exc:
            raise RuntimeError("Não foi possível carregar o exportador FlexSim.") from exc
        predictions = result.get("predictions", result.get("scans", {}))
        if isinstance(predictions, Mapping):
            # layout/FlexScript são definidos para uma lista de objetos; para
            # um dataset, exportamos todos os scans com ids preservados.
            objects: list[dict[str, Any]] = []
            for scan_id, values in predictions.items():
                for value in values or []:
                    item = dict(value)
                    item.setdefault("scan_id", scan_id)
                    objects.append(item)
        elif isinstance(predictions, list):
            objects = [dict(value) for value in predictions]
        else:
            objects = []
        config.output_dir.mkdir(parents=True, exist_ok=True)
        exportar_layout(objects, str(config.output_dir))
        script_path = config.output_dir / "build_flexsim.txt"
        gerar_flexscript(objects, str(script_path))
        files = [str(config.output_dir / "layout.json"),
                 str(config.output_dir / "layout.csv"), str(script_path)]
        return {"output_dir": str(config.output_dir), "files": files, "objects": len(objects)}


class GuiController:
    """Estado e handlers da página NiceGUI."""

    def __init__(self) -> None:
        self.adapter = CoreAdapter()
        self.dataset: DatasetInfo | None = None
        self.scan_files: list[Path] = []
        self.mode = "single"
        self.selected_scan = ""
        self.range_start = 0
        self.range_end = 0
        self.preset = "equilibrado"
        self.params = dict(PARAMETER_PRESETS[self.preset])
        self.output_dir: Path | None = None
        self.result: dict[str, Any] | None = None
        self.metrics: dict[str, Any] | None = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lidar2flexsim")
        self.future: Future[Any] | None = None
        self.cancel_event = threading.Event()
        self.progress = ProgressState()
        self.refs: dict[str, Any] = {}
        self._timer: Any = None

    def build(self) -> None:
        assert ui is not None
        ui.colors(primary="#2563eb", secondary="#0f766e", accent="#7c3aed")
        ui.add_head_html("""<style>
            body { background: #f3f6fb; }
            .section-card { border-radius: 14px; }
            .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
        </style>""")
        with ui.header().classes("items-center justify-between bg-slate-900 text-white px-6"):
            ui.label("LiDAR → FlexSim").classes("text-xl font-semibold")
            ui.label("Reconstrução de ambientes industriais").classes("text-sm opacity-70")
        with ui.column().classes("w-full max-w-7xl mx-auto p-4 gap-4"):
            self._build_dataset_card()
            with ui.row().classes("w-full items-start gap-4 flex-wrap"):
                self._build_processing_card()
                self._build_progress_card()
            self._build_results_card()
        self._timer = ui.timer(0.35, self._poll_future)

    def _build_dataset_card(self) -> None:
        assert ui is not None
        with ui.card().classes("section-card w-full p-5"):
            ui.label("1. Dados do dataset").classes("text-lg font-semibold")
            ui.label("Selecione a pasta que contém bin/, label/ e vis/. Os arquivos permanecem locais.").classes("text-sm text-gray-600")
            with ui.row().classes("w-full items-end gap-2 mt-3"):
                self.refs["dataset_input"] = ui.input("Pasta do dataset", placeholder="/caminho/para/warehouse").classes("grow").props("clearable")
                self.refs["dataset_input"].on("keydown.enter", lambda _: self.validate_dataset_from_ui())
                ui.button("Selecionar pasta", icon="folder_open", on_click=self.choose_dataset_folder).props("outline")
                ui.button("Validar", icon="check_circle", on_click=self.validate_dataset_from_ui)
            with ui.row().classes("gap-3 mt-3"):
                for key, title in (("bin", "bin/"), ("label", "label/"), ("vis", "vis/")):
                    self.refs[f"status_{key}"] = ui.chip(f"{title}: não verificado", icon="help_outline").props("outline")
            self.refs["dataset_message"] = ui.label("Selecione uma pasta para começar.").classes("text-sm text-gray-600")

    def _build_processing_card(self) -> None:
        assert ui is not None
        with ui.card().classes("section-card flex-1 min-w-[420px] p-5"):
            ui.label("2. Configuração").classes("text-lg font-semibold")
            self.refs["mode"] = ui.radio({"single": "Um scan", "range": "Intervalo", "all": "Dataset completo"}, value="single").props("inline")
            self.refs["mode"].on_value_change(self._on_mode_change)
            with ui.row().classes("w-full gap-2 mt-3"):
                self.refs["scan_select"] = ui.select(options={}, label="Scan", value=None).classes("grow")
                self.refs["start_input"] = ui.number("Início", value=0, min=0, step=1).classes("w-28")
                self.refs["end_input"] = ui.number("Fim", value=0, min=0, step=1).classes("w-28")
            self.refs["scan_select"].on_value_change(lambda _: self._sync_selection())
            self.refs["start_input"].on_value_change(lambda _: self._sync_selection())
            self.refs["end_input"].on_value_change(lambda _: self._sync_selection())
            with ui.row().classes("w-full items-end gap-2 mt-3"):
                self.refs["preset"] = ui.select(options={key: key.capitalize() for key in PARAMETER_PRESETS}, value=self.preset, label="Predefinição").classes("grow")
                self.refs["preset"].on_value_change(self._on_preset_change)
                self.refs["output_input"] = ui.input("Pasta de saída", value="saida/gui").classes("grow")
            with ui.expansion("Parâmetros avançados", icon="tune").classes("w-full mt-2"):
                with ui.row().classes("w-full gap-2 flex-wrap"):
                    self.refs["voxel"] = ui.number("Voxel (m)", value=self.params["voxel"], min=0.001, step=0.01).classes("w-32")
                    self.refs["eps"] = ui.number("DBSCAN eps (m)", value=self.params["eps"], min=0.001, step=0.01).classes("w-36")
                    self.refs["min_points"] = ui.number("Mín. pontos", value=self.params["min_points"], min=1, step=1).classes("w-32")
                    self.refs["plane_distance"] = ui.number("Plano RANSAC (m)", value=self.params["plane_distance"], min=0.001, step=0.01).classes("w-40")
                    self.refs["oriented_box"] = ui.switch("Bounding box orientada", value=self.params["oriented_box"])
                    self.refs["max_ground_tilt_deg"] = ui.number("Inclinação máx. piso (°)", value=self.params["max_ground_tilt_deg"], min=1, max=89, step=1).classes("w-48")
                    self.refs["ground_quantile"] = ui.number("Quantil do piso", value=self.params["ground_quantile"], min=0.01, max=0.9, step=0.05).classes("w-36")
                    self.refs["outlier_neighbors"] = ui.number("Vizinhos do filtro", value=self.params["outlier_neighbors"], min=0, step=1).classes("w-40")
                    self.refs["outlier_std_ratio"] = ui.number("Tolerância do filtro", value=self.params["outlier_std_ratio"], min=0.1, step=0.1).classes("w-40")
                    self.refs["remove_outliers"] = ui.switch("Remover outliers", value=self.params["remove_outliers"])
                    self.refs["cluster_mode"] = ui.select({"3d": "3D (padrão)", "bev": "BEV (experimental)"}, value=self.params["cluster_mode"], label="Espaço da clusterização").classes("w-64")
                    self.refs["detector_backend"] = ui.select({"heuristic": "Heurístico", "pointnet2": "PointNet++ (segmentação)"}, value=self.params["detector_backend"], label="Backend de detecção").classes("w-64")
                    self.refs["model_checkpoint"] = ui.input("Checkpoint PointNet++", value=self.params["model_checkpoint"] or "", placeholder="/caminho/modelo.pt").classes("grow")
                    self.refs["device"] = ui.select({"auto": "Auto", "cpu": "CPU", "cuda": "CUDA", "mps": "Apple MPS"}, value=self.params["device"], label="Dispositivo").classes("w-40")
                    self.refs["score_threshold"] = ui.number("Confiança mínima", value=self.params["score_threshold"], min=0, max=1, step=0.05).classes("w-40")
                    self.refs["num_points"] = ui.number("Pontos por amostra", value=self.params["num_points"], min=32, step=256).classes("w-44")
            with ui.row().classes("gap-2 mt-4"):
                self.refs["start_button"] = ui.button("Iniciar processamento", icon="play_arrow", on_click=self.start_processing).props("color=primary")
                self.refs["cancel_button"] = ui.button("Cancelar", icon="stop", on_click=self.cancel_processing).props("outline color=negative")
                self.refs["cancel_button"].disable()

    def _build_progress_card(self) -> None:
        assert ui is not None
        with ui.card().classes("section-card flex-1 min-w-[360px] p-5"):
            ui.label("3. Progresso").classes("text-lg font-semibold")
            self.refs["progress_message"] = ui.label("Aguardando processamento").classes("text-sm")
            self.refs["progress_bar"] = ui.linear_progress(value=0).classes("w-full mt-3")
            self.refs["progress_counter"] = ui.label("0 / 0 scans").classes("text-sm text-gray-600")
            self.refs["progress_error"] = ui.label("").classes("text-sm text-red-600")

    def _build_results_card(self) -> None:
        assert ui is not None
        with ui.card().classes("section-card w-full p-5"):
            with ui.row().classes("w-full justify-between items-center"):
                ui.label("4. Resultados").classes("text-lg font-semibold")
                with ui.row().classes("gap-2"):
                    self.refs["export_button"] = ui.button("Exportar para FlexSim", icon="file_download", on_click=self.export_results).props("outline")
                    self.refs["open_button"] = ui.button("Abrir pasta de saída", icon="folder_open", on_click=self.open_output_folder).props("outline")
                    self.refs["export_button"].disable()
                    self.refs["open_button"].disable()
            self.refs["result_summary"] = ui.label("Nenhum processamento concluído.").classes("text-sm text-gray-600 mt-2")
            self.refs["metrics"] = ui.label("").classes("text-sm mt-2")
            self.refs["diagnostics"] = ui.label("").classes("text-sm text-gray-600 mt-2 whitespace-pre-line")
            self.refs["result_table"] = ui.table(columns=[
                {"name": "scan", "label": "Scan", "field": "scan"},
                {"name": "objects", "label": "Objetos", "field": "objects"},
                {"name": "classes", "label": "Classes", "field": "classes"},
            ], rows=[]).classes("w-full mt-3")
            self.refs["preview"] = ui.image("").classes("max-h-72 max-w-full object-contain mt-3")
            self.refs["preview"].set_visibility(False)

    def _set_chip(self, key: str, text: str, color: str) -> None:
        chip = self.refs.get(f"status_{key}")
        if chip is not None:
            chip.text = text
            chip.props(f"color={color}")

    def choose_dataset_folder(self) -> None:
        """Abre seletor nativo quando Tk está disponível; texto continua aceito."""

        selected: str | None = None
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            selected = filedialog.askdirectory(title="Selecione a pasta do dataset") or None
            root.destroy()
        except Exception:
            # Ambientes sem Tk (servidor Linux/empacotamento) usam o input manual.
            selected = None
        if selected:
            self.refs["dataset_input"].value = selected
            self.validate_dataset_from_ui()

    def validate_dataset_from_ui(self) -> None:
        path = str(self.refs["dataset_input"].value or "").strip()
        if not path:
            self._show_dataset_message("Informe ou selecione uma pasta.", error=True)
            return
        self.dataset = validate_dataset(path)
        info = self.dataset
        self._set_chip("bin", f"bin/: {info.bin_count} scans" if info.bin_dir else "bin/: ausente", "positive" if info.ready else "negative")
        self._set_chip("label", f"label/: {info.label_count} labels" if info.label_dir else "label/: ausente", "positive" if info.label_dir and info.label_count else "warning")
        self._set_chip("vis", f"vis/: {info.vis_count} imagens" if info.vis_dir else "vis/: ausente", "positive" if info.vis_dir and info.vis_count else "warning")
        if info.ready:
            self.scan_files = list_scan_files(info.bin_dir)  # type: ignore[arg-type]
            self.refs["scan_select"].options = {path.stem: path.stem for path in self.scan_files}
            self.refs["scan_select"].value = self.scan_files[0].stem if self.scan_files else None
            self.refs["start_input"].value = 0
            self.refs["end_input"].value = max(0, len(self.scan_files) - 1)
            # Mantém o dataset somente para leitura; resultados ficam fora da
            # pasta baixada por padrão e podem ser alterados pelo usuário.
            self.output_dir = Path.cwd() / "saida_gui" / info.root.name
            self.refs["output_input"].value = str(self.output_dir)
            self._show_dataset_message("Dataset pronto para processamento." + (f" {' '.join(info.warnings)}" if info.warnings else ""))
            self._sync_selection()
        else:
            self.scan_files = []
            self._show_dataset_message(" ".join(info.errors), error=True)
        self._update_buttons()

    def _show_dataset_message(self, message: str, error: bool = False) -> None:
        label = self.refs.get("dataset_message")
        if label is not None:
            label.text = message
            label.classes(add="text-red-600" if error else "text-gray-600", remove="text-gray-600" if error else "text-red-600")

    def _on_mode_change(self, event: Any) -> None:
        self.mode = str(getattr(event, "value", event) or "single")
        self._sync_selection()

    def _on_preset_change(self, event: Any) -> None:
        value = str(getattr(event, "value", event) or "equilibrado").lower()
        self.preset = value if value in PARAMETER_PRESETS else "equilibrado"
        self.params.update(PARAMETER_PRESETS[self.preset])
        for key, value in self.params.items():
            ref = self.refs.get(key)
            if ref is not None:
                ref.value = value

    def _sync_selection(self) -> None:
        if self.mode == "single" and self.scan_files:
            self.selected_scan = str(self.refs["scan_select"].value or self.scan_files[0].stem)
        self._update_buttons()

    def _selected_paths(self) -> list[Path]:
        if not self.scan_files:
            return []
        if self.mode == "single":
            selected = self.selected_scan or str(self.refs["scan_select"].value or self.scan_files[0].stem)
            return [path for path in self.scan_files if path.stem == selected][:1]
        if self.mode == "range":
            start = max(0, int(self.refs["start_input"].value or 0))
            end = min(len(self.scan_files) - 1, int(self.refs["end_input"].value or 0))
            if end < start:
                return []
            return self.scan_files[start:end + 1]
        return list(self.scan_files)

    def _read_params(self) -> dict[str, Any]:
        values = {
            "voxel": float(self.refs["voxel"].value),
            "eps": float(self.refs["eps"].value),
            "min_points": int(self.refs["min_points"].value),
            "plane_distance": float(self.refs["plane_distance"].value),
            "oriented_box": bool(self.refs["oriented_box"].value),
            "max_ground_tilt_deg": float(self.refs["max_ground_tilt_deg"].value),
            "ground_quantile": float(self.refs["ground_quantile"].value),
            "remove_outliers": bool(self.refs["remove_outliers"].value),
            "outlier_neighbors": int(self.refs["outlier_neighbors"].value),
            "outlier_std_ratio": float(self.refs["outlier_std_ratio"].value),
            "cluster_mode": str(self.refs["cluster_mode"].value or "3d"),
            "detector_backend": str(self.refs["detector_backend"].value or "heuristic"),
            "model_checkpoint": str(self.refs["model_checkpoint"].value or "").strip() or None,
            "device": str(self.refs["device"].value or "auto"),
            "score_threshold": float(self.refs["score_threshold"].value),
            "num_points": int(self.refs["num_points"].value),
        }
        if (values["voxel"] <= 0 or values["eps"] <= 0 or
                values["plane_distance"] <= 0 or values["min_points"] < 1 or
                not 0 < values["max_ground_tilt_deg"] < 90 or
                not 0.01 <= values["ground_quantile"] <= 0.9 or
                values["outlier_neighbors"] < 0 or values["outlier_std_ratio"] <= 0 or
                not 0 <= values["score_threshold"] <= 1 or values["num_points"] < 32):
            raise ValueError("Revise os parâmetros: valores positivos, quantil entre 0,01 e 0,90, confiança entre 0 e 1 e pelo menos 32 pontos.")
        if values["detector_backend"] == "pointnet2" and not values["model_checkpoint"]:
            raise ValueError("Informe o checkpoint .pt/.pth para usar o PointNet++.")
        self.params = values
        return values

    def _build_config(self) -> ProcessingConfig:
        if self.dataset is None or not self.dataset.ready or self.dataset.bin_dir is None:
            raise ValueError("Valide uma pasta de dataset antes de processar.")
        scans = self._selected_paths()
        if not scans:
            raise ValueError("Nenhum scan foi selecionado; revise o modo e o intervalo.")
        output_text = str(self.refs["output_input"].value or "saida_gui").strip()
        output = Path(output_text).expanduser()
        if not output.is_absolute():
            output = Path.cwd() / output
        return ProcessingConfig(
            dataset_dir=self.dataset.root,
            bin_dir=self.dataset.bin_dir,
            label_dir=self.dataset.label_dir,
            vis_dir=self.dataset.vis_dir,
            output_dir=output.resolve(),
            scan_paths=scans,
            params=self._read_params(),
        )

    def start_processing(self) -> None:
        if self.future is not None and not self.future.done():
            return
        try:
            config = self._build_config()
        except (ValueError, OSError) as exc:
            self._notify(str(exc), error=True)
            return
        self.cancel_event = threading.Event()
        self.progress = ProgressState(total=len(config.scan_paths), running=True, message="Iniciando…")
        self.result = None
        self.metrics = None
        self.output_dir = config.output_dir
        self.future = self.executor.submit(self._worker, config)
        self._update_buttons()

    def _worker(self, config: ProcessingConfig) -> dict[str, Any]:
        def callback(*args: Any, **kwargs: Any) -> None:
            current = total = None
            message = None
            fraction = None
            if kwargs:
                current = kwargs.get("current", kwargs.get("index"))
                total = kwargs.get("total")
                message = kwargs.get("message", kwargs.get("scan"))
                percent = kwargs.get("percent")
                if percent is not None:
                    fraction = float(percent) / 100.0
            if args:
                if len(args) >= 1 and isinstance(args[0], (int, float)):
                    current = int(args[0])
                if len(args) >= 2 and isinstance(args[1], (int, float)):
                    total = int(args[1])
                if len(args) >= 3:
                    message = str(args[2])
                elif len(args) == 1 and isinstance(args[0], Mapping):
                    payload = args[0]
                    current = payload.get("current", payload.get("index"))
                    total = payload.get("total")
                    message = payload.get("message", payload.get("scan"))
                    percent = payload.get("percent")
                    if percent is not None:
                        fraction = float(percent) / 100.0
                elif len(args) == 1:
                    # O serviço oficial envia ``core.models.Progress``. Não
                    # exigimos a classe aqui para manter o adaptador leve e
                    # compatível com callbacks de dicionário/objetos simples.
                    evento = args[0]
                    current = getattr(evento, "current", current)
                    total = getattr(evento, "total", total)
                    message = getattr(evento, "message", message)
                    percent = getattr(evento, "percent", None)
                    if percent is not None:
                        fraction = float(percent) / 100.0
            self.progress.update(current=current, total=total, fraction=fraction, message=message)
        try:
            result = self.adapter.process(config, callback, self.cancel_event)
            if not self.cancel_event.is_set():
                self.progress.update(current=len(config.scan_paths), total=len(config.scan_paths), fraction=1.0, message="Processamento concluído")
            result = dict(result)
            # Métricas são uma etapa complementar: uma detecção válida não
            # deve ser descartada apenas porque labels estão incompletos ou a
            # avaliação não pôde ser executada naquele ambiente.
            try:
                metrics = self.adapter.evaluate(result, config)
                if metrics is not None:
                    result["metrics"] = metrics
            except Exception as exc:
                result["evaluation_error"] = f"{type(exc).__name__}: {exc}"
            return result
        except Exception as exc:
            self.progress.error = f"{type(exc).__name__}: {exc}"
            raise

    def cancel_processing(self) -> None:
        if self.future is not None and not self.future.done():
            self.cancel_event.set()
            with self.progress.lock:
                self.progress.cancelled = True
                self.progress.message = "Cancelamento solicitado…"
            self._notify("O cancelamento foi solicitado; o scan atual será concluído.")

    def _poll_future(self) -> None:
        snapshot = self.progress.snapshot()
        for key, value in (("progress_bar", snapshot["fraction"]),):
            ref = self.refs.get(key)
            if ref is not None:
                ref.value = value
        if self.refs.get("progress_message") is not None:
            self.refs["progress_message"].text = snapshot["message"]
        if self.refs.get("progress_counter") is not None:
            total = snapshot["total"]
            self.refs["progress_counter"].text = f"{snapshot['current']} / {total} scans" if total else "0 scans"
        if self.refs.get("progress_error") is not None:
            self.refs["progress_error"].text = snapshot["error"] or ""
        future = self.future
        if future is not None and future.done() and snapshot["running"]:
            with self.progress.lock:
                self.progress.running = False
            try:
                self.result = future.result()
                self.metrics = self.result.get("metrics") if isinstance(self.result, Mapping) else None
                if snapshot["cancelled"]:
                    self._notify("Processamento cancelado; resultados parciais foram preservados.")
                else:
                    self._notify("Processamento concluído.")
                self._render_results()
            except Exception as exc:
                self._notify(f"Falha no processamento: {exc}", error=True)
            self._update_buttons()

    def _render_results(self) -> None:
        result = self.result or {}
        predictions = result.get("predictions", result.get("scans", {}))
        rows: list[dict[str, Any]] = []
        total_objects = 0
        if isinstance(predictions, Mapping):
            for scan_id, objects in sorted(predictions.items()):
                values = list(objects or [])
                classes = ", ".join(sorted({str(item.get("classe", "desconhecido")) for item in values})) or "—"
                rows.append({"scan": str(scan_id), "objects": len(values), "classes": classes})
                total_objects += len(values)
        elif isinstance(predictions, list):
            rows.append({"scan": "selecionado", "objects": len(predictions), "classes": ", ".join(sorted({str(item.get("classe", "desconhecido")) for item in predictions}))})
            total_objects = len(predictions)
        self.refs["result_table"].rows = rows[:200]
        self.refs["result_summary"].text = f"{len(rows)} scans processados · {total_objects} objetos detectados"
        metrics = self.metrics
        if metrics:
            self.refs["metrics"].text = self._format_metrics(metrics)
        elif result.get("evaluation_error"):
            self.refs["metrics"].text = f"Métricas indisponíveis: {result['evaluation_error']}"
        else:
            self.refs["metrics"].text = ""
        diagnosticos = result.get("diagnosticos", {})
        if isinstance(diagnosticos, Mapping) and diagnosticos:
            self.refs["diagnostics"].text = self._format_diagnostics(diagnosticos)
        elif isinstance(result.get("diagnostico"), Mapping):
            self.refs["diagnostics"].text = self._format_diagnostics({str(result.get("scan_id", "scan")): result["diagnostico"]})
        else:
            self.refs["diagnostics"].text = ""
        output = result.get("output_dir") or self.output_dir
        if output:
            self.output_dir = Path(output)
        self._update_buttons()
        self._render_preview()

    @staticmethod
    def _format_metrics(metrics: Mapping[str, Any]) -> str:
        # Aceita o formato do avaliar_deteccoes e formatos de serviço menores.
        parts: list[str] = []
        thresholds = metrics.get("por_threshold", metrics.get("thresholds", metrics))
        if isinstance(thresholds, Mapping):
            for threshold, values in thresholds.items():
                if not isinstance(values, Mapping):
                    continue
                f1 = values.get("f1")
                precision = values.get("precisao", values.get("precision"))
                recall = values.get("revocacao", values.get("recall"))
                if f1 is not None or precision is not None:
                    parts.append(f"IoU {threshold}: precisão={_fmt(precision)} · recall={_fmt(recall)} · F1={_fmt(f1)}")
        return "\n".join(parts) or json.dumps(metrics, ensure_ascii=False, default=str)[:800]

    @staticmethod
    def _format_diagnostics(diagnostics: Mapping[str, Any]) -> str:
        """Resume a qualidade da segmentação sem despejar JSON na tela."""
        values = list(diagnostics.values())
        if not values:
            return ""
        valid = [item for item in values if isinstance(item, Mapping)]
        if not valid:
            return ""
        total_scans = len(valid)
        clusters = sum(int(item.get("clusters", 0) or 0) for item in valid)
        floor_points = sum(int(item.get("pontos_chao", 0) or 0) for item in valid)
        object_points = sum(int(item.get("pontos_objetos", 0) or 0) for item in valid)
        fallback = sum(1 for item in valid if item.get("metodo") == "fallback_z_horizontal")
        tilts = [float(item["inclinacao_deg"]) for item in valid if item.get("inclinacao_deg") is not None]
        floors = [float(item["z_chao"]) for item in valid if item.get("z_chao") is not None]
        parts = [
            f"Diagnóstico: {total_scans} scan(s) · {clusters} clusters · "
            f"piso={floor_points:,} pontos · objetos={object_points:,} pontos",
        ]
        if tilts:
            parts.append(f"inclinação média do plano: {sum(tilts) / len(tilts):.1f}°")
        if floors:
            parts.append(f"altura média do piso: {sum(floors) / len(floors):.3f} m")
        if fallback:
            parts.append(f"aviso: fallback horizontal usado em {fallback} scan(s)")
        return "\n".join(parts)

    def _render_preview(self) -> None:
        if self.dataset is None or self.dataset.vis_dir is None:
            return
        scan_id = self.selected_scan or (self.scan_files[0].stem if self.scan_files else "")
        candidates = [self.dataset.vis_dir / f"{scan_id}{ext}" for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp")]
        image = next((candidate for candidate in candidates if candidate.exists()), None)
        if image is not None:
            self.refs["preview"].source = image.as_uri()
            self.refs["preview"].set_visibility(True)

    def export_results(self) -> None:
        if not self.result or self.output_dir is None or self.dataset is None:
            return
        try:
            config = self._build_config()
            exported = self.adapter.export(self.result, config)
            self.result.update(exported)
            self._notify("Arquivos para o FlexSim exportados.")
        except Exception as exc:
            self._notify(f"Falha ao exportar: {exc}", error=True)

    def open_output_folder(self) -> None:
        if self.output_dir is None:
            return
        path = self.output_dir.expanduser().resolve()
        if not path.exists():
            self._notify("A pasta de saída ainda não existe.", error=True)
            return
        try:
            if hasattr(os, "startfile"):
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", str(path)])
            elif platform.system() == "Linux":
                subprocess.Popen(["xdg-open", str(path)])
            else:
                webbrowser.open(path.as_uri())
        except Exception as exc:
            self._notify(f"Não foi possível abrir a pasta: {exc}", error=True)

    def _update_buttons(self) -> None:
        running = self.future is not None and not self.future.done()
        if self.refs.get("start_button") is not None:
            (self.refs["start_button"].disable if running or not self.scan_files else self.refs["start_button"].enable)()
        if self.refs.get("cancel_button") is not None:
            (self.refs["cancel_button"].enable if running else self.refs["cancel_button"].disable)()
        has_result = self.result is not None
        if self.refs.get("export_button") is not None:
            (self.refs["export_button"].enable if has_result else self.refs["export_button"].disable)()
        if self.refs.get("open_button") is not None:
            (self.refs["open_button"].enable if self.output_dir is not None else self.refs["open_button"].disable)()

    @staticmethod
    def _notify(message: str, error: bool = False) -> None:
        if ui is None:
            return
        try:
            ui.notify(message, type="negative" if error else "positive")
        except Exception:
            pass


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def run_app() -> int:
    """Inicia a interface; retorna um código útil para scripts e testes."""

    if ui is None:
        print("NiceGUI não está instalado. Instale com 'python -m pip install nicegui' e execute app.py novamente.")
        return 2
    controller = GuiController()

    @ui.page("/")
    def index() -> None:
        controller.build()

    options: dict[str, Any] = {"title": "LiDAR → FlexSim", "reload": False}
    try:
        import webview  # type: ignore  # opcional, fornecido por requirements-desktop
        del webview
        native_available = True
    except ImportError:
        native_available = False

    if native_available:
        try:
            # O parâmetro native está disponível nas versões recentes do NiceGUI.
            ui.run(native=True, **options)
        except (TypeError, ImportError, ModuleNotFoundError, RuntimeError, OSError) as exc:
            print(f"Modo nativo indisponível ({exc}); abrindo no navegador.")
            ui.run(native=False, **options)
    else:
        print("Backend nativo não instalado; abrindo no navegador local.")
        ui.run(native=False, **options)
    return 0


def main() -> int:
    return run_app()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

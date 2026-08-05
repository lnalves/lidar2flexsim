"""Interface NiceGUI para treino, inferência e benchmark PointNet++."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from ml.classes import WAREHOUSE_CLASS_NAMES
from ml.gui import (
    DatasetInfo,
    ProcessResult,
    ProcessRunner,
    available_devices,
    build_benchmark_command,
    build_infer_command,
    build_train_command,
    discover_checkpoints,
    format_duration,
    run_dir_for_checkpoint,
    validate_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parent
CLASS_COLORS = {
    "Box": "#5eead4",
    "ELFplusplus": "#60a5fa",
    "CargoBike": "#fbbf24",
    "FTS": "#c084fc",
    "ForkLift": "#fb7185",
}
TRAINING_PRESETS = {
    # Os scans do Warehouse têm entre 3,5k e 9k pontos e menos de 7% deles caem
    # dentro de caixas. Amostrar 8192 preserva o scan quase inteiro; valores
    # menores descartam a maior parte dos pontos de objeto.
    "quick": {"epochs": 2, "batch": 2, "scans": 12, "points": 8192},
    "recommended": {"epochs": 15, "batch": 2, "scans": 300, "points": 8192},
    "complete": {"epochs": 20, "batch": 4, "scans": 0, "points": 8192},
}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _percent(value: Any) -> str:
    if value is None:
        return "—"
    return f"{100.0 * _float(value):.1f}%"


class PointNetApplication:
    def __init__(self) -> None:
        self.runner = ProcessRunner(PROJECT_ROOT)
        self.dataset: DatasetInfo | None = None
        self.devices = available_devices()
        self.refs: dict[str, Any] = {}
        self.operation_pending = False
        self.active_mode = ""
        self.expected_epochs = 1

    def build(self) -> None:
        from nicegui import ui

        ui.dark_mode().enable()
        ui.add_head_html(
            """
            <style>
              :root {
                --pn-bg: #07111f;
                --pn-panel: rgba(16, 31, 49, .88);
                --pn-border: rgba(148, 163, 184, .14);
                --pn-muted: #8ea3b8;
                --pn-teal: #5eead4;
              }
              body {
                background:
                  radial-gradient(circle at 8% -10%, rgba(45, 212, 191, .14), transparent 34rem),
                  radial-gradient(circle at 95% 10%, rgba(59, 130, 246, .11), transparent 32rem),
                  var(--pn-bg);
                color: #e5edf5;
              }
              .nicegui-content { padding: 0 !important; }
              .pn-shell { width: min(1120px, calc(100vw - 32px)); margin: 0 auto; }
              .pn-header {
                background: rgba(7, 17, 31, .78) !important;
                border-bottom: 1px solid var(--pn-border);
                backdrop-filter: blur(18px);
              }
              .pn-card {
                background: linear-gradient(145deg, rgba(19, 37, 57, .94), rgba(11, 24, 40, .94)) !important;
                border: 1px solid var(--pn-border);
                border-radius: 18px !important;
                box-shadow: 0 20px 60px rgba(0, 0, 0, .18);
              }
              .pn-soft {
                background: rgba(6, 16, 29, .45);
                border: 1px solid rgba(148, 163, 184, .10);
                border-radius: 14px;
              }
              .pn-kpi {
                min-width: 130px;
                background: rgba(6, 16, 29, .50);
                border: 1px solid rgba(148, 163, 184, .11);
                border-radius: 14px;
                padding: 14px 16px;
              }
              .pn-eyebrow { color: var(--pn-teal); letter-spacing: .14em; text-transform: uppercase; font-size: .72rem; font-weight: 700; }
              .pn-muted { color: var(--pn-muted); }
              .pn-title { letter-spacing: -.035em; }
              .pn-primary {
                min-height: 52px;
                font-size: 1rem;
                border-radius: 12px;
              }
              .q-expansion-item {
                border-radius: 14px;
                overflow: hidden;
              }
              .q-field--outlined .q-field__control:before { border-color: rgba(148, 163, 184, .22); }
              .q-table__container { background: transparent !important; box-shadow: none !important; }
              .q-table thead tr { color: #8ea3b8; }
              .q-table tbody td { border-color: rgba(148, 163, 184, .09); }
              @media (max-width: 800px) {
                .pn-shell { width: min(100vw - 20px, 1120px); }
                .pn-kpi { min-width: calc(50% - 8px); }
              }
            </style>
            """
        )

        with ui.header().classes("pn-header h-20 items-center"):
            with ui.row().classes("pn-shell items-center justify-between no-wrap"):
                with ui.row().classes("items-center gap-3 no-wrap"):
                    ui.icon("scatter_plot", size="36px").style("color: var(--pn-teal)")
                    with ui.column().classes("gap-0"):
                        ui.label("POINTNET LAB").classes("pn-eyebrow")
                        ui.label("Ambiente de percepção Warehouse").classes(
                            "text-lg font-semibold pn-title"
                        )
                with ui.row().classes("items-center gap-2"):
                    ui.badge("PointNet++", color="teal-5").props("outline")
                    self.refs["header_device"] = ui.badge(
                        self.devices.get("cpu", "CPU"), color="blue-grey-5"
                    ).props("outline")

        with ui.column().classes("pn-shell gap-5 py-6"):
            self._build_intro()
            self._build_dataset_bar()
            self._build_workspace()
            self._build_operation_bar()

        ui.timer(0.35, self._refresh_operation_status)
        self.refresh_checkpoints(show_notification=False)
        self.validate_dataset_path(show_notification=False)

    def _build_intro(self) -> None:
        from nicegui import ui

        with ui.row().classes("w-full items-center justify-between gap-4"):
            with ui.column().classes("gap-1"):
                ui.label("ANÁLISE POINTNET++").classes("pn-eyebrow")
                ui.label("Analise um scan em três passos.").classes(
                    "text-2xl md:text-3xl font-bold pn-title"
                )
                ui.label(
                    "Escolha os dados, o scan e o modelo. O restante já vem configurado."
                ).classes("pn-muted")
            ui.label("5 classes + background").classes(
                "pn-soft pn-muted px-3 py-2 text-xs"
            )

    def _build_dataset_bar(self) -> None:
        from nicegui import ui

        with ui.card().classes("pn-card w-full p-4"):
            with ui.row().classes("w-full items-center gap-3"):
                ui.icon("database", size="26px").style("color: var(--pn-teal)")
                self.refs["dataset_input"] = (
                    ui.input(
                        "Dataset",
                        value=str(PROJECT_ROOT / "dados" / "warehouse"),
                    )
                    .props("outlined dense")
                    .classes("grow min-w-64")
                )
                ui.button(
                    icon="folder_open",
                    on_click=lambda: self.choose_folder(self.refs["dataset_input"]),
                ).props("flat round").tooltip("Selecionar pasta")
                ui.button(
                    "Validar",
                    on_click=self.validate_dataset_path,
                ).props("flat color=teal-4")
            with ui.row().classes("w-full items-center gap-2 px-1"):
                self.refs["dataset_status"] = ui.label("Aguardando validação").classes(
                    "font-semibold"
                )
                ui.label("·").classes("pn-muted")
                self.refs["dataset_scans"] = ui.label("— scans").classes(
                    "pn-muted text-sm"
                )
                ui.label("·").classes("pn-muted")
                self.refs["dataset_labels"] = ui.label("— labels").classes(
                    "pn-muted text-sm"
                )
                self.refs["dataset_pairs"] = ui.label("").classes("hidden")
                ui.label("·").classes("pn-muted")
                self.refs["checkpoint_count"] = ui.label("— modelos").classes(
                    "pn-muted text-sm"
                )

    @staticmethod
    def _kpi(title: str, value: str) -> Any:
        from nicegui import ui

        with ui.column().classes("pn-kpi gap-0 grow"):
            ui.label(title).classes("pn-muted text-xs uppercase tracking-wider")
            label = ui.label(value).classes("text-xl font-bold")
        return label

    def _build_workspace(self) -> None:
        from nicegui import ui

        with ui.card().classes("pn-card w-full p-5"):
            self._build_infer_panel()

        with ui.expansion(
            "Ferramentas do modelo",
            icon="construction",
            caption="Treinamento e benchmark",
        ).classes("pn-card w-full"):
            with ui.column().classes("w-full gap-5 p-4"):
                with ui.expansion(
                    "Treinar um modelo",
                    icon="model_training",
                    caption="Use um preset ou ajuste os parâmetros",
                ).classes("pn-soft w-full"):
                    self._build_train_panel()
                with ui.expansion(
                    "Medir qualidade",
                    icon="query_stats",
                    caption="Gere métricas no conjunto de teste",
                ).classes("pn-soft w-full"):
                    self._build_benchmark_panel()

    def _device_select(self, key: str, value: str = "cpu") -> Any:
        from nicegui import ui

        control = (
            ui.select(self.devices, value=value, label="Dispositivo")
            .props("outlined dense")
            .classes("w-full")
        )
        self.refs[key] = control
        return control

    def _checkpoint_select(self, key: str) -> Any:
        from nicegui import ui

        control = (
            ui.select({}, label="Checkpoint")
            .props("outlined dense use-input")
            .classes("min-w-64")
        )
        self.refs[key] = control
        return control

    def _build_train_panel(self) -> None:
        from nicegui import ui

        with ui.column().classes("w-full gap-4 p-4"):
            ui.label("Escolha quanto tempo e dados quer dedicar ao treino.").classes(
                "pn-muted"
            )
            with ui.row().classes("w-full items-end gap-3"):
                self.refs["train_preset"] = (
                    ui.select(
                        {
                            "quick": "Teste rápido · 12 scans",
                            "recommended": "Recomendado · 300 scans",
                            "complete": "Completo · todos os scans",
                        },
                        value="recommended",
                        label="Perfil de treinamento",
                        on_change=lambda event: self._apply_training_preset(
                            event.value
                        ),
                    )
                    .props("outlined dense")
                    .classes("grow min-w-64")
                )
                self.refs["train_start"] = ui.button(
                    "Iniciar treinamento",
                    icon="play_arrow",
                    on_click=self.start_training,
                ).props("unelevated color=teal-6")

            with ui.expansion(
                "Configuração avançada",
                icon="tune",
                caption="Épocas, pontos, pastas e dispositivo",
            ).classes("w-full"):
                with ui.grid(columns=2).classes("w-full gap-3"):
                    self.refs["train_config"] = ui.input(
                        "Configuração", value="ml/configs/pointnet2_seg.yaml"
                    ).props("outlined dense")
                    self.refs["train_output"] = ui.input(
                        "Pasta de execuções", value="runs"
                    ).props("outlined dense")
                    self.refs["train_run_name"] = ui.input(
                        "Nome da execução", value=self._new_run_name()
                    ).props("outlined dense")
                    self.refs["train_resume"] = ui.input(
                        "Retomar (opcional)", placeholder="runs/... ou last.pt"
                    ).props("outlined dense")
                with ui.grid(columns=3).classes("w-full gap-3"):
                    self.refs["train_epochs"] = ui.number(
                        "Épocas", value=15, min=1, precision=0
                    ).props("outlined dense")
                    self.refs["train_batch"] = ui.number(
                        "Batch", value=2, min=1, precision=0
                    ).props("outlined dense")
                    self.refs["train_scans"] = ui.number(
                        "Máx. scans", value=300, min=2, precision=0
                    ).props("outlined dense")
                    self.refs["train_points"] = ui.number(
                        "Pontos/scan", value=8192, min=32, precision=0
                    ).props("outlined dense")
                    self.refs["train_weights"] = ui.input(
                        "Pesos de classe", value="auto"
                    ).props("outlined dense")
                    self._device_select("train_device")

            with ui.column().classes("w-full gap-3"):
                self.refs["train_progress"] = ui.linear_progress(
                    value=0, show_value=False
                ).props("rounded color=teal-5")
                self.refs["train_progress_text"] = ui.label(
                    "Pronto para iniciar."
                ).classes("pn-muted text-sm")
                with ui.row().classes("w-full gap-2"):
                    self.refs["train_epoch_kpi"] = self._kpi("Época", "—")
                    self.refs["train_miou_kpi"] = self._kpi("Val. mIoU", "—")

            with ui.expansion("Curva de aprendizado", icon="show_chart").classes(
                "w-full"
            ):
                self.refs["train_chart"] = ui.echart(
                    self._training_chart_options([])
                ).classes("w-full h-72")

    def _apply_training_preset(self, preset_name: str) -> None:
        preset = TRAINING_PRESETS.get(preset_name, TRAINING_PRESETS["recommended"])
        for key, field in (
            ("train_epochs", "epochs"),
            ("train_batch", "batch"),
            ("train_scans", "scans"),
            ("train_points", "points"),
        ):
            control = self.refs.get(key)
            if control is not None:
                control.value = preset[field]
                control.update()

    def _build_infer_panel(self) -> None:
        from nicegui import ui

        with ui.column().classes("w-full gap-4"):
            with ui.row().classes("w-full items-center justify-between gap-3"):
                with ui.column().classes("gap-0"):
                    ui.label("ANÁLISE").classes("pn-eyebrow")
                    ui.label("Scan e modelo").classes("text-xl font-bold pn-title")
                ui.badge("Configuração automática", color="teal-5").props("outline")

            with ui.row().classes("w-full items-end gap-3"):
                self.refs["infer_scan"] = (
                    ui.select({}, label="Scan")
                    .props("outlined dense use-input")
                    .classes("grow min-w-56")
                )
                self._checkpoint_select("infer_checkpoint").classes(
                    "grow min-w-64"
                )
                ui.button(
                    icon="refresh", on_click=self.refresh_checkpoints
                ).props("flat round").tooltip("Atualizar modelos")

            self.refs["infer_start"] = ui.button(
                "Analisar scan",
                icon="radar",
                on_click=self.start_inference,
            ).props("unelevated color=teal-6").classes("pn-primary w-full")

            with ui.expansion(
                "Ajustes avançados",
                icon="tune",
                caption="Confiança, agrupamento e calibração",
            ).classes("w-full"):
                with ui.grid(columns=3).classes("w-full gap-3 p-2"):
                    self._device_select("infer_device")
                    self.refs["infer_points"] = ui.number(
                        "Pontos/scan", value=8192, min=32, precision=0
                    ).props("outlined dense")
                    self.refs["infer_threshold"] = ui.number(
                        "Confiança", value=0.50, min=0, max=1, step=0.05
                    ).props("outlined dense")
                    self.refs["infer_eps"] = ui.number(
                        "Raio do cluster", value=0.35, min=0.01, step=0.05
                    ).props("outlined dense")
                    self.refs["infer_min_points"] = ui.number(
                        "Mín. pontos", value=5, min=1, precision=0
                    ).props("outlined dense")
                    self.refs["infer_calibration"] = ui.input(
                        "Calibração (opcional)", placeholder="calibration.json"
                    ).props("outlined dense")

            with ui.column().classes("pn-soft p-4 gap-3 w-full"):
                with ui.row().classes("w-full items-center gap-3"):
                    ui.label("Resultado").classes("font-semibold grow")
                    self.refs["infer_boxes_kpi"] = self._kpi("Caixas", "—")
                    self.refs["infer_object_points_kpi"] = self._kpi(
                        "Pontos objeto", "—"
                    )
                self.refs["infer_checkpoint_label"] = ui.label(
                    "Nenhum resultado ainda."
                ).classes("pn-muted text-sm break-all")

                self.refs["infer_table"] = ui.table(
                    columns=[
                        {"name": "classe", "label": "Classe", "field": "classe", "align": "left"},
                        {"name": "score", "label": "Conf.", "field": "score"},
                        {"name": "centro", "label": "Centro XYZ", "field": "centro"},
                        {"name": "dimensoes", "label": "Dimensões", "field": "dimensoes"},
                        {"name": "pontos", "label": "Pontos", "field": "pontos"},
                    ],
                    rows=[],
                    row_key="id",
                ).classes("w-full")
                self.refs["infer_table"].set_visibility(False)

            with ui.expansion(
                "Mapa e detalhes técnicos",
                icon="map",
                caption="Vista superior dos centros detectados",
            ).classes("w-full"):
                self.refs["infer_chart"] = ui.echart(
                    self._prediction_chart_options([])
                ).classes("w-full h-72")

    def _build_benchmark_panel(self) -> None:
        from nicegui import ui

        with ui.column().classes("w-full gap-4 p-4"):
            ui.label("Compare o checkpoint no conjunto de teste.").classes(
                "pn-muted"
            )
            with ui.row().classes("w-full items-end gap-3"):
                self._checkpoint_select("benchmark_checkpoint").classes("grow")
                self.refs["benchmark_scans"] = (
                    ui.select(
                        {3: "Rápido · 3 scans", 30: "Normal · 30 scans", 300: "Robusto · 300 scans"},
                        value=30,
                        label="Tamanho do teste",
                    )
                    .props("outlined dense")
                    .classes("min-w-56")
                )
                self.refs["benchmark_start"] = ui.button(
                    "Executar benchmark",
                    icon="query_stats",
                    on_click=self.start_benchmark,
                ).props("unelevated color=teal-6")

            with ui.expansion(
                "Configuração avançada",
                icon="tune",
                caption="Saída, manifesto, seed e dispositivo",
            ).classes("w-full"):
                with ui.grid(columns=2).classes("w-full gap-3 p-2"):
                    self.refs["benchmark_output"] = ui.input(
                        "Pasta do relatório", value="benchmark/gui-evaluation"
                    ).props("outlined dense")
                    self.refs["benchmark_manifest"] = ui.input(
                        "Manifesto existente (opcional)"
                    ).props("outlined dense")
                    self.refs["benchmark_seed"] = ui.number(
                        "Seed", value=42, min=0, precision=0
                    ).props("outlined dense")
                    self._device_select("benchmark_device")

            with ui.column().classes("pn-soft p-4 gap-3 w-full"):
                ui.label("Resultado de teste").classes("font-semibold")
                with ui.row().classes("w-full gap-2"):
                    self.refs["benchmark_accuracy"] = self._kpi("Acurácia", "—")
                    self.refs["benchmark_miou"] = self._kpi("mIoU", "—")
                    self.refs["benchmark_f1_25"] = self._kpi("Box F1 · .25", "—")
                    self.refs["benchmark_f1_50"] = self._kpi("Box F1 · .50", "—")

                self.refs["benchmark_table"] = ui.table(
                    columns=[
                        {"name": "class", "label": "Classe", "field": "class", "align": "left"},
                        {"name": "support", "label": "Suporte", "field": "support"},
                        {"name": "iou", "label": "IoU", "field": "iou"},
                        {"name": "precision", "label": "Precisão", "field": "precision"},
                        {"name": "recall", "label": "Recall", "field": "recall"},
                        {"name": "f1", "label": "F1", "field": "f1"},
                    ],
                    rows=[],
                    row_key="class",
                ).classes("w-full")
                self.refs["benchmark_table"].set_visibility(False)

    def _build_operation_bar(self) -> None:
        from nicegui import ui

        with ui.card().classes("pn-card w-full p-4"):
            with ui.row().classes("w-full items-center gap-4"):
                self.refs["operation_icon"] = ui.icon("check_circle", size="24px").style(
                    "color: var(--pn-teal)"
                )
                with ui.column().classes("gap-0 grow"):
                    self.refs["operation_status"] = ui.label(
                        "Interface pronta"
                    ).classes("font-semibold")
                    self.refs["operation_detail"] = ui.label(
                        "Nenhuma operação em execução."
                    ).classes("pn-muted text-sm")
                self.refs["cancel_button"] = ui.button(
                    "Cancelar", icon="stop", on_click=self.cancel_operation
                ).props("outline color=red-4")
                self.refs["cancel_button"].set_visibility(False)

    def validate_dataset_path(self, show_notification: bool = True) -> bool:
        from nicegui import ui

        try:
            info = validate_dataset(self.refs["dataset_input"].value)
        except (FileNotFoundError, ValueError) as exc:
            self.dataset = None
            self.refs["dataset_status"].set_text("Dataset inválido")
            self.refs["dataset_scans"].set_text("— scans")
            self.refs["dataset_labels"].set_text("— labels")
            self.refs["dataset_pairs"].set_text("")
            if show_notification:
                ui.notify(str(exc), type="negative")
            return False
        self.dataset = info
        pairs = len(info.scans) - len(info.missing_label_ids)
        self.refs["dataset_status"].set_text(info.root.name)
        scan_count = f"{len(info.scans):,}".replace(",", ".")
        label_count = f"{info.label_count:,}".replace(",", ".")
        self.refs["dataset_scans"].set_text(f"{scan_count} scans")
        self.refs["dataset_labels"].set_text(f"{label_count} labels")
        self.refs["dataset_pairs"].set_text(f"{pairs:,}".replace(",", "."))
        options = {str(path): path.stem for path in info.scans}
        self.refs["infer_scan"].options = options
        if self.refs["infer_scan"].value not in options:
            self.refs["infer_scan"].value = str(info.scans[0])
        self.refs["infer_scan"].update()
        if show_notification:
            warning = (
                f" · {len(info.missing_label_ids)} sem label"
                if info.missing_label_ids
                else ""
            )
            ui.notify(f"{len(info.scans)} scans validados{warning}", type="positive")
        return True

    def refresh_checkpoints(self, show_notification: bool = True) -> None:
        from nicegui import ui

        checkpoints = discover_checkpoints(PROJECT_ROOT)
        options = {
            str(path): str(path.relative_to(PROJECT_ROOT))
            for path in checkpoints
        }
        for key in ("infer_checkpoint", "benchmark_checkpoint"):
            control = self.refs.get(key)
            if control is None:
                continue
            previous = control.value
            control.options = options
            if previous not in options:
                # discover_checkpoints já ordena do mais recente para o mais
                # antigo, então o primeiro é o treino mais atual.
                control.value = next(iter(options), None)
            control.update()
        self.refs["checkpoint_count"].set_text(f"{len(checkpoints)} modelos")
        if show_notification:
            ui.notify(f"{len(checkpoints)} checkpoints disponíveis", type="info")

    async def choose_folder(self, input_control: Any) -> None:
        from nicegui import app as nicegui_app, ui

        window = getattr(nicegui_app.native, "main_window", None)
        if window is None:
            ui.notify(
                "No navegador, informe o caminho local no campo e clique em Validar.",
                type="info",
            )
            return
        try:
            import webview
        except ImportError:
            ui.notify("O seletor nativo requer pywebview.", type="warning")
            return
        selected = await window.create_file_dialog(
            dialog_type=webview.FileDialog.FOLDER,
            directory=str(PROJECT_ROOT),
        )
        if selected:
            input_control.value = str(selected[0])
            input_control.update()
            self.validate_dataset_path()

    def _ensure_ready(self) -> bool:
        from nicegui import ui

        if self.operation_pending or self.runner.snapshot().running:
            ui.notify("Aguarde ou cancele a operação atual.", type="warning")
            return False
        if self.dataset is None and not self.validate_dataset_path():
            return False
        return True

    async def _execute(
        self, mode: str, command: list[str]
    ) -> ProcessResult | None:
        from nicegui import run, ui

        self.operation_pending = True
        self.active_mode = mode
        self._set_action_buttons(False)
        self.refs["operation_status"].set_text(
            {
                "training": "Treinamento em execução",
                "inference": "Inferência em execução",
                "benchmark": "Benchmark em execução",
            }[mode]
        )
        self.refs["operation_detail"].set_text("Preparando o processo…")
        self.refs["operation_icon"].set_name("sync")
        self.refs["cancel_button"].set_visibility(True)
        try:
            result = await run.io_bound(self.runner.run, command)
        except Exception as exc:
            ui.notify(f"Falha ao iniciar: {exc}", type="negative")
            self.refs["operation_status"].set_text("Falha na execução")
            self.refs["operation_detail"].set_text(str(exc))
            return None
        finally:
            self.operation_pending = False
            self._set_action_buttons(True)
            self.refs["cancel_button"].set_visibility(False)
        if result.cancelled:
            self.refs["operation_icon"].set_name("cancel")
            self.refs["operation_status"].set_text("Operação cancelada")
            self.refs["operation_detail"].set_text(
                f"Interrompida após {format_duration(result.elapsed_seconds)}."
            )
            ui.notify("Operação cancelada.", type="warning")
        elif result.returncode != 0:
            detail = result.logs[-1] if result.logs else "Erro sem saída."
            self.refs["operation_icon"].set_name("error")
            self.refs["operation_status"].set_text("A operação falhou")
            self.refs["operation_detail"].set_text(detail)
            ui.notify(detail, type="negative", timeout=8000)
        else:
            self.refs["operation_icon"].set_name("check_circle")
            self.refs["operation_status"].set_text("Operação concluída")
            completion = "finalizada" if mode == "inference" else "finalizado"
            self.refs["operation_detail"].set_text(
                f"{self._mode_label(mode)} {completion} em "
                f"{format_duration(result.elapsed_seconds)}."
            )
            ui.notify("Operação concluída.", type="positive")
        return result

    def _set_action_buttons(self, enabled: bool) -> None:
        for key in ("train_start", "infer_start", "benchmark_start"):
            button = self.refs.get(key)
            if button is None:
                continue
            button.enable() if enabled else button.disable()

    @staticmethod
    def _mode_label(mode: str) -> str:
        return {
            "training": "Treinamento",
            "inference": "Inferência",
            "benchmark": "Benchmark",
        }.get(mode, mode)

    def _show_inference(self, payload: dict[str, Any], checkpoint: str) -> None:
        predictions = payload.get("predictions", [])
        diagnostics = payload.get("diagnostics", {})
        self.refs["infer_boxes_kpi"].set_text(str(len(predictions)))
        self.refs["infer_object_points_kpi"].set_text(
            f"{int(diagnostics.get('object_points', 0)):,}".replace(",", ".")
        )
        self.refs["infer_checkpoint_label"].set_text(
            f"{Path(checkpoint).name} · {diagnostics.get('device', '—')}"
        )
        rows = []
        for index, item in enumerate(predictions):
            center = [_float(value) for value in item.get("centro", (0, 0, 0))]
            dimensions = [
                _float(value) for value in item.get("dimensoes", (0, 0, 0))
            ]
            rows.append(
                {
                    "id": index,
                    "classe": item.get("classe", "—"),
                    "score": _percent(item.get("score")),
                    "centro": " · ".join(f"{value:.2f}" for value in center),
                    "dimensoes": " × ".join(f"{value:.2f}" for value in dimensions),
                    "pontos": int(item.get("num_pontos", 0)),
                }
            )
        self.refs["infer_table"].rows = rows
        self.refs["infer_table"].set_visibility(bool(rows))
        self.refs["infer_table"].update()
        self.refs["infer_chart"].options.clear()
        self.refs["infer_chart"].options.update(
            self._prediction_chart_options(predictions)
        )
        self.refs["infer_chart"].update()

    def _show_benchmark(self, payload: dict[str, Any]) -> None:
        test = ((payload.get("segmentation") or {}).get("test") or {})
        boxes = payload.get("boxes") or {}
        thresholds = boxes.get("por_threshold", {})
        self.refs["benchmark_accuracy"].set_text(_percent(test.get("accuracy")))
        self.refs["benchmark_miou"].set_text(_percent(test.get("miou")))
        self.refs["benchmark_f1_25"].set_text(
            _percent((thresholds.get("0.25") or {}).get("f1"))
        )
        self.refs["benchmark_f1_50"].set_text(
            _percent((thresholds.get("0.5") or {}).get("f1"))
        )
        rows = []
        supports = test.get("support_per_class", {})
        for class_name in WAREHOUSE_CLASS_NAMES:
            rows.append(
                {
                    "class": class_name,
                    "support": supports.get(class_name, 0),
                    "iou": _percent((test.get("iou_per_class") or {}).get(class_name)),
                    "precision": _percent(
                        (test.get("precision_per_class") or {}).get(class_name)
                    ),
                    "recall": _percent(
                        (test.get("recall_per_class") or {}).get(class_name)
                    ),
                    "f1": _percent((test.get("f1_per_class") or {}).get(class_name)),
                }
            )
        self.refs["benchmark_table"].rows = rows
        self.refs["benchmark_table"].set_visibility(True)
        self.refs["benchmark_table"].update()

    @staticmethod
    def _training_chart_options(records: list[dict[str, Any]]) -> dict[str, Any]:
        epochs = [int(item.get("epoch", 0)) for item in records]
        return {
            "backgroundColor": "transparent",
            "tooltip": {"trigger": "axis"},
            "legend": {"data": ["loss", "val_loss"], "textStyle": {"color": "#9fb1c4"}},
            "grid": {"left": 48, "right": 24, "top": 48, "bottom": 36},
            "xAxis": {
                "type": "category",
                "data": epochs,
                "axisLabel": {"color": "#8ea3b8"},
                "axisLine": {"lineStyle": {"color": "#334155"}},
            },
            "yAxis": {
                "type": "value",
                "axisLabel": {"color": "#8ea3b8"},
                "splitLine": {"lineStyle": {"color": "rgba(148,163,184,.10)"}},
            },
            "series": [
                {
                    "name": "loss",
                    "type": "line",
                    "smooth": True,
                    "showSymbol": False,
                    "lineStyle": {"color": "#5eead4", "width": 3},
                    "data": [_float(item.get("loss")) for item in records],
                },
                {
                    "name": "val_loss",
                    "type": "line",
                    "smooth": True,
                    "showSymbol": False,
                    "lineStyle": {"color": "#60a5fa", "width": 3},
                    "data": [_float(item.get("val_loss")) for item in records],
                },
            ],
        }

    @staticmethod
    def _prediction_chart_options(
        predictions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        series = []
        for class_name, color in CLASS_COLORS.items():
            values = []
            for item in predictions:
                if item.get("classe") != class_name:
                    continue
                center = item.get("centro", [0, 0, 0])
                dimensions = item.get("dimensoes", [1, 1, 1])
                values.append(
                    [
                        _float(center[0]),
                        _float(center[1]),
                        max(8.0, min(34.0, 8.0 + 4.0 * _float(dimensions[0], 1))),
                        _float(item.get("score")),
                    ]
                )
            series.append(
                {
                    "name": class_name,
                    "type": "scatter",
                    "data": values,
                    "symbolSize": 16,
                    "itemStyle": {"color": color, "opacity": 0.88},
                }
            )
        return {
            "backgroundColor": "transparent",
            "tooltip": {"trigger": "item"},
            "legend": {
                "data": list(CLASS_COLORS),
                "textStyle": {"color": "#9fb1c4"},
                "top": 0,
            },
            "grid": {"left": 48, "right": 24, "top": 48, "bottom": 42},
            "xAxis": {
                "name": "X (m)",
                "nameTextStyle": {"color": "#8ea3b8"},
                "axisLabel": {"color": "#8ea3b8"},
                "axisLine": {"lineStyle": {"color": "#334155"}},
                "splitLine": {"lineStyle": {"color": "rgba(148,163,184,.10)"}},
            },
            "yAxis": {
                "name": "Y (m)",
                "nameTextStyle": {"color": "#8ea3b8"},
                "axisLabel": {"color": "#8ea3b8"},
                "axisLine": {"lineStyle": {"color": "#334155"}},
                "splitLine": {"lineStyle": {"color": "rgba(148,163,184,.10)"}},
            },
            "series": series,
        }

    @staticmethod
    def _new_run_name() -> str:
        return f"gui-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def run_app(*, native: bool = False, port: int = 8080) -> None:
    try:
        from nicegui import ui
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "NiceGUI não está instalado. Execute: python -m pip install -r requirements.txt"
        ) from exc
    application = PointNetApplication()
    application.build()
    ui.run(
        title="PointNet Lab",
        reload=False,
        native=native,
        port=int(port),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Interface PointNet++")
    parser.add_argument("--native", action="store_true", help="abre em janela nativa")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)
    run_app(native=args.native, port=args.port)
    return 0


if __name__ in {"__main__", "__mp_main__"}:  # pragma: no cover
    main()

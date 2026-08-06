"""Testes dos construtores de painel do app NiceGUI.

Cada painel registra os controles que o resto da aplicação consome via
``self.refs``. Um painel que deixe de registrar uma chave quebra os handlers
com ``KeyError`` só em runtime, então o contrato testado aqui é justamente o
conjunto de refs que cada builder precisa publicar.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app import PROJECT_ROOT, TRAINING_PRESETS, PointNetApplication


def test_application_starts_without_dataset_or_pending_operation() -> None:
    application = PointNetApplication()

    assert application.dataset is None
    assert application.operation_pending is False
    assert application.active_mode == ""
    assert application.expected_epochs == 1
    assert application.refs == {}
    assert "cpu" in application.devices
    assert application.runner.cwd == PROJECT_ROOT


def test_dataset_bar_publishes_the_refs_the_validation_writes_to(
    ui_context: None,
) -> None:
    application = PointNetApplication()

    application._build_dataset_bar()

    for key in (
        "dataset_input",
        "dataset_status",
        "dataset_scans",
        "dataset_labels",
        "dataset_pairs",
        "checkpoint_count",
    ):
        assert key in application.refs, key
    assert application.refs["dataset_input"].value.endswith("dados/warehouse")


def test_train_panel_publishes_every_ref_the_train_command_reads(
    ui_context: None,
) -> None:
    application = PointNetApplication()

    application._build_train_panel()

    for key in (
        "train_output",
        "train_config",
        "train_run_name",
        "train_resume",
        "train_epochs",
        "train_batch",
        "train_scans",
        "train_points",
        "train_device",
        "train_weights",
        "train_start",
        "train_chart",
        "train_epoch_kpi",
        "train_miou_kpi",
        "train_progress",
        "train_progress_text",
    ):
        assert key in application.refs, key
    # Cada abertura do painel propõe um nome novo; runs com o mesmo nome
    # fazem o train_command abortar com FileExistsError.
    assert application.refs["train_run_name"].value.startswith("gui-")


def test_infer_panel_publishes_every_ref_the_infer_command_reads(
    ui_context: None,
) -> None:
    application = PointNetApplication()

    application._build_infer_panel()

    for key in (
        "infer_scan",
        "infer_checkpoint",
        "infer_device",
        "infer_points",
        "infer_threshold",
        "infer_eps",
        "infer_min_points",
        "infer_calibration",
        "infer_start",
    ):
        assert key in application.refs, key


def test_benchmark_panel_publishes_every_ref_the_benchmark_command_reads(
    ui_context: None,
) -> None:
    application = PointNetApplication()

    application._build_benchmark_panel()

    for key in (
        "benchmark_checkpoint",
        "benchmark_output",
        "benchmark_device",
        "benchmark_scans",
        "benchmark_seed",
        "benchmark_manifest",
        "benchmark_start",
    ):
        assert key in application.refs, key


def test_build_assembles_every_panel_and_the_operation_bar(
    ui_context: None,
) -> None:
    application = PointNetApplication()

    application.build()

    # build() é o único ponto que monta a árvore inteira; se algum painel
    # deixar de ser chamado, os handlers correspondentes perdem seus refs.
    for key in (
        "dataset_input",
        "train_start",
        "infer_start",
        "benchmark_start",
        "operation_status",
        "operation_detail",
        "operation_icon",
        "cancel_button",
    ):
        assert key in application.refs, key


def test_applying_a_preset_overwrites_the_training_inputs(
    ui_context: None,
) -> None:
    application = PointNetApplication()
    application._build_train_panel()

    application._apply_training_preset("complete")

    preset = TRAINING_PRESETS["complete"]
    assert int(application.refs["train_epochs"].value) == preset["epochs"]
    assert int(application.refs["train_batch"].value) == preset["batch"]
    assert int(application.refs["train_scans"].value) == preset["scans"]
    assert int(application.refs["train_points"].value) == preset["points"]


def test_validate_dataset_path_fills_the_counters_and_the_scan_options(
    ui_context: None, warehouse_dataset: Path
) -> None:
    application = PointNetApplication()
    application.build()
    application.refs["dataset_input"].value = str(warehouse_dataset)

    assert application.validate_dataset_path(show_notification=False) is True

    assert application.dataset is not None
    assert application.refs["dataset_status"].text == warehouse_dataset.name
    assert application.refs["dataset_scans"].text == "6 scans"
    assert application.refs["dataset_labels"].text == "6 labels"
    assert application.refs["dataset_pairs"].text == "6"
    assert len(application.refs["infer_scan"].options) == 6
    # Sem um scan pré-selecionado a inferência abriria com o campo vazio.
    assert application.refs["infer_scan"].value in application.refs["infer_scan"].options


def test_validate_dataset_path_resets_the_counters_on_an_invalid_root(
    ui_context: None, tmp_path: Path
) -> None:
    application = PointNetApplication()
    application.build()
    application.refs["dataset_input"].value = str(tmp_path / "inexistente")

    assert application.validate_dataset_path(show_notification=False) is False

    assert application.dataset is None
    assert application.refs["dataset_status"].text == "Dataset inválido"
    assert application.refs["dataset_scans"].text == "— scans"
    assert application.refs["dataset_labels"].text == "— labels"
    assert application.refs["dataset_pairs"].text == ""


def test_refresh_checkpoints_reports_how_many_models_exist(
    ui_context: None,
) -> None:
    application = PointNetApplication()
    application.build()

    application.refresh_checkpoints(show_notification=False)

    assert application.refs["checkpoint_count"].text.endswith("modelos")


def test_show_inference_fills_the_table_and_the_kpis(ui_context: None) -> None:
    application = PointNetApplication()
    application.build()

    application._show_inference(
        {
            "predictions": [
                {
                    "classe": "ForkLift",
                    "score": 0.91,
                    "num_pontos": 42,
                    "centro": [1.0, 2.0, 0.5],
                    "dimensoes": [2.0, 1.0, 1.5],
                }
            ],
            "diagnostics": {"object_points": 1234, "device": "cpu"},
        },
        checkpoint="runs/exemplo/checkpoints/best.pt",
    )

    assert application.refs["infer_boxes_kpi"].text == "1"
    assert application.refs["infer_object_points_kpi"].text == "1.234"
    assert application.refs["infer_checkpoint_label"].text == "best.pt · cpu"
    row = application.refs["infer_table"].rows[0]
    assert row["classe"] == "ForkLift"
    assert row["score"] == "91.0%"
    assert row["centro"] == "1.00 · 2.00 · 0.50"
    assert row["dimensoes"] == "2.00 × 1.00 × 1.50"
    assert row["pontos"] == 42


def test_show_inference_hides_the_table_when_nothing_was_detected(
    ui_context: None,
) -> None:
    application = PointNetApplication()
    application.build()

    application._show_inference({"predictions": [], "diagnostics": {}}, checkpoint="b.pt")

    assert application.refs["infer_boxes_kpi"].text == "0"
    assert application.refs["infer_table"].rows == []
    assert application.refs["infer_table"].visible is False


def test_show_benchmark_reports_one_row_per_class_including_absent_ones(
    ui_context: None,
) -> None:
    from ml.classes import WAREHOUSE_CLASS_NAMES

    application = PointNetApplication()
    application.build()

    application._show_benchmark(
        {
            "segmentation": {
                "test": {
                    "accuracy": 0.8,
                    "miou": 0.5,
                    "support_per_class": {"Box": 10},
                    "iou_per_class": {"Box": 0.5, "ForkLift": None},
                    "precision_per_class": {"Box": 0.6},
                    "recall_per_class": {"Box": 0.7},
                    "f1_per_class": {"Box": 0.65},
                }
            },
            "boxes": {"por_threshold": {"0.25": {"f1": 0.4}, "0.5": {"f1": 0.2}}},
        }
    )

    assert application.refs["benchmark_accuracy"].text == "80.0%"
    assert application.refs["benchmark_miou"].text == "50.0%"
    assert application.refs["benchmark_f1_25"].text == "40.0%"
    assert application.refs["benchmark_f1_50"].text == "20.0%"
    rows = application.refs["benchmark_table"].rows
    assert [row["class"] for row in rows] == list(WAREHOUSE_CLASS_NAMES)
    box = next(row for row in rows if row["class"] == "Box")
    assert box["support"] == 10
    assert box["iou"] == "50.0%"
    # Uma classe sem suporte precisa aparecer como "—" e não como 0%, senão o
    # relatório sugere que o modelo errou onde não havia nada para acertar.
    forklift = next(row for row in rows if row["class"] == "ForkLift")
    assert forklift["iou"] == "—"
    assert forklift["support"] == 0


def test_show_benchmark_survives_an_empty_payload(ui_context: None) -> None:
    application = PointNetApplication()
    application.build()

    application._show_benchmark({})

    assert application.refs["benchmark_accuracy"].text == "—"
    assert application.refs["benchmark_table"].rows


def _snapshot(**overrides: object) -> object:
    from ml.gui import ProcessSnapshot

    defaults = {
        "running": True,
        "cancelled": False,
        "returncode": None,
        "elapsed_seconds": 65.0,
        "line_count": 3,
        "last_line": "",
        "events": (),
    }
    defaults.update(overrides)
    return ProcessSnapshot(**defaults)  # type: ignore[arg-type]


def test_refresh_operation_status_advances_the_training_progress(
    ui_context: None,
) -> None:
    application = PointNetApplication()
    application.build()
    application.active_mode = "training"
    application.expected_epochs = 4
    application.runner.snapshot = lambda: _snapshot(  # type: ignore[method-assign]
        events=({"epoch": 2, "loss": 0.5, "val_miou": 0.42},)
    )

    application._refresh_operation_status()

    assert application.refs["operation_detail"].text == "1m 05s · 3 atualizações"
    assert application.refs["train_progress"].value == pytest.approx(0.5)
    assert application.refs["train_epoch_kpi"].text == "2"
    assert application.refs["train_miou_kpi"].text == "42.0%"
    assert "Época 2/4" in application.refs["train_progress_text"].text


def test_refresh_operation_status_is_a_noop_when_nothing_is_running(
    ui_context: None,
) -> None:
    application = PointNetApplication()
    application.build()
    application.refs["operation_detail"].set_text("intocado")
    application.runner.snapshot = lambda: _snapshot(running=False)  # type: ignore[method-assign]

    application._refresh_operation_status()

    assert application.refs["operation_detail"].text == "intocado"


def test_cancel_operation_reports_progress_only_when_a_process_was_killed(
    ui_context: None,
) -> None:
    application = PointNetApplication()
    application.build()
    application.runner.cancel = lambda: True  # type: ignore[method-assign]

    application.cancel_operation()

    assert application.refs["operation_status"].text == "Cancelando…"

    application.refs["operation_status"].set_text("intocado")
    application.runner.cancel = lambda: False  # type: ignore[method-assign]

    application.cancel_operation()

    # Sem processo vivo o texto não pode mudar, senão a barra mente sobre o
    # estado da operação.
    assert application.refs["operation_status"].text == "intocado"


@pytest.mark.asyncio
async def test_execute_reports_success_and_re_enables_the_action_buttons(
    ui_client: object,
) -> None:
    import sys

    with ui_client:
        application = PointNetApplication()
        application.build()

        result = await application._execute(
            "inference",
            [sys.executable, "-c", "import json; print(json.dumps({'ok': True}))"],
        )

        assert result is not None
        assert result.returncode == 0
        assert result.payload == {"ok": True}
        assert application.refs["operation_status"].text == "Operação concluída"
        assert application.refs["operation_icon"].name == "check_circle"
        assert "Inferência finalizada" in application.refs["operation_detail"].text
        # O finally precisa liberar a UI mesmo no caminho feliz.
        assert application.operation_pending is False
        assert application.refs["cancel_button"].visible is False
        assert application.refs["infer_start"].enabled is True


@pytest.mark.asyncio
async def test_execute_surfaces_the_last_log_line_when_the_process_fails(
    ui_client: object,
) -> None:
    import sys

    with ui_client:
        application = PointNetApplication()
        application.build()

        result = await application._execute(
            "training",
            [sys.executable, "-c", "import sys; print('boom detalhado'); sys.exit(3)"],
        )

        assert result is not None
        assert result.returncode == 3
        assert application.refs["operation_status"].text == "A operação falhou"
        assert application.refs["operation_icon"].name == "error"
        assert application.refs["operation_detail"].text == "boom detalhado"
        assert application.operation_pending is False
        assert application.refs["train_start"].enabled is True


@pytest.mark.asyncio
async def test_execute_releases_the_ui_when_the_process_cannot_start(
    ui_client: object,
) -> None:
    with ui_client:
        application = PointNetApplication()
        application.build()

        result = await application._execute("benchmark", ["/binario/que/nao/existe"])

        assert result is None
        assert application.refs["operation_status"].text == "Falha na execução"
        # Sem o finally a aplicação ficaria travada com os botões desabilitados.
        assert application.operation_pending is False
        assert application.refs["benchmark_start"].enabled is True
        assert application.refs["cancel_button"].visible is False


def _capture_execute(application: PointNetApplication, result: object) -> list[list[str]]:
    """Substitui _execute e devolve a lista de comandos disparados."""

    captured: list[list[str]] = []

    async def fake_execute(mode: str, command: list[str]) -> object:
        captured.append(command)
        return result

    application._execute = fake_execute  # type: ignore[method-assign]
    return captured


def _result(**overrides: object) -> object:
    from ml.gui import ProcessResult

    defaults = {
        "command": (),
        "returncode": 0,
        "cancelled": False,
        "payload": None,
        "events": (),
        "logs": (),
        "elapsed_seconds": 1.0,
    }
    defaults.update(overrides)
    return ProcessResult(**defaults)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_start_training_builds_the_command_and_refreshes_the_panel(
    ui_client: object, warehouse_dataset: Path
) -> None:
    with ui_client:
        application = PointNetApplication()
        application.build()
        application.refs["dataset_input"].value = str(warehouse_dataset)
        application.validate_dataset_path(show_notification=False)
        application.refs["train_epochs"].value = 3
        application.refs["train_run_name"].value = "run-antigo"
        commands = _capture_execute(
            application,
            _result(events=({"epoch": 3, "loss": 0.1, "val_miou": 0.75},)),
        )

        await application.start_training()

        assert len(commands) == 1
        command = commands[0]
        assert command[1:4] == ["-m", "ml.cli", "train"]
        assert command[command.index("--epochs") + 1] == "3"
        assert command[command.index("--dataset") + 1] == str(warehouse_dataset)
        assert application.expected_epochs == 3
        assert application.refs["train_epoch_kpi"].text == "3"
        assert application.refs["train_miou_kpi"].text == "75.0%"
        # Manter o nome antigo faria o próximo treino abortar com FileExistsError.
        assert application.refs["train_run_name"].value != "run-antigo"


@pytest.mark.asyncio
async def test_start_training_refuses_a_non_positive_epoch_count(
    ui_client: object, warehouse_dataset: Path
) -> None:
    with ui_client:
        application = PointNetApplication()
        application.build()
        application.refs["dataset_input"].value = str(warehouse_dataset)
        application.validate_dataset_path(show_notification=False)
        application.refs["train_epochs"].value = 0
        commands = _capture_execute(application, _result())

        await application.start_training()

        assert commands == []


@pytest.mark.asyncio
async def test_start_training_does_nothing_without_a_valid_dataset(
    ui_client: object, tmp_path: Path
) -> None:
    with ui_client:
        application = PointNetApplication()
        application.build()
        # build() já valida o dataset padrão; zerar força _ensure_ready a
        # revalidar o caminho digitado, que é o cenário sob teste.
        application.dataset = None
        application.refs["dataset_input"].value = str(tmp_path / "inexistente")
        commands = _capture_execute(application, _result())

        await application.start_training()

        assert commands == []


@pytest.mark.asyncio
async def test_start_inference_renders_the_payload_it_receives(
    ui_client: object, warehouse_dataset: Path
) -> None:
    with ui_client:
        application = PointNetApplication()
        application.build()
        application.refs["dataset_input"].value = str(warehouse_dataset)
        application.validate_dataset_path(show_notification=False)
        # ui.select ignora um value fora de options, então o checkpoint só
        # "existe" para a UI depois de entrar na lista.
        checkpoint = "runs/x/checkpoints/best.pt"
        application.refs["infer_checkpoint"].options = {checkpoint: "best.pt"}
        application.refs["infer_checkpoint"].value = checkpoint
        commands = _capture_execute(
            application,
            _result(
                payload={
                    "predictions": [
                        {
                            "classe": "Box",
                            "score": 0.5,
                            "num_pontos": 9,
                            "centro": [0.0, 0.0, 0.0],
                            "dimensoes": [1.0, 1.0, 1.0],
                        }
                    ],
                    "diagnostics": {"object_points": 9, "device": "cpu"},
                }
            ),
        )

        await application.start_inference()

        assert commands[0][1:4] == ["-m", "ml.cli", "infer"]
        assert application.refs["infer_boxes_kpi"].text == "1"
        assert len(application.refs["infer_table"].rows) == 1


@pytest.mark.asyncio
async def test_start_inference_requires_both_a_scan_and_a_checkpoint(
    ui_client: object, warehouse_dataset: Path
) -> None:
    with ui_client:
        application = PointNetApplication()
        application.build()
        application.refs["dataset_input"].value = str(warehouse_dataset)
        application.validate_dataset_path(show_notification=False)
        application.refs["infer_checkpoint"].value = None
        commands = _capture_execute(application, _result())

        await application.start_inference()

        assert commands == []


@pytest.mark.asyncio
async def test_start_benchmark_reuses_the_split_of_the_run_that_owns_the_checkpoint(
    ui_client: object, warehouse_dataset: Path, tmp_path: Path
) -> None:
    run = tmp_path / "runs" / "exemplo"
    (run / "checkpoints").mkdir(parents=True)
    checkpoint = run / "checkpoints" / "best.pt"
    checkpoint.write_bytes(b"")
    (run / "metadata.json").write_text("{}", encoding="utf-8")

    with ui_client:
        application = PointNetApplication()
        application.build()
        application.refs["dataset_input"].value = str(warehouse_dataset)
        application.validate_dataset_path(show_notification=False)
        application.refs["benchmark_checkpoint"].options = {str(checkpoint): "best.pt"}
        application.refs["benchmark_checkpoint"].value = str(checkpoint)
        application.refs["benchmark_scans"].value = 25
        commands = _capture_execute(
            application,
            _result(payload={"segmentation": {"test": {"accuracy": 0.9}}}),
        )

        await application.start_benchmark()

        command = commands[0]
        assert command[1:4] == ["-m", "ml.cli", "benchmark"]
        assert command[command.index("--from-run") + 1] == str(run)
        # Com um split gravado, recortar por --max-scans inventaria outro
        # conjunto de teste e invalidaria a comparação com o treino.
        assert "--max-scans" not in command
        assert application.refs["benchmark_accuracy"].text == "90.0%"


@pytest.mark.asyncio
async def test_start_benchmark_requires_a_checkpoint(
    ui_client: object, warehouse_dataset: Path
) -> None:
    with ui_client:
        application = PointNetApplication()
        application.build()
        application.refs["dataset_input"].value = str(warehouse_dataset)
        application.validate_dataset_path(show_notification=False)
        application.refs["benchmark_checkpoint"].value = None
        commands = _capture_execute(application, _result())

        await application.start_benchmark()

        assert commands == []

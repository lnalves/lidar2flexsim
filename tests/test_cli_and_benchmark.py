"""Parser da CLI e benchmark completo com um checkpoint real."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.cli import build_parser


def test_parser_binds_a_handler_to_every_subcommand() -> None:
    parser = build_parser()

    for command in ("train", "infer", "benchmark"):
        namespace = parser.parse_args(_minimal_args(command))
        # O handler só é alcançado via set_defaults; sem ele main() estoura
        # com AttributeError em runtime.
        assert callable(namespace.handler)
        assert namespace.command == command


def _minimal_args(command: str) -> list[str]:
    return {
        "train": ["train", "--dataset", "d"],
        "infer": ["infer", "--scan", "s.bin", "--checkpoint", "c.pt"],
        "benchmark": ["benchmark", "--dataset", "d"],
    }[command]


def test_parser_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


@pytest.mark.parametrize(
    ("command", "argv"),
    [
        ("train", ["train"]),
        ("infer", ["infer", "--scan", "s.bin"]),
        ("infer", ["infer", "--checkpoint", "c.pt"]),
        ("benchmark", ["benchmark"]),
    ],
)
def test_parser_rejects_a_subcommand_missing_its_required_flags(
    command: str, argv: list[str]
) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)


def test_train_defaults_match_the_documented_behaviour() -> None:
    args = build_parser().parse_args(["train", "--dataset", "dados/warehouse"])

    assert args.output == "runs"
    assert args.device is None
    assert args.epochs is None
    assert args.run_name is None
    assert args.resume is None


def test_infer_and_benchmark_defaults_match_the_documented_behaviour() -> None:
    infer = build_parser().parse_args(
        ["infer", "--scan", "s.bin", "--checkpoint", "c.pt"]
    )
    benchmark = build_parser().parse_args(["benchmark", "--dataset", "d"])

    assert infer.device == "cpu"
    assert infer.score_threshold == pytest.approx(0.5)
    assert infer.cluster_eps == pytest.approx(0.35)
    assert infer.min_cluster_points == 5
    assert benchmark.seed == 42
    assert benchmark.output == "benchmark"


def test_main_turns_a_domain_error_into_a_usage_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from ml.cli import main

    with pytest.raises(SystemExit):
        main(["train", "--dataset", "/pasta/que/nao/existe"])

    assert "error" in capsys.readouterr().err.lower()


def _config_file(root: Path) -> Path:
    path = root / "config.json"
    path.write_text(
        json.dumps({
            "model": {
                "input_points": 8,
                "sa1_points": 4,
                "sa2_points": 2,
                "neighbors": 2,
                "hidden_channels": 8,
            },
            "training": {"epochs": 1, "batch_size": 1, "device": "cpu"},
        }),
        encoding="utf-8",
    )
    return path


def test_train_command_writes_config_metadata_and_checkpoints(
    warehouse_dataset: Path, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    from ml.cli import main

    runs = tmp_path / "runs"

    code = main([
        "train",
        "--dataset", str(warehouse_dataset),
        "--config", str(_config_file(tmp_path)),
        "--output", str(runs),
        "--run-name", "exec1",
    ])

    run_dir = runs / "exec1"
    assert code == 0
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "metadata.json").is_file()
    assert (run_dir / "history.jsonl").is_file()
    assert list((run_dir / "checkpoints").glob("*.pt"))


def test_train_command_refuses_to_overwrite_an_existing_run(
    warehouse_dataset: Path, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    from ml.cli import main

    runs = tmp_path / "runs"
    (runs / "exec1").mkdir(parents=True)

    # Reusar o diretório misturaria histórico e checkpoints de dois treinos.
    with pytest.raises(SystemExit):
        main([
            "train",
            "--dataset", str(warehouse_dataset),
            "--config", str(_config_file(tmp_path)),
            "--output", str(runs),
            "--run-name", "exec1",
        ])


def test_train_command_applies_the_command_line_overrides(
    warehouse_dataset: Path, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    from ml.cli import main

    runs = tmp_path / "runs"

    code = main([
        "train",
        "--dataset", str(warehouse_dataset),
        "--config", str(_config_file(tmp_path)),
        "--output", str(runs),
        "--run-name", "override",
        "--input-points", "16",
        "--epochs", "2",
        "--device", "cpu",
    ])

    config = json.loads((runs / "override" / "config.json").read_text(encoding="utf-8"))
    history = (runs / "override" / "history.jsonl").read_text(encoding="utf-8")
    assert code == 0
    assert config["model"]["input_points"] == 16
    assert history.count("\n") == 2


def test_train_command_rejects_malformed_class_weights(
    warehouse_dataset: Path, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    from ml.cli import main

    with pytest.raises(SystemExit):
        main([
            "train",
            "--dataset", str(warehouse_dataset),
            "--config", str(_config_file(tmp_path)),
            "--output", str(tmp_path / "runs"),
            "--run-name", "pesos",
            "--class-weights", "1,2",
        ])


def test_train_command_derives_automatic_class_weights(
    warehouse_dataset: Path, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    from ml.cli import main

    runs = tmp_path / "runs"

    code = main([
        "train",
        "--dataset", str(warehouse_dataset),
        "--config", str(_config_file(tmp_path)),
        "--output", str(runs),
        "--run-name", "auto",
        "--class-weights", "auto",
    ])

    metadata = json.loads((runs / "auto" / "metadata.json").read_text(encoding="utf-8"))
    weights = metadata["class_weights"]
    assert code == 0
    assert len(weights) == 6
    assert all(weight > 0 for weight in weights)


def test_train_command_resumes_reusing_the_recorded_split(
    warehouse_dataset: Path, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    from ml.cli import main

    runs = tmp_path / "runs"
    config = _config_file(tmp_path)
    main([
        "train",
        "--dataset", str(warehouse_dataset),
        "--config", str(config),
        "--output", str(runs),
        "--run-name", "resumo",
    ])
    first = json.loads((runs / "resumo" / "metadata.json").read_text(encoding="utf-8"))

    code = main([
        "train",
        "--dataset", str(warehouse_dataset),
        "--config", str(config),
        "--output", str(runs),
        "--epochs", "2",
        "--resume", str(runs / "resumo"),
    ])
    second = json.loads((runs / "resumo" / "metadata.json").read_text(encoding="utf-8"))

    assert code == 0
    # Recalcular o split ao retomar contaminaria a validação com scans já vistos.
    assert second["train_scan_ids"] == first["train_scan_ids"]
    assert second["test_scan_ids"] == first["test_scan_ids"]


def _trained_run(dataset: Path, tmp_path: Path) -> Path:
    from ml.cli import main

    runs = tmp_path / "runs"
    main([
        "train",
        "--dataset", str(dataset),
        "--config", str(_config_file(tmp_path)),
        "--output", str(runs),
        "--run-name", "bench",
    ])
    return runs / "bench"


def test_run_benchmark_without_a_checkpoint_only_records_the_split(
    warehouse_dataset: Path, tmp_path: Path
) -> None:
    from ml.benchmark import run_benchmark

    report = run_benchmark(
        warehouse_dataset, output_dir=tmp_path / "saida", seed=7, max_scans=4
    )

    assert report["segmentation"] is None
    assert report["boxes"] is None
    assert report["baseline"] is None
    assert report["manifest"]["max_scans"] == 4
    assert (tmp_path / "saida" / "manifest.json").is_file()
    assert (tmp_path / "saida" / "benchmark.json").is_file()


def test_run_benchmark_with_a_checkpoint_scores_every_split(
    warehouse_dataset: Path, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    from ml.benchmark import run_benchmark

    run_dir = _trained_run(warehouse_dataset, tmp_path)
    checkpoint = next((run_dir / "checkpoints").glob("*.pt"))

    report = run_benchmark(
        warehouse_dataset,
        checkpoint=checkpoint,
        from_run=run_dir,
        output_dir=tmp_path / "saida",
    )

    segmentation = report["segmentation"]
    assert segmentation is not None
    assert {"train", "validation", "test"} <= set(segmentation)
    assert segmentation["test"]["miou"] is not None
    assert segmentation["elapsed_seconds"] >= 0.0
    assert report["baseline"]["format_version"] == 2
    assert report["boxes"] is not None


def test_run_benchmark_from_run_reuses_the_split_the_checkpoint_trained_on(
    warehouse_dataset: Path, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    from ml.benchmark import run_benchmark

    run_dir = _trained_run(warehouse_dataset, tmp_path)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))

    report = run_benchmark(
        warehouse_dataset,
        checkpoint=next((run_dir / "checkpoints").glob("*.pt")),
        from_run=run_dir,
    )

    assert report["manifest"]["test_scan_ids"] == metadata["test_scan_ids"]
    assert report["manifest"]["train_scan_ids"] == metadata["train_scan_ids"]


def test_run_benchmark_rejects_manifest_and_from_run_together(
    warehouse_dataset: Path, tmp_path: Path
) -> None:
    from ml.benchmark import run_benchmark

    with pytest.raises(ValueError, match="não os dois"):
        run_benchmark(
            warehouse_dataset,
            manifest={"train_scan_ids": [], "validation_scan_ids": [], "test_scan_ids": []},
            from_run=tmp_path,
        )


def test_run_benchmark_accepts_an_inline_manifest(warehouse_dataset: Path) -> None:
    from ml.benchmark import run_benchmark

    report = run_benchmark(
        warehouse_dataset,
        manifest={
            "train_scan_ids": ["000000"],
            "validation_scan_ids": ["000001"],
            "test_scan_ids": ["000002"],
        },
    )

    assert report["manifest"]["test_scan_ids"] == ["000002"]

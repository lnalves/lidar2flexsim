"""Contratos leves da interface PointNet++, sem iniciar servidor web."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from app import TRAINING_PRESETS, PointNetApplication
from ml.gui import (
    ProcessRunner,
    build_benchmark_command,
    build_infer_command,
    build_train_command,
    discover_checkpoints,
    format_duration,
    validate_dataset,
)


def _dataset(root: Path) -> Path:
    bin_dir = root / "bin"
    label_dir = root / "label"
    bin_dir.mkdir(parents=True)
    label_dir.mkdir()
    points = np.zeros((2, 4), dtype=np.float32)
    for scan_id in ("000002", "000001"):
        (bin_dir / f"{scan_id}.bin").write_bytes(points.tobytes())
    (label_dir / "000001.txt").write_text(
        "Box 0 0 0 1 1 1 0\n", encoding="utf-8"
    )
    return root


def test_validate_dataset_reports_scans_labels_and_missing_pairs(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "warehouse")

    info = validate_dataset(root)

    assert info.root == root
    assert [item.stem for item in info.scans] == ["000001", "000002"]
    assert info.label_count == 1
    assert info.missing_label_ids == ("000002",)
    assert info.ready is True


def test_validate_dataset_accepts_bin_directory(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "warehouse")

    info = validate_dataset(root / "bin")

    assert info.root == root
    assert info.bin_dir == root / "bin"


def test_discover_checkpoints_prefers_best_and_last_inside_runs(tmp_path: Path) -> None:
    legacy = tmp_path / "checkpoints"
    run = tmp_path / "runs" / "experiment" / "checkpoints"
    legacy.mkdir()
    run.mkdir(parents=True)
    for path in (
        legacy / "pointnet2_epoch_0001.pt",
        run / "best.pt",
        run / "last.pt",
        run / "pointnet2_epoch_0001.pt",
    ):
        path.write_bytes(b"checkpoint")

    values = {path.name for path in discover_checkpoints(tmp_path)}

    assert values == {"pointnet2_epoch_0001.pt", "best.pt", "last.pt"}
    assert run / "pointnet2_epoch_0001.pt" not in discover_checkpoints(tmp_path)


def test_command_builders_match_the_public_cli() -> None:
    train = build_train_command(
        dataset="dados/warehouse",
        output="runs",
        config="ml/configs/pointnet2_seg.yaml",
        run_name="gui-test",
        resume=None,
        epochs=2,
        batch_size=1,
        max_scans=12,
        input_points=1024,
        device="cpu",
        class_weights="auto",
        python="python",
    )
    infer = build_infer_command(
        scan="dados/warehouse/bin/000000.bin",
        checkpoint="runs/test/checkpoints/best.pt",
        device="cpu",
        num_points=1024,
        score_threshold=0.5,
        cluster_eps=0.35,
        min_cluster_points=5,
        python="python",
    )
    benchmark = build_benchmark_command(
        dataset="dados/warehouse",
        checkpoint="runs/test/checkpoints/best.pt",
        output="benchmark/test",
        device="cpu",
        max_scans=30,
        python="python",
    )

    assert train[:5] == ["python", "-m", "ml.cli", "train", "--dataset"]
    assert train[train.index("--max-scans") + 1] == "12"
    assert infer[:4] == ["python", "-m", "ml.cli", "infer"]
    assert infer[infer.index("--score-threshold") + 1] == "0.5"
    assert benchmark[:4] == ["python", "-m", "ml.cli", "benchmark"]


def test_process_runner_captures_json_payload(tmp_path: Path) -> None:
    runner = ProcessRunner(tmp_path)

    result = runner.run(
        [
            sys.executable,
            "-c",
            "import json; print(json.dumps({'epoch': 1, 'loss': 0.25}))",
        ]
    )

    assert result.returncode == 0
    assert result.payload == {"epoch": 1, "loss": 0.25}
    assert runner.snapshot().running is False


def test_chart_contracts_accept_empty_and_populated_results() -> None:
    training = PointNetApplication._training_chart_options(
        [{"epoch": 1, "loss": 1.0, "val_loss": 1.2}]
    )
    predictions = PointNetApplication._prediction_chart_options(
        [
            {
                "classe": "ForkLift",
                "centro": [1.0, 2.0, 0.5],
                "dimensoes": [2.0, 1.0, 1.5],
                "score": 0.9,
            }
        ]
    )

    assert training["xAxis"]["data"] == [1]
    forklift = next(
        item for item in predictions["series"] if item["name"] == "ForkLift"
    )
    assert forklift["data"][0][:2] == [1.0, 2.0]
    assert format_duration(65) == "1m 05s"


def test_training_presets_keep_the_primary_flow_concise() -> None:
    assert TRAINING_PRESETS["quick"]["scans"] == 12
    assert TRAINING_PRESETS["recommended"] == {
        "epochs": 15,
        "batch": 2,
        "scans": 300,
        "points": 1024,
    }
    assert TRAINING_PRESETS["complete"]["scans"] == 0

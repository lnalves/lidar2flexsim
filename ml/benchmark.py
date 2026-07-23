"""Benchmark reproduzível para segmentação e caixas do Warehouse.

O benchmark separa três artefatos: manifesto congelado dos scans, métricas de
segmentação ponto a ponto e métricas geométricas de caixas. Assim uma mudança
no clustering não fica mascarada por uma mudança no classificador.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .checkpoints import load_checkpoint
from .config import PointNet2Config
from .data import WarehouseSegmentationDataset
from .dependencies import require_torch
from .models.pointnet2_seg import PointNet2Segmentation
from .training import _confusion_from_batch, _metrics_from_confusion, _set_seed


def _scan_files(dataset: str | Path) -> tuple[Path, Path, list[Path]]:
    root = Path(dataset).expanduser()
    bin_dir = root if root.name.casefold() in {"bin", "bins"} else root / "bin"
    dataset_root = bin_dir.parent if bin_dir.name.casefold() in {"bin", "bins"} else root
    scans = sorted(bin_dir.glob("*.bin"), key=lambda path: path.stem)
    if not scans:
        raise FileNotFoundError(f"Nenhum scan .bin encontrado em {bin_dir}")
    return dataset_root, dataset_root / "label", scans


def build_benchmark_manifest(
    scans: Sequence[str | Path],
    *,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.1,
    seed: int = 42,
    max_scans: int | None = None,
) -> dict[str, Any]:
    """Cria splits temporais determinísticos e serializáveis."""

    ordered = sorted((Path(scan) for scan in scans), key=lambda path: path.stem)
    if max_scans is not None:
        if int(max_scans) < 3:
            raise ValueError("max_scans deve permitir pelo menos treino, validação e teste")
        ordered = ordered[: int(max_scans)]
    if len(ordered) < 3:
        raise ValueError("O benchmark requer pelo menos três scans")
    validation_count = max(1, int(round(len(ordered) * float(validation_fraction))))
    test_count = max(1, int(round(len(ordered) * float(test_fraction))))
    if validation_count + test_count >= len(ordered):
        validation_count = 1
        test_count = 1
    train_end = len(ordered) - validation_count - test_count
    return {
        "format": "lidar2flexsim-benchmark-v1",
        "seed": int(seed),
        "split_strategy": "temporal_sorted_stem",
        "validation_fraction": float(validation_fraction),
        "test_fraction": float(test_fraction),
        "all_scan_ids": [path.stem for path in ordered],
        "train_scan_ids": [path.stem for path in ordered[:train_end]],
        "validation_scan_ids": [path.stem for path in ordered[train_end:train_end + validation_count]],
        "test_scan_ids": [path.stem for path in ordered[train_end + validation_count:]],
    }


def save_benchmark_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(manifest), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def load_benchmark_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or value.get("format") != "lidar2flexsim-benchmark-v1":
        raise ValueError(f"Manifesto de benchmark inválido: {source}")
    for key in ("train_scan_ids", "validation_scan_ids", "test_scan_ids"):
        if not isinstance(value.get(key), list):
            raise ValueError(f"Manifesto sem split válido: {key}")
    return dict(value)


def _paths_for_ids(scans: Sequence[Path], ids: Sequence[str]) -> list[Path]:
    by_id = {path.stem: path for path in scans}
    missing = [scan_id for scan_id in ids if scan_id not in by_id]
    if missing:
        raise FileNotFoundError("Scans ausentes no manifesto: " + ", ".join(missing[:5]))
    return [by_id[scan_id] for scan_id in ids]


def _segmentation_benchmark(
    train_paths: Sequence[Path],
    validation_paths: Sequence[Path],
    test_paths: Sequence[Path],
    label_dir: Path,
    checkpoint: str | Path,
    *,
    device: str,
    seed: int,
) -> dict[str, Any]:
    torch = require_torch("executar benchmark PointNet++")
    payload = load_checkpoint(checkpoint, map_location=device)
    config = PointNet2Config.from_mapping(payload.get("config") or {})
    model = PointNet2Segmentation(config)
    load_checkpoint(checkpoint, model=model, map_location=device)
    model.eval()
    model.to(device)
    result: dict[str, Any] = {}
    started = time.perf_counter()
    for name, paths in (("train", train_paths), ("validation", validation_paths), ("test", test_paths)):
        if not paths:
            continue
        dataset = WarehouseSegmentationDataset(
            paths,
            label_dir=label_dir,
            class_names=config.class_names,
            num_points=config.input_points,
            seed=seed,
            preprocessing=config.preprocessing,
        )
        confusion = torch.zeros((config.num_classes, config.num_classes), dtype=torch.long)
        with torch.no_grad():
            for index in range(len(dataset)):
                batch = dataset[index]
                points = batch["points"].unsqueeze(0).to(device=device, dtype=torch.float32)
                labels = batch["labels"].unsqueeze(0).to(device=device, dtype=torch.long)
                logits = model(points)
                confusion += _confusion_from_batch(logits.argmax(-1), labels, config.num_classes)
        result[name] = _metrics_from_confusion(confusion, config.num_classes, config.class_names)
    result["elapsed_seconds"] = float(time.perf_counter() - started)
    result["checkpoint"] = str(checkpoint)
    return result


def _box_benchmark(
    test_paths: Sequence[Path],
    label_dir: Path,
    checkpoint: str | Path,
    *,
    device: str,
) -> dict[str, Any]:
    from avaliar_deteccoes import predicao_para_caixa, resumir_scans
    from .inference import inferir_scan

    predictions: dict[str, list[dict[str, Any]]] = {}
    started = time.perf_counter()
    for path in test_paths:
        raw_predictions = inferir_scan(
            path,
            checkpoint=checkpoint,
            device=device,
        )["predictions"]
        predictions[path.stem] = [predicao_para_caixa(item) for item in raw_predictions]
    summary = resumir_scans(predictions, label_dir, [0.25, 0.5], None)
    summary["elapsed_seconds"] = float(time.perf_counter() - started)
    return summary


def run_benchmark(
    dataset: str | Path,
    *,
    checkpoint: str | Path | None = None,
    manifest: str | Path | Mapping[str, Any] | None = None,
    output_dir: str | Path | None = None,
    device: str = "cpu",
    seed: int = 42,
    max_scans: int | None = None,
) -> dict[str, Any]:
    """Executa e opcionalmente salva um benchmark completo."""

    torch = require_torch("executar benchmark PointNet++") if checkpoint else None
    if torch is not None:
        _set_seed(seed, torch)
    dataset_root, label_dir, scans = _scan_files(dataset)
    if manifest is None:
        manifest_value = build_benchmark_manifest(scans, seed=seed, max_scans=max_scans)
    elif isinstance(manifest, Mapping):
        manifest_value = dict(manifest)
    else:
        manifest_value = load_benchmark_manifest(manifest)
    train_paths = _paths_for_ids(scans, manifest_value["train_scan_ids"])
    validation_paths = _paths_for_ids(scans, manifest_value["validation_scan_ids"])
    test_paths = _paths_for_ids(scans, manifest_value["test_scan_ids"])
    report: dict[str, Any] = {
        "manifest": manifest_value,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": getattr(torch, "__version__", None) if torch is not None else None,
        },
        "dataset": str(dataset_root),
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "segmentation": None,
        "boxes": None,
        "baseline": None,
    }
    if checkpoint is not None:
        checkpoint_payload = load_checkpoint(checkpoint, map_location=device)
        report["baseline"] = {
            "epoch": checkpoint_payload.get("epoch"),
            "metrics": checkpoint_payload.get("metrics", {}),
            "format_version": checkpoint_payload.get("format_version"),
        }
        report["segmentation"] = _segmentation_benchmark(
            train_paths, validation_paths, test_paths, label_dir, checkpoint,
            device=device, seed=seed,
        )
        if label_dir.is_dir():
            report["boxes"] = _box_benchmark(test_paths, label_dir, checkpoint, device=device)
    if output_dir is not None:
        target = Path(output_dir).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        save_benchmark_manifest(target / "manifest.json", manifest_value)
        (target / "benchmark.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        report["output_dir"] = str(target)
    return report


__all__ = [
    "build_benchmark_manifest",
    "load_benchmark_manifest",
    "run_benchmark",
    "save_benchmark_manifest",
]

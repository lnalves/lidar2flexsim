"""CLI para treinar, inferir e avaliar o PointNet++.

``python -m ml.cli train --dataset dados/warehouse --output checkpoints``
``python -m ml.cli infer --scan dados/warehouse/bin/000000.bin --checkpoint ...``
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import PointNet2Config, TrainingConfig, load_config
from .data import WarehousePointDataset, select_scan_subset, temporal_three_way_split
from .benchmark import run_benchmark
from .dependencies import require_torch
from .inference import inferir_scan
from .models.pointnet2_seg import PointNet2Segmentation
from .training import train_model


def _config(path: str | None) -> tuple[PointNet2Config, TrainingConfig]:
    return load_config(path) if path else (PointNet2Config(), TrainingConfig())


def _class_weights(value: str | None, expected: int) -> list[float] | None:
    if not value:
        return None
    if value.strip().casefold() == "auto":
        return None
    try:
        weights = [float(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise ValueError("--class-weights deve ser uma lista numérica separada por vírgulas") from exc
    if len(weights) != expected or any(item <= 0 for item in weights):
        raise ValueError(
            f"--class-weights deve conter {expected} pesos positivos (incluindo background)"
        )
    return weights


def _automatic_class_weights(dataset: Any, torch: Any, num_classes: int) -> list[float]:
    counts = torch.zeros(int(num_classes), dtype=torch.float64)
    for index in range(len(dataset)):
        labels = dataset[index]["labels"]
        counts += torch.bincount(labels.to(dtype=torch.long), minlength=int(num_classes)).to(dtype=torch.float64)
    present = counts > 0
    if not bool(present.any()):
        return [1.0] * int(num_classes)
    total = counts[present].sum()
    weights = torch.ones_like(counts)
    weights[present] = total / (present.sum().to(dtype=torch.float64) * counts[present])
    # Keep the mean weight of observed classes at one for stable CE scaling.
    weights[present] /= weights[present].mean()
    return [float(value) for value in weights.tolist()]


def _resolve_run_dir(output: str, run_name: str | None, resume: str | None) -> Path:
    if resume:
        source = Path(resume).expanduser()
        if source.is_dir():
            return source
        return source.parent.parent if source.parent.name == "checkpoints" else source.parent
    root = Path(output).expanduser()
    name = run_name or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_pointnet2"
    candidate = root / name
    if candidate.exists():
        raise FileExistsError(
            f"A execução já existe: {candidate}. Use --run-name único ou --resume."
        )
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate

def _recorded_split(run_dir: Path) -> dict[str, list[str]] | None:
    """Lê os splits gravados por uma execução anterior, se existirem.

    Recalcular o split ao retomar reparticionaria o treino sempre que uma
    fração padrão mudasse, contaminando a validação com scans já vistos.
    """

    metadata_path = run_dir / "metadata.json"
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(metadata, dict):
        return None
    keys = ("train_scan_ids", "validation_scan_ids", "test_scan_ids")
    if not all(isinstance(metadata.get(key), list) for key in keys):
        return None
    return {key: [str(item) for item in metadata[key]] for key in keys}

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Treino e inferência PointNet++ para LiDAR Warehouse")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="treina segmentador supervisionado")
    train.add_argument("--dataset", required=True, help="raiz com bin/ e label/")
    train.add_argument("--output", default="runs", help="raiz das execuções isoladas")
    train.add_argument("--run-name", help="nome único da execução")
    train.add_argument("--resume", help="checkpoint ou diretório de execução para retomar")
    train.add_argument("--config", help="arquivo JSON/YAML com model/training")
    train.add_argument("--epochs", type=int)
    train.add_argument("--batch-size", type=int, help="batch menor para testes rápidos")
    train.add_argument("--max-scans", type=int, help="limita scans, preservando cobertura temporal")
    train.add_argument("--input-points", type=int, help="pontos por scan, por exemplo 1024")
    train.add_argument("--device", default=None, help="cpu, cuda ou auto")
    train.add_argument(
        "--test-fraction",
        type=float,
        help="fração final reservada para o benchmark (padrão 0.1)",
    )
    train.add_argument(
        "--class-weights",
        help="pesos CE, 'auto' ou lista como 0.1,1,1,1,1,1 (background + 5 classes)",
    )
    train.set_defaults(handler=train_command)
    infer = commands.add_parser("infer", help="executa detecção em um scan")
    infer.add_argument("--scan", required=True)
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--device", default="cpu")
    infer.add_argument("--num-points", type=int)
    infer.add_argument("--score-threshold", type=float, default=0.5)
    infer.add_argument("--cluster-eps", type=float, default=0.35)
    infer.add_argument("--min-cluster-points", type=int, default=5)
    infer.add_argument("--calibration", help="JSON de filtros/NMS por classe")
    infer.set_defaults(handler=infer_command)
    benchmark = commands.add_parser("benchmark", help="executa benchmark reproduzível")
    benchmark.add_argument("--dataset", required=True)
    benchmark.add_argument("--checkpoint")
    benchmark.add_argument("--manifest")
    benchmark.add_argument(
        "--from-run",
        help="diretório de execução cujo split real deve ser reutilizado",
    )
    benchmark.add_argument("--output", default="benchmark")
    benchmark.add_argument("--device", default="cpu")
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.add_argument("--max-scans", type=int)
    benchmark.set_defaults(handler=benchmark_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

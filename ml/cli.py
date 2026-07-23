"""CLI opcional para treinar e inferir o PointNet++.

Os comandos são mantidos separados da CLI geométrica existente:

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
from .data import WarehouseSegmentationDataset, temporal_split
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


def train_command(args: argparse.Namespace) -> int:
    torch = require_torch("usar o comando ml train")
    model_config, train_config = _config(args.config)
    if args.epochs is not None:
        train_config = TrainingConfig.from_mapping(train_config.to_dict(), epochs=args.epochs)
    if args.device is not None:
        train_config = TrainingConfig.from_mapping(train_config.to_dict(), device=args.device)
    root = Path(args.dataset).expanduser()
    bin_dir = root if root.name.casefold() in {"bin", "bins"} else root / "bin"
    dataset_root = bin_dir.parent if bin_dir.name.casefold() in {"bin", "bins"} else root
    scans = sorted(bin_dir.glob("*.bin"))
    if not scans:
        raise FileNotFoundError(f"Nenhum .bin encontrado em {bin_dir}")
    if len(scans) < 2:
        raise ValueError("O treinamento requer pelo menos dois scans para o split temporal.")
    train_scans, validation_scans = temporal_split(scans, train_config.validation_fraction)
    class_weights = _class_weights(args.class_weights, model_config.num_classes)
    train_dataset = WarehouseSegmentationDataset(
        train_scans,
        label_dir=dataset_root / "label",
        class_names=model_config.class_names,
        num_points=model_config.input_points,
        seed=train_config.seed,
        augment=True,
        preprocessing=model_config.preprocessing,
    )
    validation_dataset = WarehouseSegmentationDataset(
        validation_scans,
        label_dir=dataset_root / "label",
        class_names=model_config.class_names,
        num_points=model_config.input_points,
        seed=train_config.seed,
        preprocessing=model_config.preprocessing,
    )
    if args.class_weights and args.class_weights.strip().casefold() == "auto":
        class_weights = _automatic_class_weights(train_dataset, torch, model_config.num_classes)
    loader_kwargs = {"batch_size": train_config.batch_size, "num_workers": 0}
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, shuffle=False, **loader_kwargs)
    model = PointNet2Segmentation(model_config)
    run_dir = _resolve_run_dir(args.output, args.run_name, args.resume)
    metadata = {
        "dataset": str(root),
        "all_scan_ids": [scan.stem for scan in scans],
        "train_scan_ids": [scan.stem for scan in train_scans],
        "validation_scan_ids": [scan.stem for scan in validation_scans],
        "class_weights": class_weights,
    }
    result = train_model(
        train_loader,
        model,
        config=train_config,
        validation_loader=validation_loader,
        class_weights=class_weights,
        run_dir=run_dir,
        resume=args.resume,
        experiment_metadata=metadata,
        callback=lambda record: print(json.dumps(record, ensure_ascii=False), flush=True),
    )
    print(json.dumps({"scans": len(scans), **result}, ensure_ascii=False, default=str))
    return 0


def infer_command(args: argparse.Namespace) -> int:
    calibration = None
    if args.calibration:
        calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    result = inferir_scan(
        args.scan,
        checkpoint=args.checkpoint,
        device=args.device,
        score_threshold=args.score_threshold,
        cluster_eps=args.cluster_eps,
        min_cluster_points=args.min_cluster_points,
        num_points=args.num_points,
        calibration=calibration,
    )
    # Clusters are retained by the Python API for STL export, but printing all
    # sampled points would make the command output unnecessarily large.
    output = dict(result)
    output.pop("clusters", None)
    print(json.dumps(output, ensure_ascii=False, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)))
    return 0


def benchmark_command(args: argparse.Namespace) -> int:
    report = run_benchmark(
        args.dataset,
        checkpoint=args.checkpoint,
        manifest=args.manifest,
        output_dir=args.output,
        device=args.device,
        seed=args.seed,
        max_scans=args.max_scans,
    )
    print(json.dumps(report, ensure_ascii=False, default=str))
    return 0


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
    train.add_argument("--device", default=None, help="cpu, cuda ou auto")
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

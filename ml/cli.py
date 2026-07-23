"""CLI opcional para treinar e inferir o PointNet++.

Os comandos são mantidos separados da CLI geométrica existente:

``python -m ml.cli train --dataset dados/warehouse --output checkpoints``
``python -m ml.cli infer --scan dados/warehouse/bin/000000.bin --checkpoint ...``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import PointNet2Config, TrainingConfig, load_config
from .data import WarehouseSegmentationDataset, temporal_split
from .dependencies import require_torch
from .inference import inferir_scan
from .models.pointnet2_seg import PointNet2Segmentation
from .training import train_model


def _config(path: str | None) -> tuple[PointNet2Config, TrainingConfig]:
    return load_config(path) if path else (PointNet2Config(), TrainingConfig())


def _class_weights(value: str | None, expected: int) -> list[float] | None:
    if not value:
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
    )
    validation_dataset = WarehouseSegmentationDataset(
        validation_scans,
        label_dir=dataset_root / "label",
        class_names=model_config.class_names,
        num_points=model_config.input_points,
        seed=train_config.seed,
    )
    loader_kwargs = {"batch_size": train_config.batch_size, "num_workers": 0}
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, shuffle=False, **loader_kwargs)
    model = PointNet2Segmentation(model_config)
    result = train_model(
        train_loader,
        model,
        config=train_config,
        validation_loader=validation_loader,
        class_weights=class_weights,
        checkpoint_dir=args.output,
        callback=lambda record: print(json.dumps(record, ensure_ascii=False), flush=True),
    )
    print(json.dumps({"scans": len(scans), **result}, ensure_ascii=False, default=str))
    return 0


def infer_command(args: argparse.Namespace) -> int:
    result = inferir_scan(
        args.scan,
        checkpoint=args.checkpoint,
        device=args.device,
        score_threshold=args.score_threshold,
        cluster_eps=args.cluster_eps,
        min_cluster_points=args.min_cluster_points,
        num_points=args.num_points,
    )
    # Clusters are retained by the Python API for STL export, but printing all
    # sampled points would make the command output unnecessarily large.
    output = dict(result)
    output.pop("clusters", None)
    print(json.dumps(output, ensure_ascii=False, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Treino e inferência PointNet++ para LiDAR Warehouse")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="treina segmentador supervisionado")
    train.add_argument("--dataset", required=True, help="raiz com bin/ e label/")
    train.add_argument("--output", default="checkpoints", help="pasta dos checkpoints")
    train.add_argument("--config", help="arquivo JSON/YAML com model/training")
    train.add_argument("--epochs", type=int)
    train.add_argument("--device", default=None, help="cpu, cuda ou auto")
    train.add_argument(
        "--class-weights",
        help="pesos CE, por exemplo 0.1,1,1,1,1,1 (background + 5 classes)",
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
    infer.set_defaults(handler=infer_command)
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

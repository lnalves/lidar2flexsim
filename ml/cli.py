"""CLI para treinar, inferir, avaliar e transmitir para o FlexSim.

``python -m ml.cli train --dataset dados/warehouse --output checkpoints``
``python -m ml.cli infer --scan dados/warehouse/bin/000000.bin --checkpoint ...``
``python -m ml.cli stream --dataset dados/warehouse --checkpoint ... --serve``
"""

from __future__ import annotations

import argparse
import json
import signal
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


def train_command(args: argparse.Namespace) -> int:
    torch = require_torch("usar o comando ml train")
    model_config, train_config = _config(args.config)
    if args.input_points is not None:
        model_config = PointNet2Config.from_mapping(
            model_config.to_dict(), input_points=args.input_points
        )
    if args.batch_size is not None:
        train_config = TrainingConfig.from_mapping(
            train_config.to_dict(), batch_size=args.batch_size
        )
    if args.epochs is not None:
        train_config = TrainingConfig.from_mapping(train_config.to_dict(), epochs=args.epochs)
    if args.device is not None:
        train_config = TrainingConfig.from_mapping(train_config.to_dict(), device=args.device)
    if args.test_fraction is not None:
        train_config = TrainingConfig.from_mapping(
            train_config.to_dict(), test_fraction=args.test_fraction
        )
    root = Path(args.dataset).expanduser()
    bin_dir = root if root.name.casefold() in {"bin", "bins"} else root / "bin"
    dataset_root = bin_dir.parent if bin_dir.name.casefold() in {"bin", "bins"} else root
    available = sorted(bin_dir.glob("*.bin"))
    if not available:
        raise FileNotFoundError(f"Nenhum .bin encontrado em {bin_dir}")
    run_dir = _resolve_run_dir(args.output, args.run_name, args.resume)
    recorded = _recorded_split(run_dir) if args.resume else None
    if recorded is not None:
        by_id = {scan.stem: scan for scan in available}
        missing = [
            scan_id
            for ids in recorded.values()
            for scan_id in ids
            if scan_id not in by_id
        ]
        if missing:
            raise FileNotFoundError(
                "Scans do split gravado não estão no dataset: " + ", ".join(missing[:5])
            )
        train_scans = [by_id[scan_id] for scan_id in recorded["train_scan_ids"]]
        validation_scans = [by_id[scan_id] for scan_id in recorded["validation_scan_ids"]]
        test_scans = [by_id[scan_id] for scan_id in recorded["test_scan_ids"]]
        scans = sorted(
            train_scans + validation_scans + test_scans, key=lambda item: item.stem
        )
    else:
        scans = select_scan_subset(available, args.max_scans)
        if len(scans) < 2:
            raise ValueError(
                "O treinamento requer pelo menos dois scans para o split temporal."
            )
        train_scans, validation_scans, test_scans = temporal_three_way_split(
            scans, train_config.validation_fraction, train_config.test_fraction
        )
    if not train_scans:
        raise ValueError("O split temporal não deixou scans de treino.")
    class_weights = _class_weights(args.class_weights, model_config.num_classes)
    train_dataset = WarehousePointDataset(
        dataset_root,
        label_dir=dataset_root / "label",
        class_names=model_config.class_names,
        num_points=model_config.input_points,
        scan_ids=[scan.stem for scan in train_scans],
        random_seed=train_config.seed,
        return_tensors=True,
        augment=True,
        preprocessing=model_config.preprocessing,
    )
    validation_dataset = WarehousePointDataset(
        dataset_root,
        label_dir=dataset_root / "label",
        class_names=model_config.class_names,
        num_points=model_config.input_points,
        scan_ids=[scan.stem for scan in validation_scans],
        random_seed=train_config.seed,
        return_tensors=True,
        preprocessing=model_config.preprocessing,
    )
    if args.class_weights and args.class_weights.strip().casefold() == "auto":
        class_weights = _automatic_class_weights(train_dataset, torch, model_config.num_classes)
    loader_kwargs = {"batch_size": train_config.batch_size, "num_workers": 0}
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, shuffle=False, **loader_kwargs)
    model = PointNet2Segmentation(model_config)
    metadata = {
        "dataset": str(root),
        "split_strategy": "temporal_sorted_stem",
        "selection_strategy": "uniform_over_sequence",
        "max_scans": int(args.max_scans) if args.max_scans else None,
        "input_points": int(model_config.input_points),
        "validation_fraction": float(train_config.validation_fraction),
        "test_fraction": float(train_config.test_fraction),
        "all_scan_ids": [scan.stem for scan in scans],
        "train_scan_ids": [scan.stem for scan in train_scans],
        "validation_scan_ids": [scan.stem for scan in validation_scans],
        # Bloco reservado: nenhum dos loaders acima recebe estes scans, então o
        # benchmark pode reusá-los como conjunto de teste real do checkpoint.
        "test_scan_ids": [scan.stem for scan in test_scans],
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
    # Clusters remain available through the Python API; omit their raw points
    # from terminal JSON to keep the output compact.
    output = dict(result)
    output.pop("clusters", None)
    print(json.dumps(output, ensure_ascii=False, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)))
    return 0


def _sensor_translation(value: str | None) -> list[float] | None:
    if not value:
        return None
    parts = [item.strip() for item in str(value).replace(";", ",").split(",")]
    try:
        numbers = [float(item) for item in parts if item]
    except ValueError as exc:
        raise ValueError("--sensor-translation deve ser 'x,y,z' em metros") from exc
    if len(numbers) != 3:
        raise ValueError("--sensor-translation deve conter exatamente três valores")
    return numbers


def stream_command(args: argparse.Namespace) -> int:
    """Laço em tempo real: fonte → PointNet++ → tracking → FlexSim."""

    from .flexsim.pipeline import build_pipeline
    from .flexsim.server import SceneServer
    from .flexsim.sources import ReplayPointSource

    calibration = None
    if args.calibration:
        calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    source = ReplayPointSource(
        args.dataset,
        rate_hz=args.rate,
        loop=args.loop,
        max_frames=args.max_frames,
        realtime=not args.no_realtime,
    )
    server = (
        SceneServer(host=args.serve_host, port=args.serve_port).start()
        if args.serve
        else None
    )
    pipeline = build_pipeline(
        args.checkpoint,
        device=args.device,
        num_points=args.num_points,
        score_threshold=args.score_threshold,
        cluster_eps=args.cluster_eps,
        min_cluster_points=args.min_cluster_points,
        calibration=calibration,
        container=args.container,
        tracking={
            key: value
            for key, value in {
                "max_distance": args.track_max_distance,
                "max_age": args.track_max_age,
                "min_hits": args.track_min_hits,
                "smoothing": args.track_smoothing,
            }.items()
            if value is not None
        },
        placement={
            key: value
            for key, value in {
                "translation": _sensor_translation(args.sensor_translation),
                "yaw_deg": args.sensor_yaw,
                "scale": args.sensor_scale,
            }.items()
            if value is not None
        },
        object_map=args.flexsim_map,
        output_dir=args.output_dir,
        server=server,
    )

    def emit(record: dict[str, Any]) -> None:
        print(json.dumps(record, ensure_ascii=False), flush=True)

    # Ctrl+C encerra pelo caminho normal, publicando o resumo e fechando o
    # servidor; matar o processo deixaria o FlexSim lendo uma cena congelada
    # sem nenhuma indicação de que o produtor morreu.
    def request_stop(*_: Any) -> None:
        pipeline.stop()
        source.close()

    previous: dict[Any, Any] = {}
    for name in ("SIGINT", "SIGTERM"):
        number = getattr(signal, name, None)
        if number is None:
            continue
        try:
            previous[number] = signal.signal(number, lambda *_: request_stop())
        except ValueError:
            # Fora da thread principal não há como instalar handlers; o laço
            # ainda encerra por --max-frames ou pelo fim do dataset.
            break
    try:
        pipeline.prepare()
        emit(
            {
                "event": "ready",
                "checkpoint": str(args.checkpoint),
                "scans": len(source.scans),
                "rate_hz": source.rate_hz,
                "server_url": server.url if server else None,
                "output_dir": str(args.output_dir) if args.output_dir else None,
                "container": args.container,
            }
        )
        summary = pipeline.run(
            source,
            on_frame=None if args.quiet else (lambda report: emit(report.to_event())),
        )
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
        if server is not None:
            server.stop()
    emit(summary)
    return 0


def benchmark_command(args: argparse.Namespace) -> int:
    report = run_benchmark(
        args.dataset,
        checkpoint=args.checkpoint,
        manifest=args.manifest,
        from_run=args.from_run,
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
    stream = commands.add_parser(
        "stream", help="transmite detecções em tempo real para o FlexSim"
    )
    stream.add_argument("--dataset", required=True, help="raiz do dataset usado no replay")
    stream.add_argument("--checkpoint", required=True)
    stream.add_argument("--device", default="cpu")
    stream.add_argument("--num-points", type=int)
    stream.add_argument("--score-threshold", type=float, default=0.5)
    stream.add_argument("--cluster-eps", type=float, default=0.35)
    stream.add_argument("--min-cluster-points", type=int, default=5)
    stream.add_argument("--calibration", help="JSON de filtros/NMS por classe")
    stream.add_argument("--rate", type=float, default=10.0, help="quadros por segundo do replay")
    stream.add_argument("--loop", action="store_true", help="reinicia ao fim do dataset")
    stream.add_argument("--max-frames", type=int, help="encerra após N quadros")
    stream.add_argument(
        "--no-realtime",
        action="store_true",
        help="processa o mais rápido possível, sem esperar o intervalo do sensor",
    )
    stream.add_argument(
        "--output-dir", help="pasta onde gravar scene.json, scene.csv e lidar_bridge.txt"
    )
    stream.add_argument("--serve", action="store_true", help="publica a cena por HTTP")
    stream.add_argument("--serve-host", default="127.0.0.1")
    stream.add_argument("--serve-port", type=int, default=8765)
    stream.add_argument(
        "--container", default="LidarScene", help="nome do container no modelo FlexSim"
    )
    stream.add_argument("--flexsim-map", help="JSON com 'flexsim_objects' por classe")
    stream.add_argument("--sensor-translation", help="origem do sensor no modelo, 'x,y,z'")
    stream.add_argument("--sensor-yaw", type=float, help="rotação do sensor em graus")
    stream.add_argument("--sensor-scale", type=float, help="metros do sensor por unidade do modelo")
    stream.add_argument("--track-max-distance", type=float)
    stream.add_argument("--track-max-age", type=int)
    stream.add_argument("--track-min-hits", type=int)
    stream.add_argument("--track-smoothing", type=float)
    stream.add_argument("--quiet", action="store_true", help="omite o evento por quadro")
    stream.set_defaults(handler=stream_command)
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

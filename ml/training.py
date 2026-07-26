"""Laço de treinamento pequeno e reproduzível para o segmentador."""

from __future__ import annotations

import random
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from .checkpoints import load_checkpoint, save_checkpoint
from .config import PointNet2Config, TrainingConfig, save_config
from .dependencies import MissingOptionalDependency, require_torch, torch as _torch
from .models.pointnet2_seg import PointNet2Segmentation


def _device(torch: Any, requested: str) -> Any:
    name = str(requested).strip().lower()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    if name == "mps":
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is None or not bool(mps.is_available()):
            return torch.device("cpu")
    return torch.device(name)


def _set_seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _confusion_from_batch(predicted: Any, target: Any, num_classes: int) -> Any:
    """Acumula uma matriz de confusão sem depender do tamanho do batch."""

    torch = _torch or require_torch("calcular métricas")
    values = target.detach().reshape(-1).to(dtype=torch.long) * int(num_classes)
    values = values + predicted.detach().reshape(-1).to(dtype=torch.long)
    return torch.bincount(values, minlength=int(num_classes) ** 2).reshape(
        int(num_classes), int(num_classes)
    ).to(device="cpu", dtype=torch.long)


def _metrics_from_confusion(
    confusion: Any,
    num_classes: int,
    class_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Calcula métricas globais e preserva classes sem suporte como ``None``."""

    matrix = np.asarray(confusion, dtype=np.int64)
    names = tuple(class_names or (str(index) for index in range(num_classes)))
    if len(names) != num_classes:
        names = tuple(str(index) for index in range(num_classes))
    true_counts = matrix.sum(axis=1)
    predicted_counts = matrix.sum(axis=0)
    supports: dict[str, int] = {}
    iou: dict[str, float | None] = {}
    precision: dict[str, float | None] = {}
    recall: dict[str, float | None] = {}
    f1: dict[str, float | None] = {}
    for class_id, name in enumerate(names):
        key = str(name)
        tp = int(matrix[class_id, class_id])
        union = int(true_counts[class_id] + predicted_counts[class_id] - tp)
        supports[key] = int(true_counts[class_id])
        iou[key] = float(tp / union) if union else None
        precision[key] = float(tp / predicted_counts[class_id]) if predicted_counts[class_id] else None
        recall[key] = float(tp / true_counts[class_id]) if true_counts[class_id] else None
        if precision[key] is not None and recall[key] is not None and precision[key] + recall[key]:
            f1[key] = float(2 * precision[key] * recall[key] / (precision[key] + recall[key]))
        else:
            f1[key] = None
    present = [value for value in iou.values() if value is not None]
    foreground = [iou[str(names[index])] for index in range(1, num_classes) if iou[str(names[index])] is not None]
    total = int(matrix.sum())
    return {
        "accuracy": float(np.trace(matrix) / total) if total else 0.0,
        # ``miou`` remains foreground-only for compatibility with old logs.
        "miou": float(np.mean(foreground)) if foreground else 0.0,
        "miou_macro": float(np.mean(present)) if present else 0.0,
        "iou_per_class": iou,
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
        "support_per_class": supports,
        "present_classes": [str(names[index]) for index, value in enumerate(iou.values()) if value is not None],
        "confusion_matrix": matrix.tolist(),
    }


def _batch_metrics(predicted: Any, target: Any, num_classes: int) -> dict[str, float]:
    """Compatibilidade para consumidores antigos; métricas novas são globais."""

    metrics = _metrics_from_confusion(
        _confusion_from_batch(predicted, target, num_classes), num_classes
    )
    return {"accuracy": float(metrics["accuracy"]), "miou": float(metrics["miou"])}


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), ensure_ascii=False, default=str) + "\n")


def _read_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    history: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            history.append(dict(value))
    return history


def _prepare_run_directory(
    run_dir: str | Path | None,
    checkpoint_dir: str | Path | None,
    resume: str | Path | None,
) -> tuple[Path | None, Path | None]:
    if run_dir is not None:
        root = Path(run_dir).expanduser()
        checkpoint_path = root / "checkpoints"
    elif resume is not None:
        resume_path = Path(resume).expanduser()
        root = resume_path.parent.parent if resume_path.parent.name == "checkpoints" else resume_path.parent
        checkpoint_path = root / "checkpoints" if root.name != "checkpoints" else root
    elif checkpoint_dir is not None:
        root = None
        checkpoint_path = Path(checkpoint_dir).expanduser()
    else:
        return None, None
    root_to_create = root if root is not None else checkpoint_path
    root_to_create.mkdir(parents=True, exist_ok=True)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    return root, checkpoint_path


def train_model(
    train_loader: Iterable[Mapping[str, Any]],
    model: PointNet2Segmentation,
    *,
    config: TrainingConfig | Mapping[str, Any] | None = None,
    validation_loader: Iterable[Mapping[str, Any]] | None = None,
    class_weights: Iterable[float] | None = None,
    checkpoint_dir: str | Path | None = None,
    run_dir: str | Path | None = None,
    resume: str | Path | None = None,
    experiment_metadata: Mapping[str, Any] | None = None,
    callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Treina o modelo e retorna o histórico por época.

    O ``DataLoader`` pode usar ``WarehousePointDataset`` ou
    qualquer iterável que forneça ``points`` e ``labels``.
    """

    torch = require_torch("treinar o modelo PointNet++")
    train_config = config if isinstance(config, TrainingConfig) else TrainingConfig.from_mapping(config)
    _set_seed(train_config.seed, torch)
    device = _device(torch, train_config.device)
    model.to(device)
    weights = None
    if class_weights is not None:
        weights = torch.as_tensor(list(class_weights), dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    history: list[dict[str, Any]] = []
    run_path, checkpoint_path = _prepare_run_directory(run_dir, checkpoint_dir, resume)
    history_path = run_path / "history.jsonl" if run_path is not None else None
    if run_path is not None:
        config_path = run_path / "config.json"
        if not config_path.exists():
            save_config(config_path, model.config, train_config)
        metadata_path = run_path / "metadata.json"
        if not metadata_path.exists():
            metadata = {
                "run_id": run_path.name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "python": sys.version,
                "torch": getattr(torch, "__version__", "unknown"),
                **dict(experiment_metadata or {}),
            }
            metadata_path.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "\n",
                encoding="utf-8",
            )
        history = _read_history(history_path) if history_path is not None else []

    start_epoch = 1
    if resume is not None:
        resume_path = Path(resume).expanduser()
        if resume_path.is_dir():
            candidate = resume_path / "checkpoints" / "last.pt"
            resume_path = candidate if candidate.exists() else resume_path / "last.pt"
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint de retomada não encontrado: {resume_path}")
        payload = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            map_location=device,
        )
        start_epoch = int(payload.get("epoch", 0) or 0) + 1

    best_metric = max(
        (float(item.get("val_miou", item.get("miou", float("-inf")))) for item in history),
        default=float("-inf"),
    )

    for epoch in range(start_epoch, train_config.epochs + 1):
        model.train()
        losses: list[float] = []
        train_confusion = torch.zeros(
            (model.config.num_classes, model.config.num_classes), dtype=torch.long
        )
        for batch in train_loader:
            points = batch["points"].to(device=device, dtype=torch.float32)
            labels = batch["labels"].to(device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits = model(points)
            loss = criterion(logits.reshape(-1, model.config.num_classes), labels.reshape(-1))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            train_confusion += _confusion_from_batch(
                logits.detach().argmax(-1), labels, model.config.num_classes
            )

        train_metrics = _metrics_from_confusion(
            train_confusion, model.config.num_classes, model.config.class_names
        )
        record: dict[str, Any] = {
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else 0.0,
            **train_metrics,
        }
        if validation_loader is not None:
            model.eval()
            val_losses: list[float] = []
            val_confusion = torch.zeros(
                (model.config.num_classes, model.config.num_classes), dtype=torch.long
            )
            with torch.no_grad():
                for batch in validation_loader:
                    points = batch["points"].to(device=device, dtype=torch.float32)
                    labels = batch["labels"].to(device=device, dtype=torch.long)
                    logits = model(points)
                    val_losses.append(float(criterion(logits.reshape(-1, model.config.num_classes), labels.reshape(-1)).cpu().item()))
                    val_confusion += _confusion_from_batch(
                        logits.argmax(-1), labels, model.config.num_classes
                    )
            val_metrics = _metrics_from_confusion(
                val_confusion, model.config.num_classes, model.config.class_names
            )
            record["val_loss"] = float(np.mean(val_losses)) if val_losses else 0.0
            record.update({f"val_{key}": value for key, value in val_metrics.items()})
        history.append(record)
        if history_path is not None:
            _append_jsonl(history_path, record)
        if callback is not None:
            callback(dict(record))
        if checkpoint_path and epoch % train_config.checkpoint_every == 0:
            save_kwargs = {
                "optimizer": optimizer,
                "epoch": epoch,
                "config": model.config,
                "metrics": record,
            }
            save_checkpoint(checkpoint_path / f"pointnet2_epoch_{epoch:04d}.pt", model, **save_kwargs)
            save_checkpoint(checkpoint_path / "last.pt", model, **save_kwargs)
            metric = float(record.get("val_miou", record.get("miou", 0.0)))
            if metric > best_metric:
                best_metric = metric
                save_checkpoint(checkpoint_path / "best.pt", model, **save_kwargs)
    result: dict[str, Any] = {
        "history": history,
        "device": str(device),
        "config": train_config.to_dict(),
    }
    if run_path is not None:
        result.update({
            "run_dir": str(run_path),
            "checkpoint_dir": str(checkpoint_path),
            "last_checkpoint": str(checkpoint_path / "last.pt"),
            "best_checkpoint": str(checkpoint_path / "best.pt") if (checkpoint_path / "best.pt").exists() else None,
        })
    return result


def smoke_train(steps: int = 1, *, num_points: int = 32) -> dict[str, Any]:
    """Executa um passo mínimo em CPU para validar a instalação do modelo."""

    torch = require_torch("executar o smoke test do PointNet++")
    config = PointNet2Config(
        input_points=num_points,
        sa1_points=min(16, num_points),
        sa2_points=min(4, num_points),
        neighbors=min(8, num_points),
    )
    model = PointNet2Segmentation(config)
    loader = [
        {
            "points": torch.randn(1, num_points, config.in_channels),
            "labels": torch.randint(0, config.num_classes, (1, num_points)),
        }
        for _ in range(max(1, int(steps)))
    ]
    result = train_model(loader, model, config=TrainingConfig(epochs=1, batch_size=1))
    return {"model": model, **result}

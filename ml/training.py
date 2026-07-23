"""Laço de treinamento pequeno e reproduzível para o segmentador."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from .checkpoints import save_checkpoint
from .config import PointNet2Config, TrainingConfig
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


def _batch_metrics(predicted: Any, target: Any, num_classes: int) -> dict[str, float]:
    accuracy = float((predicted == target).float().mean().item())
    ious: list[float] = []
    for class_id in range(1, num_classes):
        intersection = ((predicted == class_id) & (target == class_id)).sum().item()
        union = ((predicted == class_id) | (target == class_id)).sum().item()
        if union:
            ious.append(float(intersection / union))
    return {"accuracy": accuracy, "miou": float(np.mean(ious)) if ious else 0.0}


def train_model(
    train_loader: Iterable[Mapping[str, Any]],
    model: PointNet2Segmentation,
    *,
    config: TrainingConfig | Mapping[str, Any] | None = None,
    validation_loader: Iterable[Mapping[str, Any]] | None = None,
    class_weights: Iterable[float] | None = None,
    checkpoint_dir: str | Path | None = None,
    callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Treina o modelo e retorna o histórico por época.

    O ``DataLoader`` pode ser o adaptador ``WarehouseSegmentationDataset`` ou
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
    checkpoint_path = Path(checkpoint_dir).expanduser() if checkpoint_dir else None
    if checkpoint_path:
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, train_config.epochs + 1):
        model.train()
        losses: list[float] = []
        train_metrics: list[dict[str, float]] = []
        for batch in train_loader:
            points = batch["points"].to(device=device, dtype=torch.float32)
            labels = batch["labels"].to(device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits = model(points)
            loss = criterion(logits.reshape(-1, model.config.num_classes), labels.reshape(-1))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            train_metrics.append(_batch_metrics(logits.detach().argmax(-1), labels, model.config.num_classes))

        record: dict[str, Any] = {
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else 0.0,
            "accuracy": float(np.mean([item["accuracy"] for item in train_metrics])) if train_metrics else 0.0,
            "miou": float(np.mean([item["miou"] for item in train_metrics])) if train_metrics else 0.0,
        }
        if validation_loader is not None:
            model.eval()
            val_losses: list[float] = []
            val_metrics: list[dict[str, float]] = []
            with torch.no_grad():
                for batch in validation_loader:
                    points = batch["points"].to(device=device, dtype=torch.float32)
                    labels = batch["labels"].to(device=device, dtype=torch.long)
                    logits = model(points)
                    val_losses.append(float(criterion(logits.reshape(-1, model.config.num_classes), labels.reshape(-1)).cpu().item()))
                    val_metrics.append(_batch_metrics(logits.argmax(-1), labels, model.config.num_classes))
            record["val_loss"] = float(np.mean(val_losses)) if val_losses else 0.0
            record["val_accuracy"] = float(np.mean([item["accuracy"] for item in val_metrics])) if val_metrics else 0.0
            record["val_miou"] = float(np.mean([item["miou"] for item in val_metrics])) if val_metrics else 0.0
        history.append(record)
        if callback is not None:
            callback(dict(record))
        if checkpoint_path and epoch % train_config.checkpoint_every == 0:
            save_checkpoint(
                checkpoint_path / f"pointnet2_epoch_{epoch:04d}.pt",
                model,
                optimizer=optimizer,
                epoch=epoch,
                config=model.config,
                metrics=record,
            )
    return {"history": history, "device": str(device), "config": train_config.to_dict()}


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

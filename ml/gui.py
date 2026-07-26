"""Contratos e execução de processos usados pela interface PointNet++."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class DatasetInfo:
    root: Path
    bin_dir: Path
    label_dir: Path
    scans: tuple[Path, ...]
    label_count: int
    missing_label_ids: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return bool(self.scans) and self.label_dir.is_dir()


def validate_dataset(path: str | Path) -> DatasetInfo:
    """Valida a estrutura mínima usada por treino, inferência e benchmark."""

    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {source}")
    if not source.is_dir():
        raise ValueError(f"O dataset deve ser uma pasta: {source}")
    if source.name.casefold() in {"bin", "bins"}:
        root, bin_dir = source.parent, source
    else:
        root = source
        bin_dir = next(
            (root / name for name in ("bin", "bins") if (root / name).is_dir()),
            root / "bin",
        )
    label_dir = next(
        (root / name for name in ("label", "labels") if (root / name).is_dir()),
        root / "label",
    )
    if not bin_dir.is_dir():
        raise FileNotFoundError(f"Pasta de scans não encontrada: {bin_dir}")
    scans = tuple(sorted(bin_dir.glob("*.bin"), key=lambda item: item.stem))
    if not scans:
        raise FileNotFoundError(f"Nenhum scan .bin encontrado em {bin_dir}")
    labels = {item.stem for item in label_dir.glob("*.txt")} if label_dir.is_dir() else set()
    missing = tuple(scan.stem for scan in scans if scan.stem not in labels)
    return DatasetInfo(
        root=root,
        bin_dir=bin_dir,
        label_dir=label_dir,
        scans=scans,
        label_count=len(labels),
        missing_label_ids=missing,
    )


def discover_checkpoints(project_root: str | Path) -> list[Path]:
    """Encontra checkpoints relevantes sem inundar a seleção com toda época."""

    root = Path(project_root).expanduser().resolve()
    candidates: set[Path] = set()
    candidates.update((root / "checkpoints").glob("*.pt"))
    for run_root in (root / "runs",):
        candidates.update(run_root.glob("**/best.pt"))
        candidates.update(run_root.glob("**/last.pt"))
    return sorted(
        (item for item in candidates if item.is_file()),
        key=lambda item: (-item.stat().st_mtime_ns, str(item)),
    )


def available_devices() -> dict[str, str]:
    """Retorna dispositivos realmente utilizáveis pelo PyTorch local."""

    devices = {"cpu": "CPU"}
    try:
        import torch
    except (ImportError, OSError):
        return devices
    if torch.cuda.is_available():
        devices["cuda"] = f"CUDA · {torch.cuda.get_device_name(0)}"
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and bool(mps.is_available()):
        devices["mps"] = "Apple MPS"
    devices["auto"] = "Automático"
    return devices


def build_train_command(
    *,
    dataset: str | Path,
    output: str | Path,
    config: str | Path | None,
    run_name: str | None,
    resume: str | Path | None,
    epochs: int,
    batch_size: int,
    max_scans: int | None,
    input_points: int,
    device: str,
    class_weights: str,
    python: str = sys.executable,
) -> list[str]:
    command = [
        python,
        "-m",
        "ml.cli",
        "train",
        "--dataset",
        str(dataset),
        "--output",
        str(output),
        "--epochs",
        str(int(epochs)),
        "--batch-size",
        str(int(batch_size)),
        "--input-points",
        str(int(input_points)),
        "--device",
        str(device),
        "--class-weights",
        str(class_weights),
    ]
    if config:
        command.extend(["--config", str(config)])
    if run_name:
        command.extend(["--run-name", str(run_name)])
    if resume:
        command.extend(["--resume", str(resume)])
    if max_scans:
        command.extend(["--max-scans", str(int(max_scans))])
    return command


def build_infer_command(
    *,
    scan: str | Path,
    checkpoint: str | Path,
    device: str,
    num_points: int,
    score_threshold: float,
    cluster_eps: float,
    min_cluster_points: int,
    calibration: str | Path | None = None,
    python: str = sys.executable,
) -> list[str]:
    command = [
        python,
        "-m",
        "ml.cli",
        "infer",
        "--scan",
        str(scan),
        "--checkpoint",
        str(checkpoint),
        "--device",
        str(device),
        "--num-points",
        str(int(num_points)),
        "--score-threshold",
        str(float(score_threshold)),
        "--cluster-eps",
        str(float(cluster_eps)),
        "--min-cluster-points",
        str(int(min_cluster_points)),
    ]
    if calibration:
        command.extend(["--calibration", str(calibration)])
    return command


def build_benchmark_command(
    *,
    dataset: str | Path,
    checkpoint: str | Path,
    output: str | Path,
    device: str,
    max_scans: int | None,
    seed: int = 42,
    manifest: str | Path | None = None,
    python: str = sys.executable,
) -> list[str]:
    command = [
        python,
        "-m",
        "ml.cli",
        "benchmark",
        "--dataset",
        str(dataset),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--device",
        str(device),
        "--seed",
        str(int(seed)),
    ]
    if max_scans:
        command.extend(["--max-scans", str(int(max_scans))])
    if manifest:
        command.extend(["--manifest", str(manifest)])
    return command


@dataclass(frozen=True)
class ProcessSnapshot:
    running: bool
    cancelled: bool
    returncode: int | None
    elapsed_seconds: float
    line_count: int
    last_line: str
    events: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ProcessResult:
    command: tuple[str, ...]
    returncode: int
    cancelled: bool
    payload: dict[str, Any] | None
    events: tuple[dict[str, Any], ...]
    logs: tuple[str, ...]
    elapsed_seconds: float


@dataclass
class ProcessRunner:
    """Executa uma CLI por vez e permite cancelamento seguro."""

    cwd: Path
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _process: subprocess.Popen[str] | None = field(default=None, init=False)
    _running: bool = field(default=False, init=False)
    _cancelled: bool = field(default=False, init=False)
    _returncode: int | None = field(default=None, init=False)
    _started_at: float = field(default=0.0, init=False)
    _finished_at: float = field(default=0.0, init=False)
    _lines: list[str] = field(default_factory=list, init=False)
    _events: list[dict[str, Any]] = field(default_factory=list, init=False)

    def run(self, command: Sequence[str]) -> ProcessResult:
        with self._lock:
            if self._running:
                raise RuntimeError("Já existe uma operação em execução.")
            self._running = True
            self._cancelled = False
            self._returncode = None
            self._started_at = time.monotonic()
            self._finished_at = 0.0
            self._lines = []
            self._events = []
        try:
            process = subprocess.Popen(
                [str(item) for item in command],
                cwd=self.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            with self._lock:
                self._process = process
            assert process.stdout is not None
            for raw_line in process.stdout:
                line = raw_line.rstrip()
                if not line:
                    continue
                event = _parse_json_object(line)
                with self._lock:
                    self._lines.append(line)
                    if event is not None:
                        self._events.append(event)
            returncode = int(process.wait())
        finally:
            with self._lock:
                process = self._process
                returncode = (
                    int(process.returncode)
                    if process is not None and process.returncode is not None
                    else int(self._returncode or -1)
                )
                self._process = None
                self._running = False
                self._returncode = returncode
                self._finished_at = time.monotonic()
                elapsed = self._finished_at - self._started_at
                logs = tuple(self._lines)
                events = tuple(self._events)
                cancelled = self._cancelled
        payload = events[-1] if events else None
        return ProcessResult(
            command=tuple(str(item) for item in command),
            returncode=returncode,
            cancelled=cancelled,
            payload=payload,
            events=events,
            logs=logs,
            elapsed_seconds=elapsed,
        )

    def cancel(self) -> bool:
        with self._lock:
            process = self._process
            if process is None or not self._running:
                return False
            self._cancelled = True
            process.terminate()
            return True

    def snapshot(self) -> ProcessSnapshot:
        with self._lock:
            end = time.monotonic() if self._running else self._finished_at
            elapsed = max(0.0, end - self._started_at) if self._started_at else 0.0
            return ProcessSnapshot(
                running=self._running,
                cancelled=self._cancelled,
                returncode=self._returncode,
                elapsed_seconds=elapsed,
                line_count=len(self._lines),
                last_line=self._lines[-1] if self._lines else "",
                events=tuple(self._events),
            )


def _parse_json_object(line: str) -> dict[str, Any] | None:
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return None
    return dict(value) if isinstance(value, Mapping) else None


def format_duration(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    minutes, second = divmod(total, 60)
    hours, minute = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h {minute:02d}m"
    if minutes:
        return f"{minutes:d}m {second:02d}s"
    return f"{second:d}s"


__all__ = [
    "DatasetInfo",
    "ProcessResult",
    "ProcessRunner",
    "ProcessSnapshot",
    "available_devices",
    "build_benchmark_command",
    "build_infer_command",
    "build_train_command",
    "discover_checkpoints",
    "format_duration",
    "validate_dataset",
]

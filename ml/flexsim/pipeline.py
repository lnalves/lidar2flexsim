"""Serviço de inferência contínua que alimenta o FlexSim.

Este é o laço que a integração em tempo real exigia e que não existia: uma
fonte entrega quadros, o modelo — carregado **uma vez** — segmenta, o
tracker dá identidade aos objetos e a cena resultante é publicada.

O modo antigo, um ``python -m ml.cli infer`` por scan, não serve aqui. A
inferência em si custa poucas dezenas de milissegundos; o que dominava era
subir o interpretador e importar o PyTorch a cada chamada. Mantendo o
processo vivo, o orçamento por quadro passa a ser o do modelo.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from ..calibration import PredictionCalibrationConfig
from ..inference import inferir_scan, load_segmentation_model
from .export import DEFAULT_CONTAINER, write_flexscript_bridge, write_scene_files
from .scene import DEFAULT_FLEXSIM_OBJECTS, build_scene, load_object_map
from .server import SceneServer, SceneStore
from .sources import PointFrame, describe_source
from .tracking import ObjectTracker, TrackingConfig
from .transform import SensorPlacement


@dataclass
class StreamConfig:
    """Ajustes do laço em tempo real, separados dos ajustes de detecção."""

    device: str = "cpu"
    num_points: int | None = None
    score_threshold: float = 0.50
    cluster_eps: float = 0.35
    min_cluster_points: int = 5
    calibration: PredictionCalibrationConfig | Mapping[str, Any] | None = None
    container: str = DEFAULT_CONTAINER

    def to_dict(self) -> dict[str, Any]:
        calibration = self.calibration
        if isinstance(calibration, PredictionCalibrationConfig):
            calibration = calibration.to_dict()
        return {
            "device": self.device,
            "num_points": self.num_points,
            "score_threshold": self.score_threshold,
            "cluster_eps": self.cluster_eps,
            "min_cluster_points": self.min_cluster_points,
            "calibration": dict(calibration) if calibration else None,
            "container": self.container,
        }


@dataclass
class FrameReport:
    """O que aconteceu em um quadro, para log e para a interface."""

    frame: int
    scan_id: str | None
    objects: int
    removed: int
    detections: int
    inference_ms: float
    total_ms: float
    lag_ms: float
    fps: float

    def to_event(self) -> dict[str, Any]:
        return {
            "event": "scene",
            "frame": self.frame,
            "scan_id": self.scan_id,
            "objects": self.objects,
            "removed": self.removed,
            "detections": self.detections,
            "inference_ms": round(self.inference_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "lag_ms": round(self.lag_ms, 2),
            "fps": round(self.fps, 2),
        }


class StreamPipeline:
    """Segmenta, rastreia e publica cenas enquanto a fonte entregar quadros."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        config: StreamConfig | None = None,
        tracking: TrackingConfig | Mapping[str, Any] | None = None,
        placement: SensorPlacement | Mapping[str, Any] | None = None,
        object_map: Mapping[str, str] | None = None,
        output_dir: str | Path | None = None,
        server: SceneServer | None = None,
        model: Any | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser()
        self.config = config or StreamConfig()
        self.tracker = ObjectTracker(tracking)
        self.placement = (
            placement
            if isinstance(placement, SensorPlacement)
            else SensorPlacement.from_mapping(placement)
        )
        self.object_map = dict(object_map or DEFAULT_FLEXSIM_OBJECTS)
        self.output_dir = Path(output_dir).expanduser() if output_dir else None
        self.server = server
        self.store: SceneStore | None = server.store if server is not None else None
        # Um modelo já carregado dispensa o checkpoint em disco; é o que
        # permite a interface reaproveitar o que ela mesma abriu.
        self._model: Any | None = model
        self._model_config: Any | None = getattr(model, "config", None)
        self._running = False
        self._stopping = False
        self._frames = 0
        self._last_scene: dict[str, Any] | None = None
        self._source_label: str | None = None

    @property
    def running(self) -> bool:
        return self._running

    @property
    def last_scene(self) -> dict[str, Any] | None:
        return dict(self._last_scene) if self._last_scene is not None else None

    def prepare(self) -> "StreamPipeline":
        """Carrega o modelo e grava os artefatos fixos da ponte."""

        if self._model is None:
            self._model, self._model_config = load_segmentation_model(
                self.checkpoint, device=self.config.device
            )
        if self.output_dir is not None:
            write_flexscript_bridge(
                self.output_dir,
                container=self.config.container,
                object_map=self.object_map,
            )
        return self

    def process(self, frame: PointFrame) -> tuple[dict[str, Any], FrameReport]:
        """Roda um quadro completo e devolve a cena publicada e o relatório."""

        if self._model is None:
            self.prepare()
        started = time.perf_counter()
        result = inferir_scan(
            frame.points,
            model=self._model,
            device=self.config.device,
            num_points=self.config.num_points,
            score_threshold=self.config.score_threshold,
            cluster_eps=self.config.cluster_eps,
            min_cluster_points=self.config.min_cluster_points,
            calibration=self.config.calibration,
        )
        inference_seconds = time.perf_counter() - started
        predictions = result["predictions"]
        tracked = self.tracker.update(predictions, timestamp=frame.timestamp)
        scene = build_scene(
            tracked.tracks,
            tracked.removed,
            frame=frame.index,
            timestamp=frame.timestamp,
            placement=self.placement,
            object_map=self.object_map,
            scan_id=frame.scan_id,
            source=self._source_label,
            diagnostics={
                "input_points": frame.num_points,
                "detections": len(predictions),
                "tracking": tracked.stats,
                "inference_ms": round(inference_seconds * 1000, 2),
                "checkpoint": str(self.checkpoint),
                "device": self.config.device,
            },
        )
        self._publish(scene)
        total_seconds = time.perf_counter() - started
        self._frames += 1
        self._last_scene = scene
        report = FrameReport(
            frame=frame.index,
            scan_id=frame.scan_id,
            objects=len(scene["objects"]),
            removed=len(scene["removed"]),
            detections=len(predictions),
            inference_ms=inference_seconds * 1000,
            total_ms=total_seconds * 1000,
            lag_ms=max(0.0, time.time() - frame.timestamp) * 1000,
            fps=1.0 / total_seconds if total_seconds > 0 else 0.0,
        )
        return scene, report

    def run(
        self,
        source: Any,
        *,
        max_frames: int | None = None,
        on_frame: Callable[[FrameReport], None] | None = None,
    ) -> dict[str, Any]:
        """Consome a fonte até ela acabar, ``max_frames`` ou :meth:`stop`."""

        self.prepare()
        self._running = True
        self._stopping = False
        self._source_label = str(describe_source(source).get("name") or "")
        limit = int(max_frames) if max_frames else None
        started = time.perf_counter()
        processed = 0
        total_inference = 0.0
        try:
            for frame in _iter_frames(source):
                if self._stopping:
                    break
                _, report = self.process(frame)
                processed += 1
                total_inference += report.inference_ms
                if on_frame is not None:
                    on_frame(report)
                if limit is not None and processed >= limit:
                    break
        finally:
            self._running = False
            close = getattr(source, "close", None)
            if callable(close):
                close()
        elapsed = time.perf_counter() - started
        return {
            "event": "summary",
            "frames": processed,
            "elapsed_seconds": round(elapsed, 3),
            "fps": round(processed / elapsed, 2) if elapsed > 0 else 0.0,
            "mean_inference_ms": round(total_inference / processed, 2) if processed else 0.0,
            "tracks_active": self.tracker.active_tracks,
            "source": describe_source(source),
            "checkpoint": str(self.checkpoint),
            "stream": self.config.to_dict(),
            "tracking": self.tracker.config.to_dict(),
            "sensor": self.placement.to_dict(),
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "server_url": self.server.url if self.server else None,
        }

    def stop(self) -> None:
        """Pede o encerramento do laço no próximo quadro."""

        self._stopping = True

    def _publish(self, scene: Mapping[str, Any]) -> None:
        if self.store is not None:
            self.store.publish(scene)
        if self.output_dir is not None:
            write_scene_files(self.output_dir, scene)


def _iter_frames(source: Any) -> Iterable[PointFrame]:
    frames = getattr(source, "frames", None)
    if callable(frames):
        return frames()
    return source


def build_pipeline(
    checkpoint: str | Path,
    *,
    device: str = "cpu",
    num_points: int | None = None,
    score_threshold: float = 0.50,
    cluster_eps: float = 0.35,
    min_cluster_points: int = 5,
    calibration: Mapping[str, Any] | None = None,
    container: str = DEFAULT_CONTAINER,
    tracking: Mapping[str, Any] | None = None,
    placement: Mapping[str, Any] | None = None,
    object_map: str | Path | Mapping[str, Any] | None = None,
    output_dir: str | Path | None = None,
    server: SceneServer | None = None,
) -> StreamPipeline:
    """Atalho para montar o pipeline a partir de valores simples da CLI."""

    return StreamPipeline(
        checkpoint,
        config=StreamConfig(
            device=device,
            num_points=num_points,
            score_threshold=score_threshold,
            cluster_eps=cluster_eps,
            min_cluster_points=min_cluster_points,
            calibration=calibration,
            container=container,
        ),
        tracking=TrackingConfig.from_mapping(tracking),
        placement=SensorPlacement.from_mapping(placement),
        object_map=load_object_map(object_map),
        output_dir=output_dir,
        server=server,
    )


__all__ = ["FrameReport", "StreamConfig", "StreamPipeline", "build_pipeline"]

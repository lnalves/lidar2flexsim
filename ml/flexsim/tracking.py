"""Identidade temporal das caixas detectadas quadro a quadro.

A segmentação é feita de forma independente em cada scan: a mesma
empilhadeira vira uma caixa nova a cada rodada, sem nenhuma ligação com a
caixa anterior. Isso é suficiente para avaliar o modelo, mas não para
alimentar uma simulação — se cada atualização apagasse e recriasse os
objetos, o FlexSim perderia estatísticas e tarefas em andamento a cada 100 ms.

Este módulo associa detecções a faixas persistentes (``track_id``), suaviza a
geometria para reduzir o tremor típico de PCA em nuvens esparsas e estima a
velocidade de cada objeto, que é o que permite ao FlexSim mover um recurso em
vez de reposicioná-lo bruscamente.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DEFAULT_FRAME_INTERVAL = 0.1


@dataclass(frozen=True)
class TrackingConfig:
    """Parâmetros de associação e suavização.

    ``max_distance`` é o raio de associação no plano ``xy``, em metros: um
    objeto de armazém a 1 m/s percorre 10 cm entre quadros a 10 Hz, então o
    padrão dá margem confortável sem confundir objetos vizinhos.
    """

    max_distance: float = 1.5
    max_age: int = 5
    min_hits: int = 2
    smoothing: float = 0.5
    match_class: bool = True

    def __post_init__(self) -> None:
        distance = float(self.max_distance)
        if not math.isfinite(distance) or distance <= 0:
            raise ValueError("max_distance deve ser maior que zero")
        object.__setattr__(self, "max_distance", distance)
        age = int(self.max_age)
        if age < 0:
            raise ValueError("max_age não pode ser negativo")
        object.__setattr__(self, "max_age", age)
        hits = int(self.min_hits)
        if hits < 1:
            raise ValueError("min_hits deve ser maior que zero")
        object.__setattr__(self, "min_hits", hits)
        smoothing = float(self.smoothing)
        if not 0 < smoothing <= 1:
            raise ValueError("smoothing deve estar entre 0 (exclusivo) e 1")
        object.__setattr__(self, "smoothing", smoothing)
        object.__setattr__(self, "match_class", bool(self.match_class))

    @classmethod
    def from_mapping(
        cls, value: "TrackingConfig | Mapping[str, Any] | None" = None, **overrides: Any
    ) -> "TrackingConfig":
        if isinstance(value, cls):
            data: dict[str, Any] = {
                "max_distance": value.max_distance,
                "max_age": value.max_age,
                "min_hits": value.min_hits,
                "smoothing": value.smoothing,
                "match_class": value.match_class,
            }
        else:
            data = dict(value or {})
        data.update({key: item for key, item in overrides.items() if item is not None})
        allowed = {"max_distance", "max_age", "min_hits", "smoothing", "match_class"}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError("Opções desconhecidas do tracking: " + ", ".join(unknown))
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_distance": self.max_distance,
            "max_age": self.max_age,
            "min_hits": self.min_hits,
            "smoothing": self.smoothing,
            "match_class": self.match_class,
        }


@dataclass(frozen=True)
class TrackedObject:
    """Estado publicado de um objeto persistente, no frame do sensor."""

    track_id: int
    classe: str
    class_id: int
    score: float
    center: tuple[float, float, float]
    dimensions: tuple[float, float, float]
    yaw_rad: float
    velocity: tuple[float, float, float]
    num_points: int
    age: int
    hits: int
    misses: int
    state: str


@dataclass(frozen=True)
class RemovedTrack:
    """Faixa que saiu de cena.

    Carrega a classe junto do identificador porque o nome do objeto dentro do
    FlexSim é derivado dos dois; sem a classe não seria possível reconstruir
    o nome a destruir.
    """

    track_id: int
    classe: str


@dataclass(frozen=True)
class TrackingResult:
    """Faixas ativas confirmadas e as que deixaram de existir neste quadro."""

    tracks: tuple[TrackedObject, ...]
    removed: tuple[RemovedTrack, ...]
    stats: dict[str, Any]


@dataclass
class _Track:
    track_id: int
    classe: str
    class_id: int
    score: float
    center: np.ndarray
    dimensions: np.ndarray
    yaw_rad: float
    velocity: np.ndarray
    num_points: int
    age: int = 0
    hits: int = 1
    misses: int = 0
    published: bool = False


def _align_yaw(new_yaw: float, reference: float) -> float:
    """Aproxima ``new_yaw`` de ``reference`` a menos de meia volta.

    A caixa vem de um eixo principal de PCA, que é definido apenas a menos de
    180°. Sem esse alinhamento, um objeto parado alterna entre ``yaw`` e
    ``yaw ± π`` e a suavização produz uma rotação intermediária inexistente.
    """

    delta = new_yaw - reference
    if not math.isfinite(delta):
        return reference
    shifted = delta - math.pi * round(delta / math.pi)
    return reference + shifted


def _as_center(value: Any) -> np.ndarray:
    center = np.asarray(value, dtype=np.float64).reshape(-1)
    if center.size != 3 or not np.isfinite(center).all():
        raise ValueError("centro da predição deve ter três valores finitos")
    return center


def _as_dimensions(value: Any) -> np.ndarray:
    dimensions = np.asarray(value, dtype=np.float64).reshape(-1)
    if dimensions.size != 3 or not np.isfinite(dimensions).all():
        raise ValueError("dimensoes da predição devem ter três valores finitos")
    return np.maximum(dimensions, 1e-3)


class ObjectTracker:
    """Associa detecções a faixas persistentes por proximidade no plano ``xy``.

    A associação é gulosa por distância crescente. Com a dezena de objetos
    típica de um armazém, o custo é irrelevante e o resultado é idêntico ao
    de um algoritmo húngaro na esmagadora maioria dos quadros; um matcher
    ótimo só compensaria com dezenas de candidatos ambíguos simultâneos.
    """

    def __init__(self, config: TrackingConfig | Mapping[str, Any] | None = None) -> None:
        self.config = TrackingConfig.from_mapping(config)
        self._tracks: list[_Track] = []
        self._next_id = 1
        self._last_timestamp: float | None = None

    @property
    def active_tracks(self) -> int:
        return len(self._tracks)

    def reset(self) -> None:
        """Descarta todas as faixas mantendo a numeração já usada.

        Reaproveitar ``track_id`` depois de um reinício faria o FlexSim
        associar um objeto novo ao histórico de outro que já saiu de cena.
        """

        self._tracks.clear()
        self._last_timestamp = None

    def update(
        self,
        predictions: Iterable[Mapping[str, Any]],
        *,
        timestamp: float | None = None,
    ) -> TrackingResult:
        """Consome as predições de um quadro e devolve o estado persistente."""

        detections = [dict(item) for item in predictions]
        interval = self._interval(timestamp)
        for track in self._tracks:
            track.age += 1
        matches, unmatched = self._associate(detections)
        for track, detection in matches:
            self._apply_match(track, detection, interval)
        for detection in unmatched:
            self._spawn(detection)
        removed: list[RemovedTrack] = []
        survivors: list[_Track] = []
        for track in self._tracks:
            if track.misses > self.config.max_age:
                if track.published:
                    removed.append(RemovedTrack(track.track_id, track.classe))
                continue
            survivors.append(track)
        self._tracks = survivors
        published: list[TrackedObject] = []
        for track in self._tracks:
            if track.hits < self.config.min_hits:
                continue
            state = "new" if not track.published else ("updated" if track.misses == 0 else "coasting")
            track.published = True
            published.append(
                TrackedObject(
                    track_id=track.track_id,
                    classe=track.classe,
                    class_id=track.class_id,
                    score=float(track.score),
                    center=(float(track.center[0]), float(track.center[1]), float(track.center[2])),
                    dimensions=(
                        float(track.dimensions[0]),
                        float(track.dimensions[1]),
                        float(track.dimensions[2]),
                    ),
                    yaw_rad=float(track.yaw_rad),
                    velocity=(
                        float(track.velocity[0]),
                        float(track.velocity[1]),
                        float(track.velocity[2]),
                    ),
                    num_points=int(track.num_points),
                    age=int(track.age),
                    hits=int(track.hits),
                    misses=int(track.misses),
                    state=state,
                )
            )
        published.sort(key=lambda item: item.track_id)
        stats = {
            "detections": len(detections),
            "matched": len(matches),
            "spawned": len(unmatched),
            "active": len(self._tracks),
            "published": len(published),
            "removed": len(removed),
            "interval_seconds": interval,
        }
        removed.sort(key=lambda item: item.track_id)
        return TrackingResult(tuple(published), tuple(removed), stats)

    def _interval(self, timestamp: float | None) -> float:
        if timestamp is None:
            return DEFAULT_FRAME_INTERVAL
        current = float(timestamp)
        previous = self._last_timestamp
        self._last_timestamp = current
        if previous is None:
            return DEFAULT_FRAME_INTERVAL
        interval = current - previous
        # Um relógio que anda para trás ou congela tornaria a velocidade
        # infinita; nesses casos o intervalo nominal é a estimativa honesta.
        return interval if interval > 1e-6 else DEFAULT_FRAME_INTERVAL

    def _associate(
        self, detections: Sequence[Mapping[str, Any]]
    ) -> tuple[list[tuple[_Track, Mapping[str, Any]]], list[Mapping[str, Any]]]:
        if not detections:
            for track in self._tracks:
                track.misses += 1
            return [], []
        if not self._tracks:
            return [], list(detections)
        gate = self.config.max_distance
        candidates: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self._tracks):
            predicted = track.center[:2] + track.velocity[:2] * DEFAULT_FRAME_INTERVAL
            for detection_index, detection in enumerate(detections):
                if self.config.match_class and str(detection.get("classe")) != track.classe:
                    continue
                center = _as_center(detection.get("centro"))
                distance = float(np.linalg.norm(center[:2] - predicted))
                if distance <= gate:
                    candidates.append((distance, track_index, detection_index))
        candidates.sort()
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        matches: list[tuple[_Track, Mapping[str, Any]]] = []
        for _, track_index, detection_index in candidates:
            if track_index in used_tracks or detection_index in used_detections:
                continue
            used_tracks.add(track_index)
            used_detections.add(detection_index)
            matches.append((self._tracks[track_index], detections[detection_index]))
        for index, track in enumerate(self._tracks):
            if index not in used_tracks:
                track.misses += 1
        unmatched = [
            detection
            for index, detection in enumerate(detections)
            if index not in used_detections
        ]
        return matches, unmatched

    def _apply_match(
        self, track: _Track, detection: Mapping[str, Any], interval: float
    ) -> None:
        weight = self.config.smoothing
        center = _as_center(detection.get("centro"))
        dimensions = _as_dimensions(detection.get("dimensoes"))
        yaw = _align_yaw(float(detection.get("rotacao", track.yaw_rad)), track.yaw_rad)
        previous_center = track.center
        blended_center = weight * center + (1 - weight) * previous_center
        instant_velocity = (blended_center - previous_center) / max(interval, 1e-6)
        track.center = blended_center
        track.dimensions = weight * dimensions + (1 - weight) * track.dimensions
        track.yaw_rad = weight * yaw + (1 - weight) * track.yaw_rad
        track.velocity = weight * instant_velocity + (1 - weight) * track.velocity
        track.score = float(detection.get("score", track.score))
        track.num_points = int(detection.get("num_pontos", track.num_points) or 0)
        track.class_id = int(detection.get("class_id", track.class_id) or 0)
        track.hits += 1
        track.misses = 0

    def _spawn(self, detection: Mapping[str, Any]) -> None:
        track = _Track(
            track_id=self._next_id,
            classe=str(detection.get("classe", "unknown")),
            class_id=int(detection.get("class_id", 0) or 0),
            score=float(detection.get("score", 0.0)),
            center=_as_center(detection.get("centro")),
            dimensions=_as_dimensions(detection.get("dimensoes")),
            yaw_rad=float(detection.get("rotacao", 0.0)),
            velocity=np.zeros(3, dtype=np.float64),
            num_points=int(detection.get("num_pontos", 0) or 0),
        )
        self._next_id += 1
        self._tracks.append(track)


__all__ = [
    "DEFAULT_FRAME_INTERVAL",
    "ObjectTracker",
    "RemovedTrack",
    "TrackedObject",
    "TrackingConfig",
    "TrackingResult",
]

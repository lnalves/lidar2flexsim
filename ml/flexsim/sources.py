"""Fontes de nuvens de pontos consumidas pelo pipeline em tempo real.

O sensor ainda não existe, mas a forma dos dados já é conhecida: uma matriz
``[N, 4]`` de ``x, y, z, intensidade`` por quadro. Toda a integração com o
FlexSim é escrita contra :class:`PointSource`, de modo que o driver do
sensor, quando chegar, entra como uma implementação nova sem tocar em
tracking, exportador ou servidor.

Duas implementações já vêm prontas:

:class:`ReplayPointSource`
    Reproduz o Warehouse Dataset no ritmo de um sensor real. É com ela que
    dá para validar a ponte inteira com o FlexSim antes de comprar hardware.

:class:`LivePointSource`
    Fila alimentada de fora. É o ponto de encaixe do driver: o receptor UDP
    monta um quadro por rotação e chama :meth:`LivePointSource.push`.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Protocol, Sequence, runtime_checkable

import numpy as np

from ..data import load_bin


@dataclass(frozen=True)
class PointFrame:
    """Um quadro de nuvem de pontos com a sua marcação de tempo."""

    points: np.ndarray
    index: int
    timestamp: float
    scan_id: str | None = None

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError("points deve ser uma matriz [N, >=3]")
        object.__setattr__(self, "points", points)

    @property
    def num_points(self) -> int:
        return int(len(self.points))


@runtime_checkable
class PointSource(Protocol):
    """Produtor de quadros; a implementação define de onde eles vêm."""

    name: str

    def frames(self) -> Iterator[PointFrame]:
        """Itera quadros até a fonte se esgotar ou ser fechada."""

    def close(self) -> None:
        """Libera recursos e faz :meth:`frames` terminar."""


@dataclass
class ReplayPointSource:
    """Reproduz scans ``.bin`` de um dataset no ritmo de um sensor.

    O intervalo é medido contra um relógio monotônico acumulado desde o
    início, e não com um ``sleep`` fixo por quadro: assim o atraso de um
    quadro lento não desloca permanentemente todos os seguintes.
    """

    dataset: str | Path
    rate_hz: float = 10.0
    loop: bool = False
    max_frames: int | None = None
    realtime: bool = True
    name: str = "replay"
    _closed: threading.Event = field(default_factory=threading.Event, init=False, repr=False)

    def __post_init__(self) -> None:
        rate = float(self.rate_hz)
        if not rate > 0:
            raise ValueError("rate_hz deve ser maior que zero")
        self.rate_hz = rate
        root = Path(self.dataset).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"Dataset não encontrado: {root}")
        bin_dir = root if root.name.casefold() in {"bin", "bins"} else root / "bin"
        if not bin_dir.is_dir():
            raise FileNotFoundError(f"Pasta de scans não encontrada: {bin_dir}")
        scans = sorted(bin_dir.glob("*.bin"), key=lambda item: item.stem)
        if not scans:
            raise FileNotFoundError(f"Nenhum scan .bin encontrado em {bin_dir}")
        self._scans: tuple[Path, ...] = tuple(scans)

    @property
    def scans(self) -> tuple[Path, ...]:
        return self._scans

    @property
    def interval(self) -> float:
        return 1.0 / self.rate_hz

    def frames(self) -> Iterator[PointFrame]:
        started = time.monotonic()
        index = 0
        limit = int(self.max_frames) if self.max_frames else None
        while not self._closed.is_set():
            for scan in self._scans:
                if self._closed.is_set() or (limit is not None and index >= limit):
                    return
                if self.realtime:
                    due = started + index * self.interval
                    delay = due - time.monotonic()
                    if delay > 0 and self._closed.wait(delay):
                        return
                points = load_bin(scan, num_features=4, return_features=4)
                yield PointFrame(
                    points=points,
                    index=index,
                    timestamp=time.time(),
                    scan_id=scan.stem,
                )
                index += 1
            if not self.loop:
                return

    def close(self) -> None:
        self._closed.set()


@dataclass
class LivePointSource:
    """Fonte alimentada por um driver externo, mantendo só o quadro mais novo.

    Um sensor não espera o consumidor. Se a inferência atrasar, guardar a
    fila inteira só aumentaria a defasagem entre o armazém real e a
    simulação; a fila descarta o quadro antigo e conta o descarte em
    :attr:`dropped`, que é o número honesto a mostrar na interface.
    """

    name: str = "live"
    #: A fila é ilimitada mas nunca acumula: :meth:`push` esvazia o que está
    #: pendente antes de inserir. Uma fila com ``maxsize=1`` obrigaria
    #: :meth:`close` a remover um quadro para caber a sentinela, perdendo
    #: justamente a última leitura do sensor.
    _queue: "queue.Queue[PointFrame | None]" = field(
        default_factory=queue.Queue, init=False, repr=False
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _index: int = field(default=0, init=False, repr=False)
    _dropped: int = field(default=0, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def dropped(self) -> int:
        with self._lock:
            return self._dropped

    def push(
        self,
        points: np.ndarray | Sequence[Sequence[float]],
        *,
        timestamp: float | None = None,
        scan_id: str | None = None,
    ) -> PointFrame | None:
        """Publica um quadro montado pelo driver. Devolve ``None`` se fechada."""

        with self._lock:
            if self._closed:
                return None
            index = self._index
            self._index += 1
        frame = PointFrame(
            points=np.asarray(points, dtype=np.float32),
            index=index,
            timestamp=float(timestamp if timestamp is not None else time.time()),
            scan_id=scan_id,
        )
        dropped = 0
        while True:
            try:
                stale = self._queue.get_nowait()
            except queue.Empty:
                break
            if stale is None:  # pragma: no cover - fechamento concorrente
                self._queue.put_nowait(None)
                return None
            dropped += 1
        self._queue.put_nowait(frame)
        if dropped:
            with self._lock:
                self._dropped += dropped
        return frame

    def frames(self) -> Iterator[PointFrame]:
        while True:
            frame = self._queue.get()
            if frame is None:
                return
            yield frame

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        # A sentinela desbloqueia um consumidor parado em ``get`` e entra
        # atrás do último quadro, que continua sendo entregue antes do fim.
        self._queue.put_nowait(None)


def describe_source(source: Any) -> dict[str, Any]:
    """Resumo da fonte para os diagnósticos publicados na cena."""

    info: dict[str, Any] = {"name": str(getattr(source, "name", type(source).__name__))}
    if isinstance(source, ReplayPointSource):
        info.update(
            {
                "kind": "replay",
                "dataset": str(source.dataset),
                "scans": len(source.scans),
                "rate_hz": source.rate_hz,
                "loop": bool(source.loop),
            }
        )
    elif isinstance(source, LivePointSource):
        info.update({"kind": "live", "dropped": source.dropped})
    else:
        info.setdefault("kind", "custom")
    return info


__all__ = [
    "LivePointSource",
    "PointFrame",
    "PointSource",
    "ReplayPointSource",
    "describe_source",
]

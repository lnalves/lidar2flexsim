"""Geometria de caixas 3D orientadas usadas pelo Warehouse LiDAR.

O dataset fornece caixas no formato ``centro + dimensões + yaw``.  Este
módulo implementa a geometria com NumPy para ser compartilhado pelo treino,
pela inferência e pelos testes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


def _vector3(value: Sequence[float] | np.ndarray, name: str) -> tuple[float, float, float]:
    """Normaliza um vetor de três números para uma tupla imutável."""

    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} deve conter exatamente três valores.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contém valores não finitos.")
    return tuple(float(item) for item in array)


@dataclass(frozen=True)
class OrientedBox:
    """Uma bounding box 3D alinhada em ``z`` e rotacionada pelo yaw.

    ``dimensions`` são os comprimentos dos eixos locais ``x, y, z``.  O yaw
    é medido em radianos no plano ``xy`` e segue a convenção do dataset.
    """

    class_name: str
    center: tuple[float, float, float]
    dimensions: tuple[float, float, float]
    yaw_rad: float = 0.0

    def __post_init__(self) -> None:
        if not str(self.class_name).strip():
            raise ValueError("class_name não pode ser vazio.")
        object.__setattr__(self, "class_name", str(self.class_name).strip())
        object.__setattr__(self, "center", _vector3(self.center, "center"))
        dims = _vector3(self.dimensions, "dimensions")
        if any(value <= 0.0 for value in dims):
            raise ValueError("dimensions deve conter valores maiores que zero.")
        object.__setattr__(self, "dimensions", dims)
        try:
            yaw = float(self.yaw_rad)
        except (TypeError, ValueError) as exc:
            raise ValueError("yaw_rad deve ser numérico.") from exc
        if not math.isfinite(yaw):
            raise ValueError("yaw_rad deve ser finito.")
        object.__setattr__(self, "yaw_rad", yaw)

    @property
    def volume(self) -> float:
        """Volume da caixa em metros cúbicos."""

        return float(np.prod(self.dimensions))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "OrientedBox":
        """Cria uma caixa a partir de um mapping de label/JSON.

        São aceitos nomes em português e inglês, o que permite reutilizar
        predições exportadas pelo pipeline atual.
        """

        def first(*names: str, default: object = None) -> object:
            for name in names:
                if name in value:
                    return value[name]
            return default

        class_name = first("class_name", "classe", "class", "label")
        center = first("center", "centro")
        dimensions = first("dimensions", "dimensoes", "size")
        yaw = first("yaw_rad", "yaw", "rotation_z", default=0.0)
        if class_name is None or center is None or dimensions is None:
            raise ValueError("Mapping de caixa requer classe, centro e dimensões.")
        # O pipeline exporta rotation_z em graus; somente esse nome é tratado
        # como grau.  ``yaw``/``yaw_rad`` seguem a unidade do dataset.
        if "rotation_z" in value and "yaw_rad" not in value and "yaw" not in value:
            yaw = math.radians(float(yaw))
        return cls(str(class_name), center, dimensions, float(yaw))  # type: ignore[arg-type]


def _coerce_box(box: OrientedBox | Mapping[str, object] | Sequence[object]) -> OrientedBox:
    """Converte formas de caixa públicas para :class:`OrientedBox`."""

    if isinstance(box, OrientedBox):
        return box
    if isinstance(box, Mapping):
        return OrientedBox.from_mapping(box)
    values = list(box)
    if len(values) != 8:
        raise ValueError("Caixa sequencial deve conter classe + 7 valores.")
    return OrientedBox(str(values[0]), values[1:4], values[4:7], float(values[7]))  # type: ignore[arg-type]


def points_in_oriented_box(
    points: np.ndarray | Sequence[Sequence[float]],
    box: OrientedBox | Mapping[str, object] | Sequence[object],
    *,
    tolerance: float = 0.0,
) -> np.ndarray:
    """Retorna uma máscara indicando os pontos dentro de uma caixa orientada.

    A rotação é feita no sentido inverso do yaw para levar os pontos ao
    referencial local da caixa.  A tolerância, em metros, é útil para reduzir
    perdas na fronteira devido ao ruído de quantização do LiDAR.
    """

    if tolerance < 0 or not math.isfinite(float(tolerance)):
        raise ValueError("tolerance deve ser um número não negativo.")
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError("points deve ter formato (N, >=3).")
    current = _coerce_box(box)
    center = np.asarray(current.center, dtype=np.float64)
    dimensions = np.asarray(current.dimensions, dtype=np.float64)
    delta = array[:, :3] - center
    c, s = math.cos(current.yaw_rad), math.sin(current.yaw_rad)
    local_x = c * delta[:, 0] + s * delta[:, 1]
    local_y = -s * delta[:, 0] + c * delta[:, 1]
    half = dimensions / 2.0 + float(tolerance)
    return (
        (np.abs(local_x) <= half[0])
        & (np.abs(local_y) <= half[1])
        & (np.abs(delta[:, 2]) <= half[2])
    )

points_inside_oriented_box = points_in_oriented_box
pontos_na_caixa_orientada = points_in_oriented_box

__all__ = [
    "OrientedBox",
    "points_in_oriented_box",
    "points_inside_oriented_box",
    "pontos_na_caixa_orientada",
]

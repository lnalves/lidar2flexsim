"""Transformação do frame do sensor para o frame do modelo FlexSim.

O detector devolve caixas em metros no referencial do LiDAR: origem no
sensor, ``yaw`` em radianos no plano ``xy`` e ``z`` para cima. O FlexSim
trabalha em graus, com a origem do modelo em outro ponto e — dependendo da
montagem — com o sensor girado em relação ao armazém. Esta camada é a única
que conhece essa diferença; todo o resto do pacote continua em metros e
radianos.

Um ponto merece atenção: o FlexSim posiciona objetos pelo **canto** da
bounding box, não pelo centro, e a rotação acontece em torno desse canto. A
função :func:`flexsim_corner` faz a conversão explicitamente para que o
FlexScript gerado não precise adivinhar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


def _vector3(value: Sequence[float] | np.ndarray, name: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} deve conter exatamente três valores.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contém valores não finitos.")
    return (float(array[0]), float(array[1]), float(array[2]))


@dataclass(frozen=True)
class SensorPlacement:
    """Posição e orientação do sensor dentro do modelo FlexSim.

    ``translation`` e ``scale`` são aplicados depois de ``yaw_deg``: um ponto
    do sensor vira ``translation + scale * R(yaw) @ ponto``. ``scale`` existe
    para modelos que não usam metros; mantenha em 1,0 no caso normal.
    """

    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw_deg: float = 0.0
    scale: float = 1.0
    #: Nome do referencial de destino, apenas informativo no payload.
    frame: str = "flexsim_model"

    def __post_init__(self) -> None:
        object.__setattr__(self, "translation", _vector3(self.translation, "translation"))
        yaw = float(self.yaw_deg)
        if not math.isfinite(yaw):
            raise ValueError("yaw_deg deve ser finito.")
        object.__setattr__(self, "yaw_deg", yaw)
        scale = float(self.scale)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("scale deve ser maior que zero.")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "frame", str(self.frame).strip() or "flexsim_model")

    @classmethod
    def from_mapping(
        cls, value: "SensorPlacement | Mapping[str, Any] | None" = None, **overrides: Any
    ) -> "SensorPlacement":
        if isinstance(value, cls):
            data: dict[str, Any] = {
                "translation": value.translation,
                "yaw_deg": value.yaw_deg,
                "scale": value.scale,
                "frame": value.frame,
            }
        else:
            data = dict(value or {})
        data.update({key: item for key, item in overrides.items() if item is not None})
        allowed = {"translation", "yaw_deg", "scale", "frame"}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError("Opções desconhecidas do sensor: " + ", ".join(unknown))
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "translation": list(self.translation),
            "yaw_deg": self.yaw_deg,
            "scale": self.scale,
            "frame": self.frame,
        }

    @property
    def yaw_rad(self) -> float:
        return math.radians(self.yaw_deg)

    def transform_center(
        self, center: Sequence[float] | np.ndarray
    ) -> tuple[float, float, float]:
        """Leva o centro de uma caixa do frame do sensor para o do modelo."""

        x, y, z = _vector3(center, "center")
        cosine, sine = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        return (
            self.translation[0] + self.scale * (cosine * x - sine * y),
            self.translation[1] + self.scale * (sine * x + cosine * y),
            self.translation[2] + self.scale * z,
        )

    def transform_dimensions(
        self, dimensions: Sequence[float] | np.ndarray
    ) -> tuple[float, float, float]:
        """Escala as dimensões; a rotação em ``z`` não altera os comprimentos."""

        dx, dy, dz = _vector3(dimensions, "dimensions")
        return (self.scale * dx, self.scale * dy, self.scale * dz)

    def transform_yaw_deg(self, yaw_rad: float) -> float:
        """Converte o yaw do sensor em graus no referencial do modelo."""

        value = float(yaw_rad)
        if not math.isfinite(value):
            raise ValueError("yaw_rad deve ser finito.")
        return _wrap_degrees(math.degrees(value) + self.yaw_deg)

    def transform_velocity(
        self, velocity: Sequence[float] | np.ndarray
    ) -> tuple[float, float, float]:
        """Rotaciona e escala uma velocidade; a translação não se aplica."""

        vx, vy, vz = _vector3(velocity, "velocity")
        cosine, sine = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        return (
            self.scale * (cosine * vx - sine * vy),
            self.scale * (sine * vx + cosine * vy),
            self.scale * vz,
        )


IDENTITY_PLACEMENT = SensorPlacement()


def _wrap_degrees(value: float) -> float:
    """Normaliza um ângulo para ``[0, 360)``, como o FlexSim reporta."""

    wrapped = math.fmod(float(value), 360.0)
    return wrapped + 360.0 if wrapped < 0 else wrapped


def flexsim_corner(
    center: Sequence[float] | np.ndarray,
    size: Sequence[float] | np.ndarray,
    rotation_deg: float,
) -> tuple[float, float, float]:
    """Converte ``centro + dimensões + rotação`` no ponto de ``Object.location``.

    Um objeto FlexSim de tamanho ``(sx, sy, sz)`` ancorado em ``location``
    ocupa ``x .. x + sx``, ``y - sy .. y`` e ``z .. z + sz`` no seu próprio
    referencial, e a rotação em ``z`` gira em torno de ``location``. Logo o
    centro está em ``location + R(rot) @ (sx/2, -sy/2, sz/2)`` e o canto sai
    subtraindo esse mesmo vetor já rotacionado.

    Se o modelo do professor usar outra convenção de âncora, este é o único
    lugar a mudar: o payload continua carregando ``center`` intacto.
    """

    cx, cy, cz = _vector3(center, "center")
    sx, sy, sz = _vector3(size, "size")
    angle = math.radians(float(rotation_deg))
    cosine, sine = math.cos(angle), math.sin(angle)
    local_x, local_y = 0.5 * sx, -0.5 * sy
    return (
        cx - (cosine * local_x - sine * local_y),
        cy - (sine * local_x + cosine * local_y),
        cz - 0.5 * sz,
    )


__all__ = [
    "IDENTITY_PLACEMENT",
    "SensorPlacement",
    "flexsim_corner",
]

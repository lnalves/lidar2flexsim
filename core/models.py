"""Modelos de dados compartilhados pelos serviços da aplicação.

Este módulo não importa Open3D nem qualquer outra dependência pesada. Assim,
componentes de interface podem importar os tipos e validar parâmetros mesmo em
um ambiente que ainda não tenha o runtime de processamento instalado.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping


class ParameterError(ValueError):
    """Indica que um parâmetro do pipeline está fora do domínio válido."""


class PipelineServiceError(RuntimeError):
    """Erro de alto nível produzido por um serviço do pipeline."""


def _coerce_bool(value: Any) -> bool:
    """Converte valores de controles/JSON em bool sem tratar ``"false"`` como true."""
    if isinstance(value, str):
        normalizado = value.strip().casefold()
        if normalizado in {"false", "0", "no", "nao", "não", "off", ""}:
            return False
        if normalizado in {"true", "1", "yes", "sim", "on"}:
            return True
    return bool(value)


@dataclass(frozen=True)
class PipelineParameters:
    """Parâmetros geométricos usados no processamento de um scan.

    Os valores padrão são os sugeridos para o Warehouse LiDAR Dataset. Além
    dos parâmetros do DBSCAN, a camada de serviço expõe a tolerância e o
    quantil do piso, o filtro de outliers e o espaço de clusterização. O
    ``detector_backend=pointnet2`` ativa o segmentador supervisionado e usa
    ``model_checkpoint`` para localizar os pesos treinados.
    """

    voxel: float = 0.05
    eps: float = 0.25
    min_points: int = 20
    plane_distance: float = 0.05
    oriented_box: bool = False
    max_ground_tilt_deg: float = 25.0
    ground_quantile: float = 0.30
    remove_outliers: bool = True
    outlier_neighbors: int = 12
    outlier_std_ratio: float = 2.5
    cluster_mode: str = "3d"
    detector_backend: str = "heuristic"
    model_checkpoint: str | None = None
    device: str = "auto"
    score_threshold: float = 0.50
    num_points: int = 4096

    def __post_init__(self) -> None:
        """Valida invariantes para falhar antes de iniciar Open3D."""
        valores_positivos = {
            "voxel": self.voxel,
            "eps": self.eps,
            "plane_distance": self.plane_distance,
            "max_ground_tilt_deg": self.max_ground_tilt_deg,
            "outlier_std_ratio": self.outlier_std_ratio,
        }
        for nome, valor in valores_positivos.items():
            try:
                numerico = float(valor)
            except (TypeError, ValueError) as exc:
                raise ParameterError(f"{nome} deve ser numérico.") from exc
            if not math.isfinite(numerico) or numerico <= 0:
                raise ParameterError(f"{nome} deve ser maior que zero.")
            object.__setattr__(self, nome, numerico)

        if not math.isfinite(float(self.max_ground_tilt_deg)) or self.max_ground_tilt_deg >= 90:
            raise ParameterError("max_ground_tilt_deg deve ser menor que 90 graus.")
        if not math.isfinite(float(self.ground_quantile)) or not 0.01 <= float(self.ground_quantile) <= 0.9:
            raise ParameterError("ground_quantile deve estar entre 0,01 e 0,90.")
        object.__setattr__(self, "ground_quantile", float(self.ground_quantile))

        try:
            pontos = int(self.min_points)
        except (TypeError, ValueError) as exc:
            raise ParameterError("min_points deve ser um inteiro positivo.") from exc
        if pontos < 1:
            raise ParameterError("min_points deve ser maior que zero.")
        object.__setattr__(self, "min_points", pontos)

        try:
            vizinhos = int(self.outlier_neighbors)
        except (TypeError, ValueError) as exc:
            raise ParameterError("outlier_neighbors deve ser um inteiro não negativo.") from exc
        if vizinhos < 0:
            raise ParameterError("outlier_neighbors não pode ser negativo.")
        object.__setattr__(self, "outlier_neighbors", vizinhos)
        object.__setattr__(self, "oriented_box", _coerce_bool(self.oriented_box))
        object.__setattr__(self, "remove_outliers", _coerce_bool(self.remove_outliers))

        modo = str(self.cluster_mode).strip().casefold()
        aliases_modo = {"3d": "3d", "xyz": "3d", "bev": "bev", "xy": "bev", "2d": "bev"}
        if modo not in aliases_modo:
            raise ParameterError("cluster_mode deve ser '3d' ou 'bev'.")
        object.__setattr__(self, "cluster_mode", aliases_modo[modo])

        backend = str(self.detector_backend).strip().casefold()
        aliases_backend = {
            "heuristic": "heuristic", "heuristica": "heuristic",
            "heurístico": "heuristic", "pointnet2": "pointnet2",
            "pointnet++": "pointnet2", "pointnet": "pointnet2",
        }
        if backend not in aliases_backend:
            raise ParameterError("detector_backend deve ser 'heuristic' ou 'pointnet2'.")
        object.__setattr__(self, "detector_backend", aliases_backend[backend])

        checkpoint = self.model_checkpoint
        if checkpoint is not None:
            checkpoint = str(checkpoint).strip() or None
        object.__setattr__(self, "model_checkpoint", checkpoint)

        dispositivo = str(self.device).strip().casefold()
        if dispositivo not in {"auto", "cpu", "cuda", "mps"}:
            raise ParameterError("device deve ser 'auto', 'cpu', 'cuda' ou 'mps'.")
        object.__setattr__(self, "device", dispositivo)
        if not math.isfinite(float(self.score_threshold)) or not 0.0 <= float(self.score_threshold) <= 1.0:
            raise ParameterError("score_threshold deve estar entre 0 e 1.")
        object.__setattr__(self, "score_threshold", float(self.score_threshold))
        try:
            quantidade = int(self.num_points)
        except (TypeError, ValueError) as exc:
            raise ParameterError("num_points deve ser um inteiro positivo.") from exc
        if quantidade < 32:
            raise ParameterError("num_points deve ser pelo menos 32.")
        object.__setattr__(self, "num_points", quantidade)

        # A dataclass é imutável depois da construção; a normalização acima
        # usa ``object.__setattr__`` apenas durante ``__post_init__`` para que
        # criação direta e criação via ``from_value`` tenham o mesmo contrato.

    @classmethod
    def from_value(
        cls,
        value: "PipelineParameters | Mapping[str, Any] | None" = None,
        **overrides: Any,
    ) -> "PipelineParameters":
        """Cria parâmetros a partir de dataclass, mapping ou valores avulsos.

        ``oriented`` é aceito como alias histórico de ``oriented_box``. Os
        aliases ``plane_dist`` e ``min_cluster_points`` também são aceitos
        para facilitar integração com controles da interface.
        """
        if value is None:
            dados: dict[str, Any] = {}
        elif isinstance(value, cls):
            dados = {
                "voxel": value.voxel,
                "eps": value.eps,
                "min_points": value.min_points,
                "plane_distance": value.plane_distance,
                "oriented_box": value.oriented_box,
                "max_ground_tilt_deg": value.max_ground_tilt_deg,
                "ground_quantile": value.ground_quantile,
                "remove_outliers": value.remove_outliers,
                "outlier_neighbors": value.outlier_neighbors,
                "outlier_std_ratio": value.outlier_std_ratio,
                "cluster_mode": value.cluster_mode,
                "detector_backend": value.detector_backend,
                "model_checkpoint": value.model_checkpoint,
                "device": value.device,
                "score_threshold": value.score_threshold,
                "num_points": value.num_points,
            }
        elif isinstance(value, Mapping):
            dados = dict(value)
        else:
            raise ParameterError(
                "parametros deve ser PipelineParameters, mapping ou None."
            )

        dados.update(overrides)
        aliases = {
            "oriented": "oriented_box",
            "oriented-box": "oriented_box",
            "plane_dist": "plane_distance",
            "min_cluster_points": "min_points",
            "max_ground_tilt": "max_ground_tilt_deg",
            "ground_tilt": "max_ground_tilt_deg",
            "ground_q": "ground_quantile",
            "outlier_k": "outlier_neighbors",
            "outlier_std": "outlier_std_ratio",
            "cluster_space": "cluster_mode",
            "cluster-mode": "cluster_mode",
            "backend": "detector_backend",
            "model": "model_checkpoint",
            "checkpoint": "model_checkpoint",
            "threshold": "score_threshold",
            "points": "num_points",
        }
        for antigo, novo in aliases.items():
            if antigo in dados and novo not in dados:
                dados[novo] = dados.pop(antigo)

        campos = {
            "voxel", "eps", "min_points", "plane_distance", "oriented_box",
            "max_ground_tilt_deg", "ground_quantile", "remove_outliers",
            "outlier_neighbors", "outlier_std_ratio", "cluster_mode",
            "detector_backend", "model_checkpoint", "device",
            "score_threshold", "num_points",
        }
        desconhecidos = sorted(set(dados) - campos)
        if desconhecidos:
            raise ParameterError(
                "Parâmetros desconhecidos: " + ", ".join(desconhecidos)
            )

        try:
            return cls(
                voxel=float(dados.get("voxel", cls.voxel)),
                eps=float(dados.get("eps", cls.eps)),
                min_points=int(dados.get("min_points", cls.min_points)),
                plane_distance=float(
                    dados.get("plane_distance", cls.plane_distance)
                ),
                oriented_box=_coerce_bool(dados.get("oriented_box", cls.oriented_box)),
                max_ground_tilt_deg=float(
                    dados.get("max_ground_tilt_deg", cls.max_ground_tilt_deg)
                ),
                ground_quantile=float(
                    dados.get("ground_quantile", cls.ground_quantile)
                ),
                remove_outliers=_coerce_bool(
                    dados.get("remove_outliers", cls.remove_outliers)
                ),
                outlier_neighbors=int(
                    dados.get("outlier_neighbors", cls.outlier_neighbors)
                ),
                outlier_std_ratio=float(
                    dados.get("outlier_std_ratio", cls.outlier_std_ratio)
                ),
                cluster_mode=str(dados.get("cluster_mode", cls.cluster_mode)),
                detector_backend=str(
                    dados.get("detector_backend", cls.detector_backend)
                ),
                model_checkpoint=dados.get("model_checkpoint", cls.model_checkpoint),
                device=str(dados.get("device", cls.device)),
                score_threshold=float(
                    dados.get("score_threshold", cls.score_threshold)
                ),
                num_points=int(dados.get("num_points", cls.num_points)),
            )
        except (TypeError, ValueError) as exc:
            raise ParameterError(f"Parâmetros inválidos: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        """Retorna representação JSON-serializável dos parâmetros."""
        return {
            "voxel": float(self.voxel),
            "eps": float(self.eps),
            "min_points": int(self.min_points),
            "plane_distance": float(self.plane_distance),
            "oriented_box": bool(self.oriented_box),
            "max_ground_tilt_deg": float(self.max_ground_tilt_deg),
            "ground_quantile": float(self.ground_quantile),
            "remove_outliers": bool(self.remove_outliers),
            "outlier_neighbors": int(self.outlier_neighbors),
            "outlier_std_ratio": float(self.outlier_std_ratio),
            "cluster_mode": self.cluster_mode,
            "detector_backend": self.detector_backend,
            "model_checkpoint": self.model_checkpoint,
            "device": self.device,
            "score_threshold": float(self.score_threshold),
            "num_points": int(self.num_points),
        }


@dataclass
class Progress:
    """Evento de progresso enviado pelos serviços.

    ``current`` e ``total`` são contadores de scans. ``percent`` é sempre
    normalizado para o intervalo [0, 100]. ``stage`` identifica a etapa atual
    (por exemplo, ``carregando`` ou ``clusterizando``), e ``scan_id`` pode ser
    nulo no início/fim do processamento.
    """

    current: int = 0
    total: int = 0
    message: str = ""
    stage: str = ""
    scan_id: str | None = None
    percent: float = 0.0
    result: dict[str, Any] | None = None
    error: str | None = None
    cancelled: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normaliza o percentual e evita valores inválidos na UI."""
        if self.total > 0 and self.percent == 0.0:
            self.percent = (float(self.current) / float(self.total)) * 100.0
        self.percent = min(100.0, max(0.0, float(self.percent)))

    def to_dict(self) -> dict[str, Any]:
        """Converte o evento em dicionário adequado para JSON/UI."""
        dados: dict[str, Any] = {
            "current": int(self.current),
            "total": int(self.total),
            "percent": self.percent,
            "message": self.message,
            "stage": self.stage,
            "scan_id": self.scan_id,
            "cancelled": bool(self.cancelled),
        }
        if self.result is not None:
            dados["result"] = self.result
        if self.error is not None:
            dados["error"] = self.error
        dados.update(self.extra)
        return dados

    # ``Progress`` é passado diretamente aos callbacks. Estes métodos
    # permitem que uma UI trate o evento como dataclass (``evento.percent``)
    # ou como mapping (``evento["percent"]``/``evento.get(...)``).
    def __getitem__(self, chave: str) -> Any:
        return self.to_dict()[chave]

    def get(self, chave: str, padrao: Any = None) -> Any:
        return self.to_dict().get(chave, padrao)


ProgressCallback = Callable[[Progress], None]


def caminho_str(caminho: str | Path) -> str:
    """Normaliza um caminho para texto sem resolver symlinks."""
    return str(Path(caminho).expanduser())


__all__ = [
    "ParameterError",
    "PipelineParameters",
    "PipelineServiceError",
    "Progress",
    "ProgressCallback",
    "caminho_str",
]

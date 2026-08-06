"""Ponte entre o detector PointNet++ e uma simulação FlexSim ao vivo.

O caminho completo é:

``PointSource`` → ``StreamPipeline`` → ``ObjectTracker`` → ``build_scene``
→ ``SceneServer`` (HTTP) e/ou ``write_scene_files`` (disco) → FlexScript.

As predições continuam saindo de :func:`ml.inference.inferir_scan` no
formato já documentado (``classe``, ``class_id``, ``score``, ``centro``,
``dimensoes``, ``rotacao``, ``num_pontos``). Este pacote acrescenta o que
faltava para uma simulação: identidade dos objetos entre quadros,
transformação para o referencial do modelo e transporte.

Nada aqui importa PyTorch no nível do módulo; só :mod:`ml.flexsim.pipeline`
carrega o modelo, e apenas quando é usado.
"""

from __future__ import annotations

from .export import (
    BRIDGE_SCRIPT_NAME,
    DEFAULT_CONTAINER,
    SCENE_CSV_NAME,
    SCENE_JSON_NAME,
    build_flexscript_bridge,
    write_atomic,
    write_box_stl,
    write_flexscript_bridge,
    write_object_map,
    write_scene_files,
)
from .scene import (
    CSV_COLUMNS,
    DEFAULT_FLEXSIM_OBJECTS,
    SCENE_FORMAT,
    build_scene,
    empty_scene,
    flexsim_class_for,
    load_object_map,
    object_name,
    scene_to_csv,
    scene_to_json,
)
from .server import SceneServer, SceneStore
from .sources import (
    LivePointSource,
    PointFrame,
    PointSource,
    ReplayPointSource,
    describe_source,
)
from .tracking import (
    ObjectTracker,
    RemovedTrack,
    TrackedObject,
    TrackingConfig,
    TrackingResult,
)
from .transform import IDENTITY_PLACEMENT, SensorPlacement, flexsim_corner

__all__ = [
    "BRIDGE_SCRIPT_NAME",
    "CSV_COLUMNS",
    "DEFAULT_CONTAINER",
    "DEFAULT_FLEXSIM_OBJECTS",
    "IDENTITY_PLACEMENT",
    "LivePointSource",
    "ObjectTracker",
    "PointFrame",
    "PointSource",
    "RemovedTrack",
    "ReplayPointSource",
    "SCENE_CSV_NAME",
    "SCENE_FORMAT",
    "SCENE_JSON_NAME",
    "SceneServer",
    "SceneStore",
    "SensorPlacement",
    "TrackedObject",
    "TrackingConfig",
    "TrackingResult",
    "build_flexscript_bridge",
    "build_scene",
    "describe_source",
    "empty_scene",
    "flexsim_class_for",
    "flexsim_corner",
    "load_object_map",
    "object_name",
    "scene_to_csv",
    "scene_to_json",
    "write_atomic",
    "write_box_stl",
    "write_flexscript_bridge",
    "write_object_map",
    "write_scene_files",
]


def __getattr__(name: str):
    """Expõe o pipeline sem forçar o import do PyTorch ao carregar o pacote."""

    if name in {"FrameReport", "StreamConfig", "StreamPipeline", "build_pipeline"}:
        from . import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

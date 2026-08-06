"""Contrato de cena trocado com o FlexSim.

Uma cena descreve o armazém em um instante: quais objetos existem, onde
estão no referencial do modelo e quais deixaram de ser vistos. O payload é
publicado em duas serializações equivalentes:

``scene.json``
    Forma canônica, com tudo que o pipeline sabe. É o formato para logs,
    testes e qualquer consumidor que não seja o FlexSim.

``scene.csv``
    Uma tabela plana e sem aspas. O FlexScript lê arquivos linha a linha com
    facilidade, mas não tem um parser de JSON garantido em todas as versões;
    o CSV existe para que a ponte funcione em qualquer instalação do FlexSim
    sem depender de módulo opcional.

As duas carregam a mesma informação e são geradas do mesmo dicionário, então
não há como uma divergir da outra.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..classes import WAREHOUSE_CLASS_NAMES
from .tracking import RemovedTrack, TrackedObject
from .transform import IDENTITY_PLACEMENT, SensorPlacement, flexsim_corner


SCENE_FORMAT = "flexsim-scene-v1"

#: Tipo de objeto FlexSim criado para cada classe do Warehouse Dataset.
#: As quatro classes móveis viram ``Transporter`` porque o que interessa à
#: simulação é o recurso que se desloca; ``Box`` é carga estática e vira um
#: ``VisualTool``. O mapa é substituível por JSON — veja :func:`load_object_map`.
DEFAULT_FLEXSIM_OBJECTS: dict[str, str] = {
    "Box": "VisualTool",
    "ELFplusplus": "Transporter",
    "CargoBike": "Transporter",
    "FTS": "Transporter",
    "ForkLift": "Transporter",
    "unknown": "VisualTool",
}

CSV_COLUMNS: tuple[str, ...] = (
    "name",
    "track_id",
    "classe",
    "flexsim_class",
    "state",
    "score",
    "loc_x",
    "loc_y",
    "loc_z",
    "center_x",
    "center_y",
    "center_z",
    "size_x",
    "size_y",
    "size_z",
    "rot_z",
    "vel_x",
    "vel_y",
    "vel_z",
    "num_points",
)


def load_object_map(source: str | Path | Mapping[str, Any] | None = None) -> dict[str, str]:
    """Carrega o mapa classe → objeto FlexSim, partindo do padrão.

    Aceita tanto o JSON completo usado pela CLI (com a chave
    ``flexsim_objects``) quanto um dicionário direto de classe para tipo.
    Chaves ausentes mantêm o valor padrão, de modo que sobrescrever uma
    classe não obriga a redeclarar as outras.
    """

    mapping: dict[str, str] = dict(DEFAULT_FLEXSIM_OBJECTS)
    if source is None:
        return mapping
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Mapa FlexSim não encontrado: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = dict(source)
    if not isinstance(data, Mapping):
        raise ValueError("O mapa FlexSim deve ser um objeto JSON.")
    entries = data.get("flexsim_objects", data)
    if not isinstance(entries, Mapping):
        raise ValueError("A configuração deve conter 'flexsim_objects' como objeto JSON.")
    for key, value in entries.items():
        name = str(key).strip()
        target = str(value).strip()
        if not name or not target:
            raise ValueError("Nomes de classe e de objeto FlexSim não podem ser vazios.")
        mapping[name] = target
    return mapping


def flexsim_class_for(classe: str, object_map: Mapping[str, str] | None = None) -> str:
    mapping = object_map or DEFAULT_FLEXSIM_OBJECTS
    return str(
        mapping.get(str(classe), mapping.get("unknown", DEFAULT_FLEXSIM_OBJECTS["unknown"]))
    )


def object_name(track_id: int, classe: str) -> str:
    """Nome estável do objeto dentro do modelo FlexSim.

    O FlexSim identifica objetos por nome dentro do container, então o nome
    é a chave que liga um quadro ao seguinte. Só entram caracteres seguros.
    """

    safe = "".join(char if char.isalnum() else "_" for char in str(classe)).strip("_")
    return f"LiDAR_{safe or 'unknown'}_{int(track_id)}"


def build_scene(
    tracks: Iterable[TrackedObject],
    removed: Sequence[RemovedTrack] = (),
    *,
    frame: int = 0,
    timestamp: float = 0.0,
    placement: SensorPlacement | Mapping[str, Any] | None = None,
    object_map: Mapping[str, str] | None = None,
    scan_id: str | None = None,
    source: str | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Monta o payload de uma cena a partir das faixas persistentes."""

    sensor = (
        placement
        if isinstance(placement, SensorPlacement)
        else SensorPlacement.from_mapping(placement)
        if placement is not None
        else IDENTITY_PLACEMENT
    )
    mapping = dict(object_map or DEFAULT_FLEXSIM_OBJECTS)
    objects: list[dict[str, Any]] = []
    departed = [
        {"track_id": int(item.track_id), "name": object_name(item.track_id, item.classe)}
        for item in removed
    ]
    for track in tracks:
        center = sensor.transform_center(track.center)
        size = sensor.transform_dimensions(track.dimensions)
        rotation = sensor.transform_yaw_deg(track.yaw_rad)
        velocity = sensor.transform_velocity(track.velocity)
        objects.append(
            {
                "name": object_name(track.track_id, track.classe),
                "track_id": int(track.track_id),
                "classe": str(track.classe),
                "class_id": int(track.class_id),
                "flexsim_class": flexsim_class_for(track.classe, mapping),
                "state": str(track.state),
                "score": round(float(track.score), 4),
                "location": [round(value, 4) for value in flexsim_corner(center, size, rotation)],
                "center": [round(value, 4) for value in center],
                "size": [round(value, 4) for value in size],
                "rotation": [0.0, 0.0, round(rotation, 3)],
                "velocity": [round(value, 4) for value in velocity],
                "num_points": int(track.num_points),
                "age": int(track.age),
                "hits": int(track.hits),
                "misses": int(track.misses),
            }
        )
    scene: dict[str, Any] = {
        "format": SCENE_FORMAT,
        "frame": int(frame),
        "timestamp": float(timestamp),
        "scan_id": str(scan_id) if scan_id is not None else None,
        "source": str(source) if source is not None else None,
        "units": {"length": "m", "rotation": "deg", "velocity": "m/s"},
        "anchor": "corner",
        "sensor": sensor.to_dict(),
        "classes": list(WAREHOUSE_CLASS_NAMES),
        "flexsim_objects": mapping,
        "objects": objects,
        "removed": departed,
        "stats": {
            "objects": len(objects),
            "removed": len(departed),
        },
    }
    if diagnostics is not None:
        scene["diagnostics"] = dict(diagnostics)
    return scene


def empty_scene(
    *,
    frame: int = 0,
    timestamp: float = 0.0,
    placement: SensorPlacement | Mapping[str, Any] | None = None,
    object_map: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Cena válida sem objetos, publicada antes do primeiro quadro chegar."""

    return build_scene(
        (), (), frame=frame, timestamp=timestamp, placement=placement, object_map=object_map
    )


def _csv_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    text = str(value)
    # O formato é deliberadamente sem aspas para que o FlexScript possa
    # separar por vírgula sem um parser de verdade.
    return text.replace(",", " ").replace("\n", " ").strip()


def scene_to_csv(scene: Mapping[str, Any]) -> str:
    """Serializa a cena na tabela plana consumida pelo FlexScript.

    Linha 1: ``formato,frame,timestamp,linhas``. Linha 2: cabeçalho das
    colunas. Demais linhas: um objeto cada, incluindo os removidos com
    ``state=removed`` — assim o FlexSim descobre o que destruir lendo a
    mesma tabela, sem uma segunda estrutura.
    """

    objects = list(scene.get("objects") or [])
    departed = list(scene.get("removed") or [])
    rows: list[str] = []
    for item in objects:
        location = list(item.get("location") or [0, 0, 0])
        center = list(item.get("center") or [0, 0, 0])
        size = list(item.get("size") or [0, 0, 0])
        rotation = list(item.get("rotation") or [0, 0, 0])
        velocity = list(item.get("velocity") or [0, 0, 0])
        rows.append(
            ",".join(
                _csv_value(value)
                for value in (
                    item.get("name", ""),
                    int(item.get("track_id", 0)),
                    item.get("classe", ""),
                    item.get("flexsim_class", ""),
                    item.get("state", ""),
                    float(item.get("score", 0.0)),
                    *(float(value) for value in location),
                    *(float(value) for value in center),
                    *(float(value) for value in size),
                    float(rotation[2] if len(rotation) > 2 else 0.0),
                    *(float(value) for value in velocity),
                    int(item.get("num_points", 0)),
                )
            )
        )
    for item in departed:
        rows.append(
            ",".join(
                _csv_value(value)
                for value in (
                    item.get("name", ""),
                    int(item.get("track_id", 0)),
                    "",
                    "",
                    "removed",
                    0.0,
                    *(0.0,) * 9,
                    0.0,
                    *(0.0,) * 3,
                    0,
                )
            )
        )
    header = ",".join(
        _csv_value(value)
        for value in (
            scene.get("format", SCENE_FORMAT),
            int(scene.get("frame", 0)),
            float(scene.get("timestamp", 0.0)),
            len(rows),
        )
    )
    return "\n".join([header, ",".join(CSV_COLUMNS), *rows]) + "\n"


def scene_to_json(scene: Mapping[str, Any]) -> str:
    return json.dumps(scene, ensure_ascii=False, allow_nan=False)


__all__ = [
    "CSV_COLUMNS",
    "DEFAULT_FLEXSIM_OBJECTS",
    "SCENE_FORMAT",
    "build_scene",
    "empty_scene",
    "flexsim_class_for",
    "load_object_map",
    "object_name",
    "scene_to_csv",
    "scene_to_json",
]

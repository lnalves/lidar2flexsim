"""Artefatos gravados em disco para o FlexSim consumir.

Três coisas saem daqui:

* ``scene.json`` e ``scene.csv``, escritos de forma atômica para que o
  FlexSim nunca leia meio arquivo;
* ``lidar_bridge.txt``, o FlexScript que o professor cola uma vez no modelo
  e que passa a atualizar a cena sozinho a cada tick;
* ``caixa.stl``, uma malha de caixa unitária em NumPy puro, para quem
  preferir um shape importado ao cubo padrão do FlexSim.

A geração de STL não usa Open3D de propósito: a dependência saiu do projeto
na simplificação do pipeline e uma caixa são doze triângulos.
"""

from __future__ import annotations

import json
import os
import struct
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .scene import DEFAULT_FLEXSIM_OBJECTS, SCENE_FORMAT, scene_to_csv, scene_to_json


SCENE_JSON_NAME = "scene.json"
SCENE_CSV_NAME = "scene.csv"
BRIDGE_SCRIPT_NAME = "lidar_bridge.txt"
DEFAULT_CONTAINER = "LidarScene"


def write_atomic(path: str | Path, content: str, *, encoding: str = "utf-8") -> Path:
    """Grava um arquivo por rename atômico dentro da pasta de destino.

    O FlexSim lê em um timer, sem coordenação com o produtor. Escrever no
    lugar final direto significaria, mais cedo ou mais tarde, uma leitura de
    arquivo truncado; ``os.replace`` no mesmo sistema de arquivos troca o
    conteúdo em uma operação só.
    """

    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding=encoding,
        newline="\n",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    )
    try:
        with handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, target)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
    return target


def write_scene_files(
    directory: str | Path,
    scene: Mapping[str, Any],
    *,
    json_name: str = SCENE_JSON_NAME,
    csv_name: str = SCENE_CSV_NAME,
) -> dict[str, Path]:
    """Publica a cena nas duas serializações e devolve os caminhos escritos."""

    root = Path(directory).expanduser()
    return {
        "json": write_atomic(root / json_name, scene_to_json(scene) + "\n"),
        "csv": write_atomic(root / csv_name, scene_to_csv(scene)),
    }


def write_box_stl(
    path: str | Path,
    dimensions: Sequence[float] = (1.0, 1.0, 1.0),
    *,
    centered: bool = True,
) -> Path:
    """Grava uma caixa como STL binário, sem dependências externas.

    O FlexSim redimensiona o shape para o tamanho do objeto, então a caixa
    unitária padrão já serve para todas as classes; ``dimensions`` existe
    para quem quiser um shape com proporção fixa por classe.
    """

    size = np.asarray(dimensions, dtype=np.float64).reshape(-1)
    if size.size != 3 or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError("dimensions deve conter três valores positivos e finitos.")
    low = -size / 2 if centered else np.zeros(3)
    high = low + size
    corners = np.array(
        [
            [low[0], low[1], low[2]],
            [high[0], low[1], low[2]],
            [high[0], high[1], low[2]],
            [low[0], high[1], low[2]],
            [low[0], low[1], high[2]],
            [high[0], low[1], high[2]],
            [high[0], high[1], high[2]],
            [low[0], high[1], high[2]],
        ],
        dtype=np.float32,
    )
    faces = (
        ((0, 2, 1), (0, 3, 2), (0.0, 0.0, -1.0)),
        ((4, 5, 6), (4, 6, 7), (0.0, 0.0, 1.0)),
        ((0, 1, 5), (0, 5, 4), (0.0, -1.0, 0.0)),
        ((1, 2, 6), (1, 6, 5), (1.0, 0.0, 0.0)),
        ((2, 3, 7), (2, 7, 6), (0.0, 1.0, 0.0)),
        ((3, 0, 4), (3, 4, 7), (-1.0, 0.0, 0.0)),
    )
    triangles: list[tuple[tuple[float, float, float], np.ndarray]] = []
    for first, second, normal in faces:
        triangles.append((normal, corners[list(first)]))
        triangles.append((normal, corners[list(second)]))
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = bytearray()
    payload += b"lidar2flexsim box".ljust(80, b"\0")
    payload += struct.pack("<I", len(triangles))
    for normal, vertices in triangles:
        payload += struct.pack("<3f", *normal)
        for vertex in vertices:
            payload += struct.pack("<3f", float(vertex[0]), float(vertex[1]), float(vertex[2]))
        payload += struct.pack("<H", 0)
    target.write_bytes(bytes(payload))
    return target


def build_flexscript_bridge(
    *,
    csv_path: str | Path,
    container: str = DEFAULT_CONTAINER,
    object_map: Mapping[str, str] | None = None,
) -> str:
    """Gera o FlexScript que sincroniza o modelo com o arquivo de cena.

    O script é **idempotente**: ele cria o que falta, atualiza o que já
    existe pelo nome e destrói apenas o que a cena marcou como ``removed``.
    Era esse o ponto fraco do exportador antigo, que sempre recriava tudo e
    zerava o estado da simulação a cada atualização.

    Duas linhas dependem da versão do FlexSim e estão marcadas com
    ``AJUSTE``: a criação de objetos e a divisão da linha em campos. Elas
    ficam isoladas no topo justamente para não obrigar a caçar nada no meio
    da lógica.
    """

    mapping = dict(object_map or DEFAULT_FLEXSIM_OBJECTS)
    source = str(csv_path).replace("\\", "/")
    catalogo = "\n".join(
        f"//   {classe:<14} -> {tipo}" for classe, tipo in sorted(mapping.items())
    )
    return f"""\
// ---------------------------------------------------------------------------
// Ponte LiDAR -> FlexSim  ({SCENE_FORMAT})
// Gerado por ml.flexsim.export.build_flexscript_bridge
//
// Instalação (uma vez):
//   1. Crie no modelo um container vazio chamado "{container}"
//      (um Group ou um VisualTool serve como pai dos objetos detectados).
//   2. Cole este script em Tools > User Commands como "atualizarDoLidar".
//   3. Chame atualizarDoLidar() num Process Flow em loop, ou no evento
//      OnReset/OnTick do modelo, com o intervalo que quiser (0,1 s acompanha
//      um sensor de 10 Hz).
//
// Mapa de classes em uso:
{catalogo}
// ---------------------------------------------------------------------------

// AJUSTE 1 - criação de objeto. A primeira forma vale para FlexSim 2021+.
// Em versões anteriores, troque pela linha comentada abaixo dela.
Object lidarCriar(string tipo, Object pai, string nome)
{{
    Object novo = Object.create(tipo, pai);
    // Object novo = createinstance(library().find("?" + tipo), pai);
    novo.name = nome;
    return novo;
}}

int atualizarDoLidar()
{{
    string caminho = "{source}";
    Object pai = Model.find("{container}");
    if (!pai) {{
        print("[lidar] container '{container}' nao existe no modelo.");
        return 0;
    }}

    int arquivo = fileopen(caminho, "r");
    if (!arquivo) {{
        // Sem arquivo ainda: o pipeline pode nao ter iniciado. Nao e erro.
        return 0;
    }}

    string cabecalho = filereadline();
    // AJUSTE 2 - separação de campos. `split` existe no FlexSim moderno.
    Array meta = cabecalho.split(",");
    if (meta.length < 4 || meta[1] != "{SCENE_FORMAT}") {{
        fileclose();
        print("[lidar] formato inesperado: " + cabecalho);
        return 0;
    }}
    int totalLinhas = meta[4];
    filereadline();  // descarta a linha de nomes de coluna

    int aplicados = 0;
    for (int i = 1; i <= totalLinhas; i++) {{
        string linha = filereadline();
        if (linha == "") continue;
        Array campo = linha.split(",");
        if (campo.length < 20) continue;

        string nome     = campo[1];
        string classe   = campo[3];
        string tipo     = campo[4];
        string estado   = campo[5];

        Object alvo = pai.find(nome);

        if (estado == "removed") {{
            if (alvo) destroyobject(alvo);
            continue;
        }}

        if (!alvo) alvo = lidarCriar(tipo, pai, nome);
        if (!alvo) continue;

        alvo.setLocation(campo[7], campo[8], campo[9]);
        alvo.setSize(campo[13], campo[14], campo[15]);
        alvo.setRotation(0, 0, campo[16]);
        // Velocidade em m/s nos campos 17..19 e a contagem de pontos em 20,
        // disponiveis para logica propria (por exemplo, mover um Transporter
        // em vez de reposiciona-lo).
        aplicados++;
    }}
    fileclose();
    return aplicados;
}}
"""


def write_flexscript_bridge(
    directory: str | Path,
    *,
    csv_path: str | Path | None = None,
    container: str = DEFAULT_CONTAINER,
    object_map: Mapping[str, str] | None = None,
    name: str = BRIDGE_SCRIPT_NAME,
) -> Path:
    """Grava o FlexScript ao lado dos arquivos de cena."""

    root = Path(directory).expanduser()
    target_csv = Path(csv_path) if csv_path is not None else (root / SCENE_CSV_NAME)
    script = build_flexscript_bridge(
        csv_path=target_csv.resolve(), container=container, object_map=object_map
    )
    return write_atomic(root / name, script)


def write_object_map(path: str | Path, object_map: Mapping[str, str] | None = None) -> Path:
    """Grava o mapa classe → objeto FlexSim no formato aceito pela CLI."""

    mapping = dict(object_map or DEFAULT_FLEXSIM_OBJECTS)
    payload = json.dumps({"flexsim_objects": mapping}, ensure_ascii=False, indent=2)
    return write_atomic(path, payload + "\n")


__all__ = [
    "BRIDGE_SCRIPT_NAME",
    "DEFAULT_CONTAINER",
    "SCENE_CSV_NAME",
    "SCENE_JSON_NAME",
    "build_flexscript_bridge",
    "write_atomic",
    "write_box_stl",
    "write_flexscript_bridge",
    "write_object_map",
    "write_scene_files",
]

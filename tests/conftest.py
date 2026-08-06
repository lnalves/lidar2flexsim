"""Fixtures compartilhadas pelos testes.

O ``ui_context`` monta os elementos NiceGUI sem subir servidor web: basta um
``Client`` ativo para que ``ui.*`` tenha um slot onde inserir os elementos.
Isso permite exercitar os builders de painel do ``app.py`` como código comum.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pytest


def _new_client() -> Any:
    from nicegui.client import Client
    from nicegui.page import page

    return Client(page("/"), request=None)


@pytest.fixture
def ui_context() -> Iterator[None]:
    with _new_client():
        yield


@pytest.fixture
def ui_client() -> Any:
    """Client ainda não ativado, para testes async.

    O slot stack do NiceGUI é guardado por task asyncio, então um ``with``
    aberto na fixture síncrona não vale dentro da task do teste async: cada
    teste async precisa entrar no client no próprio corpo.
    """

    return _new_client()


@pytest.fixture
def warehouse_dataset(tmp_path: Path) -> Path:
    """Dataset mínimo com pares bin/label válidos."""

    root = tmp_path / "warehouse"
    bin_dir = root / "bin"
    label_dir = root / "label"
    bin_dir.mkdir(parents=True)
    label_dir.mkdir()
    points = np.tile(
        np.asarray(
            [
                [0.0, 0.0, 0.0, 0.10],
                [0.3, 0.1, 0.0, 0.20],
                [5.0, 5.0, 1.0, 0.30],
                [0.1, 0.2, 0.1, 0.40],
            ],
            dtype=np.float32,
        ),
        (4, 1),
    )
    for index in range(6):
        (bin_dir / f"{index:06d}.bin").write_bytes(points.tobytes())
        (label_dir / f"{index:06d}.txt").write_text(
            "Box 0 0 0 1 1 1 0\n", encoding="utf-8"
        )
    return root


@pytest.fixture
def tiny_model_config() -> Any:
    """Configuração pequena o bastante para rodar em CPU dentro do teste."""

    from ml.config import PointNet2Config

    return PointNet2Config(
        input_points=8,
        sa1_points=4,
        sa2_points=2,
        neighbors=2,
        hidden_channels=8,
    )

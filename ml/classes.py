"""Ontologia e conversões de classes do Warehouse LiDAR Dataset."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


# O índice zero é reservado ao fundo para segmentação ponto a ponto.
WAREHOUSE_CLASS_NAMES: tuple[str, ...] = (
    "background",
    "Box",
    "ELFplusplus",
    "CargoBike",
    "FTS",
    "ForkLift",
)

WAREHOUSE_CLASS_TO_INDEX: dict[str, int] = {
    name: index for index, name in enumerate(WAREHOUSE_CLASS_NAMES)
}
WAREHOUSE_INDEX_TO_CLASS: dict[int, str] = {
    index: name for index, name in enumerate(WAREHOUSE_CLASS_NAMES)
}

# O dataset usa grafias estáveis, mas entradas de usuários e alguns datasets
# derivados variam em caixa, separadores ou escrevem ForkLift como Forklift.
_ALIASES: dict[str, str] = {
    "background": "background",
    "bg": "background",
    "box": "Box",
    "metalbox": "Box",
    "metal_box": "Box",
    "elfplusplus": "ELFplusplus",
    "elf++": "ELFplusplus",
    "elf": "ELFplusplus",
    "cargobike": "CargoBike",
    "cargo_bike": "CargoBike",
    "cargo-bike": "CargoBike",
    "fts": "FTS",
    "forklift": "ForkLift",
    "fork_lift": "ForkLift",
    "fork-lift": "ForkLift",
}


def _normalized_key(value: object) -> str:
    return "".join(str(value).strip().casefold().split())


def normalize_class_name(name: str, *, strict: bool = True) -> str:
    """Retorna a grafia canônica de uma classe.

    Com ``strict=False``, classes desconhecidas são mantidas (após trim), o
    que facilita carregar labels de um dataset estendido sem perder os dados.
    """

    text = str(name).strip()
    canonical = _ALIASES.get(_normalized_key(text))
    if canonical is not None:
        return canonical
    if strict:
        classes = ", ".join(WAREHOUSE_CLASS_NAMES[1:])
        raise ValueError(f"Classe desconhecida {text!r}; esperadas: {classes}.")
    return text


def class_name_to_index(
    name: str,
    *,
    mapping: Mapping[str, int] | None = None,
    strict: bool = True,
) -> int:
    """Converte o nome de uma classe para um índice inteiro."""

    classes = mapping or WAREHOUSE_CLASS_TO_INDEX
    canonical = normalize_class_name(name, strict=strict)
    if canonical in classes:
        return int(classes[canonical])
    # Mappings personalizados podem escolher aliases em vez do nome canônico.
    for key, value in classes.items():
        if _normalized_key(key) == _normalized_key(canonical):
            return int(value)
    if strict:
        raise ValueError(f"Classe {name!r} não está no mapeamento fornecido.")
    return 0


def class_index_to_name(
    index: int,
    *,
    mapping: Mapping[int, str] | None = None,
    strict: bool = True,
) -> str:
    """Converte um índice para o nome canônico da classe."""

    classes = mapping or WAREHOUSE_INDEX_TO_CLASS
    try:
        value = classes[int(index)]
    except (KeyError, TypeError, ValueError):
        if strict:
            raise ValueError(f"Índice de classe inválido: {index!r}.") from None
        return "unknown"
    return str(value)


__all__ = [
    "WAREHOUSE_CLASS_NAMES",
    "WAREHOUSE_CLASS_TO_INDEX",
    "WAREHOUSE_INDEX_TO_CLASS",
    "class_name_to_index",
    "class_index_to_name",
    "normalize_class_name",
]

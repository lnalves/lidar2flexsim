"""Leitura e rotulagem das amostras do Warehouse LiDAR Dataset.

O pacote mantém os adaptadores legados em :mod:`warehouse` e expõe a API
PointNet++ em :mod:`pointnet`. Nenhum dos dois exige PyTorch para ser
importado; o dataset genérico só importa o framework quando solicitado.
"""

from .pointnet import (
    DEFAULT_BIN_FEATURES,
    DEFAULT_NUM_POINTS,
    PointSample,
    WarehouseDataset,
    WarehousePointDataset,
    WarehouseScan,
    assign_point_labels,
    carregar_bin,
    label_points,
    ler_labels,
    load_bin,
    load_labels,
    load_scan,
    load_warehouse_bin,
    prepare_point_sample,
    prepare_sample,
    read_label_file,
    rotular_pontos,
    sample_fixed_points,
)

from .warehouse import (
    WarehouseBox,
    WarehouseSegmentationDataset,
    load_bin_points,
    load_label_boxes,
    points_to_segmentation_labels,
    temporal_split,
)

__all__ = [
    "DEFAULT_BIN_FEATURES",
    "DEFAULT_NUM_POINTS",
    "PointSample",
    "WarehouseBox",
    "WarehouseDataset",
    "WarehousePointDataset",
    "WarehouseScan",
    "WarehouseSegmentationDataset",
    "assign_point_labels",
    "carregar_bin",
    "label_points",
    "ler_labels",
    "load_bin",
    "load_bin_points",
    "load_label_boxes",
    "load_labels",
    "load_scan",
    "load_warehouse_bin",
    "points_to_segmentation_labels",
    "prepare_point_sample",
    "prepare_sample",
    "read_label_file",
    "rotular_pontos",
    "sample_fixed_points",
    "temporal_split",
]

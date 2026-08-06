"""Treinamento, inferência e avaliação PointNet++ para nuvens LiDAR.

As importações deste pacote são leves e não exigem PyTorch. O dataset só
carrega torch quando solicitado com ``return_tensors=True``.
"""

from .classes import (
    WAREHOUSE_CLASS_NAMES,
    WAREHOUSE_CLASS_TO_INDEX,
    WAREHOUSE_INDEX_TO_CLASS,
    class_index_to_name,
    class_name_to_index,
    normalize_class_name,
)
from .data import (
    DEFAULT_BIN_FEATURES,
    DEFAULT_NUM_POINTS,
    PointSample,
    WarehouseDataset,
    WarehousePointDataset,
    WarehouseScan,
    assign_point_labels,
    load_bin,
    load_scan,
    prepare_point_sample,
    read_label_file,
    select_scan_subset,
    temporal_split,
    temporal_three_way_split,
)
from .geometry import (
    OrientedBox,
    points_in_oriented_box,
    points_inside_oriented_box,
    pontos_na_caixa_orientada,
)
from .config import PointNet2Config, TrainingConfig, load_config, save_config
from .preprocessing import PointPreprocessingConfig, preprocess_points
from .dependencies import MissingOptionalDependency, torch_available
from .checkpoints import load_checkpoint, save_checkpoint
from .inference import inferir_scan, predict_points
from .models.pointnet2_seg import PointNet2Segmentation
from .training import smoke_train, train_model
from .benchmark import (
    build_benchmark_manifest,
    load_benchmark_manifest,
    manifest_from_run,
    run_benchmark,
    save_benchmark_manifest,
)
from .calibration import PredictionCalibrationConfig, calibrate_predictions

__all__ = [
    "DEFAULT_BIN_FEATURES",
    "DEFAULT_NUM_POINTS",
    "OrientedBox",
    "PointSample",
    "WAREHOUSE_CLASS_NAMES",
    "WAREHOUSE_CLASS_TO_INDEX",
    "WAREHOUSE_INDEX_TO_CLASS",
    "WarehouseDataset",
    "WarehousePointDataset",
    "WarehouseScan",
    "assign_point_labels",
    "build_benchmark_manifest",
    "calibrate_predictions",
    "class_index_to_name",
    "class_name_to_index",
    "load_bin",
    "load_scan",
    "normalize_class_name",
    "pontos_na_caixa_orientada",
    "points_in_oriented_box",
    "points_inside_oriented_box",
    "prepare_point_sample",
    "preprocess_points",
    "read_label_file",
    "select_scan_subset",
    "temporal_split",
    "temporal_three_way_split",
    "MissingOptionalDependency",
    "PointNet2Config",
    "PointPreprocessingConfig",
    "PredictionCalibrationConfig",
    "PointNet2Segmentation",
    "TrainingConfig",
    "inferir_scan",
    "load_checkpoint",
    "load_benchmark_manifest",
    "load_config",
    "manifest_from_run",
    "predict_points",
    "save_checkpoint",
    "save_benchmark_manifest",
    "save_config",
    "smoke_train",
    "run_benchmark",
    "train_model",
    "torch_available",
]

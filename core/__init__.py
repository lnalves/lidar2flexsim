"""Serviços reutilizáveis do aplicativo LiDAR → FlexSim.

As funções são exportadas aqui para que a interface possa depender de uma API
pequena e estável, sem importar diretamente os scripts de linha de comando.
"""

from .dataset_service import (
    listar_scans,
    localizar_diretorios,
    validar_dataset,
    validate_dataset,
)
from .models import (
    ParameterError,
    PipelineParameters,
    PipelineServiceError,
    Progress,
    ProgressCallback,
)
from .pipeline_service import (
    avaliar_predicoes,
    evaluate_dataset,
    evaluate_predictions,
    export_flexsim,
    exportar_flexsim,
    process_dataset,
    process_scan,
    processar_dataset,
    processar_scan,
)

__all__ = [
    "ParameterError",
    "PipelineParameters",
    "PipelineServiceError",
    "Progress",
    "ProgressCallback",
    "avaliar_predicoes",
    "evaluate_dataset",
    "evaluate_predictions",
    "export_flexsim",
    "exportar_flexsim",
    "listar_scans",
    "localizar_diretorios",
    "process_dataset",
    "process_scan",
    "processar_dataset",
    "processar_scan",
    "validar_dataset",
    "validate_dataset",
]

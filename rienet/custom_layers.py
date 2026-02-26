"""
Backward-compatible custom layer exports.

Preferred modules:
- `rienet.trainable_layers` for layers with trainable parameters.
- `rienet.ops_layers` for deterministic operation layers.
"""

from .ops_layers import (
    NormalizationModeType,
    StandardDeviationLayer,
    CovarianceLayer,
    SpectralDecompositionLayer,
    DimensionAwareLayer,
    CustomNormalizationLayer,
    EigenvectorRescalingLayer,
    EigenProductLayer,
    EigenWeightsLayer,
    NormalizedSum,
)
from .trainable_layers import (
    LagTransformVariant,
    RecurrentCellType,
    RecurrentDirectionType,
    CorrelationTransformOutput,
    CorrelationTransformOutputType,
    DeepLayer,
    DeepRecurrentLayer,
    CorrelationEigenTransformLayer,
    LagTransformLayer,
)

__all__ = [
    "LagTransformVariant",
    "RecurrentCellType",
    "RecurrentDirectionType",
    "NormalizationModeType",
    "CorrelationTransformOutput",
    "CorrelationTransformOutputType",
    "StandardDeviationLayer",
    "CovarianceLayer",
    "SpectralDecompositionLayer",
    "DimensionAwareLayer",
    "DeepLayer",
    "DeepRecurrentLayer",
    "CustomNormalizationLayer",
    "EigenvectorRescalingLayer",
    "EigenProductLayer",
    "EigenWeightsLayer",
    "CorrelationEigenTransformLayer",
    "NormalizedSum",
    "LagTransformLayer",
]

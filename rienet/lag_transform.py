"""
Public lag-transformation module for RIEnet.

This module exposes `LagTransformLayer` as a standalone component that can be used
outside `RIEnetLayer` for return pre-processing.

The layer supports two parameterizations:
- `variant="compact"`: 5 scalar parameters and dynamic lookback length.
- `variant="per_lag"`: one parameter pair per lag index and fixed lookback inferred
  from the first built input (the time axis must be static at build time).
"""

from .trainable_layers import LagTransformLayer, LagTransformVariant

__all__ = [
    "LagTransformLayer",
    "LagTransformVariant",
]

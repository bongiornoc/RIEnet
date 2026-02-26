"""
Public package entrypoint for RIEnet.

This package exposes:
- `RIEnetLayer`: end-to-end GMV pipeline layer.
- `LagTransformLayer`: standalone lag/return non-linearity module.
- `CorrelationEigenTransformLayer`: standalone correlation cleaning module.
- `EigenWeightsLayer`: standalone weight-construction module.
- `variance_loss_function`: GMV variance objective.
"""

from .trainable_layers import (
    RIEnetLayer,
    LagTransformLayer,
    LagTransformVariant,
    CorrelationEigenTransformLayer,
)
from .ops_layers import EigenWeightsLayer
from .losses import variance_loss_function
from . import trainable_layers, ops_layers, custom_layers, losses, lag_transform
from .version import __version__

# Author information
__author__ = "Christian Bongiorno"
__email__ = "christian.bongiorno@centralesupelec.fr"

# Public API
__all__ = [
    'RIEnetLayer',
    'LagTransformLayer',
    'LagTransformVariant',
    'EigenWeightsLayer',
    'CorrelationEigenTransformLayer',
    'variance_loss_function',
    'print_citation',
    'trainable_layers',
    'ops_layers',
    'custom_layers',
    'losses',
    'lag_transform',
    '__version__'
]

# Citation reminder
def print_citation():
    """Print citation information for academic use."""
    citation = """
    Please cite the following references when using RIEnet:

    Bongiorno, C., Manolakis, E., & Mantegna, R. N. (2025).
    Neural Network-Driven Volatility Drag Mitigation under Aggressive Leverage.
    Proceedings of the 6th ACM International Conference on AI in Finance (ICAIF '25).

    Bongiorno, C., Manolakis, E., & Mantegna, R. N. (2025).
    End-to-End Large Portfolio Optimization for Variance Minimization with Neural Networks through Covariance Cleaning.
    arXiv preprint arXiv:2507.01918.

    For software citation:

    @software{rienet2025,
        title={RIEnet: A Compact Rotational Invariant Estimator Network for Global Minimum-Variance Optimisation},
        author={Christian Bongiorno},
        year={2025},
        version={VERSION},
        url={https://github.com/bongiornoc/RIEnet}
    }
    """
    print(citation.replace("VERSION", __version__))

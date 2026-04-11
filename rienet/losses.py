"""
Loss functions module for RIEnet.

This module provides the variance loss used to train RIEnet for
global minimum-variance (GMV) portfolio optimisation.

References:
-----------
Bongiorno, C., Manolakis, E., & Mantegna, R. N. (2025).
"Neural Network-Driven Volatility Drag Mitigation under Aggressive Leverage."
Proceedings of the 6th ACM International Conference on AI in Finance (ICAIF '25).
Also see Bongiorno, C., Manolakis, E., & Mantegna, R. N. (2025). "End-to-End Large Portfolio
Optimization for Variance Minimization with Neural Networks through Covariance Cleaning"
(arXiv:2507.01918) for a broader treatment.

Copyright (c) 2025
"""

import tensorflow as tf

from .dtype_utils import ensure_float32


def _require_rank_3(tensor: tf.Tensor, name: str, expected_shape: str) -> None:
    rank = tensor.shape.rank
    if rank is not None and rank != 3:
        raise ValueError(
            f"{name} must have rank 3 with shape {expected_shape}; "
            f"got shape {tensor.shape}."
        )


def _static_dim(tensor: tf.Tensor, axis: int):
    value = tensor.shape[axis]
    return None if value is None else int(value)


def _add_assertion(assertions: list, assertion) -> None:
    if assertion is not None:
        assertions.append(assertion)


@tf.keras.utils.register_keras_serializable(package='rienet')
def variance_loss_function(covariance_true: tf.Tensor,
                          weights_predicted: tf.Tensor) -> tf.Tensor:
    """
    Portfolio variance loss function for training RIEnet models.
    
    This loss function computes the global minimum-variance objective using the true
    covariance matrix and predicted portfolio weights.
    
    The portfolio variance is calculated as:
    variance = weights^T @ Σ @ weights
    
    where Σ is the true covariance matrix and weights are the predicted portfolio weights.
    
    Parameters
    ----------
    covariance_true : tf.Tensor
        Covariance matrices with shape ``(batch_size, n_assets, n_assets)``.
        Each matrix should be symmetric positive semi-definite.
    weights_predicted : tf.Tensor
        Predicted weights with shape ``(batch_size, n_assets, 1)``.
        The function assumes they already satisfy the portfolio constraint
        (typically sum to 1 across assets).
        
    Returns
    -------
    tf.Tensor
        Per-sample portfolio variance tensor with shape ``(batch_size, 1, 1)``.
        
    Notes
    -----
    The loss function assumes:
    - Daily returns data (annualized by factor of 252 in preprocessing)
    - Portfolio weights are expected to sum to one (enforced by the layer)
    - Covariance matrices are positive (semi) definite
    
    Examples
    --------
    >>> import tensorflow as tf
    >>> from rienet.losses import variance_loss_function
    >>> 
    >>> # Sample data: 32 batches, 10 assets
    >>> covariance = tf.random.normal((32, 10, 10))
    >>> covariance = tf.matmul(covariance, covariance, transpose_b=True)  # PSD
    >>> weights = tf.random.normal((32, 10, 1))
    >>> weights = weights / tf.reduce_sum(weights, axis=1, keepdims=True)  # Normalize
    >>> 
    >>> # Compute loss
    >>> loss = variance_loss_function(covariance, weights)
    >>> print(f"Portfolio variance: {loss.shape}")  # (32, 1, 1)
    
    """
    covariance_true = tf.convert_to_tensor(covariance_true)
    weights_predicted = tf.convert_to_tensor(weights_predicted)

    _require_rank_3(covariance_true, "covariance_true", "(batch_size, n_assets, n_assets)")
    _require_rank_3(weights_predicted, "weights_predicted", "(batch_size, n_assets, 1)")

    covariance_true, _ = ensure_float32(covariance_true)
    weights_predicted, _ = ensure_float32(weights_predicted)

    dtype = weights_predicted.dtype
    covariance_true = tf.cast(covariance_true, dtype)

    if covariance_true.shape.rank == 3:
        cov_rows = _static_dim(covariance_true, -2)
        cov_cols = _static_dim(covariance_true, -1)
        if cov_rows is not None and cov_cols is not None and cov_rows != cov_cols:
            raise ValueError(
                "covariance_true must be square on the last two dimensions; "
                f"got shape {covariance_true.shape}."
            )

    if weights_predicted.shape.rank == 3:
        weight_last_dim = _static_dim(weights_predicted, -1)
        if weight_last_dim is not None and weight_last_dim != 1:
            raise ValueError(
                "weights_predicted must have singleton last dimension; "
                f"got shape {weights_predicted.shape}."
            )

    if covariance_true.shape.rank == 3 and weights_predicted.shape.rank == 3:
        cov_batch = _static_dim(covariance_true, 0)
        weights_batch = _static_dim(weights_predicted, 0)
        if cov_batch is not None and weights_batch is not None and cov_batch != weights_batch:
            raise ValueError(
                "Batch mismatch between covariance_true and weights_predicted: "
                f"covariance_true batch={cov_batch}, weights_predicted batch={weights_batch}."
            )

        cov_assets = _static_dim(covariance_true, -1)
        weights_assets = _static_dim(weights_predicted, -2)
        if cov_assets is not None and weights_assets is not None and cov_assets != weights_assets:
            raise ValueError(
                "Asset-dimension mismatch between covariance_true and weights_predicted: "
                f"covariance_true assets={cov_assets}, weights_predicted assets={weights_assets}."
            )

    assertions = []
    _add_assertion(
        assertions,
        tf.debugging.assert_rank(
            covariance_true,
            3,
            message="covariance_true must have rank 3.",
        ),
    )
    _add_assertion(
        assertions,
        tf.debugging.assert_rank(
            weights_predicted,
            3,
            message="weights_predicted must have rank 3.",
        ),
    )
    _add_assertion(
        assertions,
        tf.debugging.assert_equal(
            tf.shape(covariance_true)[0],
            tf.shape(weights_predicted)[0],
            message="Batch mismatch between covariance_true and weights_predicted.",
        ),
    )
    _add_assertion(
        assertions,
        tf.debugging.assert_equal(
            tf.shape(covariance_true)[-2],
            tf.shape(covariance_true)[-1],
            message="covariance_true must be square on the last two dimensions.",
        ),
    )
    _add_assertion(
        assertions,
        tf.debugging.assert_equal(
            tf.shape(covariance_true)[-1],
            tf.shape(weights_predicted)[-2],
            message="Asset-dimension mismatch between covariance_true and weights_predicted.",
        ),
    )
    _add_assertion(
        assertions,
        tf.debugging.assert_equal(
            tf.shape(weights_predicted)[-1],
            1,
            message="weights_predicted must have singleton last dimension.",
        ),
    )

    if assertions:
        with tf.control_dependencies(assertions):
            covariance_true = tf.identity(covariance_true)
            weights_predicted = tf.identity(weights_predicted)

    covariance_true = tf.debugging.check_numerics(
        covariance_true,
        "covariance_true must contain finite values",
    )
    weights_predicted = tf.debugging.check_numerics(
        weights_predicted,
        "weights_predicted must contain finite values",
    )

    n = tf.cast(tf.shape(covariance_true)[-1], dtype=dtype)

    # Portfolio variance: n * w^T Σ w (Eq. 6 in the paper)
    portfolio_variance = n * tf.matmul(
        weights_predicted,
        tf.matmul(covariance_true, weights_predicted),
        transpose_a=True
    )

    return portfolio_variance

__all__ = [
    'variance_loss_function',
]

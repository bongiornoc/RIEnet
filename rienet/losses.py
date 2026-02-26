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
    dtype = weights_predicted.dtype
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

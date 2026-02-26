"""
Deterministic operation layers for RIEnet.

This module groups layers that do not own trainable parameters and only
perform tensor operations, statistics, matrix algebra, or normalization.
"""

import tensorflow as tf
from keras import backend as K
from keras import layers
from typing import List, Literal, Optional, Tuple

from .dtype_utils import ensure_float32, restore_dtype, epsilon_for_dtype

NormalizationModeType = Literal["inverse", "sum"]

@tf.keras.utils.register_keras_serializable(package='rienet')
class StandardDeviationLayer(layers.Layer):
    """
    Layer for computing sample standard deviation and mean.
    
    This layer computes the sample standard deviation and mean along a specified axis,
    with optional demeaning for statistical preprocessing.
    
    Parameters
    ----------
    axis : int, default 1
        Axis along which to compute statistics
    demean : bool, default False
        Whether to use an unbiased denominator (n-1)
    epsilon : float, optional
        Small epsilon for numerical stability
    name : str, optional
        Layer name
    """
    
    def __init__(self,
                 axis: int = 1,
                 demean: bool = False,
                 epsilon: Optional[float] = None,
                 name: Optional[str] = None,
                 **kwargs):
        if name is None:
            raise ValueError("StandardDeviationLayer must have a name.")
        super().__init__(name=name, **kwargs)
        self.axis = axis
        self.demean = demean
        self.epsilon = float(epsilon if epsilon is not None else K.epsilon())

    def call(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute per-axis sample standard deviation and mean.
        
        Parameters
        ----------
        x : tf.Tensor
            Input tensor. Statistics are computed along ``self.axis`` while
            preserving dimensions (`keepdims=True`), so the outputs can be
            broadcast back to ``x``.
            
        Returns
        -------
        tuple of tf.Tensor
            ``(std, mean)`` where both tensors have the same rank as ``x`` and
            singleton size on ``self.axis``.
        """
        dtype = x.dtype
        epsilon = epsilon_for_dtype(dtype, self.epsilon)

        sample_size = tf.cast(tf.shape(x)[self.axis], dtype)
        sample_size = tf.maximum(sample_size, 1.0)

        mean = tf.reduce_mean(x, axis=self.axis, keepdims=True)
        centered = x - mean

        if self.demean:
            denom = tf.maximum(sample_size - 1.0, 1.0)
        else:
            denom = sample_size

        variance = tf.reduce_sum(tf.square(centered), axis=self.axis, keepdims=True) / denom
        std = tf.sqrt(tf.maximum(variance, epsilon))

        return std, mean

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'axis': self.axis,
            'demean': self.demean,
            'epsilon': self.epsilon,
        })
        return config


@tf.keras.utils.register_keras_serializable(package='rienet')
class CovarianceLayer(layers.Layer):
    """
    Layer for computing covariance matrices.
    
    This layer computes sample covariance matrices from input data with optional
    normalization and dimension expansion.
    
    Parameters
    ----------
    expand_dims : bool, default False
        Whether to expand dimensions of output
    normalize : bool, default True  
        Whether to normalize by sample size
    name : str, optional
        Layer name
    """
    
    def __init__(self, expand_dims: bool = False, normalize: bool = True, 
                 name: Optional[str] = None, **kwargs):
        if name is None:
            raise ValueError("CovarianceLayer must have a name.")
        super().__init__(name=name, **kwargs)
        self.expand_dims = expand_dims
        self.normalize = normalize

    def call(self, returns: tf.Tensor) -> tf.Tensor:
        """
        Compute batched covariance/correlation-like matrices.
        
        Parameters
        ----------
        returns : tf.Tensor
            Return tensor of shape ``(..., n_assets, n_observations)``.
            The last axis is interpreted as the sample axis.
            
        Returns
        -------
        tf.Tensor
            Tensor of shape ``(..., n_assets, n_assets)`` if
            ``expand_dims=False``; otherwise ``(..., 1, n_assets, n_assets)``.
        """
        if self.normalize:
            sample_size = tf.cast(tf.shape(returns)[-1], returns.dtype)
            covariance = tf.matmul(returns, returns, transpose_b=True) / sample_size
        else:
            covariance = tf.matmul(returns, returns, transpose_b=True)
            
        if self.expand_dims:
            covariance = tf.expand_dims(covariance, axis=-3)
            
        return covariance

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'expand_dims': self.expand_dims,
            'normalize': self.normalize
        })
        return config


@tf.keras.utils.register_keras_serializable(package='rienet')
class SpectralDecompositionLayer(layers.Layer):
    """
    Layer for eigenvalue decomposition of symmetric matrices.
    
    This layer performs eigenvalue decomposition using tf.linalg.eigh,
    which is optimized for symmetric/Hermitian matrices like covariance matrices.
    
    Parameters
    ----------
    name : str, optional
        Layer name
    """
    
    def __init__(self, name: Optional[str] = None, **kwargs):
        if name is None:
            raise ValueError("SpectralDecompositionLayer must have a name.")
        super().__init__(name=name, **kwargs)

    def call(self, covariance_matrix: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Perform eigenvalue decomposition.
        
        Parameters
        ----------
        covariance_matrix : tf.Tensor
            Symmetric matrix tensor of shape ``(..., n, n)``.
            
        Returns
        -------
        tuple of tf.Tensor
            ``(eigenvalues, eigenvectors)`` where:
            - ``eigenvalues`` has shape ``(..., n, 1)`` in ascending order.
            - ``eigenvectors`` has shape ``(..., n, n)``.
        """
        covariance32, original_dtype = ensure_float32(covariance_matrix)
        eigenvalues, eigenvectors = tf.linalg.eigh(covariance32)
        # Expand dims to make eigenvalues [..., n, 1] for compatibility
        eigenvalues = tf.expand_dims(eigenvalues, axis=-1)
        return (
            restore_dtype(eigenvalues, original_dtype),
            restore_dtype(eigenvectors, original_dtype)
        )

    def get_config(self) -> dict:
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package='rienet')
class DimensionAwareLayer(layers.Layer):
    """
    Layer that builds per-asset dimensional attributes.
    
    This layer returns only attribute channels (no eigenvalues appended) with
    shape ``(batch, n_stocks, k)``, where ``k == len(features)``.
    
    Parameters
    ----------
    features : list of str
        List of features to add: 'n_stocks', 'n_days', 'q', 'rsqrt_n_days'
    name : str, optional
        Layer name
    """
    
    def __init__(self, features: List[str], name: Optional[str] = None, **kwargs):
        if name is None:
            raise ValueError("DimensionAwareLayer must have a name.")
        super().__init__(name=name, **kwargs)
        self.features = features

    def _set_attribute(self, value: tf.Tensor, shape: tf.Tensor, dtype: tf.dtypes.DType) -> tf.Tensor:
        """Broadcast scalar value to target shape with the target dtype."""
        value = tf.cast(value, dtype)
        value = tf.expand_dims(value, axis=-1)
        value = tf.broadcast_to(value, shape)
        return value

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        """
        Build dimensional attributes for each stock.
        
        Parameters
        ----------
        inputs : list of tf.Tensor
            Two-element list ``[standardized_returns, correlation_matrix]``:
            - ``standardized_returns``: shape ``(batch, n_stocks, n_days)``
            - ``correlation_matrix``: shape ``(batch, n_stocks, n_stocks)``
            The second tensor is used to infer the asset axis for broadcasting.
            
        Returns
        -------
        tf.Tensor
            Attribute tensor with shape ``(batch, n_stocks, k)``,
            where ``k == len(self.features)``.
        """
        standardized_returns, correlation_matrix = inputs
        n_stocks_raw = tf.shape(correlation_matrix)[-1]
        n_stocks = tf.cast(n_stocks_raw, tf.float32)
        n_days = tf.cast(tf.shape(standardized_returns)[-1], tf.float32)
        target_dtype = standardized_returns.dtype
        final_shape = tf.stack(
            [
                tf.shape(correlation_matrix)[0],
                n_stocks_raw,
                tf.constant(1, dtype=tf.int32),
            ]
        )

        tensors_to_concat = []
        
        if 'q' in self.features:
            q = n_days / n_stocks
            tensors_to_concat.append(self._set_attribute(q, final_shape, target_dtype))
            
        if 'n_stocks' in self.features:
            tensors_to_concat.append(self._set_attribute(tf.sqrt(n_stocks), final_shape, target_dtype))
            
        if 'n_days' in self.features:
            tensors_to_concat.append(self._set_attribute(tf.sqrt(n_days), final_shape, target_dtype))
            
        if 'rsqrt_n_days' in self.features:
            rsqrt_n_days = tf.math.rsqrt(n_days)
            tensors_to_concat.append(self._set_attribute(rsqrt_n_days, final_shape, target_dtype))

        if not tensors_to_concat:
            empty_shape = tf.stack(
                [
                    tf.shape(correlation_matrix)[0],
                    n_stocks_raw,
                    tf.constant(0, dtype=tf.int32),
                ]
            )
            return tf.zeros(empty_shape, dtype=target_dtype)

        return tf.concat(tensors_to_concat, axis=-1)

    def compute_output_shape(self, input_shape: Tuple[Tuple, Tuple]) -> Tuple:
        """Compute output shape."""
        _, correlation_shape = input_shape
        batch_size = correlation_shape[0]
        n_stocks = correlation_shape[-1]
        return (batch_size, n_stocks, len(self.features))

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({'features': self.features})
        return config



@tf.keras.utils.register_keras_serializable(package='rienet')
class CustomNormalizationLayer(layers.Layer):
    """
    Custom normalization layer with different modes.
    
    This layer applies different types of normalization along specified axes,
    including sum normalization and inverse normalization.
    
    Parameters
    ----------
    mode : Literal['sum', 'inverse'], default 'sum'
        Normalization rule.
    axis : int, default -2
        Axis along which to normalize
    inverse_power : float, default 1.0
        Exponent ``p`` used for inverse normalization. In ``mode='inverse'``,
        outputs are rescaled so that the mean of ``x^{-p}`` along ``axis`` is 1.
        Larger ``p`` emphasizes small values more strongly.
    epsilon : float, optional
        Small epsilon for numerical stability
    name : str, optional
        Layer name
    """
    
    def __init__(self,
                 mode: NormalizationModeType = 'sum',
                 axis: int = -2,
                 inverse_power: float = 1.0,
                 epsilon: Optional[float] = None,
                 name: Optional[str] = None,
                 **kwargs):
        if name is None:
            raise ValueError("CustomNormalizationLayer must have a name.")
        super().__init__(name=name, **kwargs)
        self.mode = mode
        self.axis = axis
        if inverse_power <= 0:
            raise ValueError("inverse_power must be positive")
        self.inverse_power = float(inverse_power)
        self.epsilon = float(epsilon if epsilon is not None else K.epsilon())

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Apply normalization along the configured axis.
        
        Parameters
        ----------
        x : tf.Tensor
            Input tensor.
            
        Returns
        -------
        tf.Tensor
            Normalized tensor with the same shape as ``x``.

        Notes
        -----
        - ``mode='sum'`` enforces ``sum(x)=n`` along ``axis``.
        - ``mode='inverse'`` enforces ``mean(x^{-p})=1`` along ``axis``.
        """
        dtype = x.dtype
        epsilon = epsilon_for_dtype(dtype, self.epsilon)
        n = tf.cast(tf.shape(x)[self.axis], dtype)

        denom_axis = tf.reduce_sum(x, axis=self.axis, keepdims=True)

        if self.mode == 'sum':
            x = n * x / tf.maximum(denom_axis, epsilon)
        elif self.mode == 'inverse':
            x = tf.maximum(x, epsilon)
            inv = tf.math.pow(x, -self.inverse_power)
            inv_total = tf.reduce_sum(inv, axis=self.axis, keepdims=True)
            inv_normalized = n * inv / tf.maximum(inv_total, epsilon)
            power = tf.cast(-1.0 / self.inverse_power, dtype)
            x = tf.math.pow(tf.maximum(inv_normalized, epsilon), power)
        
        return x

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'mode': self.mode,
            'axis': self.axis,
            'inverse_power': self.inverse_power,
            'epsilon': self.epsilon,
        })
        return config


@tf.keras.utils.register_keras_serializable(package='rienet')
class EigenvectorRescalingLayer(layers.Layer):
    """
    Layer that rescales eigenvectors to enforce unit diagonals.

    Given eigenvectors ``V`` and eigenvalues ``λ`` this layer computes the diagonal
    elements of ``V diag(λ) Vᵀ`` and divides each eigenvector row by the square
    root of the corresponding diagonal entry. The operation matches::

        d = einsum('...ij,...j,...ij->...i', V, λ, V)
        V_rescaled = V / sqrt(d)[..., None]

    Parameters
    ----------
    epsilon : float, optional
        Minimum value used to avoid division-by-zero during normalization.
    name : str, optional
        Layer name.
    """

    def __init__(self, epsilon: Optional[float] = None, name: Optional[str] = None, **kwargs):
        if name is None:
            raise ValueError("EigenvectorRescalingLayer must have a name.")
        super().__init__(name=name, **kwargs)
        self.epsilon = float(epsilon if epsilon is not None else K.epsilon())

    def build(self, input_shape) -> None:
        # Nothing to build, but override for Keras compatibility
        super().build(input_shape)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Rescale eigenvectors based on eigenvalues.

        Parameters
        ----------
        inputs : tuple
            Tuple ``(eigenvectors, eigenvalues)`` where:
            - ``eigenvectors`` has shape ``(..., n, n)``
            - ``eigenvalues`` has shape ``(..., n)`` or ``(..., n, 1)``

        Returns
        -------
        tf.Tensor
            Rescaled eigenvectors with the same shape as the input eigenvectors.
        """
        eigenvectors, eigenvalues = inputs
        dtype = eigenvectors.dtype
        eigenvectors = tf.convert_to_tensor(eigenvectors, dtype=dtype)
        eigenvalues = tf.convert_to_tensor(eigenvalues, dtype=dtype)

        target_shape = tf.shape(eigenvectors)[:-1]
        eigenvalues = tf.reshape(eigenvalues, target_shape)

        diag = tf.einsum('...ij,...j,...ij->...i', eigenvectors, eigenvalues, eigenvectors)
        eps = epsilon_for_dtype(dtype, self.epsilon)
        diag = tf.maximum(diag, eps)
        inv_sqrt = tf.math.rsqrt(diag)
        scaling = tf.expand_dims(inv_sqrt, axis=-1)
        return eigenvectors * scaling

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({'epsilon': self.epsilon})
        return config


@tf.keras.utils.register_keras_serializable(package='rienet')
class EigenProductLayer(layers.Layer):
    """
    Layer for reconstructing matrices from eigenvalue decomposition.
    
    This layer implements the vanilla reconstruction ``V diag(λ) Vᵀ`` without
    any diagonal post-scaling. It assumes eigenvectors have already been
    preprocessed (e.g., via :class:`EigenvectorRescalingLayer`) when diagonal
    control is required.
    """
    
    def __init__(self, name: Optional[str] = None, **kwargs):
        if name is None:
            raise ValueError("EigenProductLayer must have a name.")
        super().__init__(name=name, **kwargs)

    def call(self, eigenvalues: tf.Tensor, eigenvectors: tf.Tensor) -> tf.Tensor:
        """
        Reconstruct matrix from eigenvalue decomposition.
        
        Parameters
        ----------
        eigenvalues : tf.Tensor
            Eigenvalues tensor with shape ``(..., n)`` or ``(..., n, 1)``.
        eigenvectors : tf.Tensor
            Eigenvectors tensor with shape ``(..., n, n)``.
            
        Returns
        -------
        tf.Tensor
            Reconstructed matrix ``V diag(λ) V^T`` with shape ``(..., n, n)``.
        """
        dtype = eigenvectors.dtype
        eigenvalues = tf.convert_to_tensor(eigenvalues, dtype=dtype)
        eigenvectors = tf.convert_to_tensor(eigenvectors, dtype=dtype)

        target_shape = tf.shape(eigenvectors)[:-1]
        eigenvalues = tf.reshape(eigenvalues, target_shape)

        scaled_vectors = eigenvectors * tf.expand_dims(eigenvalues, axis=-2)
        return tf.matmul(scaled_vectors, eigenvectors, transpose_b=True)

    def get_config(self) -> dict:
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package='rienet')
class EigenWeightsLayer(layers.Layer):
    """
    Compute GMV-like portfolio weights from eigensystem quantities.

    This layer is intended for direct external use and accepts explicit inputs:
    eigenvectors, inverse eigenvalues, and optionally inverse standard deviations.

    Let ``V`` be eigenvectors and ``lambda_inv`` inverse eigenvalues. Define::

        c_k = sum_i V_{ik}
        s_k = lambda_inv_k * c_k

    The raw weights are computed as:
    - with ``inverse_std`` provided:
      ``raw_i = (sum_k V_{ik} s_k) * inverse_std_i``
    - with ``inverse_std=None``:
      ``raw_i = sum_k V_{ik} s_k``

    Then the output is normalized to sum to one across assets.

    Parameters
    ----------
    epsilon : float, optional
        Numerical stability term used for safe normalization.
    name : str, optional
        Layer name.

    Call Arguments
    --------------
    eigenvectors : tf.Tensor
        Tensor of shape ``(..., n_assets, n_assets)``.
    inverse_eigenvalues : tf.Tensor
        Tensor of shape ``(..., n_assets)`` or ``(..., n_assets, 1)``.
    inverse_std : tf.Tensor, optional
        Tensor of shape ``(..., n_assets)`` or ``(..., n_assets, 1)``.
        If omitted, the layer computes the covariance-eigensystem branch directly
        without materializing a ones vector.

    Returns
    -------
    tf.Tensor
        Normalized weights with shape ``(..., n_assets, 1)``.
    """

    def __init__(self, epsilon: Optional[float] = None, name: Optional[str] = None, **kwargs):
        if name is None:
            raise ValueError("EigenWeightsLayer must have a name.")
        super().__init__(name=name, **kwargs)
        self.epsilon = float(epsilon if epsilon is not None else K.epsilon())

    def build(self, input_shape) -> None:
        super().build(input_shape)

    def call(self,
             eigenvectors: tf.Tensor,
             inverse_eigenvalues: tf.Tensor,
             inverse_std: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Compute normalized portfolio weights from spectral quantities.

        Parameters
        ----------
        eigenvectors : tf.Tensor
            Tensor of shape ``(..., n_assets, n_assets)``.
        inverse_eigenvalues : tf.Tensor
            Tensor of shape ``(..., n_assets)`` or ``(..., n_assets, 1)``.
        inverse_std : tf.Tensor, optional
            Optional tensor of shape ``(..., n_assets)`` or ``(..., n_assets, 1)``
            representing inverse standard deviations.

        Returns
        -------
        tf.Tensor
            Normalized weights with shape ``(..., n_assets, 1)``.
        """
        dtype = eigenvectors.dtype

        eigenvectors = tf.convert_to_tensor(eigenvectors, dtype=dtype)
        inverse_eigenvalues = tf.convert_to_tensor(inverse_eigenvalues, dtype=dtype)

        eigenvector_sum = tf.reduce_sum(eigenvectors, axis=-2)
        target_shape = tf.shape(eigenvector_sum)

        inverse_eigenvalues = tf.reshape(inverse_eigenvalues, target_shape)
        spectral_term = inverse_eigenvalues * eigenvector_sum
        raw_weights = tf.linalg.matvec(eigenvectors, spectral_term)

        if inverse_std is not None:
            inverse_std = tf.convert_to_tensor(inverse_std, dtype=dtype)
            inverse_std = tf.reshape(inverse_std, target_shape)
            raw_weights = raw_weights * inverse_std

        denom = tf.reduce_sum(raw_weights, axis=-1, keepdims=True)
        epsilon = epsilon_for_dtype(dtype, self.epsilon)
        sign = tf.where(denom >= 0, tf.ones_like(denom), -tf.ones_like(denom))
        safe_denom = tf.where(tf.abs(denom) < epsilon, sign * epsilon, denom)
        weights = raw_weights / safe_denom

        return tf.expand_dims(weights, axis=-1)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({'epsilon': self.epsilon})
        return config



@tf.keras.utils.register_keras_serializable(package='rienet')
class NormalizedSum(layers.Layer):
    """
    Layer for computing normalized sums along specified axes.
    
    This layer computes sums along one axis and then normalizes by the sum
    along another axis, commonly used for portfolio weight computation.
    
    Parameters
    ----------
    axis_1 : int, default -1
        First axis for summation
    axis_2 : int, default -2
        Second axis for normalization
    epsilon : float, optional
        Small epsilon for numerical stability
    name : str, optional
        Layer name
    """
    
    def __init__(self,
                 axis_1: int = -1,
                 axis_2: int = -2,
                 epsilon: Optional[float] = None,
                 name: Optional[str] = None,
                 **kwargs):
        if name is None:
            raise ValueError("NormalizedSum must have a name.")
        super().__init__(name=name, **kwargs)
        self.axis_1 = axis_1
        self.axis_2 = axis_2
        self.epsilon = float(epsilon if epsilon is not None else K.epsilon())

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute normalized sum.
        
        Parameters
        ----------
        x : tf.Tensor
            Input tensor. The layer first sums along ``axis_1``, then normalizes
            by the sum along ``axis_2``.
            
        Returns
        -------
        tf.Tensor
            Tensor with the same rank as ``x`` where the aggregated values are
            normalized by a safe denominator.
        """
        dtype = x.dtype
        epsilon = epsilon_for_dtype(dtype, self.epsilon)
        w = tf.reduce_sum(x, axis=self.axis_1, keepdims=True)
        denominator = tf.reduce_sum(w, axis=self.axis_2, keepdims=True)
        sign = tf.where(denominator >= 0, tf.ones_like(denominator), -tf.ones_like(denominator))
        safe_denominator = tf.where(
            tf.abs(denominator) < epsilon,
            sign * epsilon,
            denominator
        )
        result = w / safe_denominator
        return result

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'axis_1': self.axis_1,
            'axis_2': self.axis_2,
            'epsilon': self.epsilon,
        })
        return config


__all__ = [
    "NormalizationModeType",
    "StandardDeviationLayer",
    "CovarianceLayer",
    "SpectralDecompositionLayer",
    "DimensionAwareLayer",
    "CustomNormalizationLayer",
    "EigenvectorRescalingLayer",
    "EigenProductLayer",
    "EigenWeightsLayer",
    "NormalizedSum",
]

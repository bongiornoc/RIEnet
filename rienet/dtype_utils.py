"""
Utility helpers to keep sensitive operations in float32 under mixed precision.

These functions centralise the logic for casting tensors to float32 when the
active mixed-precision policy would otherwise request lower precision (e.g.
float16 or bfloat16) and for restoring the original dtype afterwards. They also
provide dtype-aware epsilon values so that stability constants track the active
precision.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from typing import Optional, Tuple, Union

_LOWER_PRECISION_DTYPES = (tf.float16, tf.bfloat16)


def ensure_dense_tensor(
    tensor: Union[tf.Tensor, tf.SparseTensor],
) -> tf.Tensor:
    """
    Convert sparse tensors to dense while preserving known static shape metadata.

    Keras may route symbolic inputs through ``tf.SparseTensor`` placeholders even
    when the logical input is dense, for example after scalar arithmetic on a
    dense ``KerasTensor``. RIEnet layers operate on dense tensors, so sparse
    placeholders must be densified before further TensorFlow ops such as
    ``tf.convert_to_tensor`` are applied.
    """
    if isinstance(tensor, tf.SparseTensor):
        static_shape = tensor.shape
        tensor = tf.sparse.to_dense(tensor)
        tensor = tf.ensure_shape(tensor, static_shape)
    return tf.convert_to_tensor(tensor)


def ensure_float32(tensor: tf.Tensor) -> Tuple[tf.Tensor, Optional[tf.dtypes.DType]]:
    """
    Cast ``tensor`` to float32 when it comes from a lower-precision dtype.

    Returns
    -------
    Tuple[tf.Tensor, Optional[tf.DType]]
        The possibly cast tensor and the original dtype so callers can restore it.
    """
    dtype = getattr(tensor, "dtype", None)
    if dtype in _LOWER_PRECISION_DTYPES:
        return tf.cast(tensor, tf.float32), dtype
    return tensor, dtype


def restore_dtype(tensor: tf.Tensor, dtype: Optional[tf.dtypes.DType]) -> tf.Tensor:
    """
    Cast ``tensor`` back to ``dtype`` if it differs from the current dtype.
    """
    if dtype is None or tensor.dtype == dtype:
        return tensor
    return tf.cast(tensor, dtype)


def epsilon_for_dtype(
    dtype: tf.dtypes.DType,
    base_value: float,
) -> tf.Tensor:
    """
    Return a stability epsilon scaled to the ``dtype``.

    The helper guarantees the epsilon is at least as large as the machine epsilon
    of ``dtype`` while respecting the requested ``base_value`` (interpreted as the
    float32 baseline).
    """
    dtype = tf.dtypes.as_dtype(dtype)
    if not dtype.is_floating:
        raise TypeError("epsilon_for_dtype requires a floating-point dtype")

    if dtype == tf.bfloat16:
        # bfloat16 has 7 mantissa bits -> machine epsilon = 2^-7.
        dtype_eps = 2.0 ** -7
    else:
        dtype_eps = float(np.finfo(dtype.as_numpy_dtype).eps)
    scaled_base = base_value
    # Ensure the epsilon is never smaller than the machine epsilon of the dtype.
    scaled = max(scaled_base, dtype_eps)
    return tf.cast(scaled, dtype)

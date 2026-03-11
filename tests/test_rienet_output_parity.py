"""Parity tests for the optimized RIEnetLayer output dispatch."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import tensorflow as tf
from keras import backend as K

from rienet import RIEnetLayer
from rienet.dtype_utils import ensure_float32, epsilon_for_dtype, restore_dtype


def _legacy_normalize_raw_weights(
    raw_weights: tf.Tensor,
    original_dtype: tf.dtypes.DType | None,
) -> tf.Tensor:
    dtype = raw_weights.dtype
    epsilon = epsilon_for_dtype(dtype, K.epsilon())
    denom = tf.reduce_sum(raw_weights, axis=-1, keepdims=True)
    sign = tf.where(denom >= 0, tf.ones_like(denom), -tf.ones_like(denom))
    safe_denom = tf.where(tf.abs(denom) < epsilon, sign * epsilon, denom)
    weights = raw_weights / safe_denom
    return restore_dtype(tf.expand_dims(weights, axis=-1), original_dtype)


def _legacy_exact_weights_from_rescaled_eigensystem(
    direct_eigenvectors: tf.Tensor,
    inverse_eigenvalues: tf.Tensor,
    inverse_std: tf.Tensor,
) -> tf.Tensor:
    direct_eigenvectors = tf.convert_to_tensor(direct_eigenvectors)
    vectors_work, original_dtype = ensure_float32(direct_eigenvectors)
    dtype = vectors_work.dtype

    target_shape = tf.shape(vectors_work)[:-1]
    inverse_eigenvalues = tf.cast(tf.convert_to_tensor(inverse_eigenvalues), dtype)
    inverse_eigenvalues = tf.reshape(inverse_eigenvalues, target_shape)
    inverse_std = tf.cast(tf.convert_to_tensor(inverse_std), dtype)
    inverse_std = tf.reshape(inverse_std, target_shape)

    row_norm_sq = tf.reduce_sum(tf.square(vectors_work), axis=-1)
    eps = epsilon_for_dtype(dtype, K.epsilon())
    inverse_row_norm_sq = tf.math.reciprocal(tf.maximum(row_norm_sq, eps))
    spectral_rhs = tf.linalg.matvec(
        vectors_work,
        inverse_row_norm_sq * inverse_std,
        transpose_a=True,
    )
    spectral_rhs = inverse_eigenvalues * spectral_rhs
    raw_weights = inverse_std * inverse_row_norm_sq * tf.linalg.matvec(
        vectors_work,
        spectral_rhs,
    )
    return _legacy_normalize_raw_weights(raw_weights, original_dtype)


def _legacy_weights_from_inverse_correlation(
    inverse_correlation: tf.Tensor,
    inverse_std: tf.Tensor,
) -> tf.Tensor:
    inverse_corr_work, inverse_corr_original_dtype = ensure_float32(inverse_correlation)
    inverse_std = tf.cast(tf.convert_to_tensor(inverse_std), inverse_corr_work.dtype)
    inverse_std = tf.reshape(inverse_std, tf.shape(inverse_corr_work)[:-1])
    raw_weights = inverse_std * tf.linalg.matvec(inverse_corr_work, inverse_std)
    return _legacy_normalize_raw_weights(raw_weights, inverse_corr_original_dtype)


def legacy_reference_forward(
    layer: RIEnetLayer,
    inputs: tf.Tensor,
    training: bool | None = None,
) -> Any:
    """Replicate the pre-optimization RIEnetLayer dispatch for parity checks."""
    need_precision = layer._need_precision
    need_covariance = layer._need_covariance
    need_correlation = layer._need_correlation
    need_weights = layer._need_weights
    need_eigenvalues = layer._need_eigenvalues
    need_eigenvectors = layer._need_eigenvectors
    need_transformed_std = layer._need_transformed_std
    need_pipeline_outputs = layer._need_pipeline_outputs

    scaled_inputs = inputs * layer._annualization_factor
    input_transformed = layer.lag_transform(scaled_inputs)

    results: dict[str, tf.Tensor] = {}
    if "input_transformed" in layer.output_components:
        results["input_transformed"] = input_transformed

    if not need_pipeline_outputs:
        return (
            results[layer.output_components[0]]
            if len(layer.output_components) == 1
            else results
        )

    std, mean = layer.std_layer(input_transformed)

    need_inverse_std = need_precision or need_covariance or need_weights or need_transformed_std
    transformed_inverse_std = None
    std_for_structural = None
    transformed_std = None
    if need_inverse_std:
        if layer.std_transform is None:
            raise RuntimeError("Internal error: missing std_transform.")
        transformed_inverse_std = layer.std_transform(std)
        std_for_structural = transformed_inverse_std
        if layer.std_normalization is not None and need_inverse_std:
            std_for_structural = layer.std_normalization(transformed_inverse_std)

        std_work, std_original_dtype = ensure_float32(std_for_structural)
        std_eps = epsilon_for_dtype(std_work.dtype, K.epsilon())
        transformed_std_work = tf.math.reciprocal(tf.maximum(std_work, std_eps))
        transformed_std = restore_dtype(transformed_std_work, std_original_dtype)

    if need_transformed_std:
        results["transformed_std"] = transformed_std

    need_spectral_branch = layer._need_spectral_branch
    if not need_spectral_branch:
        return (
            results[layer.output_components[0]]
            if len(layer.output_components) == 1
            else results
        )

    zscores = (input_transformed - mean) / std
    correlation_matrix = layer.covariance_layer(zscores)

    attributes = None
    if layer._dimensional_features:
        attributes = layer.dimension_aware([zscores, correlation_matrix])

    spectral_components = []
    need_exact_weight_eigensystem = need_weights and not need_precision
    if need_eigenvectors or need_exact_weight_eigensystem:
        spectral_components.append("eigenvectors")
    if need_eigenvalues:
        spectral_components.append("eigenvalues")
    if need_exact_weight_eigensystem:
        spectral_components.append("inverse_eigenvalues")
    if need_covariance or need_correlation:
        spectral_components.append("correlation")
    if need_precision:
        spectral_components.append("inverse_correlation")

    deduped_components = []
    seen_components = set()
    for component in spectral_components:
        if component not in seen_components:
            deduped_components.append(component)
            seen_components.add(component)

    spectral_outputs = layer.correlation_eigen_transform(
        correlation_matrix,
        attributes=attributes,
        output_type=deduped_components,
        training=training,
    )
    if isinstance(spectral_outputs, dict):
        spectral_results = spectral_outputs
    else:
        spectral_results = {deduped_components[0]: spectral_outputs}

    eigenvectors = spectral_results.get("eigenvectors")
    transformed_inverse_eigenvalues = spectral_results.get("inverse_eigenvalues")
    if transformed_inverse_eigenvalues is not None:
        transformed_inverse_eigenvalues = tf.squeeze(
            transformed_inverse_eigenvalues,
            axis=-1,
        )
    cleaned_correlation = spectral_results.get("correlation")

    if need_eigenvectors:
        results["eigenvectors"] = eigenvectors
    if need_eigenvalues:
        results["eigenvalues"] = spectral_results["eigenvalues"]

    inverse_correlation = spectral_results.get("inverse_correlation")

    if need_covariance:
        if layer.outer_product is None:
            raise RuntimeError("Internal error: missing outer_product layer.")
        volatility_matrix = layer.outer_product(transformed_std)
        results["covariance"] = cleaned_correlation * volatility_matrix

    if need_correlation:
        results["correlation"] = cleaned_correlation

    if need_precision:
        if layer.outer_product is None:
            raise RuntimeError("Internal error: missing outer_product layer.")
        inverse_volatility_matrix = layer.outer_product(std_for_structural)
        results["precision"] = inverse_correlation * inverse_volatility_matrix
        if need_weights:
            results["weights"] = _legacy_weights_from_inverse_correlation(
                inverse_correlation,
                std_for_structural,
            )

    if need_weights and not need_precision:
        results["weights"] = _legacy_exact_weights_from_rescaled_eigensystem(
            eigenvectors,
            transformed_inverse_eigenvalues,
            std_for_structural,
        )

    if len(layer.output_components) == 1:
        return results[layer.output_components[0]]
    return results


def assert_outputs_match(actual: Any, expected: Any, *, rtol: float, atol: float) -> None:
    """Assert parity between actual and expected RIEnet outputs."""
    assert type(actual) is type(expected)

    if isinstance(actual, dict):
        assert set(actual.keys()) == set(expected.keys())
        for key in actual:
            assert_outputs_match(actual[key], expected[key], rtol=rtol, atol=atol)
        return

    assert isinstance(actual, tf.Tensor)
    assert isinstance(expected, tf.Tensor)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(
        actual.numpy(),
        expected.numpy(),
        rtol=rtol,
        atol=atol,
    )


def assert_weight_sums_to_one(outputs: Any) -> None:
    """Check that any returned weights remain normalized."""
    tensors: list[tf.Tensor] = []
    if isinstance(outputs, dict):
        if "weights" in outputs:
            tensors.append(outputs["weights"])
    else:
        tensors.append(outputs)

    for tensor in tensors:
        weight_sums = tf.reduce_sum(tensor, axis=1)
        np.testing.assert_allclose(
            weight_sums.numpy(),
            1.0,
            rtol=1e-5,
            atol=1e-6,
        )


OUTPUT_CASES = [
    "input_transformed",
    "transformed_std",
    "eigenvalues",
    "eigenvectors",
    "correlation",
    "covariance",
    "precision",
    "weights",
    ["weights", "transformed_std"],
    ["weights", "eigenvalues"],
    ["weights", "eigenvectors"],
    ["weights", "correlation"],
    ["weights", "covariance"],
    ["weights", "precision"],
    "all",
]


def _case_id(output_type: Any) -> str:
    if isinstance(output_type, (list, tuple)):
        return "_".join(output_type)
    return str(output_type)


@pytest.mark.parametrize("output_type", OUTPUT_CASES, ids=_case_id)
def test_rienet_outputs_match_legacy_reference(output_type):
    tf.keras.utils.set_random_seed(1234)

    inputs = tf.random.normal((3, 5, 24))
    layer = RIEnetLayer(output_type=output_type, name=f"parity_{_case_id(output_type)}")

    actual = layer(inputs)
    expected = legacy_reference_forward(layer, inputs)

    assert_outputs_match(actual, expected, rtol=1e-5, atol=1e-6)
    if "weights" in layer.output_components:
        assert_weight_sums_to_one(actual)


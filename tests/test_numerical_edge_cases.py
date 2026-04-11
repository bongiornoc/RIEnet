"""Numerical edge-case and validation regression tests."""

import numpy as np
import pytest
import tensorflow as tf

from rienet import RIEnetLayer, variance_loss_function
from rienet.ops_layers import CustomNormalizationLayer, StandardDeviationLayer
from rienet.trainable_layers import CorrelationEigenTransformLayer, LagTransformLayer


def _assert_all_finite(tensor):
    assert bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy())


def _assert_weights_normalized(weights, atol=1e-5):
    _assert_all_finite(weights)
    np.testing.assert_allclose(
        tf.reduce_sum(weights, axis=1).numpy(),
        1.0,
        rtol=1e-5,
        atol=atol,
    )


def test_sum_normalization_preserves_negative_denominator_sign():
    layer = CustomNormalizationLayer(
        mode="sum",
        axis=-2,
        epsilon=1e-6,
        name="negative_sum_norm",
    )
    x = tf.constant([[[-1.0], [-2.0], [-3.0]]], dtype=tf.float32)

    y = layer(x)

    np.testing.assert_allclose(
        y.numpy(),
        np.array([[[0.5], [1.0], [1.5]]], dtype=np.float32),
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(tf.reduce_sum(y, axis=-2).numpy(), 3.0, atol=1e-7)


def test_sum_normalization_near_zero_negative_sum_uses_negative_epsilon():
    layer = CustomNormalizationLayer(
        mode="sum",
        axis=-2,
        epsilon=1e-6,
        name="near_zero_negative_sum_norm",
    )
    x = tf.constant([[[-1e-8], [-2e-8], [-3e-8]]], dtype=tf.float32)

    y = layer(x)

    _assert_all_finite(y)
    assert bool(tf.reduce_all(y > 0).numpy())
    np.testing.assert_allclose(
        y.numpy(),
        np.array([[[0.03], [0.06], [0.09]]], dtype=np.float32),
        rtol=1e-6,
        atol=1e-8,
    )


def test_custom_normalization_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode"):
        CustomNormalizationLayer(mode="bad", name="invalid_mode_norm")


def test_standard_deviation_demean_controls_denominator_not_centering():
    x = tf.constant([[[1.0, 2.0, 3.0]]], dtype=tf.float32)

    population_std, population_mean = StandardDeviationLayer(
        axis=-1,
        demean=False,
        name="population_std",
    )(x)
    sample_std, sample_mean = StandardDeviationLayer(
        axis=-1,
        demean=True,
        name="sample_std",
    )(x)

    np.testing.assert_allclose(population_mean.numpy(), sample_mean.numpy())
    np.testing.assert_allclose(population_mean.numpy(), [[[2.0]]], atol=1e-7)
    np.testing.assert_allclose(
        population_std.numpy(),
        [[[np.sqrt(2.0 / 3.0)]]],
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(sample_std.numpy(), [[[1.0]]], rtol=1e-6, atol=1e-7)


def test_rienet_weights_are_asset_permutation_equivariant():
    tf.keras.utils.set_random_seed(1234)
    layer = RIEnetLayer(output_type="weights", name="permutation_equivariant")
    x = tf.random.normal((2, 5, 16), dtype=tf.float32)
    permutation = tf.constant([2, 4, 1, 3, 0], dtype=tf.int32)

    weights = layer(x)
    permuted_weights = layer(tf.gather(x, permutation, axis=1))

    np.testing.assert_allclose(
        permuted_weights.numpy(),
        tf.gather(weights, permutation, axis=1).numpy(),
        rtol=2e-5,
        atol=2e-6,
    )


def test_rienet_single_asset_weight_is_one():
    layer = RIEnetLayer(output_type="weights", name="single_asset")
    weights = layer(tf.random.normal((3, 1, 8), dtype=tf.float32))

    np.testing.assert_allclose(weights.numpy(), 1.0, rtol=0.0, atol=1e-7)


@pytest.mark.parametrize("n_days", [1, 2])
def test_rienet_tiny_time_dimensions_are_finite_and_normalized(n_days):
    layer = RIEnetLayer(output_type="weights", name=f"tiny_time_{n_days}")
    weights = layer(tf.random.normal((2, 4, n_days), dtype=tf.float32))

    _assert_weights_normalized(weights)


def test_rienet_rank_deficient_inputs_are_finite():
    layer = RIEnetLayer(
        output_type=["weights", "precision", "covariance"],
        name="rank_deficient_inputs",
    )
    base = tf.random.normal((2, 1, 6), dtype=tf.float32)
    x = tf.concat([base, 2.0 * base, -base, tf.zeros_like(base)], axis=1)

    outputs = layer(x)

    _assert_weights_normalized(outputs["weights"])
    _assert_all_finite(outputs["precision"])
    _assert_all_finite(outputs["covariance"])


def test_correlation_transform_repeated_eigenvalues_are_finite():
    layer = CorrelationEigenTransformLayer(
        output_type=["correlation", "inverse_correlation", "eigenvalues"],
        name="repeated_eigenvalues",
    )
    outputs = layer(tf.eye(4, batch_shape=[2], dtype=tf.float32))

    _assert_all_finite(outputs["correlation"])
    _assert_all_finite(outputs["inverse_correlation"])
    _assert_all_finite(outputs["eigenvalues"])
    np.testing.assert_allclose(
        tf.linalg.diag_part(outputs["correlation"]).numpy(),
        1.0,
        rtol=1e-5,
        atol=1e-6,
    )


def test_rienet_zero_variance_assets_are_finite_and_normalized():
    layer = RIEnetLayer(
        output_type=["weights", "precision", "covariance"],
        name="zero_variance_assets",
    )
    outputs = layer(tf.zeros((2, 4, 8), dtype=tf.float32))

    _assert_weights_normalized(outputs["weights"])
    _assert_all_finite(outputs["precision"])
    _assert_all_finite(outputs["covariance"])


def test_rienet_weight_gradients_are_finite():
    tf.keras.utils.set_random_seed(5678)
    layer = RIEnetLayer(output_type="weights", name="finite_weight_gradients")
    x = tf.Variable(tf.random.normal((2, 4, 8), dtype=tf.float32))

    with tf.GradientTape() as tape:
        weights = layer(x)
        covariance = tf.eye(4, batch_shape=[2], dtype=tf.float32)
        loss = tf.reduce_sum(variance_loss_function(covariance, weights))

    gradients = tape.gradient(loss, [x] + layer.trainable_variables)

    assert all(gradient is not None for gradient in gradients)
    for gradient in gradients:
        _assert_all_finite(gradient)


def test_variance_loss_gradients_are_finite():
    covariance = tf.Variable(tf.eye(3, batch_shape=[2], dtype=tf.float32))
    weights = tf.Variable(tf.ones((2, 3, 1), dtype=tf.float32) / 3.0)

    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(variance_loss_function(covariance, weights))

    cov_gradient, weight_gradient = tape.gradient(loss, [covariance, weights])

    _assert_all_finite(cov_gradient)
    _assert_all_finite(weight_gradient)


@pytest.mark.parametrize(
    ("covariance", "weights"),
    [
        (tf.eye(3, batch_shape=[2]), tf.ones((1, 3, 1)) / 3.0),
        (tf.eye(3), tf.ones((2, 3, 1)) / 3.0),
        (tf.eye(3, batch_shape=[2]), tf.ones((2, 3)) / 3.0),
        (tf.eye(3, batch_shape=[2]), tf.ones((2, 4, 1)) / 4.0),
        (tf.eye(3, batch_shape=[2]), tf.ones((2, 3, 2)) / 3.0),
    ],
    ids=[
        "batch_mismatch",
        "rank_2_covariance",
        "rank_2_weights",
        "asset_mismatch",
        "weight_last_dim_not_one",
    ],
)
def test_variance_loss_rejects_invalid_shapes(covariance, weights):
    with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
        variance_loss_function(covariance, weights)


@pytest.mark.parametrize(
    ("covariance", "weights"),
    [
        (
            tf.constant(
                [
                    [[1.0, 0.0], [0.0, np.nan]],
                ],
                dtype=tf.float32,
            ),
            tf.ones((1, 2, 1), dtype=tf.float32) / 2.0,
        ),
        (
            tf.eye(2, batch_shape=[1], dtype=tf.float32),
            tf.constant([[[0.5], [np.inf]]], dtype=tf.float32),
        ),
    ],
    ids=["nan_covariance", "inf_weights"],
)
def test_variance_loss_rejects_non_finite_inputs(covariance, weights):
    with pytest.raises(tf.errors.InvalidArgumentError, match="finite"):
        variance_loss_function(covariance, weights)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RIEnetLayer(recurrent_layer_sizes=[]),
        lambda: RIEnetLayer(recurrent_layer_sizes=[0]),
        lambda: RIEnetLayer(std_hidden_layer_sizes=[]),
        lambda: RIEnetLayer(std_hidden_layer_sizes=[0]),
        lambda: RIEnetLayer(recurrent_cell="RNN"),
        lambda: LagTransformLayer(variant="bad"),
        lambda: CorrelationEigenTransformLayer(recurrent_layer_sizes=()),
        lambda: CorrelationEigenTransformLayer(recurrent_layer_sizes=(0,)),
        lambda: CorrelationEigenTransformLayer(recurrent_cell="RNN"),
        lambda: CorrelationEigenTransformLayer(recurrent_direction="sideways"),
        lambda: CustomNormalizationLayer(
            mode="inverse",
            inverse_power=0,
            name="bad_inverse_power",
        ),
    ],
)
def test_invalid_constructor_arguments_raise(factory):
    with pytest.raises(ValueError):
        factory()

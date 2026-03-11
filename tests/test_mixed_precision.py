"""Mixed-precision and dtype-policy regression tests."""

import pytest
import tensorflow as tf
from keras import mixed_precision

from rienet import RIEnetLayer
from rienet.losses import variance_loss_function
from rienet.trainable_layers import CorrelationEigenTransformLayer, LagTransformLayer
from tests.test_rienet_output_parity import assert_outputs_match, legacy_reference_forward


def test_rienet_forward_mixed_bfloat16_stable():
    previous_policy = mixed_precision.global_policy()
    mixed_precision.set_global_policy("mixed_bfloat16")
    try:
        layer = RIEnetLayer(output_type="weights", name="rienet_mp")
        x = tf.random.normal((2, 5, 20), dtype=tf.bfloat16)
        y = layer(x)

        assert y.dtype == tf.bfloat16
        assert bool(tf.reduce_all(tf.math.is_finite(tf.cast(y, tf.float32))).numpy())
    finally:
        mixed_precision.set_global_policy(previous_policy)


def test_layers_follow_float64_policy_without_float32_hardcoding():
    previous_policy = mixed_precision.global_policy()
    mixed_precision.set_global_policy("float64")
    try:
        lag_layer = LagTransformLayer(variant="compact", name="lag64")
        x = tf.random.normal((1, 4, 12), dtype=tf.float64)
        y = lag_layer(x)

        assert y.dtype == tf.float64
        assert all(weight.dtype == tf.float64 for weight in lag_layer.weights)

        corr_layer = CorrelationEigenTransformLayer(output_type="correlation", name="corr64")
        corr = tf.eye(4, batch_shape=[2], dtype=tf.float64)
        attrs = tf.ones((2, 4, 2), dtype=tf.float64)
        corr_out = corr_layer(corr, attributes=attrs)

        assert corr_out.dtype == tf.float64
        assert bool(tf.reduce_all(tf.math.is_finite(corr_out)).numpy())
    finally:
        mixed_precision.set_global_policy(previous_policy)


def test_variance_loss_computes_in_float32_for_low_precision_inputs():
    previous_policy = mixed_precision.global_policy()
    mixed_precision.set_global_policy("mixed_bfloat16")
    try:
        covariance_true = tf.eye(4, batch_shape=[2], dtype=tf.bfloat16)
        weights = tf.ones((2, 4, 1), dtype=tf.bfloat16) / tf.cast(4.0, tf.bfloat16)

        loss = variance_loss_function(covariance_true, weights)

        assert loss.dtype == tf.float32
        assert bool(tf.reduce_all(tf.math.is_finite(loss)).numpy())
    finally:
        mixed_precision.set_global_policy(previous_policy)


@pytest.mark.parametrize(
    "output_type",
    [
        "weights",
        ["weights", "precision"],
        "all",
    ],
    ids=["weights", "weights_precision", "all"],
)
def test_rienet_mixed_bfloat16_matches_legacy_reference(output_type):
    previous_policy = mixed_precision.global_policy()
    mixed_precision.set_global_policy("mixed_bfloat16")
    try:
        tf.keras.utils.set_random_seed(4321)
        layer_name = "rienet_mp_" + (
            output_type if isinstance(output_type, str) else "_".join(output_type)
        )
        layer = RIEnetLayer(output_type=output_type, name=layer_name)
        inputs = tf.random.normal((2, 5, 20), dtype=tf.bfloat16)

        actual = layer(inputs)
        expected = legacy_reference_forward(layer, inputs)

        assert_outputs_match(actual, expected, rtol=3e-2, atol=3e-2)
    finally:
        mixed_precision.set_global_policy(previous_policy)

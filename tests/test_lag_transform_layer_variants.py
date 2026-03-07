import numpy as np
import pytest
import tensorflow as tf

from rienet import RIEnetLayer
from rienet.trainable_layers import LagTransformLayer
from rienet.lag_transform import LagTransformLayer as LagTransformFromModule


def test_compact_supports_dynamic_T_and_variable_n_stocks():
    assert LagTransformFromModule is LagTransformLayer

    layer = LagTransformLayer(variant="compact")

    x1 = tf.random.normal((2, 5, 20))
    y1 = layer(x1)
    assert y1.shape == x1.shape

    x2 = tf.random.normal((2, 11, 30))
    y2 = layer(x2)
    assert y2.shape == x2.shape


def test_per_lag_supports_variable_n_stocks_but_fixed_T():
    layer = LagTransformLayer(variant="per_lag")

    x1 = tf.random.normal((2, 5, 20))
    y1 = layer(x1)
    assert y1.shape == x1.shape

    x2 = tf.random.normal((2, 11, 20))
    y2 = layer(x2)
    assert y2.shape == x2.shape


def test_per_lag_infers_lookback_from_first_input():
    layer = LagTransformLayer(variant="per_lag")

    x1 = tf.random.normal((2, 5, 20))
    y1 = layer(x1)
    assert y1.shape == x1.shape

    x2 = tf.random.normal((2, 5, 20))
    y2 = layer(x2)
    assert y2.shape == x2.shape

    x3 = tf.random.normal((2, 5, 21))
    with pytest.raises(ValueError, match=r"expected=20, got=21"):
        _ = layer(x3)


def test_per_lag_mismatch_T_raises():
    layer = LagTransformLayer(variant="per_lag")

    x_ref = tf.random.normal((2, 5, 20))
    _ = layer(x_ref)

    x = tf.random.normal((2, 5, 19))
    with pytest.raises(ValueError, match=r"expected=20, got=19"):
        _ = layer(x)


def test_per_lag_dynamic_T_raises():
    layer = LagTransformLayer(variant="per_lag")

    x = tf.keras.Input(shape=(5, None))
    with pytest.raises(ValueError, match=r"static time dimension|got None"):
        _ = layer(x)


def test_compact_accepts_symbolic_sparse_inputs_from_scalar_arithmetic():
    n_stocks, lookback = 5, 20
    close2close = tf.keras.Input(shape=(n_stocks, lookback), name="close2close")
    open2close = tf.keras.Input(shape=(n_stocks, lookback), name="open2close")

    scaled = tf.keras.layers.Subtract()([close2close, open2close]) * 252.0
    assert scaled.sparse is True

    layer = LagTransformLayer(variant="compact", name="lag_compact")
    outputs = layer(scaled)
    assert outputs.shape == scaled.shape
    assert outputs.sparse is False

    model = tf.keras.Model([close2close, open2close], outputs)
    x_close2close = tf.random.normal((2, n_stocks, lookback))
    x_open2close = tf.random.normal((2, n_stocks, lookback))

    y_symbolic = model([x_close2close, x_open2close])
    y_dense = layer((x_close2close - x_open2close) * 252.0)

    np.testing.assert_allclose(y_symbolic.numpy(), y_dense.numpy(), atol=1e-6)


def test_per_lag_accepts_symbolic_sparse_inputs_from_scalar_arithmetic():
    n_stocks, lookback = 5, 20
    close2close = tf.keras.Input(shape=(n_stocks, lookback), name="close2close")
    open2close = tf.keras.Input(shape=(n_stocks, lookback), name="open2close")

    scaled = tf.keras.layers.Subtract()([close2close, open2close]) * 252.0
    assert scaled.sparse is True

    layer = LagTransformLayer(variant="per_lag", name="lag_per_lag")
    outputs = layer(scaled)
    assert outputs.shape == scaled.shape
    assert outputs.sparse is False

    model = tf.keras.Model([close2close, open2close], outputs)
    x_close2close = tf.random.normal((2, n_stocks, lookback))
    x_open2close = tf.random.normal((2, n_stocks, lookback))

    y_symbolic = model([x_close2close, x_open2close])
    y_dense = layer((x_close2close - x_open2close) * 252.0)

    np.testing.assert_allclose(y_symbolic.numpy(), y_dense.numpy(), atol=1e-6)


def test_sparse_tensor_input_matches_dense_input():
    dense = tf.random.normal((2, 5, 20))
    dense = tf.where(tf.abs(dense) > 0.35, dense, tf.zeros_like(dense))
    sparse = tf.sparse.from_dense(dense)

    layer = LagTransformLayer(variant="compact")
    y_dense = layer(dense)
    y_sparse = layer(sparse)

    np.testing.assert_allclose(y_sparse.numpy(), y_dense.numpy(), atol=1e-6)


def test_serialization_roundtrip_both_variants():
    layer_c = LagTransformLayer(variant="compact")
    x_c = tf.random.normal((1, 3, 12))
    _ = layer_c(x_c)

    ser_c = tf.keras.layers.serialize(layer_c)
    deser_c = tf.keras.layers.deserialize(ser_c)
    assert isinstance(deser_c, LagTransformLayer)
    assert deser_c.variant == "compact"
    y_c = deser_c(x_c)
    assert y_c.shape == x_c.shape

    layer_p = LagTransformLayer(variant="per_lag")
    x_p = tf.random.normal((1, 4, 20))
    _ = layer_p(x_p)

    ser_p = tf.keras.layers.serialize(layer_p)
    deser_p = tf.keras.layers.deserialize(ser_p)
    assert isinstance(deser_p, LagTransformLayer)
    assert deser_p.variant == "per_lag"
    assert "lookback_days" not in deser_p.get_config()
    y_p = deser_p(x_p)
    assert y_p.shape == x_p.shape


def test_rienet_integration_per_lag_variable_n_stocks():
    model_layer = RIEnetLayer(
        lag_transform_variant="per_lag",
        output_type="weights",
    )

    x1 = tf.random.normal((2, 5, 20))
    y1 = model_layer(x1)
    assert y1.shape == (2, 5, 1)

    x2 = tf.random.normal((2, 11, 20))
    y2 = model_layer(x2)
    assert y2.shape == (2, 11, 1)

    sums_1 = tf.reduce_sum(y1, axis=1)
    sums_2 = tf.reduce_sum(y2, axis=1)
    np.testing.assert_allclose(sums_1.numpy(), 1.0, atol=1e-4)
    np.testing.assert_allclose(sums_2.numpy(), 1.0, atol=1e-4)

"""Regression tests for saving/loading RIEnet models in .keras format."""

import tensorflow as tf

from rienet.trainable_layers import (
    CorrelationEigenTransformLayer,
    LagTransformLayer,
    RIEnetLayer,
)


def _roundtrip_model(model: tf.keras.Model, save_path):
    """Save and reload a model using the native Keras format."""
    model.save(save_path)
    return tf.keras.models.load_model(save_path)


def test_rienet_model_roundtrip_keras(tmp_path):
    inputs = tf.keras.Input(shape=(6, 24), name="returns")
    outputs = RIEnetLayer(name="rienet_layer")(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="rienet_model")

    loaded = _roundtrip_model(model, tmp_path / "rienet_model.keras")

    x = tf.random.normal((2, 6, 24))
    y = loaded(x)
    assert y.shape == (2, 6, 1)


def test_correlation_eigen_model_roundtrip_keras(tmp_path):
    corr = tf.keras.Input(shape=(6, 6), name="corr")
    attrs = tf.keras.Input(shape=(6, 2), name="attrs")

    # Name intentionally omitted to validate auto-generated naming works.
    layer = CorrelationEigenTransformLayer(
        output_type=("correlation", "eigenvalues"),
    )
    outputs = layer(corr, attributes=attrs)
    model = tf.keras.Model(inputs=[corr, attrs], outputs=outputs, name="corr_model")

    loaded = _roundtrip_model(model, tmp_path / "corr_model.keras")

    x_corr = tf.random.normal((2, 6, 6))
    x_attrs = tf.random.normal((2, 6, 2))
    y = loaded([x_corr, x_attrs])
    assert isinstance(y, dict)
    assert set(y.keys()) == {"correlation", "eigenvalues"}
    assert y["correlation"].shape == (2, 6, 6)
    assert y["eigenvalues"].shape == (2, 6, 1)


def test_lag_transform_model_roundtrip_keras(tmp_path):
    inputs = tf.keras.Input(shape=(6, 24), name="returns")
    outputs = LagTransformLayer(variant="per_lag", name="lag_layer")(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="lag_model")

    loaded = _roundtrip_model(model, tmp_path / "lag_model.keras")

    x = tf.random.normal((2, 6, 24))
    y = loaded(x)
    assert y.shape == (2, 6, 24)

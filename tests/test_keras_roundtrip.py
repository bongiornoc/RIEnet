"""Regression tests for saving/loading RIEnet models in .keras format."""

import numpy as np
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


def _assert_outputs_close(actual, expected, rtol=1e-5, atol=1e-6):
    """Compare model outputs recursively across tensor or dict structures."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert set(actual.keys()) == set(expected.keys())
        for key in expected:
            _assert_outputs_close(actual[key], expected[key], rtol=rtol, atol=atol)
        return

    np.testing.assert_allclose(
        actual.numpy(),
        expected.numpy(),
        rtol=rtol,
        atol=atol,
    )


def _make_correlation_batch(batch_size: int, n_assets: int) -> tf.Tensor:
    """Generate a positive definite correlation batch for spectral tests."""
    raw = tf.random.normal((batch_size, n_assets, n_assets))
    covariance = tf.matmul(raw, raw, transpose_b=True)
    std = tf.sqrt(tf.linalg.diag_part(covariance))
    corr_scale = tf.einsum("bi,bj->bij", std, std)
    return covariance / corr_scale


def test_rienet_model_roundtrip_keras_preserves_outputs_and_names(tmp_path):
    inputs = tf.keras.Input(shape=(6, 24), name="returns")
    outputs = RIEnetLayer(
        output_type=[
            "weights",
            "precision",
            "covariance",
            "correlation",
            "eigenvalues",
            "eigenvectors",
            "transformed_std",
            "input_transformed",
        ],
        name="rienet_layer",
    )(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="rienet_model")

    x = tf.random.normal((2, 6, 24))
    expected = model(x)
    loaded = _roundtrip_model(model, tmp_path / "rienet_model.keras")
    actual = loaded(x)

    _assert_outputs_close(actual, expected)

    loaded_layer = loaded.get_layer("rienet_layer")
    assert isinstance(loaded_layer, RIEnetLayer)
    assert loaded.name == "rienet_model"
    assert loaded_layer.lag_transform.name == "rienet_layer_lag_transform"
    assert loaded_layer.std_layer.name == "rienet_layer_std"
    assert loaded_layer.covariance_layer.name == "rienet_layer_covariance"
    assert loaded_layer.dimension_aware.name == "rienet_layer_dimension_aware"
    assert loaded_layer.correlation_eigen_transform is not None
    assert loaded_layer.correlation_eigen_transform.name == "rienet_layer_corr_eigen_transform"
    assert loaded_layer.std_transform is not None
    assert loaded_layer.std_transform.name == "rienet_layer_std_transform"
    assert loaded_layer.std_normalization is not None
    assert loaded_layer.std_normalization.name == "rienet_layer_std_norm"
    assert loaded_layer.outer_product is not None
    assert loaded_layer.outer_product.name == "rienet_layer_inverse_scale_outer"


def test_compiled_rienet_model_roundtrip_keras_preserves_optimizer_state(tmp_path):
    inputs = tf.keras.Input(shape=(6, 24), name="returns")
    outputs = RIEnetLayer(output_type="weights", name="rienet_weights")(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="compiled_rienet")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")

    x = tf.random.normal((4, 6, 24))
    y = tf.random.normal((4, 6, 1))
    model.train_on_batch(x, y)
    expected = model(x)

    loaded = _roundtrip_model(model, tmp_path / "compiled_rienet.keras")
    actual = loaded(x)

    _assert_outputs_close(actual, expected)
    assert isinstance(loaded.optimizer, tf.keras.optimizers.Adam)
    assert int(loaded.optimizer.iterations.numpy()) == int(model.optimizer.iterations.numpy())


def test_correlation_eigen_model_roundtrip_keras_preserves_outputs_and_names(tmp_path):
    corr = tf.keras.Input(shape=(6, 6), name="corr")
    attrs = tf.keras.Input(shape=(6, 2), name="attrs")
    layer = CorrelationEigenTransformLayer(
        output_type=(
            "correlation",
            "inverse_correlation",
            "eigenvalues",
            "eigenvectors",
            "inverse_eigenvalues",
        ),
        name="corr_cleaner",
    )
    outputs = layer(corr, attributes=attrs)
    model = tf.keras.Model(inputs=[corr, attrs], outputs=outputs, name="corr_model")

    x_corr = _make_correlation_batch(2, 6)
    x_attrs = tf.random.normal((2, 6, 2))
    expected = model([x_corr, x_attrs])
    loaded = _roundtrip_model(model, tmp_path / "corr_model.keras")
    actual = loaded([x_corr, x_attrs])

    _assert_outputs_close(actual, expected)

    loaded_layer = loaded.get_layer("corr_cleaner")
    assert isinstance(loaded_layer, CorrelationEigenTransformLayer)
    assert loaded.name == "corr_model"
    assert loaded_layer.spectral_decomp.name == "corr_cleaner_spectral"
    assert loaded_layer.eigenvalue_transform.name == "corr_cleaner_eigenvalue_rnn"
    assert loaded_layer.eigenvector_rescaler.name == "corr_cleaner_eigenvector_rescaler"
    assert loaded_layer.correlation_product.name == "corr_cleaner_correlation"


def test_correlation_eigen_model_roundtrip_keras_with_autogenerated_name(tmp_path):
    corr = tf.keras.Input(shape=(6, 6), name="corr")
    attrs = tf.keras.Input(shape=(6, 2), name="attrs")
    outputs = CorrelationEigenTransformLayer(
        output_type=("correlation", "eigenvalues"),
    )(corr, attributes=attrs)
    model = tf.keras.Model(inputs=[corr, attrs], outputs=outputs, name="corr_model_auto")

    x_corr = _make_correlation_batch(2, 6)
    x_attrs = tf.random.normal((2, 6, 2))
    expected = model([x_corr, x_attrs])
    loaded = _roundtrip_model(model, tmp_path / "corr_model_auto.keras")
    actual = loaded([x_corr, x_attrs])

    _assert_outputs_close(actual, expected)


def test_lag_transform_model_roundtrip_keras_preserves_outputs_and_names(tmp_path):
    inputs = tf.keras.Input(shape=(6, 24), name="returns")
    outputs = LagTransformLayer(variant="per_lag", name="lag_layer")(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="lag_model")

    x = tf.random.normal((2, 6, 24))
    expected = model(x)
    loaded = _roundtrip_model(model, tmp_path / "lag_model.keras")
    actual = loaded(x)

    _assert_outputs_close(actual, expected)

    loaded_layer = loaded.get_layer("lag_layer")
    assert isinstance(loaded_layer, LagTransformLayer)
    assert loaded.name == "lag_model"
    assert loaded_layer.variant == "per_lag"


def test_rienet_load_weights_before_first_forward_has_no_unbuilt_spectral_layers(tmp_path):
    def _make_model():
        inputs = tf.keras.Input(shape=(6, 24), name="returns")
        outputs = RIEnetLayer(output_type=["correlation"], name="rienet_layer")(inputs)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name="rienet_model")

    source = _make_model()
    weights_path = tmp_path / "rienet_corr.weights.h5"
    source.save_weights(weights_path)

    target = _make_model()
    rienet_layer = next(layer for layer in target.layers if isinstance(layer, RIEnetLayer))
    corr_transform = rienet_layer.correlation_eigen_transform
    assert corr_transform is not None

    unbuilt_spectral_layers = [
        sublayer.name
        for sublayer in corr_transform._flatten_layers(include_self=False, recursive=True)
        if not sublayer.built
    ]
    assert unbuilt_spectral_layers == []

    target.load_weights(weights_path)

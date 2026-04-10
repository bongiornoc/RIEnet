"""Regression tests for saving/loading RIEnet models in .keras format."""

import inspect
import numpy as np
import pytest
import tensorflow as tf

from rienet.ops_layers import (
    CovarianceLayer,
    CustomNormalizationLayer,
    DimensionAwareLayer,
    EigenProductLayer,
    EigenvectorRescalingLayer,
    EigenWeightsLayer,
    NormalizedSum,
    SpectralDecompositionLayer,
    StandardDeviationLayer,
)
from rienet.trainable_layers import (
    CorrelationEigenTransformLayer,
    DeepLayer,
    DeepRecurrentLayer,
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


GET_CONFIG_CASES = [
    (
        StandardDeviationLayer(axis=-2, demean=True, epsilon=1e-4, name="std_cfg"),
        {"axis": -2, "demean": True, "epsilon": 1e-4},
    ),
    (
        CovarianceLayer(expand_dims=True, normalize=False, name="cov_cfg"),
        {"expand_dims": True, "normalize": False},
    ),
    (
        SpectralDecompositionLayer(name="spectral_cfg"),
        {},
    ),
    (
        DimensionAwareLayer(features=["n_days", "rsqrt_n_days"], name="dim_cfg"),
        {"features": ["n_days", "rsqrt_n_days"]},
    ),
    (
        CustomNormalizationLayer(
            mode="inverse",
            axis=-1,
            inverse_power=2.5,
            epsilon=1e-4,
            name="norm_cfg",
        ),
        {"mode": "inverse", "axis": -1, "inverse_power": 2.5, "epsilon": 1e-4},
    ),
    (
        EigenvectorRescalingLayer(epsilon=1e-4, name="rescale_cfg"),
        {"epsilon": 1e-4},
    ),
    (
        EigenProductLayer(name="product_cfg"),
        {},
    ),
    (
        EigenWeightsLayer(epsilon=1e-4, name="weights_cfg"),
        {"epsilon": 1e-4},
    ),
    (
        NormalizedSum(axis_1=-2, axis_2=-1, epsilon=1e-4, name="normsum_cfg"),
        {"axis_1": -2, "axis_2": -1, "epsilon": 1e-4},
    ),
    (
        DeepLayer(
            hidden_layer_sizes=[5, 3],
            last_activation="softplus",
            activation="relu",
            other_biases=False,
            last_bias=False,
            dropout_rate=0.2,
            kernel_initializer="he_uniform",
            name="deep_cfg",
        ),
        {
            "hidden_layer_sizes": [5, 3],
            "last_activation": "softplus",
            "activation": "relu",
            "other_biases": False,
            "last_bias": False,
            "dropout_rate": 0.2,
            "kernel_initializer": "he_uniform",
        },
    ),
    (
        DeepRecurrentLayer(
            recurrent_layer_sizes=[7],
            final_activation="sigmoid",
            final_hidden_layer_sizes=[4],
            final_hidden_activation="relu",
            direction="forward",
            dropout=0.2,
            recurrent_dropout=0.1,
            recurrent_model="GRU",
            normalize="inverse",
            normalize_inverse_power=2.0,
            name="deeprnn_cfg",
        ),
        {
            "recurrent_layer_sizes": [7],
            "final_activation": "sigmoid",
            "final_hidden_layer_sizes": [4],
            "final_hidden_activation": "relu",
            "direction": "forward",
            "dropout": 0.2,
            "recurrent_dropout": 0.1,
            "recurrent_model": "GRU",
            "normalize": "inverse",
            "normalize_inverse_power": 2.0,
        },
    ),
    (
        CorrelationEigenTransformLayer(
            recurrent_layer_sizes=(8,),
            recurrent_cell="LSTM",
            recurrent_direction="forward",
            final_hidden_layer_sizes=(4,),
            final_hidden_activation="relu",
            output_type=("correlation", "eigenvalues"),
            epsilon=1e-4,
            name="corr_cfg",
        ),
        {
            "recurrent_layer_sizes": [8],
            "recurrent_cell": "LSTM",
            "recurrent_direction": "forward",
            "final_hidden_layer_sizes": [4],
            "final_hidden_activation": "relu",
            "output_type": ["correlation", "eigenvalues"],
            "epsilon": 1e-4,
        },
    ),
    (
        LagTransformLayer(
            warm_start=False,
            eps=1e-4,
            variant="per_lag",
            name="lag_cfg",
        ),
        {"warm_start": False, "eps": 1e-4, "variant": "per_lag"},
    ),
    (
        RIEnetLayer(
            output_type=["weights", "precision"],
            recurrent_layer_sizes=[8],
            std_hidden_layer_sizes=[4],
            recurrent_cell="LSTM",
            recurrent_direction="forward",
            dimensional_features=["n_days", "rsqrt_n_days"],
            normalize_transformed_variance=False,
            lag_transform_variant="per_lag",
            annualization_factor=365.0,
            name="rienet_cfg",
        ),
        {
            "output_type": ["weights", "precision"],
            "recurrent_layer_sizes": [8],
            "std_hidden_layer_sizes": [4],
            "recurrent_cell": "LSTM",
            "recurrent_direction": "forward",
            "dimensional_features": ["n_days", "rsqrt_n_days"],
            "normalize_transformed_variance": False,
            "lag_transform_variant": "per_lag",
            "annualization_factor": 365.0,
        },
    ),
]


def _reconstruct_layer_from_config(layer: tf.keras.layers.Layer, config: dict):
    """Rebuild a layer from its config using the class-level deserializer."""
    return layer.__class__.from_config(config)


@pytest.mark.parametrize(
    ("layer", "expected_config"),
    GET_CONFIG_CASES,
    ids=[layer.name for layer, _ in GET_CONFIG_CASES],
)
def test_custom_layer_get_config_includes_all_constructor_parameters(layer, expected_config):
    """All constructor kwargs must appear in get_config for custom layers."""
    del expected_config
    config = layer.get_config()

    init_signature = inspect.signature(layer.__class__.__init__)
    init_parameter_names = {
        name
        for name, parameter in init_signature.parameters.items()
        if name not in {"self", "name", "kwargs"}
        and parameter.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }

    missing = init_parameter_names.difference(config.keys())
    assert missing == set()


@pytest.mark.parametrize(
    ("layer", "expected_config"),
    GET_CONFIG_CASES,
    ids=[layer.name for layer, _ in GET_CONFIG_CASES],
)
def test_custom_layer_get_config_covers_non_default_parameters(layer, expected_config):
    """Every custom layer should serialize all non-default constructor parameters."""
    config = layer.get_config()

    init_signature = inspect.signature(layer.__class__.__init__)
    init_parameter_names = {
        name
        for name, parameter in init_signature.parameters.items()
        if name not in {"self", "name", "kwargs"}
        and parameter.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }
    assert set(expected_config).issubset(init_parameter_names)

    for key, value in expected_config.items():
        assert config[key] == value

    restored = _reconstruct_layer_from_config(layer, config)
    restored_config = restored.get_config()
    for key, value in expected_config.items():
        assert restored_config[key] == value


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
            "input_zscores",
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


def test_rienet_model_roundtrip_keras_preserves_non_default_configuration(tmp_path):
    inputs = tf.keras.Input(shape=(6, 24), name="returns")
    outputs = RIEnetLayer(
        output_type="all",
        recurrent_layer_sizes=[12, 6],
        std_hidden_layer_sizes=[5, 3],
        recurrent_cell="LSTM",
        recurrent_direction="forward",
        dimensional_features=["n_days", "rsqrt_n_days"],
        normalize_transformed_variance=False,
        lag_transform_variant="per_lag",
        annualization_factor=365.0,
        name="rienet_custom",
    )(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="rienet_custom_model")

    x = tf.random.normal((2, 6, 24))
    expected = model(x)
    loaded = _roundtrip_model(model, tmp_path / "rienet_custom_model.keras")
    actual = loaded(x)

    _assert_outputs_close(actual, expected)

    loaded_layer = loaded.get_layer("rienet_custom")
    assert isinstance(loaded_layer, RIEnetLayer)
    assert loaded_layer.get_config()["annualization_factor"] == 365.0
    assert loaded_layer._annualization_factor == 365.0
    assert loaded_layer._recurrent_layer_sizes == [12, 6]
    assert loaded_layer._std_hidden_layer_sizes == [5, 3]
    assert loaded_layer._recurrent_model == "LSTM"
    assert loaded_layer._direction == "forward"
    assert loaded_layer._dimensional_features == ["n_days", "rsqrt_n_days"]
    assert loaded_layer._normalize_variance is False
    assert loaded_layer._lag_transform_variant == "per_lag"


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

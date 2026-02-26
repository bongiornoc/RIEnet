"""
Tests for the RIEnet package.

This module contains comprehensive tests for all components of the RIEnet
package including the main layer, loss functions, and custom layers.
"""

import pytest
import tensorflow as tf
import numpy as np
from typing import Tuple

# Import all components to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rienet import (
    RIEnetLayer,
    CorrelationEigenTransformLayer as PublicCorrelationEigenTransformLayer,
    EigenWeightsLayer as PublicEigenWeightsLayer,
    LagTransformLayer as PublicLagTransformLayer,
    variance_loss_function
)

from rienet.custom_layers import (
    StandardDeviationLayer,
    CovarianceLayer, 
    SpectralDecompositionLayer,
    DimensionAwareLayer,
    DeepLayer,
    DeepRecurrentLayer,
    CustomNormalizationLayer,
    EigenProductLayer,
    EigenvectorRescalingLayer,
    EigenWeightsLayer,
    CorrelationEigenTransformLayer,
    NormalizedSum,
    LagTransformLayer
)


class TestRIEnetLayer:
    """Test cases for the main RIEnetLayer."""
    
    def test_layer_initialization(self):
        """Test layer can be initialized with different parameters."""
        # Test default initialization
        layer1 = RIEnetLayer()
        assert layer1.output_type == 'weights'
        assert layer1._direction == 'bidirectional'
        assert layer1._dimensional_features == ['n_stocks', 'n_days', 'q']
        
        # Test with precision output
        layer2 = RIEnetLayer(output_type='precision')
        assert layer2.output_type == 'precision'
        
        # Test invalid output type
        with pytest.raises(ValueError):
            RIEnetLayer(output_type='invalid')

    def test_public_eigen_weights_layer_is_exposed(self):
        """EigenWeightsLayer should be importable from package root."""
        assert PublicEigenWeightsLayer is EigenWeightsLayer

    def test_public_correlation_eigen_transform_layer_is_exposed(self):
        """CorrelationEigenTransformLayer should be importable from package root."""
        assert PublicCorrelationEigenTransformLayer is CorrelationEigenTransformLayer

    def test_public_lag_transform_layer_is_exposed(self):
        """LagTransformLayer should be importable from package root."""
        assert PublicLagTransformLayer is LagTransformLayer
    
    def test_weights_output_shape(self):
        """Test that weights output has correct shape."""
        layer = RIEnetLayer(output_type='weights')
        
        # Test various input shapes
        test_cases = [
            (32, 10, 60),  # 32 batches, 10 stocks, 60 days
            (1, 5, 30),    # Single batch, 5 stocks, 30 days
            (16, 20, 120), # 16 batches, 20 stocks, 120 days
        ]
        
        for batch_size, n_stocks, n_days in test_cases:
            inputs = tf.random.normal((batch_size, n_stocks, n_days))
            outputs = layer(inputs)
            
            expected_shape = (batch_size, n_stocks, 1)
            assert outputs.shape == expected_shape, f"Expected {expected_shape}, got {outputs.shape}"
    
    def test_precision_output_shape(self):
        """Test that precision output has correct shape."""
        layer = RIEnetLayer(output_type='precision')
        
        batch_size, n_stocks, n_days = 16, 8, 50
        inputs = tf.random.normal((batch_size, n_stocks, n_days))
        outputs = layer(inputs)
        
        expected_shape = (batch_size, n_stocks, n_stocks)
        assert outputs.shape == expected_shape, f"Expected {expected_shape}, got {outputs.shape}"

    def test_covariance_output_shape(self):
        """Test that covariance output has correct shape."""
        layer = RIEnetLayer(output_type='covariance')

        batch_size, n_stocks, n_days = 12, 5, 40
        inputs = tf.random.normal((batch_size, n_stocks, n_days))
        outputs = layer(inputs)

        expected_shape = (batch_size, n_stocks, n_stocks)
        assert outputs.shape == expected_shape, f"Expected {expected_shape}, got {outputs.shape}"
    
    def test_correlation_output_shape(self):
        """Test that correlation output has correct shape."""
        layer = RIEnetLayer(output_type='correlation')

        batch_size, n_stocks, n_days = 7, 6, 30
        inputs = tf.random.normal((batch_size, n_stocks, n_days))
        outputs = layer(inputs)

        expected_shape = (batch_size, n_stocks, n_stocks)
        assert outputs.shape == expected_shape, f"Expected {expected_shape}, got {outputs.shape}"
    
    def test_weights_normalization(self):
        """Test that portfolio weights sum to 1."""
        layer = RIEnetLayer(output_type='weights')
        
        batch_size, n_stocks, n_days = 8, 6, 40
        inputs = tf.random.normal((batch_size, n_stocks, n_days), stddev=0.02)
        weights = layer(inputs)
        
        # Check weights sum to 1 along the stocks dimension
        weights_sum = tf.reduce_sum(weights, axis=1)  # Sum over stocks
        
        # Should be close to 1 for each sample
        np.testing.assert_allclose(weights_sum.numpy(), 1.0, rtol=1e-5)
    
    def test_input_scaling(self):
        """Test that input scaling by 252 is applied."""
        layer = RIEnetLayer(output_type='weights')
        
        # Small inputs to see scaling effect
        small_inputs = tf.ones((4, 5, 30)) * 0.001
        large_inputs = small_inputs * 252
        
        weights_small = layer(small_inputs)
        weights_large = layer(large_inputs)
        
        # After standardisation the layer is scale-invariant; outputs should match
        diff = tf.reduce_max(tf.abs(weights_small - weights_large))
        assert diff < 1e-6, f"Input scaling should preserve outputs, got {diff}"
    
    def test_layer_serialization(self):
        """Test that layer can be serialized and deserialized."""
        layer = RIEnetLayer(
            output_type='weights',
            recurrent_direction='forward',
            dimensional_features=['n_stocks', 'rsqrt_n_days'],
            name='test_layer'
        )
        config = layer.get_config()
        
        # Check config contains expected keys
        assert 'output_type' in config
        assert config['output_type'] == 'weights'
        assert config['recurrent_direction'] == 'forward'
        assert config['dimensional_features'] == ['n_stocks', 'rsqrt_n_days']
        
        # Test from_config
        new_layer = RIEnetLayer.from_config(config)
        assert new_layer.output_type == layer.output_type
        assert new_layer._direction == 'forward'
        assert new_layer._dimensional_features == ['n_stocks', 'rsqrt_n_days']

    def test_multiple_outputs(self):
        """Layer should optionally return multiple components."""
        layer = RIEnetLayer(output_type=['weights', 'precision'])

        batch_size, n_stocks, n_days = 3, 4, 20
        inputs = tf.random.normal((batch_size, n_stocks, n_days))
        outputs = layer(inputs)

        assert isinstance(outputs, dict)
        assert set(outputs.keys()) == {'weights', 'precision'}
        assert outputs['weights'].shape == (batch_size, n_stocks, 1)
        assert outputs['precision'].shape == (batch_size, n_stocks, n_stocks)

    def test_additional_spectral_outputs_shapes(self):
        """Layer should expose eigenvalues/eigenvectors/transformed_std outputs."""
        layer = RIEnetLayer(
            output_type=['eigenvalues', 'eigenvectors', 'transformed_std']
        )

        batch_size, n_stocks, n_days = 2, 5, 20
        inputs = tf.random.normal((batch_size, n_stocks, n_days))
        outputs = layer(inputs)

        assert isinstance(outputs, dict)
        assert set(outputs.keys()) == {'eigenvalues', 'eigenvectors', 'transformed_std'}
        assert outputs['eigenvalues'].shape == (batch_size, n_stocks, 1)
        assert outputs['eigenvectors'].shape == (batch_size, n_stocks, n_stocks)
        assert outputs['transformed_std'].shape == (batch_size, n_stocks, 1)

    def test_all_output_type_includes_new_components(self):
        """The special 'all' output token should include all exposed components."""
        layer = RIEnetLayer(output_type='all')

        inputs = tf.random.normal((1, 4, 12))
        outputs = layer(inputs)

        expected_keys = {
            'weights',
            'precision',
            'covariance',
            'correlation',
            'input_transformed',
            'eigenvalues',
            'eigenvectors',
            'transformed_std',
        }
        assert isinstance(outputs, dict)
        assert set(outputs.keys()) == expected_keys

    def test_additional_outputs_are_non_inverse_values(self):
        """Exposed eigenvalues/std outputs must be non-inverse quantities."""
        layer = RIEnetLayer(output_type=['eigenvalues', 'transformed_std'])

        inputs = tf.random.normal((2, 4, 16))
        outputs = layer(inputs)

        scaled_inputs = inputs * layer._annualization_factor
        input_transformed = layer.lag_transform(scaled_inputs)
        std, mean = layer.std_layer(input_transformed)

        transformed_inverse_std = layer.std_transform(std)
        std_for_structural = transformed_inverse_std
        if layer.std_normalization is not None:
            std_for_structural = layer.std_normalization(transformed_inverse_std)

        zscores = (input_transformed - mean) / std
        correlation_matrix = layer.covariance_layer(zscores)
        attributes = layer.dimension_aware([zscores, correlation_matrix])
        transformed_inverse_eigenvalues = layer.correlation_eigen_transform(
            correlation_matrix,
            attributes=attributes,
            output_type='inverse_eigenvalues',
        )

        # Check reciprocal relationship to ensure the exposed outputs are not inverse.
        np.testing.assert_allclose(
            (outputs['eigenvalues'] * transformed_inverse_eigenvalues).numpy(),
            1.0,
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            (outputs['transformed_std'] * std_for_structural).numpy(),
            1.0,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_custom_recurrent_configuration(self):
        """Custom recurrent sizes and cell types should be honoured."""
        layer = RIEnetLayer(
            output_type='weights',
            recurrent_layer_sizes=[12, 6],
            std_hidden_layer_sizes=[4, 2],
            recurrent_cell='lstm'
        )

        batch_size, n_stocks, n_days = 2, 3, 15
        inputs = tf.random.normal((batch_size, n_stocks, n_days))
        weights = layer(inputs)

        assert weights.shape == (batch_size, n_stocks, 1)
        first_block = layer.correlation_eigen_transform.eigenvalue_transform.recurrent_layers[0]
        assert isinstance(first_block, tf.keras.layers.Bidirectional)
        assert isinstance(first_block.forward_layer, tf.keras.layers.LSTM)

    def test_custom_recurrent_direction_and_dimensional_features(self):
        """Direction and dimension-aware feature options should be configurable."""
        layer = RIEnetLayer(
            output_type='weights',
            recurrent_direction='forward',
            dimensional_features=['n_stocks', 'rsqrt_n_days'],
        )

        batch_size, n_stocks, n_days = 2, 3, 15
        inputs = tf.random.normal((batch_size, n_stocks, n_days))
        weights = layer(inputs)

        assert weights.shape == (batch_size, n_stocks, 1)
        first_block = layer.correlation_eigen_transform.eigenvalue_transform.recurrent_layers[0]
        assert isinstance(first_block, tf.keras.layers.GRU)
        assert first_block.go_backwards is False
        assert layer.dimension_aware.features == ['n_stocks', 'rsqrt_n_days']

    def test_invalid_recurrent_direction_raises(self):
        with pytest.raises(ValueError, match="recurrent_direction"):
            RIEnetLayer(recurrent_direction='sideways')

    def test_invalid_dimensional_features_raises(self):
        with pytest.raises(ValueError, match="dimensional_features"):
            RIEnetLayer(dimensional_features=['n_stocks', 'bad_feature'])


class TestVarianceLoss:
    """Tests for the variance loss function."""

    def test_variance_loss_function_shape(self):
        """Variance loss should keep batch dimension and output scalar per sample."""
        batch_size, n_assets = 16, 8

        weights = tf.random.normal((batch_size, n_assets, 1))
        weights = weights / tf.reduce_sum(weights, axis=1, keepdims=True)

        covariance = tf.random.normal((batch_size, n_assets, n_assets))
        covariance = tf.matmul(covariance, covariance, transpose_b=True)

        loss = variance_loss_function(covariance, weights)

        expected_shape = (batch_size, 1, 1)
        assert loss.shape == expected_shape

    def test_variance_loss_is_non_negative(self):
        """Variance loss values should be non-negative."""
        batch_size, n_assets = 8, 5

        weights = tf.random.normal((batch_size, n_assets, 1))
        weights = weights / tf.reduce_sum(weights, axis=1, keepdims=True)

        covariance = tf.eye(n_assets, batch_shape=[batch_size]) * 0.01

        loss = variance_loss_function(covariance, weights)

        assert tf.reduce_all(loss >= 0), "Variance loss should be non-negative"


class TestCustomLayers:
    """Test cases for custom layers."""
    
    def test_standard_deviation_layer(self):
        """Test StandardDeviationLayer."""
        layer = StandardDeviationLayer(axis=-1, name='test_std')
        
        # Test with known data
        x = tf.constant([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])  # (1, 2, 3)
        std, mean = layer(x)
        
        # Check shapes
        assert std.shape == (1, 2, 1)
        assert mean.shape == (1, 2, 1)
        
        # Check values are reasonable
        assert tf.reduce_all(std > 0), "Standard deviation should be positive"
    
    def test_covariance_layer(self):
        """Test CovarianceLayer."""
        layer = CovarianceLayer(normalize=True, name='test_cov')
        
        batch_size, n_assets, n_days = 4, 3, 20
        returns = tf.random.normal((batch_size, n_assets, n_days))
        
        cov_matrix = layer(returns)
        
        # Check output shape
        expected_shape = (batch_size, n_assets, n_assets)
        assert cov_matrix.shape == expected_shape
        
        # Covariance matrix should be symmetric
        cov_transpose = tf.transpose(cov_matrix, perm=[0, 2, 1])
        np.testing.assert_allclose(cov_matrix.numpy(), cov_transpose.numpy(), rtol=1e-6)
    
    def test_spectral_decomposition_layer(self):
        """Test SpectralDecompositionLayer."""
        layer = SpectralDecompositionLayer(name='test_spectral')
        
        batch_size, n_assets = 4, 5
        
        # Create symmetric positive definite matrix
        A = tf.random.normal((batch_size, n_assets, n_assets))
        cov_matrix = tf.matmul(A, A, transpose_b=True)
        
        eigenvalues, eigenvectors = layer(cov_matrix)
        
        # Check shapes
        assert eigenvalues.shape == (batch_size, n_assets, 1)
        assert eigenvectors.shape == (batch_size, n_assets, n_assets)
        
        # Eigenvalues should be positive for PSD matrix
        assert tf.reduce_all(eigenvalues >= 0), "Eigenvalues should be non-negative for PSD matrix"
    
    def test_dimension_aware_layer(self):
        """Test DimensionAwareLayer."""
        features = ['n_stocks', 'n_days', 'q']
        layer = DimensionAwareLayer(features=features, name='test_dim_aware')
        
        batch_size, n_stocks, n_days = 4, 6, 50
        standardized_returns = tf.random.normal((batch_size, n_stocks, n_days))
        correlation_matrix = tf.eye(n_stocks, batch_shape=[batch_size])
        
        enhanced = layer([standardized_returns, correlation_matrix])
        
        # Should output only the 3 attribute channels
        expected_shape = (batch_size, n_stocks, len(features))
        assert enhanced.shape == expected_shape
    
    def test_deep_layer(self):
        """Test DeepLayer."""
        layer = DeepLayer(
            hidden_layer_sizes=[16, 8, 4],
            activation='relu',
            last_activation='linear',
            name='test_deep'
        )
        
        batch_size, input_dim = 8, 10
        inputs = tf.random.normal((batch_size, input_dim))
        
        outputs = layer(inputs)
        
        # Should output final layer size
        expected_shape = (batch_size, 4)
        assert outputs.shape == expected_shape
    
    def test_deep_recurrent_layer(self):
        """Test DeepRecurrentLayer."""
        layer = DeepRecurrentLayer(
            recurrent_layer_sizes=[32],
            recurrent_model='GRU',
            direction='bidirectional',
            name='test_deep_rnn'
        )
        
        batch_size, seq_len, input_dim = 4, 20, 8
        inputs = tf.random.normal((batch_size, seq_len, input_dim))
        
        outputs = layer(inputs)
        
        # Output should have sequence dimension
        assert len(outputs.shape) >= 2
        assert outputs.shape[0] == batch_size
    
    def test_eigen_product_layer(self):
        """Test EigenProductLayer."""
        layer = EigenProductLayer(name='test_eigen_product')

        batch_size, n_assets = 4, 5
        eigenvalues = tf.random.normal((batch_size, n_assets))
        eigenvectors = tf.random.normal((batch_size, n_assets, n_assets))

        # Make eigenvectors orthogonal (approximately)
        eigenvectors, _ = tf.linalg.qr(eigenvectors)

        reconstructed = layer(eigenvalues, eigenvectors)

        # Should reconstruct matrix of same size
        expected_shape = (batch_size, n_assets, n_assets)
        assert reconstructed.shape == expected_shape

    def test_eigenvector_rescaling_layer(self):
        """EigenvectorRescalingLayer enforces unit diagonals."""
        layer = EigenvectorRescalingLayer(name='test_eigenvector_rescaler')
        product_layer = EigenProductLayer(name='test_eigen_product_for_rescaler')

        batch_size, n_assets = 3, 4
        eigenvalues = tf.random.uniform((batch_size, n_assets), 0.5, 1.5)
        eigenvectors = tf.linalg.qr(tf.random.normal((batch_size, n_assets, n_assets)))[0]

        rescaled = layer([eigenvectors, eigenvalues])
        reconstructed = product_layer(eigenvalues, rescaled)
        diag = tf.linalg.diag_part(reconstructed)
        assert float(tf.reduce_max(tf.abs(diag - 1.0)).numpy()) < 1e-6

    def test_eigen_weights_layer(self):
        """EigenWeightsLayer matches numpy einsum formulation."""
        layer = EigenWeightsLayer(name='test_eigen_weights')

        batch_size, n_assets = 2, 4
        eigenvectors = tf.linalg.qr(tf.random.normal((batch_size, n_assets, n_assets)))[0]
        inverse_eigenvalues = tf.random.uniform((batch_size, n_assets, 1), 0.5, 1.5)
        inverse_std = tf.random.uniform((batch_size, n_assets, 1), 0.8, 1.2)

        weights = layer(eigenvectors, inverse_eigenvalues, inverse_std)

        ev = eigenvectors.numpy()
        inv_eig = inverse_eigenvalues.numpy().reshape(batch_size, n_assets)
        inv_std_np = inverse_std.numpy().reshape(batch_size, n_assets)
        c = ev.sum(axis=1)
        raw = np.einsum('bik,bk,bk,bi->bi', ev, inv_eig, c, inv_std_np)
        expected = raw / raw.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(weights.numpy().squeeze(-1), expected, rtol=1e-5, atol=1e-6)

    def test_eigen_weights_layer_without_inverse_std(self):
        """When inverse_std is omitted, no extra scaling is applied."""
        layer = EigenWeightsLayer(name='test_eigen_weights_no_std')

        batch_size, n_assets = 2, 4
        eigenvectors = tf.linalg.qr(tf.random.normal((batch_size, n_assets, n_assets)))[0]
        inverse_eigenvalues = tf.random.uniform((batch_size, n_assets, 1), 0.5, 1.5)

        weights = layer(eigenvectors, inverse_eigenvalues)

        ev = eigenvectors.numpy()
        inv_eig = inverse_eigenvalues.numpy().reshape(batch_size, n_assets)
        c = ev.sum(axis=1)
        raw = np.einsum('bik,bk,bk->bi', ev, inv_eig, c)
        expected = raw / raw.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(weights.numpy().squeeze(-1), expected, rtol=1e-5, atol=1e-6)

    def test_eigen_weights_layer_fixed_inputs_matches_reference(self):
        """Fixed inputs should reproduce the original reference formula."""
        layer = EigenWeightsLayer(name='test_eigen_weights_fixed')

        eigenvectors = tf.constant(
            [[[1.0, 0.0, 0.0],
              [0.2, 0.9, 0.1],
              [0.3, 0.2, 0.8]]],
            dtype=tf.float32,
        )
        inverse_eigenvalues = tf.constant([[[1.2], [0.7], [1.1]]], dtype=tf.float32)
        inverse_std = tf.constant([[[0.9], [1.0], [1.1]]], dtype=tf.float32)

        weights = layer(eigenvectors, inverse_eigenvalues, inverse_std)

        ev = eigenvectors.numpy()
        inv_eig = inverse_eigenvalues.numpy().reshape(1, 3)
        inv_std_np = inverse_std.numpy().reshape(1, 3)
        c = ev.sum(axis=1)
        raw = np.einsum('bik,bk,bk,bi->bi', ev, inv_eig, c, inv_std_np)
        expected = raw / raw.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(weights.numpy().squeeze(-1), expected, rtol=1e-6, atol=1e-7)

    def test_correlation_eigen_transform_layer_without_attributes(self):
        """CorrelationEigenTransformLayer should clean correlation without attributes."""
        layer = CorrelationEigenTransformLayer(name='test_corr_eig_transform')

        batch_size, n_assets = 3, 5
        A = tf.random.normal((batch_size, n_assets, n_assets))
        covariance = tf.matmul(A, A, transpose_b=True)
        diag = tf.linalg.diag_part(covariance)
        inv_std = tf.math.rsqrt(tf.maximum(diag, tf.cast(1e-6, covariance.dtype)))
        correlation = (
            covariance
            * tf.expand_dims(inv_std, axis=-1)
            * tf.expand_dims(inv_std, axis=-2)
        )

        cleaned = layer(correlation)

        assert cleaned.shape == (batch_size, n_assets, n_assets)
        np.testing.assert_allclose(
            cleaned.numpy(),
            np.transpose(cleaned.numpy(), (0, 2, 1)),
            rtol=1e-5,
            atol=1e-6,
        )
        diag_clean = tf.linalg.diag_part(cleaned)
        np.testing.assert_allclose(diag_clean.numpy(), 1.0, rtol=1e-4, atol=1e-5)

    def test_correlation_eigen_transform_layer_with_attributes(self):
        """Layer should accept optional (batch, k) attributes."""
        layer = CorrelationEigenTransformLayer(name='test_corr_eig_transform_attr')

        batch_size, n_assets, k = 2, 4, 3
        A = tf.random.normal((batch_size, n_assets, n_assets))
        covariance = tf.matmul(A, A, transpose_b=True)
        diag = tf.linalg.diag_part(covariance)
        inv_std = tf.math.rsqrt(tf.maximum(diag, tf.cast(1e-6, covariance.dtype)))
        correlation = (
            covariance
            * tf.expand_dims(inv_std, axis=-1)
            * tf.expand_dims(inv_std, axis=-2)
        )
        attributes = tf.random.normal((batch_size, k))

        cleaned_with_attr = layer(correlation, attributes=attributes)

        assert cleaned_with_attr.shape == (batch_size, n_assets, n_assets)
        np.testing.assert_allclose(
            cleaned_with_attr.numpy(),
            np.transpose(cleaned_with_attr.numpy(), (0, 2, 1)),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_correlation_eigen_transform_layer_inconsistent_feature_width_raises(self):
        """Changing attribute width on the same layer instance should fail clearly."""
        layer = CorrelationEigenTransformLayer(name='test_corr_eig_transform_feat_width')

        correlation = tf.eye(4, batch_shape=[2])
        attributes = tf.random.normal((2, 3))
        _ = layer(correlation, attributes=attributes)

        with pytest.raises(ValueError, match='Inconsistent eigenvalue feature width'):
            _ = layer(correlation)

    def test_correlation_eigen_transform_layer_batch_mismatch_raises(self):
        """Attributes batch must match correlation batch."""
        layer = CorrelationEigenTransformLayer(name='test_corr_eig_transform_mismatch')

        correlation = tf.eye(4, batch_shape=[2])
        attributes = tf.random.normal((3, 2))

        with pytest.raises(ValueError, match='Batch mismatch'):
            _ = layer(correlation, attributes=attributes)

    def test_correlation_eigen_transform_layer_multiple_outputs(self):
        """Layer should expose selectable output components."""
        layer = CorrelationEigenTransformLayer(
            output_type=[
                'correlation',
                'inverse_correlation',
                'eigenvalues',
                'eigenvectors',
                'inverse_eigenvalues',
            ],
            name='test_corr_eig_transform_outputs',
        )

        correlation = tf.eye(4, batch_shape=[2])
        attributes = tf.random.normal((2, 2))
        outputs = layer(correlation, attributes=attributes)

        assert isinstance(outputs, dict)
        assert set(outputs.keys()) == {
            'correlation',
            'inverse_correlation',
            'eigenvalues',
            'eigenvectors',
            'inverse_eigenvalues',
        }
        assert outputs['correlation'].shape == (2, 4, 4)
        assert outputs['inverse_correlation'].shape == (2, 4, 4)
        assert outputs['eigenvalues'].shape == (2, 4, 1)
        assert outputs['eigenvectors'].shape == (2, 4, 4)
        assert outputs['inverse_eigenvalues'].shape == (2, 4, 1)

    def test_correlation_eigen_transform_layer_inverse_correlation_output(self):
        """Layer should return inverse correlation when requested."""
        layer = CorrelationEigenTransformLayer(
            output_type='inverse_correlation',
            name='test_corr_eig_transform_inv_corr',
        )

        correlation = tf.eye(4, batch_shape=[2])
        inverse_correlation = layer(correlation)

        assert inverse_correlation.shape == (2, 4, 4)
        np.testing.assert_allclose(
            inverse_correlation.numpy(),
            np.transpose(inverse_correlation.numpy(), (0, 2, 1)),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_correlation_eigen_transform_layer_invalid_output_type_raises(self):
        with pytest.raises(ValueError, match='output_type'):
            CorrelationEigenTransformLayer(output_type='bad_output', name='test_corr_bad_out')

    def test_correlation_eigen_transform_layer_serialization(self):
        """CorrelationEigenTransformLayer should serialize/deserialize correctly."""
        layer = CorrelationEigenTransformLayer(
            recurrent_layer_sizes=(8,),
            recurrent_cell='GRU',
            recurrent_direction='forward',
            final_hidden_layer_sizes=(4,),
            output_type=['correlation', 'inverse_correlation', 'inverse_eigenvalues'],
            name='test_corr_eig_transform_ser'
        )

        correlation = tf.eye(4, batch_shape=[2])
        attributes = tf.random.normal((2, 2))
        _ = layer(correlation, attributes=attributes)

        serialized = tf.keras.layers.serialize(layer)
        deserialized = tf.keras.layers.deserialize(serialized)

        assert isinstance(deserialized, CorrelationEigenTransformLayer)
        output = deserialized(correlation, attributes=attributes)
        assert isinstance(output, dict)
        assert set(output.keys()) == {'correlation', 'inverse_correlation', 'inverse_eigenvalues'}
        assert output['correlation'].shape == (2, 4, 4)
        assert output['inverse_correlation'].shape == (2, 4, 4)
        assert output['inverse_eigenvalues'].shape == (2, 4, 1)

    def test_precision_normalization_diagonal_mean(self):
        """Normalized precision keeps covariance diagonal centred on one."""
        batch_size, n_assets = 2, 6
        eigenvectors = tf.eye(n_assets, batch_shape=[batch_size])

        raw_eigenvalues = tf.random.uniform((batch_size, n_assets, 1), 0.5, 1.5)
        eigen_normalizer = CustomNormalizationLayer(
            mode='inverse', axis=-2, inverse_power=1.0, name='test_eigen_norm'
        )
        cleaned_eigenvalues = tf.squeeze(eigen_normalizer(raw_eigenvalues), axis=-1)

        rescaler = EigenvectorRescalingLayer(name='test_precision_rescaler')
        inverse_layer = EigenProductLayer(name='test_precision_reconstruct')
        inverse_vectors = rescaler([eigenvectors, cleaned_eigenvalues])
        inverse_correlation = inverse_layer(cleaned_eigenvalues, inverse_vectors)

        correlation_layer = EigenProductLayer(name='test_correlation_reconstruct')
        eps = tf.cast(1e-6, cleaned_eigenvalues.dtype)
        cleaned_inverse = tf.math.reciprocal(tf.maximum(cleaned_eigenvalues, eps))
        correlation_vectors = rescaler([eigenvectors, cleaned_inverse])
        correlation = correlation_layer(cleaned_inverse, correlation_vectors)
        diag_corr = tf.linalg.diag_part(correlation)
        assert float(tf.reduce_max(tf.abs(diag_corr - 1.0)).numpy()) < 1e-6

        raw_std = tf.random.uniform((batch_size, n_assets, 1), 0.4, 2.0)
        std_normalizer = CustomNormalizationLayer(
            mode='inverse', axis=-2, inverse_power=2.0, name='test_std_norm'
        )
        inverse_std = std_normalizer(raw_std)

        scale_outer = CovarianceLayer(normalize=False, name='test_outer')
        inverse_vol = scale_outer(inverse_std)

        precision = inverse_correlation * inverse_vol
        covariance = tf.linalg.inv(precision)
        diag = tf.linalg.diag_part(covariance)
        mean_diag = tf.reduce_mean(diag)
        assert float(tf.math.abs(mean_diag - 1.0).numpy()) < 1e-4

    def test_compact_layer_input_transformed_output(self):
        """Layer can emit transformed inputs when requested."""
        layer = RIEnetLayer(output_type=['input_transformed'],
                                   normalize_transformed_variance=False,
                                   name='test_input_transformed')
        batch, n_assets, n_days = 2, 3, 5
        inputs = tf.random.normal((batch, n_assets, n_days))
        outputs = layer(inputs)
        assert outputs.shape == (batch, n_assets, n_days)

    def test_compact_layer_covariance_unit_diag_mean(self):
        """Default configuration keeps the covariance diagonal centred on one."""
        layer = RIEnetLayer(output_type=['covariance'],
                                   name='test_covariance_unit')
        batch, n_assets, n_days = 1, 4, 6
        inputs = tf.random.normal((batch, n_assets, n_days))
        covariance = layer(inputs)
        diag = tf.linalg.diag_part(covariance)
        mean_diag = tf.reduce_mean(diag)
        assert float(tf.math.abs(mean_diag - 1.0).numpy()) < 1e-4

    def test_compact_layer_skip_variance_normalization(self):
        """Disabling variance normalization leaves the layer without the normalizer."""
        layer = RIEnetLayer(output_type='precision',
                                   normalize_transformed_variance=False,
                                   name='test_no_std_norm')
        assert layer.std_normalization is None
    
    def test_normalized_sum(self):
        """Test NormalizedSum layer."""
        layer = NormalizedSum(axis_1=-1, axis_2=-2, name='test_norm_sum')
        
        batch_size, n_assets = 8, 6
        matrix = tf.random.normal((batch_size, n_assets, n_assets))
        
        weights = layer(matrix)
        
        # Should sum to 1 along the specified axis
        weights_sum = tf.reduce_sum(weights, axis=-2, keepdims=True)
        np.testing.assert_allclose(weights_sum.numpy(), 1.0, rtol=1e-5)
    
    def test_lag_transform_layer(self):
        """Test LagTransformLayer."""
        layer = LagTransformLayer(warm_start=True, name='test_lag_transform')
        
        batch_size, n_stocks, n_days = 4, 5, 30
        returns = tf.random.normal((batch_size, n_stocks, n_days), stddev=0.02)
        
        transformed = layer(returns)
        
        # Should preserve input shape
        assert transformed.shape == returns.shape
        
        # Should be different from input (transformation applied)
        assert not tf.reduce_all(tf.abs(transformed - returns) < 1e-6)


class TestIntegration:
    """Integration tests for the complete package."""
    
    def test_full_model_creation(self):
        """Test creating a complete Keras model with RIEnetLayer."""
        n_stocks = 8
        
        inputs = tf.keras.Input(shape=(n_stocks, None))
        weights = RIEnetLayer(output_type='weights')(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=weights)
        
        # Test model compilation
        model.compile(optimizer='adam', loss='mse')
        
        # Test forward pass
        batch_size, n_days = 16, 60
        test_input = tf.random.normal((batch_size, n_stocks, n_days))
        output = model(test_input)
        
        assert output.shape == (batch_size, n_stocks, 1)
    
    def test_training_compatibility(self):
        """Test that the layer works with standard Keras training."""
        n_stocks = 6
        
        # Build model
        model = tf.keras.Sequential([
            RIEnetLayer(output_type='weights')
        ])
        
        # Compile with custom loss
        def custom_loss(y_true, y_pred):
            # Simple squared difference loss for testing
            return tf.reduce_mean(tf.square(y_true - y_pred))
        
        model.compile(optimizer='adam', loss=custom_loss)
        
        # Generate dummy data
        batch_size, n_days = 32, 50
        X = tf.random.normal((batch_size, n_stocks, n_days))
        y = tf.random.normal((batch_size, n_stocks, 1))  # Dummy targets
        
        # Test that training runs without errors
        history = model.fit(X, y, epochs=1, verbose=0)
        
        assert len(history.history['loss']) == 1
        assert history.history['loss'][0] is not None


class TestMixedPrecision:
    """Tests ensuring the model works under mixed precision policies."""

    def test_layer_forward_mixed_float16(self):
        original_policy = tf.keras.mixed_precision.global_policy()
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        try:
            layer = RIEnetLayer(output_type=['weights', 'precision'],
                                       name='test_mixed_precision')
            batch_size, n_stocks, n_days = 2, 3, 12
            inputs = tf.random.normal((batch_size, n_stocks, n_days))

            outputs = layer(inputs)

            assert isinstance(outputs, dict)
            policy_dtype = tf.dtypes.as_dtype(
                tf.keras.mixed_precision.global_policy().compute_dtype
            )
            assert outputs['weights'].dtype == policy_dtype
            assert outputs['precision'].dtype == policy_dtype
        finally:
            tf.keras.mixed_precision.set_global_policy(original_policy)


# Test fixtures and utilities
@pytest.fixture
def sample_returns():
    """Generate sample returns data for testing."""
    batch_size, n_stocks, n_days = 16, 10, 60
    returns = tf.random.normal((batch_size, n_stocks, n_days), stddev=0.02)
    return returns


@pytest.fixture
def sample_covariance():
    """Generate sample covariance matrices for testing."""
    batch_size, n_stocks = 16, 10
    A = tf.random.normal((batch_size, n_stocks, n_stocks))
    covariance = tf.matmul(A, A, transpose_b=True)
    return covariance


if __name__ == "__main__":
    pytest.main([__file__])

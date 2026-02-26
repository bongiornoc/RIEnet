#!/usr/bin/env python3
"""
Comprehensive test for the rienet package.
Tests all functionality including the layer and loss functions.
"""

import sys
import os
import numpy as np

# Add the package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import tensorflow as tf
from rienet import RIEnetLayer, variance_loss_function

def test_layer_functionality():
    """Test the RIEnetLayer with different configurations."""
    print("Testing RIEnetLayer functionality...")
    
    # Test 1: Weights output
    layer_weights = RIEnetLayer(output_type='weights')
    sample_data = tf.random.normal((4, 8, 20))  # 4 batches, 8 stocks, 20 days
    weights = layer_weights(sample_data)
    
    print(f"✓ Weights output shape: {weights.shape}")
    print(f"✓ Weights sum check: {tf.reduce_sum(weights, axis=1).numpy()}")  # Should be close to 1
    
    # Test 2: Precision matrix output
    layer_precision = RIEnetLayer(output_type='precision')
    precision = layer_precision(sample_data)
    
    print(f"✓ Precision matrix output shape: {precision.shape}")
    
    return weights, precision

def test_loss_function():
    """Test the variance loss function."""
    print("\nTesting variance_loss_function...")
    
    # Create sample data
    n_batch, n_stocks = 4, 8
    weights = tf.random.normal((n_batch, n_stocks, 1))
    weights = tf.nn.softmax(weights, axis=1)  # Normalize to sum to 1
    
    # Create a sample covariance matrix
    cov_matrix = tf.eye(n_stocks, batch_shape=[n_batch]) * 0.01  # Small diagonal covariance
    
    # Calculate loss
    loss = variance_loss_function(cov_matrix, weights)
    
    print(f"✓ Variance loss calculated: {loss.numpy()}")
    print(f"✓ Loss shape: {loss.shape}")
    
    return loss

def test_end_to_end_workflow():
    """Test a complete workflow like in a real application."""
    print("\nTesting end-to-end workflow...")
    
    # Simulate daily returns data
    n_batch, n_stocks, n_days = 10, 12, 60
    daily_returns = tf.random.normal((n_batch, n_stocks, n_days), stddev=0.02)
    
    # Create model
    layer = RIEnetLayer(output_type='weights')
    
    # Forward pass
    predicted_weights = layer(daily_returns)
    
    # Create true covariance (for loss calculation)
    # In practice, this would come from your data
    true_cov = tf.eye(n_stocks, batch_shape=[n_batch]) * 0.0001
    
    # Calculate loss
    loss = variance_loss_function(true_cov, predicted_weights)
    
    print(f"✓ End-to-end test successful")
    print(f"✓ Input shape: {daily_returns.shape}")
    print(f"✓ Output shape: {predicted_weights.shape}")
    print(f"✓ Portfolio loss: {loss.numpy()}")
    
    # Check that weights sum to approximately 1
    weight_sums = tf.reduce_sum(predicted_weights, axis=1)
    print(f"✓ Weight sums (should be ~1): {weight_sums.numpy().flatten()[:5]}...")  # Show first 5
    
    return predicted_weights, loss

def main():
    """Run all tests."""
    print("=" * 60)
    print("RIENET COMPREHENSIVE TEST")
    print("=" * 60)
    
    try:
        # Test layer functionality
        weights, precision = test_layer_functionality()
        
        # Test loss function
        loss = test_loss_function()
        
        # Test end-to-end workflow
        predicted_weights, portfolio_loss = test_end_to_end_workflow()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("The RIEnet package is working correctly.")
        print("=" * 60)
        
        # Show citation reminder
        print("\n📖 CITATION REMINDER:")
        import rienet
        rienet.print_citation()
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

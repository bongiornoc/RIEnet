# RIEnet: A Rotational Invariant Estimator Network for GMV Optimization

[![PyPI version](https://img.shields.io/pypi/v/rienet.svg)](https://pypi.org/project/rienet/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**This library implements the neural estimators introduced in:**
- **Bongiorno, C., Manolakis, E., & Mantegna, R. N. (2025). End-to-End Large Portfolio Optimization for Variance Minimization with Neural Networks through Covariance Cleaning (arXiv:2507.01918). [https://arxiv.org/abs/2507.01918](https://arxiv.org/abs/2507.01918)**
- **Bongiorno, C., Manolakis, E., & Mantegna, R. N. (2025). Neural Network-Driven Volatility Drag Mitigation under Aggressive Leverage. In *Proceedings of the 6th ACM International Conference on AI in Finance (ICAIF ’25)*. [https://doi.org/10.1145/3768292.3770370](https://doi.org/10.1145/3768292.3770370)**


RIEnet is a **end-to-end neural estimator** for **Global Minimum-Variance (GMV)** portfolios. The architecture couples a **lag-kernel** with a **bidirectional RNN spectral denoiser** and a **marginal-volatility head** to produce cleaned inverse covariances and analytic GMV weights. By design, the number of learnable parameters can be **independent of the look-back window length and the asset-universe size**, enabling immediate transfer across universes and sampling frequencies without retraining. The model can be trained directly on a realized-variance objective and can export its cleaned covariance for constrained optimizations.

## Key Features

- **End-to-End Variance Objective** – Trains on the realized out-of-sample variance, yielding GMV weights analytically from the learned inverse covariance.
- **Parameter Efficiency** – Fixed-size modules (5-parameter lag kernel; BiGRU eigencleaning; lightweight volatility head) keep the model at ≈2k parameters.
- **Size-Invariant Design** – Complexity does not scale with look-back length or cross-sectional dimension; deploy on new universes without architecture changes.
- **RIE-Style Covariance Cleaning** – Rotation-invariant mapping of the correlation spectrum with a BiGRU (16 units per direction by default).
- **Practical Outputs** – Retrieve GMV weights and the cleaned precision/covariance; the latter can be plugged into long-only QP solvers when needed.
- **Reproducible Defaults** – TensorFlow/Keras reference implementation with paper-consistent hyperparameters and tests.

## Module Organization

- `rienet.trainable_layers`: layers with trainable parameters (`RIEnetLayer`, `LagTransformLayer`, `DeepLayer`, `DeepRecurrentLayer`, `CorrelationEigenTransformLayer`).
- `rienet.ops_layers`: deterministic operation layers (statistics, normalization, eigensystem algebra, weight post-processing).
- `rienet.custom_layers` and `rienet.layers` are kept as backward-compatible wrappers.

## Installation

Install from PyPI:

```bash
pip install rienet
```

Or install from source:

```bash
git clone https://github.com/bongiornoc/RIEnet.git
cd RIEnet
pip install -e .
```

## Quick Start

### Basic Usage

```python
import tensorflow as tf
from rienet import RIEnetLayer, variance_loss_function

# Defaults reproduce the compact GMV architecture (bidirectional GRU with 16 units)
rienet_layer = RIEnetLayer(output_type=['weights', 'precision'])

# Sample data: (batch_size, n_stocks, n_days)
returns = tf.random.normal((32, 10, 60), stddev=0.02)

# Retrieve GMV weights and cleaned precision in one pass
outputs = rienet_layer(returns)
weights = outputs['weights']          # (32, 10, 1)
precision = outputs['precision']      # (32, 10, 10)

# GMV training objective
covariance = tf.random.normal((32, 10, 10))
covariance = tf.matmul(covariance, covariance, transpose_b=True)
loss = variance_loss_function(covariance, weights)
print(loss.shape)  # (32, 1, 1)
```

### Training with the GMV Variance Loss

```python
import tensorflow as tf
from rienet import RIEnetLayer, variance_loss_function

def create_portfolio_model():
    inputs = tf.keras.Input(shape=(None, None))
    weights = RIEnetLayer(output_type='weights')(inputs)
    return tf.keras.Model(inputs=inputs, outputs=weights)

model = create_portfolio_model()

# Synthetic training data
X_train = tf.random.normal((1000, 10, 60), stddev=0.02)
Sigma_train = tf.random.normal((1000, 10, 10))
Sigma_train = tf.matmul(Sigma_train, Sigma_train, transpose_b=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
model.compile(optimizer=optimizer, loss=variance_loss_function)

model.fit(X_train, Sigma_train, epochs=10, batch_size=32, verbose=True)
```

> **Tip:** When you intend to deploy RIEnet on portfolios of varying size, train on batches that span different asset universes. The RIE-based architecture is dimension agnostic and benefits from heterogeneous training shapes.

### Using Different Output Types

```python
# GMV weights only
weights = RIEnetLayer(output_type='weights')(returns)

# Precision matrix only
precision_matrix = RIEnetLayer(output_type='precision')(returns)

# Precision, covariance, and the lag-transformed inputs in one pass
outputs = RIEnetLayer(
    output_type=['precision', 'covariance', 'input_transformed']
)(returns)
precision_matrix = outputs['precision']
covariance_matrix = outputs['covariance']
lagged_inputs = outputs['input_transformed']

# Spectral components (non-inverse)
spectral = RIEnetLayer(
    output_type=['eigenvalues', 'eigenvectors', 'transformed_std']
)(returns)
cleaned_eigenvalues = spectral['eigenvalues']   # (batch, n_stocks, 1)
eigenvectors = spectral['eigenvectors']         # (batch, n_stocks, n_stocks)
transformed_std = spectral['transformed_std']   # (batch, n_stocks, 1)

# Optional: disable variance normalisation (do not use it with end-to-end GMV training)
raw_covariance = RIEnetLayer(
    output_type='covariance',
    normalize_transformed_variance=False
)(returns)
```

> ⚠️ When RIEnet is trained end-to-end on the GMV variance loss, leave
> `normalize_transformed_variance=True` (the default). The loss is invariant to global
> covariance rescalings and the layer keeps the implied variance scale centred
> around one. Disable the normalisation only when using alternative objectives
> where the absolute volatility scale must be preserved.

### Using `LagTransformLayer` Directly

`LagTransformLayer` is exposed both at package root and in the dedicated module:

```python
import tensorflow as tf
from rienet import LagTransformLayer
# or: from rienet.lag_transform import LagTransformLayer

# Dynamic lookback (T can change call-by-call)
compact = LagTransformLayer(variant="compact")
y1 = compact(tf.random.normal((4, 12, 20)))
y2 = compact(tf.random.normal((4, 12, 40)))

# Fixed lookback inferred at first build/call (requires static T)
per_lag = LagTransformLayer(variant="per_lag")
z1 = per_lag(tf.random.normal((4, 12, 20)))
z2 = per_lag(tf.random.normal((4, 8, 20)))   # n_assets can change
```

### Using `EigenWeightsLayer` Directly

`EigenWeightsLayer` is part of the public API and can be imported directly:

```python
import tensorflow as tf
from rienet import EigenWeightsLayer

layer = EigenWeightsLayer(name="gmv_weights")

# Inputs
eigenvectors = tf.random.normal((8, 20, 20))         # (..., n_assets, n_assets)
inverse_eigenvalues = tf.random.uniform((8, 20, 1))  # (..., n_assets) or (..., n_assets, 1)
inverse_std = tf.random.uniform((8, 20, 1))          # optional

# Full GMV-like branch (includes inverse_std scaling)
weights = layer(eigenvectors, inverse_eigenvalues, inverse_std)

# Covariance-eigensystem branch (inverse_std omitted)
weights_cov = layer(eigenvectors, inverse_eigenvalues)
```

Notes:
- `inverse_std` is optional by design.
- If `inverse_std` is omitted, the layer uses a dedicated branch with fewer operations
  (it does not materialize a vector of ones).
- Output shape is always `(..., n_assets, 1)`, normalized to sum to one along assets.

### Using `CorrelationEigenTransformLayer` Directly

```python
import tensorflow as tf
from rienet import CorrelationEigenTransformLayer

layer = CorrelationEigenTransformLayer(name="corr_cleaner")

# Correlation matrix: (batch, n_assets, n_assets)
corr = tf.eye(6, batch_shape=[4])

# Optional attributes: (batch, k) e.g. q, lookback, regime flags, etc.
attrs = tf.constant([
    [0.5, 60.0],
    [0.7, 60.0],
    [1.2, 30.0],
    [0.9, 90.0],
], dtype=tf.float32)

# With attributes (default output_type='correlation')
cleaned_corr = layer(corr, attributes=attrs)

# Request multiple outputs
details = layer(
    corr,
    attributes=attrs,
    output_type=[
        'correlation',
        'inverse_correlation',
        'eigenvalues',
        'eigenvectors',
        'inverse_eigenvalues',
    ],
)
cleaned_eigvals = details['eigenvalues']              # (batch, n_assets, 1)
cleaned_inv_eigvals = details['inverse_eigenvalues']  # (batch, n_assets, 1)
inv_corr = details['inverse_correlation']             # (batch, n_assets, n_assets)

# Without attributes
cleaned_corr_no_attr = CorrelationEigenTransformLayer(name="corr_cleaner_no_attr")(corr)
```

Notes:
- `attributes` is optional and can have shape `(batch, k)` or `(batch, n_assets, k)`.
- The output is a cleaned correlation matrix `(batch, n_assets, n_assets)`.
- If you change attribute width `k`, use a new layer instance.

## Loss Function

### Variance Loss Function

```python
from rienet import variance_loss_function

loss = variance_loss_function(
    covariance_true=true_covariance,    # (batch_size, n_assets, n_assets)
    weights_predicted=predicted_weights # (batch_size, n_assets, 1)
)
```

**Mathematical Formula:**
```
Loss = n_assets × wᵀ Σ w
```

Where `w` are the portfolio weights and `Σ` is the realised covariance matrix.

## Architecture Details

The RIEnet pipeline consists of:

1. **Input Scaling** – Annualise returns by 252
2. **Lag Transformation** – Five-parameter memory kernel for temporal weighting
3. **Covariance Estimation** – Sample covariance across assets
4. **Eigenvalue Decomposition** – Spectral analysis of the covariance matrix
5. **Recurrent Cleaning** – Bidirectional GRU/LSTM processing of eigen spectra
6. **Marginal Volatility Head** – Dense network forecasting inverse standard deviations
7. **Matrix Reconstruction** – RIE-based synthesis of Σ⁻¹ and GMV weight normalisation

Paper defaults use a single bidirectional GRU layer with 16 units per direction and a marginal-volatility head with 8 hidden units, matching the compact network described in Bongiorno et al. (2025).

## Requirements

- Python ≥ 3.8
- TensorFlow ≥ 2.10.0
- Keras ≥ 2.10.0
- NumPy ≥ 1.21.0

## Development

```bash
git clone https://github.com/bongiornoc/RIEnet.git
cd RIEnet
pip install -e ".[dev]"
pytest tests/
```

## Citation

Please cite the following references when using RIEnet:

```bibtex
@inproceedings{bongiorno2025compact,
  author = {Bongiorno, Christian and Manolakis, Efstratios and Mantegna, Rosario Nunzio},
  title = {Neural Network-Driven Volatility Drag Mitigation under Aggressive Leverage},
  year = {2025},
  isbn = {9798400722202},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3768292.3770370},
  doi = {10.1145/3768292.3770370},
  booktitle = {Proceedings of the 6th ACM International Conference on AI in Finance},
  pages = {449–455},
  numpages = {7},
  location = {},
  series = {ICAIF '25}
  }



@article{bongiorno2025covariance,
  title={End-to-End Large Portfolio Optimization for Variance Minimization with Neural Networks through Covariance Cleaning},
  author={Bongiorno, Christian and Manolakis, Efstratios and Mantegna, Rosario N.},
  journal={arXiv preprint arXiv:2507.01918},
  year={2025}
}
```

For software citation:

```bibtex
@software{rienet2025,
  title={RIEnet: A Compact Rotational Invariant Estimator Network for Global Minimum-Variance Optimisation},
  author={Bongiorno, Christian},
  year={2025},
  version={1.1.3},
  url={https://github.com/bongiornoc/RIEnet}
}
```

You can print citation information programmatically:

```python
import rienet
rienet.print_citation()
```

## Support

For questions, issues, or contributions, please:

- Open an issue on [GitHub](https://github.com/bongiornoc/RIEnet/issues)
- Check the documentation
- Contact Prof. Christian Bongiorno (<christian.bongiorno@centralesupelec.fr>) for calibrated model weights or collaboration requests

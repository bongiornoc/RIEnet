"""
Trainable layers for RIEnet.

This module groups all layers that include trainable parameters, including the
main end-to-end `RIEnetLayer` and reusable learnable subcomponents.
"""

import math

import tensorflow as tf
from keras import backend as K
from keras import layers, initializers
from typing import List, Literal, Optional, Sequence, Tuple, Union

from .dtype_utils import ensure_float32, restore_dtype, epsilon_for_dtype
from .ops_layers import (
    NormalizationModeType,
    StandardDeviationLayer,
    CovarianceLayer,
    SpectralDecompositionLayer,
    DimensionAwareLayer,
    CustomNormalizationLayer,
    EigenvectorRescalingLayer,
    EigenProductLayer,
    EigenWeightsLayer,
)

LagTransformVariant = Literal["compact", "per_lag"]
RecurrentCellType = Literal["GRU", "LSTM"]
RecurrentDirectionType = Literal["bidirectional", "forward", "backward"]
CorrelationTransformOutput = Literal[
    "correlation",
    "inverse_correlation",
    "eigenvalues",
    "eigenvectors",
    "inverse_eigenvalues",
]
CorrelationTransformOutputType = Union[
    CorrelationTransformOutput,
    Literal["all"],
    Sequence[Union[CorrelationTransformOutput, Literal["all"]]],
]

OutputComponent = Literal[
    "weights",
    "precision",
    "covariance",
    "correlation",
    "input_transformed",
    "eigenvalues",
    "eigenvectors",
    "transformed_std",
]
OutputToken = Union[OutputComponent, Literal["all"]]
OutputType = Union[OutputToken, Sequence[OutputToken]]
RecurrentCell = RecurrentCellType
RecurrentDirection = RecurrentDirectionType
DimensionalFeature = Literal["n_stocks", "n_days", "q", "rsqrt_n_days"]

@tf.keras.utils.register_keras_serializable(package='rienet')
class DeepLayer(layers.Layer):
    """
    Multi-layer dense network with configurable activation and dropout.
    
    This layer implements a sequence of dense layers with specified activations,
    dropout, and flexible configuration for the final layer.
    
    Parameters
    ----------
    hidden_layer_sizes : list of int
        Sizes of hidden layers including output layer
    last_activation : str, default "linear"
        Activation for the final layer
    activation : str, default "leaky_relu"
        Activation for hidden layers
    other_biases : bool, default True
        Whether to use bias in hidden layers
    last_bias : bool, default True
        Whether to use bias in final layer
    dropout_rate : float, default 0.0
        Dropout rate for hidden layers
    kernel_initializer : str, default "glorot_uniform"
        Weight initialization method
    name : str, optional
        Layer name
    """
    
    def __init__(self, hidden_layer_sizes: List[int], last_activation: str = "linear",
                 activation: str = "leaky_relu", other_biases: bool = True, 
                 last_bias: bool = True, dropout_rate: float = 0., 
                 kernel_initializer: str = "glorot_uniform", name: Optional[str] = None, **kwargs):
        if name is None:
            raise ValueError("DeepLayer must have a name.")
        super().__init__(name=name, **kwargs)
        
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.last_activation = last_activation
        self.other_biases = other_biases
        self.last_bias = last_bias
        self.dropout_rate = dropout_rate
        self.kernel_initializer = kernel_initializer

        # Build hidden layers
        self.hidden_layers = []
        self.dropouts = []
        
        for i, size in enumerate(self.hidden_layer_sizes[:-1]):
            layer_name = f"{self.name}_hidden_{i}"
            dropout_name = f"{self.name}_dropout_{i}"
            
            dense = layers.Dense(
                size,
                activation=self.activation,
                use_bias=self.other_biases,
                kernel_initializer=self.kernel_initializer,
                name=layer_name
            )
            dropout = layers.Dropout(self.dropout_rate, name=dropout_name)
            
            self.hidden_layers.append(dense)
            self.dropouts.append(dropout)

        # Final layer
        self.final_dense = layers.Dense(
            self.hidden_layer_sizes[-1],
            use_bias=self.last_bias,
            activation=self.last_activation,
            kernel_initializer=self.kernel_initializer,
            name=f"{self.name}_output"
        )

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build the dense and dropout sublayers."""
        input_shape = tf.TensorShape(input_shape)
        current_shape = input_shape

        for dense, dropout in zip(self.hidden_layers, self.dropouts):
            dense.build(current_shape)
            current_shape = dense.compute_output_shape(current_shape)
            dropout.build(current_shape)

        self.final_dense.build(current_shape)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass through the network.
        
        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor of shape ``(..., features)``.
        training : bool, optional
            Keras training flag controlling dropout behavior.
            
        Returns
        -------
        tf.Tensor
            Output tensor of shape ``(..., hidden_layer_sizes[-1])``.
        """
        x = inputs
        for dense, dropout in zip(self.hidden_layers, self.dropouts):
            x = dense(x)
            x = dropout(x, training=training)
        return self.final_dense(x)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'activation': self.activation,
            'last_activation': self.last_activation,
            'other_biases': self.other_biases,
            'last_bias': self.last_bias,
            'dropout_rate': self.dropout_rate,
            'kernel_initializer': self.kernel_initializer
        })
        return config

    def compute_output_shape(self, input_shape: Tuple) -> Tuple:
        """Compute output shape."""
        output_shape = list(input_shape)
        output_shape[-1] = self.hidden_layer_sizes[-1]
        return tuple(output_shape)


@tf.keras.utils.register_keras_serializable(package='rienet')
class DeepRecurrentLayer(layers.Layer):
    """
    Deep recurrent layer with configurable RNN cells and post-processing.
    
    This layer implements a stack of recurrent layers (LSTM/GRU) with optional
    bidirectional processing, followed by dense layers for final transformation.
    
    Parameters
    ----------
    recurrent_layer_sizes : list of int
        Sizes of recurrent layers
    final_activation : str, default "softplus"
        Activation for final dense layer
    final_hidden_layer_sizes : list of int, default []
        Hidden sizes of the post-RNN MLP head before the final 1-unit output.
        For example ``[32, 8]`` builds a two-layer MLP with 32 then 8 units.
        ``[]`` means no additional hidden MLP layer.
    final_hidden_activation : str, default "leaky_relu"
        Activation for final hidden layers
    direction : Literal['bidirectional', 'forward', 'backward'], default 'bidirectional'
        RNN direction strategy.
    dropout : float, default 0.0
        Dropout rate for RNN layers
    recurrent_dropout : float, default 0.0
        Recurrent dropout rate
    recurrent_model : Literal['LSTM', 'GRU'], default 'LSTM'
        Type of recurrent cell.
    normalize : Literal['inverse', 'sum'] or None, optional
        Post-projection normalization mode applied along the sequence axis.
    normalize_inverse_power : float, default 1.0
        Exponent ``p`` used only when ``normalize='inverse'``. The layer rescales
        outputs so that the sequence-wise average of ``x^{-p}`` equals 1.
        Larger ``p`` gives more penalty to small values.
    name : str, optional
        Layer name
    """
    
    def __init__(self, recurrent_layer_sizes: List[int], final_activation: str = "softplus",
                 final_hidden_layer_sizes: List[int] = [], final_hidden_activation: str = "leaky_relu",
                 direction: RecurrentDirectionType = 'bidirectional', dropout: float = 0.,
                 recurrent_dropout: float = 0., recurrent_model: RecurrentCellType = 'LSTM',
                 normalize: Optional[NormalizationModeType] = None,
                 normalize_inverse_power: float = 1.0,
                 name: Optional[str] = None, **kwargs):
        """
        Initialize a stacked recurrent block followed by an MLP head.

        Parameters
        ----------
        recurrent_layer_sizes : list of int
            Number of units for each recurrent layer.
        final_activation : str, default "softplus"
            Activation used by the final 1-unit output layer.
        final_hidden_layer_sizes : list of int, default []
            Hidden sizes for the post-RNN MLP head.
            Example: ``[32, 8]`` means two dense hidden layers with 32 and 8 units.
            ``[]`` means the output layer is connected directly to the RNN output.
        final_hidden_activation : str, default "leaky_relu"
            Activation used by hidden layers in the MLP head.
        direction : Literal['bidirectional', 'forward', 'backward'], default 'bidirectional'
            RNN direction mode:
            - ``'bidirectional'``: process sequence in both directions and concatenate.
            - ``'forward'``: process left-to-right only.
            - ``'backward'``: process right-to-left only.
        dropout : float, default 0.0
            Input dropout used in recurrent layers and MLP hidden layers.
        recurrent_dropout : float, default 0.0
            Recurrent-state dropout used inside each recurrent cell.
        recurrent_model : Literal['LSTM', 'GRU'], default 'LSTM'
            Recurrent cell type:
            - ``'LSTM'``: Long Short-Term Memory cell.
            - ``'GRU'``: Gated Recurrent Unit cell.
        normalize : Literal['inverse', 'sum'] or None, optional
            Optional output normalization applied along the time axis:
            - ``None``: do not normalize.
            - ``'sum'``: normalize by sum.
            - ``'inverse'``: inverse-power normalization.
        normalize_inverse_power : float, default 1.0
            Inverse normalization exponent, used only when ``normalize='inverse'``.
        name : str, optional
            Keras layer name.
        **kwargs : dict
            Additional arguments passed to ``tf.keras.layers.Layer``.
        """
        if name is None:
            raise ValueError("DeepRecurrentLayer must have a name.")
        super().__init__(name=name, **kwargs)

        self.recurrent_layer_sizes = recurrent_layer_sizes
        self.final_activation = final_activation
        self.final_hidden_layer_sizes = final_hidden_layer_sizes
        self.final_hidden_activation = final_hidden_activation
        self.direction = direction
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.recurrent_model = recurrent_model
        
        if normalize not in [None, 'inverse', "sum"]:
            raise ValueError("normalize must be None, 'inverse', or 'sum'.")
        self.normalize = normalize
        if self.normalize is not None and normalize_inverse_power <= 0:
            raise ValueError("normalize_inverse_power must be positive when using inverse normalization.")
        self.normalize_inverse_power = float(normalize_inverse_power)

        # Build recurrent layers
        RNN = getattr(layers, recurrent_model)
        self.recurrent_layers = []
        
        for i, units in enumerate(self.recurrent_layer_sizes):
            layer_name = f"{self.name}_rnn_{i}"
            cell_name = f"{layer_name}_cell"
            
            if self.direction == 'bidirectional':
                cell = RNN(
                    units=units, 
                    dropout=self.dropout, 
                    recurrent_dropout=self.recurrent_dropout,
                    return_sequences=True, 
                    name=cell_name
                )
                rnn_layer = layers.Bidirectional(cell, name=layer_name)
            elif self.direction == 'forward':
                rnn_layer = RNN(
                    units=units, 
                    dropout=self.dropout, 
                    recurrent_dropout=self.recurrent_dropout,
                    return_sequences=True, 
                    name=layer_name
                )
            elif self.direction == 'backward':
                rnn_layer = RNN(
                    units=units, 
                    dropout=self.dropout, 
                    recurrent_dropout=self.recurrent_dropout,
                    return_sequences=True, 
                    go_backwards=True, 
                    name=layer_name
                )
            else:
                raise ValueError("direction must be 'bidirectional', 'forward', or 'backward'.")
                
            self.recurrent_layers.append(rnn_layer)

        # Final dense layers
        self.final_deep_dense = DeepLayer(
            final_hidden_layer_sizes + [1], 
            activation=final_hidden_activation,
            last_activation=final_activation,
            dropout_rate=dropout,
            name=f"{self.name}_finaldeep"
        )       

        if self.normalize is not None:
            inverse_power = self.normalize_inverse_power if self.normalize == 'inverse' else 1.0
            self._normalizer = CustomNormalizationLayer(
                mode=self.normalize,
                axis=-2,
                inverse_power=inverse_power,
                name=f"{self.name}_norm"
            )
        else:
            self._normalizer = None

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build the recurrent stack and final dense projection."""
        input_shape = tf.TensorShape(input_shape)
        current_shape = input_shape

        for rnn_layer in self.recurrent_layers:
            rnn_layer.build(current_shape)
            current_shape = rnn_layer.compute_output_shape(current_shape)

        self.final_deep_dense.build(current_shape)
        final_shape = self.final_deep_dense.compute_output_shape(current_shape)

        if self._normalizer is not None:
            self._normalizer.build(final_shape)

        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward pass through the recurrent stack and projection head.
        
        Parameters
        ----------
        inputs : tf.Tensor
            Input sequence tensor with shape ``(batch, timesteps, features)``.
        training : bool, optional
            Keras training flag controlling recurrent/dense dropout.
            
        Returns
        -------
        tf.Tensor
            Tensor with shape ``(batch, timesteps)`` after squeezing the
            final singleton feature axis.
        """
        x = inputs
        for layer in self.recurrent_layers:
            x = layer(x, training=training)
            
        outputs = self.final_deep_dense(x, training=training)
        
        if self._normalizer is not None:
            outputs = self._normalizer(outputs)
            
        return tf.squeeze(outputs, axis=-1)
    
    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'recurrent_layer_sizes': self.recurrent_layer_sizes,
            'final_activation': self.final_activation,
            'final_hidden_layer_sizes': self.final_hidden_layer_sizes,
            'final_hidden_activation': self.final_hidden_activation,
            'direction': self.direction,
            'dropout': self.dropout,
            'recurrent_dropout': self.recurrent_dropout,
            'recurrent_model': self.recurrent_model,
            'normalize': self.normalize,
            'normalize_inverse_power': self.normalize_inverse_power
        })
        return config



@tf.keras.utils.register_keras_serializable(package='rienet')
class CorrelationEigenTransformLayer(layers.Layer):
    """
    Transform a correlation matrix by cleaning its eigenvalues.

    The layer performs:
    1. Eigen-decomposition of the input correlation matrix.
    2. Optional enrichment of each eigenvalue with per-batch attributes ``(b, k)``.
    3. Recurrent transformation of enriched eigenvalue features in inverse-eigenvalue
       space.
    4. Reconstruction of a cleaned correlation matrix with diagonal rescaling.

    Parameters
    ----------
    recurrent_layer_sizes : tuple of int, default (16,)
        Hidden sizes of recurrent layers used to transform eigenvalues.
    recurrent_cell : Literal['GRU', 'LSTM'], default 'GRU'
        Recurrent cell family.
    recurrent_direction : Literal['bidirectional', 'forward', 'backward'], default 'bidirectional'
        Direction for recurrent processing.
    final_hidden_layer_sizes : tuple of int, default ()
        Hidden sizes of the post-recurrent MLP head used to transform eigenvalue
        features before the final scalar output.
    final_hidden_activation : str, default 'leaky_relu'
        Activation for optional dense hidden layers.
    output_type : CorrelationTransformOutputType, default 'correlation'
        Requested output component(s). Allowed values are:
        'correlation', 'inverse_correlation', 'eigenvalues', 'eigenvectors',
        'inverse_eigenvalues', or 'all'. Multiple outputs are returned as a dictionary.
    epsilon : float, optional
        Numerical epsilon used before reciprocal.
    name : str, optional
        Layer name.

    Call Arguments
    --------------
    correlation_matrix : tf.Tensor
        Correlation matrix tensor of shape ``(batch, n_assets, n_assets)``.
    attributes : tf.Tensor, optional
        Optional attribute tensor with shape ``(batch, k)`` or ``(batch, n_assets, k)``
        concatenated to each eigenvalue feature vector.
    training : bool, optional
        Keras training flag.

    Returns
    -------
    tf.Tensor or dict
        Output selected by ``output_type``.
    """

    _ALLOWED_OUTPUTS = (
        "correlation",
        "inverse_correlation",
        "eigenvalues",
        "eigenvectors",
        "inverse_eigenvalues",
    )

    def __init__(self,
                 recurrent_layer_sizes: Tuple[int, ...] = (16,),
                 recurrent_cell: RecurrentCellType = 'GRU',
                 recurrent_direction: RecurrentDirectionType = 'bidirectional',
                 final_hidden_layer_sizes: Tuple[int, ...] = (),
                 final_hidden_activation: str = 'leaky_relu',
                 output_type: CorrelationTransformOutputType = 'correlation',
                 epsilon: Optional[float] = None,
                 name: Optional[str] = None,
                 **kwargs):
        """
        Initialize the correlation-eigenvalue cleaning layer.

        Parameters
        ----------
        recurrent_layer_sizes : tuple of int, default (16,)
            Units for each recurrent layer in the eigenvalue cleaning block.
        recurrent_cell : Literal['GRU', 'LSTM'], default 'GRU'
            Recurrent cell type used in the cleaning block:
            - ``'GRU'``: Gated Recurrent Unit.
            - ``'LSTM'``: Long Short-Term Memory.
        recurrent_direction : Literal['bidirectional', 'forward', 'backward'], default 'bidirectional'
            Sequence direction mode:
            - ``'bidirectional'``: forward + backward processing.
            - ``'forward'``: forward only.
            - ``'backward'``: backward only.
        final_hidden_layer_sizes : tuple of int, default ()
            Hidden sizes of the post-recurrent MLP head.
            Example: ``(32, 8)`` adds two hidden dense layers (32 then 8 units).
            ``()`` means no extra hidden MLP layer.
        final_hidden_activation : str, default 'leaky_relu'
            Activation for hidden layers in the post-recurrent MLP head.
        output_type : CorrelationTransformOutputType, default 'correlation'
            Requested output component(s). Allowed values:
            - ``'correlation'``: cleaned correlation matrix.
            - ``'inverse_correlation'``: cleaned inverse correlation matrix.
            - ``'eigenvalues'``: cleaned (non-inverse) eigenvalues.
            - ``'eigenvectors'``: eigenvectors from decomposition.
            - ``'inverse_eigenvalues'``: transformed inverse eigenvalues.
            - ``'all'``: all components above.
            A sequence can be passed to request multiple components.
        epsilon : float, optional
            Numerical epsilon used in safe reciprocal operations.
        name : str, optional
            Keras layer name. If omitted, Keras auto-generates one.
        **kwargs : dict
            Additional arguments passed to ``tf.keras.layers.Layer``.
        """
        super().__init__(name=name, **kwargs)

        recurrent_layer_sizes = list(recurrent_layer_sizes)
        final_hidden_layer_sizes = list(final_hidden_layer_sizes)
        if not recurrent_layer_sizes:
            raise ValueError("recurrent_layer_sizes must contain at least one positive integer.")
        for units in recurrent_layer_sizes:
            if units <= 0:
                raise ValueError("recurrent_layer_sizes must contain positive integers.")
        for units in final_hidden_layer_sizes:
            if units <= 0:
                raise ValueError("final_hidden_layer_sizes must contain positive integers.")

        normalized_cell = recurrent_cell.strip().upper()
        if normalized_cell not in {'GRU', 'LSTM'}:
            raise ValueError("recurrent_cell must be either 'GRU' or 'LSTM'.")
        normalized_direction = recurrent_direction.strip().lower()
        if normalized_direction not in {'bidirectional', 'forward', 'backward'}:
            raise ValueError(
                "recurrent_direction must be 'bidirectional', 'forward', or 'backward'."
            )

        self._output_config = output_type if isinstance(output_type, str) else list(output_type)
        self.output_components = tuple(self._resolve_output_components(output_type))
        self.output_type = (
            self.output_components[0]
            if len(self.output_components) == 1
            else tuple(self.output_components)
        )

        self.recurrent_layer_sizes = recurrent_layer_sizes
        self.recurrent_cell = normalized_cell
        self.recurrent_direction = normalized_direction
        self.final_hidden_layer_sizes = final_hidden_layer_sizes
        self.final_hidden_activation = final_hidden_activation
        self.epsilon = float(epsilon if epsilon is not None else K.epsilon())
        self._feature_width: Optional[int] = None

        self.spectral_decomp = SpectralDecompositionLayer(
            name=f"{self.name}_spectral"
        )
        self.eigenvalue_transform = DeepRecurrentLayer(
            recurrent_layer_sizes=self.recurrent_layer_sizes,
            recurrent_model=self.recurrent_cell,
            direction=self.recurrent_direction,
            dropout=0.0,
            recurrent_dropout=0.0,
            final_hidden_layer_sizes=self.final_hidden_layer_sizes,
            final_hidden_activation=self.final_hidden_activation,
            final_activation='softplus',
            normalize='inverse',
            # Fixed to 1.0 to enforce true reciprocal behavior.
            normalize_inverse_power=1.0,
            name=f"{self.name}_eigenvalue_rnn",
        )
        self.eigenvector_rescaler = EigenvectorRescalingLayer(
            epsilon=self.epsilon,
            name=f"{self.name}_eigenvector_rescaler",
        )
        self.correlation_product = EigenProductLayer(
            name=f"{self.name}_correlation",
        )

    def _resolve_output_components(self, output_type: CorrelationTransformOutputType) -> List[str]:
        if isinstance(output_type, str):
            if output_type == 'all':
                return list(self._ALLOWED_OUTPUTS)
            if output_type not in self._ALLOWED_OUTPUTS:
                raise ValueError(
                    "output_type must be one of "
                    "{'correlation', 'inverse_correlation', 'eigenvalues', 'eigenvectors', "
                    "'inverse_eigenvalues', 'all'}."
                )
            return [output_type]

        output_list = list(output_type)
        if not output_list:
            raise ValueError("output_type cannot be an empty sequence.")

        expanded: List[str] = []
        for entry in output_list:
            if entry == 'all':
                expanded.extend(self._ALLOWED_OUTPUTS)
                continue
            if entry not in self._ALLOWED_OUTPUTS:
                raise ValueError(
                    "All requested outputs must be in "
                    "{'correlation', 'inverse_correlation', 'eigenvalues', 'eigenvectors', "
                    "'inverse_eigenvalues', 'all'}."
                )
            expanded.append(entry)

        deduped: List[str] = []
        seen = set()
        for entry in expanded:
            if entry not in seen:
                deduped.append(entry)
                seen.add(entry)
        return deduped

    def build(self, input_shape) -> None:
        super().build(input_shape)

    def call(self,
             correlation_matrix: tf.Tensor,
             attributes: Optional[tf.Tensor] = None,
             output_type: Optional[CorrelationTransformOutputType] = None,
             training: Optional[bool] = None) -> tf.Tensor:
        """
        Clean a correlation matrix in eigen-space.

        Parameters
        ----------
        correlation_matrix : tf.Tensor
            Correlation tensor with shape ``(batch, n_assets, n_assets)``.
        attributes : tf.Tensor, optional
            Optional auxiliary features concatenated to each eigenvalue channel.
            Accepted shapes:
            - ``(batch, k)``: batch-level attributes broadcast to all assets.
            - ``(batch, n_assets, k)``: per-asset attributes.
        output_type : CorrelationTransformOutputType, optional
            Optional per-call override of requested output components.
            If omitted, the instance-level ``output_type`` from ``__init__`` is used.
        training : bool, optional
            Keras training flag passed to the recurrent transform.

        Returns
        -------
        tf.Tensor or dict[str, tf.Tensor]
            If one component is requested, returns that tensor directly.
            If multiple components are requested, returns a dictionary keyed by:
            ``'correlation'``, ``'inverse_correlation'``, ``'eigenvalues'``,
            ``'eigenvectors'``, ``'inverse_eigenvalues'``.
        """
        components = (
            self._resolve_output_components(output_type)
            if output_type is not None
            else list(self.output_components)
        )
        need_correlation = 'correlation' in components
        need_inverse_correlation = 'inverse_correlation' in components
        need_eigenvalues = 'eigenvalues' in components
        need_eigenvectors = 'eigenvectors' in components
        need_inverse_eigenvalues = (
            'inverse_eigenvalues' in components
            or need_eigenvalues
            or need_correlation
            or need_inverse_correlation
        )

        corr_rank = correlation_matrix.shape.rank
        if corr_rank is not None and corr_rank != 3:
            raise ValueError(
                "correlation_matrix must have shape (batch, n_assets, n_assets)."
            )

        correlation_matrix = tf.convert_to_tensor(correlation_matrix)
        dtype = correlation_matrix.dtype

        if attributes is not None:
            attr_rank = attributes.shape.rank
            if attr_rank is not None and attr_rank not in {2, 3}:
                raise ValueError("attributes must have shape (batch, k) or (batch, n_assets, k).")
            attributes = tf.cast(tf.convert_to_tensor(attributes), dtype)

            corr_batch = correlation_matrix.shape[0]
            attr_batch = attributes.shape[0]
            if corr_batch is not None and attr_batch is not None and corr_batch != attr_batch:
                raise ValueError(
                    "Batch mismatch between correlation_matrix and attributes: "
                    f"correlation_matrix batch={corr_batch}, attributes batch={attr_batch}."
                )

            batch_assert = tf.debugging.assert_equal(
                tf.shape(correlation_matrix)[0],
                tf.shape(attributes)[0],
                message="Batch mismatch between correlation_matrix and attributes.",
            )
            with tf.control_dependencies([batch_assert]):
                attributes = tf.identity(attributes)

            if attr_rank == 3:
                corr_assets = correlation_matrix.shape[-1]
                attr_assets = attributes.shape[1]
                if corr_assets is not None and attr_assets is not None and corr_assets != attr_assets:
                    raise ValueError(
                        "Asset-dimension mismatch between correlation_matrix and attributes: "
                        f"correlation_matrix assets={corr_assets}, attributes assets={attr_assets}."
                    )
                assets_assert = tf.debugging.assert_equal(
                    tf.shape(correlation_matrix)[-1],
                    tf.shape(attributes)[1],
                    message="Asset-dimension mismatch between correlation_matrix and attributes.",
                )
                with tf.control_dependencies([assets_assert]):
                    attributes = tf.identity(attributes)

        eigenvalues, eigenvectors = self.spectral_decomp(correlation_matrix)
        results = {}

        if need_eigenvectors:
            results['eigenvectors'] = eigenvectors

        transformed_inverse_eigenvalues = None
        transformed_eigenvalues = None

        if need_inverse_eigenvalues:
            eigenvalue_features = eigenvalues
            if attributes is not None:
                if attributes.shape.rank == 2:
                    n_assets = tf.shape(eigenvalues)[1]
                    attr_width = tf.shape(attributes)[1]
                    attributes_expanded = tf.expand_dims(attributes, axis=1)
                    attributes_tiled = tf.broadcast_to(
                        attributes_expanded,
                        tf.stack([tf.shape(attributes)[0], n_assets, attr_width]),
                    )
                else:
                    attributes_tiled = attributes
                eigenvalue_features = tf.concat([eigenvalues, attributes_tiled], axis=-1)

            feature_width = eigenvalue_features.shape[-1]
            if self._feature_width is None and feature_width is not None:
                self._feature_width = int(feature_width)
            if (
                self._feature_width is not None
                and feature_width is not None
                and int(feature_width) != self._feature_width
            ):
                raise ValueError(
                    "Inconsistent eigenvalue feature width across calls: "
                    f"expected={self._feature_width}, got={int(feature_width)}. "
                    "Use a separate layer instance for a different attribute width."
                )

            transformed_inverse_eigenvalues = self.eigenvalue_transform(
                eigenvalue_features,
                training=training,
            )
            if 'inverse_eigenvalues' in components:
                results['inverse_eigenvalues'] = tf.expand_dims(
                    transformed_inverse_eigenvalues,
                    axis=-1,
                )

        if need_eigenvalues or need_correlation:
            inverse_eigs_work, inverse_eigs_dtype = ensure_float32(transformed_inverse_eigenvalues)
            eps = epsilon_for_dtype(inverse_eigs_work.dtype, self.epsilon)
            transformed_eigenvalues_work = tf.math.reciprocal(
                tf.maximum(inverse_eigs_work, eps)
            )
            transformed_eigenvalues = restore_dtype(transformed_eigenvalues_work, inverse_eigs_dtype)
            if need_eigenvalues:
                results['eigenvalues'] = tf.expand_dims(transformed_eigenvalues, axis=-1)

        if need_correlation:
            direct_eigenvectors = self.eigenvector_rescaler(
                [eigenvectors, transformed_eigenvalues]
            )
            results['correlation'] = self.correlation_product(
                transformed_eigenvalues,
                direct_eigenvectors,
            )

        if need_inverse_correlation:
            inverse_eigenvectors = self.eigenvector_rescaler(
                [eigenvectors, transformed_inverse_eigenvalues]
            )
            results['inverse_correlation'] = self.correlation_product(
                transformed_inverse_eigenvalues,
                inverse_eigenvectors,
            )

        if len(components) == 1:
            return results[components[0]]
        return results

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        is_multi_input = (
            isinstance(input_shape, (list, tuple))
            and bool(input_shape)
            and isinstance(input_shape[0], (list, tuple, tf.TensorShape))
        )
        corr_source = input_shape[0] if is_multi_input else input_shape
        corr_shape = tf.TensorShape(corr_source).as_list()

        batch_size = corr_shape[0]
        n_assets = corr_shape[-1]

        def shape_for(component: str):
            if component in {'correlation', 'inverse_correlation', 'eigenvectors'}:
                return (batch_size, n_assets, n_assets)
            return (batch_size, n_assets, 1)

        if len(self.output_components) == 1:
            return shape_for(self.output_components[0])
        return {component: shape_for(component) for component in self.output_components}

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'recurrent_layer_sizes': list(self.recurrent_layer_sizes),
            'recurrent_cell': self.recurrent_cell,
            'recurrent_direction': self.recurrent_direction,
            'final_hidden_layer_sizes': list(self.final_hidden_layer_sizes),
            'final_hidden_activation': self.final_hidden_activation,
            'output_type': self._output_config,
            'epsilon': self.epsilon,
        })
        return config

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package='rienet')
class LagTransformLayer(layers.Layer):
    """
    Layer that applies a lag transformation to input financial time series.
    
    This layer applies a non-linear transformation to financial returns that
    accounts for temporal dependencies and lag effects. The transformation
    uses learnable parameters to adaptively weight different time lags.
    
    Parameters
    ----------
    warm_start : bool, default True
        Whether to initialize trainable parameters near smooth deterministic
        profiles (recommended for stable optimization).
    name : str, optional
        Layer name
    eps : float, optional
        Base epsilon used in positivity constraints and safe divisions.
        If omitted, uses ``keras.backend.epsilon()``.
    variant : Literal["compact", "per_lag"], default "compact"
        Parameterization variant.
        - "compact": five-scalar parameterization with dynamic lookback support.
        - "per_lag": per-lag vectors with fixed lookback inferred at first build.
          This mode requires a static time dimension (input_shape[-1] cannot be None).

    Notes
    -----
    The transformation applied in both variants is:
    ``(alpha / (beta + eps)) * tanh(beta * R)``.
    """
    _ALLOWED_VARIANTS = {"compact", "per_lag"}

    def __init__(self,
                 warm_start: bool = True,
                 name: Optional[str] = None,
                 eps: Optional[float] = None,
                 variant: LagTransformVariant = "compact",
                 **kwargs):
        """
        Initialize the lag-transform layer.

        Parameters
        ----------
        warm_start : bool, default True
            If True, initialize parameters near smooth deterministic profiles.
            If False, use noisier random initializations.
        name : str, optional
            Keras layer name.
        eps : float, optional
            Base epsilon used in positivity constraints and safe divisions.
            If omitted, uses ``keras.backend.epsilon()``.
        variant : Literal['compact', 'per_lag'], default 'compact'
            Lag parameterization mode:
            - ``'compact'``: five global trainable scalars (dynamic lookback support).
            - ``'per_lag'``: one trainable alpha/beta pair per lag
              (requires static ``input_shape[-1]`` at first build).
        **kwargs : dict
            Additional arguments passed to ``tf.keras.layers.Layer``.
        """
        super().__init__(name=name, **kwargs)

        variant = str(variant)
        if variant not in self._ALLOWED_VARIANTS:
            raise ValueError(
                "variant must be one of {'compact', 'per_lag'}."
            )

        self.variant = variant
        self._eps_base = float(eps if eps is not None else K.epsilon())
        self.warm_start = bool(warm_start)
        self._lookback_days: Optional[int] = None

        # Target parameter values for compact parameterization.
        self._target = dict(c0=2.8, c1=0.20, c2=0.85, c3=0.50, c4=0.05)
        self._raw_alpha = None
        self._raw_beta = None

    def _inv_softplus(self, y: float) -> float:
        """Inverse softplus function for parameter initialization."""
        y = float(y)
        if y <= 0.0:
            y = max(y, 1e-12)
        return math.log(math.expm1(y))

    def _add_param(self, name: str, target: float) -> tf.Variable:
        """Add a learnable parameter with appropriate initialization."""
        mean_raw = self._inv_softplus(target - self._eps_base)

        if self.warm_start:
            init = initializers.Constant(mean_raw)
        else:
            # Add ±5% noise in raw space with a minimum scale for stability
            noise_scale = max(0.05 * abs(mean_raw), float(K.epsilon()))
            init = initializers.RandomNormal(mean_raw, noise_scale)

        return self.add_weight(
            shape=(),
            name=f"raw_{name}",
            initializer=init,
            trainable=True,
        )

    def _build_per_lag_profiles(self) -> Tuple[List[float], List[float]]:
        """
        Create smooth warm-start profiles for per-lag alpha/beta.

        alpha: monotone decaying
        beta: monotone increasing and saturating
        """
        alpha: List[float] = []
        beta: List[float] = []
        lookback_days = int(self._lookback_days)
        for lag_idx in range(1, lookback_days + 1):
            alpha.append(2.5 / (1.0 + 0.08 * (lag_idx - 1)))
            beta.append(0.25 + 0.70 * (1.0 - math.exp(-lag_idx / 8.0)))
        return alpha, beta

    def _init_vector_param(self, targets: List[float]):
        """Create initializer for vector parameters."""
        safe_targets = [max(t - self._eps_base, 1e-12) for t in targets]
        raw_targets = [self._inv_softplus(t) for t in safe_targets]
        if self.warm_start:
            return initializers.Constant(raw_targets)
        return initializers.RandomNormal(mean=0.0, stddev=0.1)

    def _runtime_lookback_error(self, got: int) -> ValueError:
        return ValueError(
            "LagTransformLayer variant='per_lag' requires fixed lookback length: "
            f"expected={self._lookback_days}, got={got}."
        )

    def build(self, input_shape: Tuple) -> None:
        """Build layer parameters."""
        input_shape = tf.TensorShape(input_shape)
        lookback_from_shape = input_shape[-1]

        if self.variant == "compact":
            self._raw_c0 = self._add_param("c0", self._target["c0"])
            self._raw_c1 = self._add_param("c1", self._target["c1"])
            self._raw_c2 = self._add_param("c2", self._target["c2"])
            self._raw_c3 = self._add_param("c3", self._target["c3"])
            self._raw_c4 = self._add_param("c4", self._target["c4"])
        else:
            if lookback_from_shape is None:
                raise ValueError(
                    "LagTransformLayer variant='per_lag' requires a static time dimension "
                    "(input_shape[-1] must be known, got None)."
                )

            if self._lookback_days is None:
                self._lookback_days = int(lookback_from_shape)

            if int(lookback_from_shape) != self._lookback_days:
                raise ValueError(
                    "LagTransformLayer variant='per_lag' got incompatible time dimension at build: "
                    f"expected={self._lookback_days}, got={int(lookback_from_shape)}."
                )

            alpha_profile, beta_profile = self._build_per_lag_profiles()
            self._raw_alpha = self.add_weight(
                shape=(self._lookback_days,),
                name="raw_alpha",
                initializer=self._init_vector_param(alpha_profile),
                trainable=True,
            )
            self._raw_beta = self.add_weight(
                shape=(self._lookback_days,),
                name="raw_beta",
                initializer=self._init_vector_param(beta_profile),
                trainable=True,
            )
        super().build(input_shape)

    def _pos(self, x: tf.Tensor) -> tf.Tensor:
        """Apply softplus + epsilon to ensure positive values."""
        return tf.nn.softplus(x) + epsilon_for_dtype(x.dtype, self._eps_base)

    def _assert_per_lag_runtime_shape(self, R: tf.Tensor) -> tf.Tensor:
        """Ensure runtime lookback matches configured fixed lookback."""
        static_t = R.shape[-1]
        if static_t is not None:
            got = int(static_t)
            if got != self._lookback_days:
                raise self._runtime_lookback_error(got)
            return R

        if tf.executing_eagerly():
            got = int(tf.shape(R)[-1].numpy())
            if got != self._lookback_days:
                raise self._runtime_lookback_error(got)
            return R

        expected = tf.cast(self._lookback_days, tf.int32)
        assertion = tf.debugging.assert_equal(
            tf.shape(R)[-1],
            expected,
            message=(
                "LagTransformLayer variant='per_lag' requires fixed lookback length "
                f"expected={self._lookback_days}."
            ),
        )
        with tf.control_dependencies([assertion]):
            return tf.identity(R)

    def call(self, R: tf.Tensor) -> tf.Tensor:
        """
        Apply lag transformation to returns.
        
        Parameters
        ----------
        R : tf.Tensor
            Returns tensor of shape ``(..., T)`` where the last axis is the
            lookback/time dimension.
            - In ``variant='compact'``, ``T`` can vary across calls.
            - In ``variant='per_lag'``, ``T`` must be fixed and equal to the
              first built static time size.
            
        Returns
        -------
        tf.Tensor
            Transformed returns with the same shape and dtype as ``R``.
        """
        R = tf.convert_to_tensor(R)
        R_work, original_dtype = ensure_float32(R)
        dtype = R_work.dtype
        eps_tensor = epsilon_for_dtype(dtype, self._eps_base)

        if self.variant == "per_lag":
            R_work = self._assert_per_lag_runtime_shape(R_work)
            T = tf.shape(R_work)[-1]

            alpha = tf.cast(self._pos(self._raw_alpha), dtype)
            beta = tf.cast(self._pos(self._raw_beta), dtype)

            ndims = tf.rank(R)
            pad_ones = tf.ones(ndims - 1, dtype=tf.int32)
            shape_T = tf.concat([pad_ones, [T]], 0)

            alpha_div_beta = tf.reshape(alpha / (beta + eps_tensor), shape_T)
            beta = tf.reshape(beta, shape_T)
            transformed = alpha_div_beta * tf.tanh(beta * R_work)
            return restore_dtype(transformed, original_dtype)

        T = tf.shape(R_work)[-1]  # Time dimension length

        # Create time indices: t = [T, T-1, ..., 1]
        t = tf.cast(tf.range(1, T + 1), dtype)  # [1, 2, ..., T]
        t = tf.reverse(t, axis=[0])  # [T, T-1, ..., 1]

        # Get positive parameters via softplus
        c0 = tf.cast(self._pos(self._raw_c0), dtype)
        c1 = tf.cast(self._pos(self._raw_c1), dtype)
        c2 = tf.cast(self._pos(self._raw_c2), dtype)
        c3 = tf.cast(self._pos(self._raw_c3), dtype)
        c4 = tf.cast(self._pos(self._raw_c4), dtype)

        # Compute lag transformation parameters
        alpha = c0 * tf.pow(t, -c1)  # (T,)
        beta = c2 - c3 * tf.exp(-c4 * t)  # (T,)

        # Reshape for broadcasting
        ndims = tf.rank(R_work)
        pad_ones = tf.ones(ndims - 1, dtype=tf.int32)
        shape_T = tf.concat([pad_ones, [T]], 0)

        alpha_div_beta = tf.reshape(alpha / (beta + eps_tensor), shape_T)
        beta = tf.reshape(beta, shape_T)

        # Apply transformation: alpha/beta * tanh(beta * R)
        transformed = alpha_div_beta * tf.tanh(beta * R_work)
        return restore_dtype(transformed, original_dtype)
    
    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'eps': self._eps_base,
            'warm_start': self.warm_start,
            'variant': self.variant,
        })
        return config

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)



@tf.keras.utils.register_keras_serializable(package='rienet')
class RIEnetLayer(layers.Layer):
    """
    Rotational Invariant Estimator (RIE) Network layer for GMV portfolios.

    This layer implements the compact network described in Bongiorno et al. (2025) for
    global minimum-variance (GMV) portfolio construction. The architecture couples
    Rotational Invariant Estimators of the covariance matrix with recurrent neural
    networks in order to clean the eigen-spectrum and learn marginal volatilities in a
    parameter-efficient way.

    The layer automatically scales daily returns by 252 (annualisation factor) and
    applies the following stages:

    - Lag transformation with a five-parameter non-linearity
    - Sample covariance estimation and eigenvalue decomposition
    - Recurrent cleaning of eigenvalues (GRU or LSTM; configurable direction)
    - Dense transformation of marginal volatilities
    - Recombination into Σ⁻¹ followed by GMV weight normalisation

    Parameters
    ----------
    output_type : OutputType, default 'weights'
        Component(s) to return. Each entry must belong to
        {'weights', 'precision', 'covariance', 'correlation', 'input_transformed',
        'eigenvalues', 'eigenvectors', 'transformed_std'} or the special string
        'all'. When multiple components are requested a dictionary mapping component name
        to tensor is returned.
    recurrent_layer_sizes : Sequence[int], optional
        Hidden sizes of the recurrent cleaning block. Defaults to [16] matching the
        compact GMV network in the paper. If a sequence with multiple integers is
        provided (e.g. [32, 16]) the recurrent cleaning head will apply multiple hidden
        layers in the given order: first a layer with 32 units, then one with 16 units.
    std_hidden_layer_sizes : Sequence[int], optional
        Hidden sizes of the dense network acting on marginal volatilities. Defaults to
        [8] matching the paper. Sequences are interpreted similarly (e.g. [64, 8] ->
        two dense hidden layers with 64 then 8 units).
    recurrent_cell : Literal['GRU', 'LSTM'], default 'GRU'
        Recurrent cell family used inside the eigenvalue cleaning block. Accepted
        values are 'GRU' and 'LSTM'.
    recurrent_direction : Literal['bidirectional', 'forward', 'backward'], default 'bidirectional'
        Direction used by the recurrent cleaning block.
    dimensional_features : Sequence[Literal['n_stocks', 'n_days', 'q', 'rsqrt_n_days']], optional
        Dimension-aware features concatenated to eigenvalues before recurrent cleaning.
        Allowed values are:
        - 'n_stocks': number of assets in the cross-section.
        - 'n_days': lookback length.
        - 'q': ratio n_days / n_stocks.
        - 'rsqrt_n_days': reciprocal sqrt of lookback length.
        Defaults to ('n_stocks', 'n_days', 'q').
    lag_transform_variant : Literal['compact', 'per_lag'], default 'compact'
        Lag transformation parameterization.
        - 'compact': five-parameter dynamic lookback transform.
        - 'per_lag': one pair of trainable parameters per lag, with fixed lookback
          inferred from the first built input (requires static input_shape[-1]).
    normalize_transformed_variance : bool, default True
        Whether to normalize the transformed inverse volatilities so that the implied
        covariance diagonal (variance) is centred on 1. Disable only when the network is
        not trained end-to-end on the GMV objective.
    name : str, optional
        Name of the Keras layer instance.
    **kwargs : dict
        Additional keyword arguments propagated to ``tf.keras.layers.Layer``.

    Input Shape
    -----------
    (batch_size, n_stocks, n_days)
        Daily return tensors for each batch element, stock and time step.

    Output Shape
    ------------
    Depends on ``output_type``:
        - 'weights' -> (batch_size, n_stocks, 1)
        - 'precision', 'covariance', or 'correlation' -> (batch_size, n_stocks, n_stocks)
        - 'input_transformed' -> (batch_size, n_stocks, n_days)
        - 'eigenvalues' -> (batch_size, n_stocks, 1) (cleaned, non-inverse)
        - 'eigenvectors' -> (batch_size, n_stocks, n_stocks)
        - 'transformed_std' -> (batch_size, n_stocks, 1) (non-inverse)
        - Multiple components -> ``dict`` mapping component name to the shapes above

    Notes
    -----
    Defaults replicate the compact RIE network optimised for GMV portfolios in the
    reference paper: a single bidirectional GRU layer with 16 units per direction and a
    dense marginal-volatility head with 8 hidden units. Inputs are annualised by 252 and
    the resulting Σ⁻¹ is symmetrised for numerical stability. Training on batches that
    span different asset universes is recommended when deploying on variable-dimension
    portfolios.
    
    Examples
    --------
    >>> import tensorflow as tf
    >>> from rienet import RIEnetLayer
    >>> 
    >>> # Create layer for portfolio weights
    >>> layer = RIEnetLayer(output_type='weights')
    >>> 
    >>> # Generate sample daily returns data  
    >>> returns = tf.random.normal((32, 10, 60))  # 32 samples, 10 stocks, 60 days
    >>> 
    >>> # Get portfolio weights
    >>> weights = layer(returns)
    >>> print(f"Portfolio weights shape: {weights.shape}")  # (32, 10, 1)
    
    References
    ----------
    Bongiorno, C., Manolakis, E., & Mantegna, R. N. (2025). Neural Network-Driven Volatility Drag Mitigation under Aggressive Leverage. In Proceedings of the 6th ACM International Conference on AI in Finance (ICAIF ’25), 449–455. https://doi.org/10.1145/3768292.3770370
    Bongiorno, C., Manolakis, E., & Mantegna, R. N. (2025). End-to-End Large Portfolio Optimization for Variance Minimization with Neural Networks through Covariance Cleaning (arXiv:2507.01918).
    """
    
    def __init__(self,
                 output_type: OutputType = 'weights',
                 recurrent_layer_sizes: Sequence[int] = (16,),
                 std_hidden_layer_sizes: Sequence[int] = (8,),
                 recurrent_cell: RecurrentCell = 'GRU',
                 recurrent_direction: RecurrentDirection = 'bidirectional',
                 dimensional_features: Sequence[DimensionalFeature] = ('n_stocks', 'n_days', 'q'),
                 normalize_transformed_variance: bool = True,
                 lag_transform_variant: LagTransformVariant = 'compact',
                 name: Optional[str] = None,
                 **kwargs):
        """
        Initialize the RIEnet layer.
        
        Parameters
        ----------
        output_type : OutputType, default 'weights'
            Requested output component(s). Allowed values:
            - ``'weights'``: GMV portfolio weights.
            - ``'precision'``: cleaned precision matrix.
            - ``'covariance'``: cleaned covariance matrix.
            - ``'correlation'``: cleaned correlation matrix.
            - ``'input_transformed'``: lag-transformed returns.
            - ``'eigenvalues'``: cleaned eigenvalues (non-inverse).
            - ``'eigenvectors'``: eigenvectors from decomposition.
            - ``'transformed_std'``: transformed standard deviations (non-inverse).
            - ``'all'``: all components above.
            A sequence can be passed to request multiple components.
        recurrent_layer_sizes : Sequence[int], optional (default (16,))
            Hidden sizes of the recurrent cleaning block (defaults to [16]).
            If multiple integers are supplied (for example [32, 16]) the recurrent
            block will create multiple hidden layers applied in sequence: first 32 units,
            then 16 units.
        std_hidden_layer_sizes : Sequence[int], optional (default (8,))
            Hidden sizes of the dense marginal-volatility block (defaults to [8]).
            A sequence such as [64, 8] will be interpreted as two dense hidden layers
            with 64 then 8 units respectively.
        recurrent_cell : Literal['GRU', 'LSTM'], default 'GRU'
            Recurrent cell type used in eigenvalue cleaning:
            - ``'GRU'``: Gated Recurrent Unit.
            - ``'LSTM'``: Long Short-Term Memory.
        recurrent_direction : Literal['bidirectional', 'forward', 'backward'], default 'bidirectional'
            Direction mode of the recurrent cleaning block:
            - ``'bidirectional'``: process the sequence in both directions.
            - ``'forward'``: process forward only.
            - ``'backward'``: process backward only.
        dimensional_features : Sequence[Literal['n_stocks', 'n_days', 'q', 'rsqrt_n_days']], optional
            Additional features concatenated before eigenvalue cleaning:
            - ``'n_stocks'``: number of assets in the universe.
            - ``'n_days'``: lookback length.
            - ``'q'``: ratio ``n_days / n_stocks``.
            - ``'rsqrt_n_days'``: ``1 / sqrt(n_days)``.
        normalize_transformed_variance : bool, default True
            If True, rescales transformed inverse volatilities so that the implied
            covariance diagonal is centered around 1.
        lag_transform_variant : Literal['compact', 'per_lag'], default 'compact'
            Lag-transform parameterization:
            - ``'compact'``: five global trainable scalars.
            - ``'per_lag'``: one trainable parameter pair per lag.
        name : str, optional
            Keras layer name.
        **kwargs : dict
            Additional arguments passed to ``tf.keras.layers.Layer``.
        """
        super().__init__(name=name, **kwargs)

        allowed_outputs = (
            'weights',
            'precision',
            'covariance',
            'correlation',
            'input_transformed',
            'eigenvalues',
            'eigenvectors',
            'transformed_std',
        )
        self._output_config = output_type if isinstance(output_type, str) else list(output_type)

        if isinstance(output_type, str):
            if output_type == 'all':
                components = list(allowed_outputs)
            else:
                if output_type not in allowed_outputs:
                    raise ValueError(
                        "output_type must be one of "
                        "'weights', 'precision', 'covariance', 'correlation', "
                        "'input_transformed', 'eigenvalues', 'eigenvectors', "
                        "'transformed_std', or 'all'"
                    )
                components = [output_type]
        else:
            output_list = list(output_type)
            if not output_list:
                raise ValueError("output_type cannot be an empty sequence")
            expanded: List[str] = []
            for entry in output_list:
                if entry == 'all':
                    expanded.extend(allowed_outputs)
                    continue
                if entry not in allowed_outputs:
                    raise ValueError(
                        "All requested outputs must be in "
                        "{'weights', 'precision', 'covariance', 'correlation', "
                        "'input_transformed', 'eigenvalues', 'eigenvectors', "
                        "'transformed_std', 'all'}"
                    )
                expanded.append(entry)
            seen = set()
            components = []
            for entry in expanded:
                if entry not in seen:
                    components.append(entry)
                    seen.add(entry)

        self.output_components = tuple(components)
        self.output_type = components[0] if len(components) == 1 else tuple(components)

        if recurrent_layer_sizes is None:
            raise ValueError("recurrent_layer_sizes cannot be None; pass a non-empty sequence of positive integers.")
        recurrent_layer_sizes = list(recurrent_layer_sizes)
        if not recurrent_layer_sizes:
            raise ValueError("recurrent_layer_sizes must contain at least one positive integer")

        if std_hidden_layer_sizes is None:
            raise ValueError("std_hidden_layer_sizes cannot be None; pass a non-empty sequence of positive integers.")
        std_hidden_layer_sizes = list(std_hidden_layer_sizes)
        if not std_hidden_layer_sizes:
            raise ValueError("std_hidden_layer_sizes must contain at least one positive integer")

        for size in recurrent_layer_sizes:
            if size <= 0:
                raise ValueError("recurrent_layer_sizes must contain positive integers")
        for size in std_hidden_layer_sizes:
            if size <= 0:
                raise ValueError("std_hidden_layer_sizes must contain positive integers")

        normalized_cell = recurrent_cell.strip().upper()
        if normalized_cell not in {'GRU', 'LSTM'}:
            raise ValueError("recurrent_cell must be either 'GRU' or 'LSTM'")
        normalized_direction = recurrent_direction.strip().lower()
        if normalized_direction not in {'bidirectional', 'forward', 'backward'}:
            raise ValueError("recurrent_direction must be 'bidirectional', 'forward', or 'backward'.")

        if dimensional_features is None:
            dimensional_features = ('n_stocks', 'n_days', 'q')
        dimensional_features = list(dimensional_features)
        allowed_features = {'n_stocks', 'n_days', 'q', 'rsqrt_n_days'}
        invalid_features = [feature for feature in dimensional_features if feature not in allowed_features]
        if invalid_features:
            raise ValueError(
                "dimensional_features entries must be in "
                "{'n_stocks', 'n_days', 'q', 'rsqrt_n_days'}; "
                f"got invalid entries: {invalid_features}."
            )

        deduped_features: List[str] = []
        seen_features = set()
        for feature in dimensional_features:
            if feature not in seen_features:
                deduped_features.append(feature)
                seen_features.add(feature)

        # Architecture parameters (paper defaults preserved if args omitted)
        self._std_hidden_layer_sizes = list(std_hidden_layer_sizes)
        self._recurrent_layer_sizes = list(recurrent_layer_sizes)
        self._recurrent_model = normalized_cell
        self._direction = normalized_direction
        self._dimensional_features = deduped_features
        self._annualization_factor = 252.0
        self._normalize_variance = bool(normalize_transformed_variance)
        self._lag_transform_variant = lag_transform_variant
        self.input_spec = layers.InputSpec(ndim=3)
        
        # Initialize component layers
        self._build_layers()

    def _build_layers(self):
        """Build the internal layers of the architecture."""
        # Input transformation and preprocessing
        self.lag_transform = LagTransformLayer(
            variant=self._lag_transform_variant,
            warm_start=True,
            name=f"{self.name}_lag_transform"
        )
        
        self.std_layer = StandardDeviationLayer(
            axis=-1, 
            name=f"{self.name}_std"
        )
        
        self.covariance_layer = CovarianceLayer(
            expand_dims=False,
            normalize=True,
            name=f"{self.name}_covariance"
        )
        
        self.dimension_aware = DimensionAwareLayer(
            features=self._dimensional_features,
            name=f"{self.name}_dimension_aware"
        )
        
        # Correlation transformation in eigen-space
        self.correlation_eigen_transform = CorrelationEigenTransformLayer(
            recurrent_layer_sizes=self._recurrent_layer_sizes,
            recurrent_cell=self._recurrent_model,
            recurrent_direction=self._direction,
            final_hidden_layer_sizes=[],
            output_type='correlation',
            name=f"{self.name}_corr_eigen_transform",
        )
        
        # Standard deviation transformation
        self.std_transform = DeepLayer(
            hidden_layer_sizes=self._std_hidden_layer_sizes + [1],
            last_activation='softplus',
            name=f"{self.name}_std_transform"
        )
        
        if self._normalize_variance:
            self.std_normalization = CustomNormalizationLayer(
                axis=-2,
                mode='inverse',
                inverse_power=2.0,
                name=f"{self.name}_std_norm"
            )
        else:
            self.std_normalization = None
        
        # Matrix reconstruction (see Eq. 13-15)
        self.eigenvector_rescaler = EigenvectorRescalingLayer(
            name=f"{self.name}_eigenvector_rescaler"
        )
        self.eigen_product = EigenProductLayer(
            name=f"{self.name}_eigen_product"
        )

        self.outer_product = CovarianceLayer(
            normalize=False,
            name=f"{self.name}_inverse_scale_outer"
        )

        self.weight_layer = EigenWeightsLayer(
            name=f"{self.name}_weights"
        )

    def build(self, input_shape: Tuple[int, int, int]) -> None:
        """Build sub-layers once input dimensionality is known."""
        input_shape = tf.TensorShape(input_shape)
        if input_shape.rank != 3:
            raise ValueError(
                "RIEnetLayer expects inputs with shape (batch, n_stocks, n_days)."
            )

        batch = input_shape[0]
        n_stocks = input_shape[1]

        covariance_shape = tf.TensorShape([batch, n_stocks, n_stocks])
        attributes_shape = tf.TensorShape([batch, n_stocks, len(self._dimensional_features)])
        std_shape = tf.TensorShape([batch, n_stocks, 1])
        eigenvalues_vector_shape = tf.TensorShape([batch, n_stocks])

        self.lag_transform.build(input_shape)
        self.std_layer.build(input_shape)
        self.covariance_layer.build(input_shape)
        self.dimension_aware.build([input_shape, covariance_shape])
        self.correlation_eigen_transform.build([covariance_shape, attributes_shape])
        self.std_transform.build(std_shape)
        if self.std_normalization is not None:
            self.std_normalization.build(std_shape)
        self.eigenvector_rescaler.build([covariance_shape, eigenvalues_vector_shape])
        self.eigen_product.build([eigenvalues_vector_shape, covariance_shape])
        self.outer_product.build(std_shape)
        self.weight_layer.build([covariance_shape, eigenvalues_vector_shape, std_shape])

        super().build(input_shape)
        
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Execute the full RIEnet pipeline.
        
        Parameters
        ----------
        inputs : tf.Tensor
            Daily returns tensor with shape ``(batch_size, n_stocks, n_days)``.
            Semantics of each axis:
            - ``batch_size``: independent samples/windows.
            - ``n_stocks``: cross-sectional asset dimension.
            - ``n_days``: lookback time dimension.
        training : bool, optional
            Keras training flag forwarded to stochastic sub-layers.
            
        Returns
        -------
        tf.Tensor
            If one output is requested, returns a single tensor.
            If multiple outputs are requested, returns a dictionary.
            Components and shapes:
            - ``weights``: ``(batch, n_stocks, 1)``
            - ``precision``: ``(batch, n_stocks, n_stocks)`` cleaned inverse covariance
            - ``covariance``: ``(batch, n_stocks, n_stocks)`` cleaned covariance
            - ``correlation``: ``(batch, n_stocks, n_stocks)`` cleaned correlation
            - ``eigenvalues``: ``(batch, n_stocks, 1)`` cleaned non-inverse eigenvalues
            - ``eigenvectors``: ``(batch, n_stocks, n_stocks)``
            - ``transformed_std``: ``(batch, n_stocks, 1)`` non-inverse transformed std
            - ``input_transformed``: ``(batch, n_stocks, n_days)``
        """
        need_precision = 'precision' in self.output_components
        need_covariance = 'covariance' in self.output_components
        need_correlation = 'correlation' in self.output_components
        need_weights = 'weights' in self.output_components
        need_eigenvalues = 'eigenvalues' in self.output_components
        need_eigenvectors = 'eigenvectors' in self.output_components
        need_transformed_std = 'transformed_std' in self.output_components

        need_structural_outputs = need_precision or need_covariance or need_correlation or need_weights
        need_spectral_outputs = need_eigenvalues or need_eigenvectors or need_transformed_std
        need_pipeline_outputs = need_structural_outputs or need_spectral_outputs

        # Scale inputs by annualization factor
        scaled_inputs = inputs * self._annualization_factor
        
        # Apply lag transformation
        input_transformed = self.lag_transform(scaled_inputs)

        results = {}
        if 'input_transformed' in self.output_components:
            results['input_transformed'] = input_transformed

        if not need_pipeline_outputs:
            return (
                results[self.output_components[0]]
                if len(self.output_components) == 1
                else results
            )
        
        # Compute standard deviation and mean
        std, mean = self.std_layer(input_transformed)
        
        # Transform standard deviations (normalize only when needed downstream)
        need_inverse_std = need_precision or need_covariance or need_weights or need_transformed_std
        transformed_inverse_std = None
        std_for_structural = None
        transformed_std = None
        if need_inverse_std:
            transformed_inverse_std = self.std_transform(std)
            std_for_structural = transformed_inverse_std
            if self.std_normalization is not None and need_inverse_std:
                std_for_structural = self.std_normalization(transformed_inverse_std)

            if need_transformed_std or need_covariance:
                std_work, std_original_dtype = ensure_float32(std_for_structural)
                std_eps = epsilon_for_dtype(std_work.dtype, K.epsilon())
                transformed_std_work = tf.math.reciprocal(tf.maximum(std_work, std_eps))
                transformed_std = restore_dtype(transformed_std_work, std_original_dtype)

        if need_transformed_std:
            results['transformed_std'] = transformed_std

        need_spectral_branch = need_precision or need_covariance or need_correlation or need_weights or need_eigenvalues or need_eigenvectors
        if not need_spectral_branch:
            return (
                results[self.output_components[0]]
                if len(self.output_components) == 1
                else results
            )

        # Standardize returns
        zscores = (input_transformed - mean) / std
        
        # Compute correlation matrix
        correlation_matrix = self.covariance_layer(zscores)

        attributes = None
        if self._dimensional_features:
            attributes = self.dimension_aware([zscores, correlation_matrix])

        spectral_components: List[str] = []
        if need_eigenvectors or need_precision or need_covariance or need_correlation or need_weights:
            spectral_components.append('eigenvectors')
        if need_precision or need_weights or need_covariance or need_correlation or need_eigenvalues:
            spectral_components.append('inverse_eigenvalues')
        if need_eigenvalues:
            spectral_components.append('eigenvalues')
        if need_covariance or need_correlation:
            spectral_components.append('correlation')

        deduped_components: List[str] = []
        seen_components = set()
        for component in spectral_components:
            if component not in seen_components:
                deduped_components.append(component)
                seen_components.add(component)

        spectral_outputs = self.correlation_eigen_transform(
            correlation_matrix,
            attributes=attributes,
            output_type=deduped_components,
            training=training,
        )
        if isinstance(spectral_outputs, dict):
            spectral_results = spectral_outputs
        else:
            spectral_results = {deduped_components[0]: spectral_outputs}

        eigenvectors = spectral_results.get('eigenvectors')
        transformed_inverse_eigenvalues = spectral_results.get('inverse_eigenvalues')
        if transformed_inverse_eigenvalues is not None:
            transformed_inverse_eigenvalues = tf.squeeze(transformed_inverse_eigenvalues, axis=-1)

        cleaned_correlation = spectral_results.get('correlation')

        if need_eigenvectors:
            results['eigenvectors'] = eigenvectors
        if need_eigenvalues:
            results['eigenvalues'] = spectral_results['eigenvalues']
       
        # Precision-specific reconstruction
        inverse_correlation = None
        if need_precision:
            inverse_eigenvectors = self.eigenvector_rescaler(
                [eigenvectors, transformed_inverse_eigenvalues]
            )
            inverse_correlation = self.eigen_product(
                transformed_inverse_eigenvalues, inverse_eigenvectors
            )
            inverse_volatility_matrix = self.outer_product(std_for_structural)
            precision_matrix = inverse_correlation * inverse_volatility_matrix
            results['precision'] = precision_matrix

        if need_covariance:
            volatility_matrix = self.outer_product(transformed_std)
            covariance = cleaned_correlation * volatility_matrix
            results['covariance'] = covariance

        if need_correlation:
            results['correlation'] = cleaned_correlation

        if need_weights:
            weights = self.weight_layer(
                eigenvectors=eigenvectors,
                inverse_eigenvalues=transformed_inverse_eigenvalues,
                inverse_std=std_for_structural,
            )
            results['weights'] = weights

        if len(self.output_components) == 1:
            return results[self.output_components[0]]

        return results
    
    def get_config(self) -> dict:
        """
        Get layer configuration for serialization.
        
        Returns
        -------
        dict
            Layer configuration dictionary
        """
        config = super().get_config()
        config.update({
            'output_type': self._output_config,
            'recurrent_layer_sizes': list(self._recurrent_layer_sizes),
            'std_hidden_layer_sizes': list(self._std_hidden_layer_sizes),
            'recurrent_cell': self._recurrent_model,
            'recurrent_direction': self._direction,
            'dimensional_features': list(self._dimensional_features),
            'normalize_transformed_variance': self._normalize_variance,
            'lag_transform_variant': self._lag_transform_variant,
        })
        return config
    
    @classmethod
    def from_config(cls, config: dict):
        """
        Create layer from configuration dictionary.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
            
        Returns
        -------
        RIEnetLayer
            Layer instance
        """
        return cls(**config)
        
    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Compute output shape given input shape.
        
        Parameters
        ----------
        input_shape : tuple
            Input shape (batch_size, n_stocks, n_days)
            
        Returns
        -------
        tuple
            Output shape
        """
        input_shape = tf.TensorShape(input_shape).as_list()
        batch_size, n_stocks, n_days = input_shape

        def shape_for(component: str) -> Tuple[int, ...]:
            if component in {'weights', 'eigenvalues', 'transformed_std'}:
                return (batch_size, n_stocks, 1)
            if component == 'input_transformed':
                return (batch_size, n_stocks, n_days)
            if component == 'eigenvectors':
                return (batch_size, n_stocks, n_stocks)
            return (batch_size, n_stocks, n_stocks)

        if len(self.output_components) == 1:
            return shape_for(self.output_components[0])

        return {component: shape_for(component) for component in self.output_components}


__all__ = [
    "LagTransformVariant",
    "RecurrentCellType",
    "RecurrentDirectionType",
    "CorrelationTransformOutput",
    "CorrelationTransformOutputType",
    "OutputComponent",
    "OutputToken",
    "OutputType",
    "RecurrentCell",
    "RecurrentDirection",
    "DimensionalFeature",
    "DeepLayer",
    "DeepRecurrentLayer",
    "CorrelationEigenTransformLayer",
    "LagTransformLayer",
    "RIEnetLayer",
]

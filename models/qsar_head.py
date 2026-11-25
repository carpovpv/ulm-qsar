from dataclasses import dataclass, field
from typing import List

import tensorflow as tf 
from tf.keras import layers 

from config import QsarCnnConfig

class SequenceMaskFromZeros(layers.Layer):
    """Derives a 2D boolean mask `(batch, seq_len)` from `ZeroOutPadding` encoder outputs."""

    def call(self, inputs):
        # True where timestep has non-zero features, False at padded positions
        return tf.reduce_any(tf.not_equal(inputs, 0.0), axis=-1)


class MaskedGlobalMaxPooling1D(layers.Layer):
    """GlobalMaxPooling1D that ignores padded timesteps (`mask == False`)."""

    def call(self, inputs, mask=None):
        # inputs: (batch, seq_len, channels), mask: (batch, seq_len)
        if mask is None:
            return tf.reduce_max(inputs, axis=1)

        mask_expanded = mask[..., tf.newaxis]  # (batch, seq_len, 1)
        # Use dtype.min instead of -inf to avoid float16 overflow or NaN gradients
        neg_val = tf.constant(tf.as_dtype(inputs.dtype).min, dtype=inputs.dtype)
        masked_inputs = tf.where(mask_expanded, inputs, neg_val)

        pooled = tf.reduce_max(masked_inputs, axis=1)
        # Guard against all-padded sequences returning dtype.min instead of 0.0
        has_valid_tokens = tf.reduce_any(mask_expanded, axis=1)
        return tf.where(has_valid_tokens, pooled, tf.zeros_like(pooled))


class HighwayLayer(layers.Layer):
    """Gated Highway layer (Srivastava et al., 2015) matching ulm-qsar."""

    def __init__(self, units: int, bias_init: float = -1.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.bias_init = bias_init
        self.transform_gate = layers.Dense(
            units=units,
            activation="sigmoid",
            bias_initializer=tf.keras.initializers.Constant(bias_init),
            name="highway_transform_gate",
        )
        self.transformed_proj = layers.Dense(
            units=units,
            activation="relu",
            name="highway_transformed_proj",
        )

    def call(self, inputs):
        t_gate = self.transform_gate(inputs)
        c_gate = 1.0 - t_gate
        h_trans = self.transformed_proj(inputs)
        return (t_gate * h_trans) + (c_gate * inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units, "bias_init": self.bias_init})
        return config


def build_qsar_cnn_extractor(
    cfg: QsarCNNConfig = QsarCNNConfig(),
    name: str = "smiles_text_cnn",
) -> tf.keras.Model:
    """Build a standalone, mask-aware Text-CNN + Highway feature extractor.

    Input:
        encoder_seq: Tensor of shape `(batch_size, seq_len, input_dim)` with padded
                     timesteps zeroed out by `ZeroOutPadding`.
    Output:
        fixed_vec: Tensor of shape `(batch_size, cfg.output_dim)` in `float32`.
    """
    if len(cfg.kernel_sizes) != len(cfg.num_filters):
        raise ValueError("`kernel_sizes` and `num_filters` must have the same length.")

    seq_in = layers.Input(shape=(None, cfg.input_dim), name="encoder_sequence")
    valid_mask = SequenceMaskFromZeros(name="extract_valid_mask")(seq_in)

    # 1. Parallel 1D Convolutions (padding='same' preserves 1-to-1 timestep alignment)
    pooled_branches = []
    for k_size, n_filt in zip(cfg.kernel_sizes, cfg.num_filters):
        conv = layers.Conv1D(
            filters=n_filt,
            kernel_size=k_size,
            padding="same",
            kernel_initializer="he_normal",
            activation="relu",
            name=f"conv1d_k{k_size}",
        )(seq_in)
        pooled = MaskedGlobalMaxPooling1D(name=f"masked_maxpool_k{k_size}")(
            conv, mask=valid_mask
        )
        pooled_branches.append(pooled)

    # Concatenate -> (batch_size, 1620)
    concat = layers.Concatenate(axis=-1, name="cnn_concat")(pooled_branches)
    dropped = layers.Dropout(rate=cfg.dropout_rate, name="cnn_dropout")(concat)

    # 2. Dense projection + Gated Highway Network -> (batch_size, 256
    dense = layers.Dense(cfg.output_dim, activation="relu", name="cnn_dense")(dropped)
    highway = HighwayLayer(
        units=cfg.output_dim, bias_init=cfg.highway_bias, name="cnn_highway"
    )(dense)

    # Cast output to float32 for downstream numerical stability under mixed_float16
    fixed_vec = layers.Activation("linear", dtype="float32", name="molecular_embedding")(highway)

    return tf.keras.Model(inputs=seq_in, outputs=fixed_vec, name=name)


from typing import Tuple
import math
import numpy as np

import tensorflow as tf 
from tensorflow.keras import layers

from config import UlmConfig 
from smi2smi.callbacks import NoamSchedule
from tokenizer import BaseTokenizer

class PositionalEncoding(layers.Layer):
    """Sinusoidal positional encoding that preserves Keras `_keras_mask`."""

    def __init__(self, d_model: int, max_len: int = 1024, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.d_model = d_model
        self.max_len = max_len
        self.embed_scale = math.sqrt(float(d_model))

        pos = np.arange(max_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1.0 / np.power(10000.0, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        self.pos_encoding = tf.constant(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def call(self, x):
        seq_len = tf.gather(tf.shape(x), 1)
        scale = tf.cast(self.embed_scale, x.dtype)
        return (x * scale) + tf.cast(self.pos_encoding[:, :seq_len, :], x.dtype)

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "max_len": self.max_len})
        return config


class ZeroOutPadding(layers.Layer):
    """Zeros out padded timesteps in the final encoder representation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is None:
            return inputs
        return inputs * tf.cast(mask[..., tf.newaxis], inputs.dtype)

    def compute_mask(self, inputs, mask=None):
        return mask
   
def masked_sparse_loss(y_true, y_pred):
    """Cross-entropy loss ignoring padding token index 0."""
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    return tf.reduce_sum(loss * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)


def masked_sparse_acc(y_true, y_pred):
    """Token-level accuracy ignoring padding token index 0."""
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    preds = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    eq = tf.cast(tf.equal(tf.cast(y_true, tf.int32), preds), tf.float32)
    return tf.reduce_sum(eq * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)


def build_smi2smi_models(
        cfg: UlmConfig, tokenizer : BaseTokenizer
) -> Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
    """Build full training model, standalone encoder, and fast cached decoder.

    Returns:
        train_model: Inputs `[src_tokens, dec_tokens]`, output `logits` (float32).
        encoder_model: Input `src_tokens`, output `encoder_hidden_states`.
        decoder_model: Inputs `[dec_tokens, encoder_hidden_states]`, output `logits`.
    """
    d_model = cfg.embedding_size
    max_seq_len = tokenizer.padded_length(cfg.max_len)
    vocab_size = tokenizer.vocab_size 

    # Shared Embedding & Positional Encoding (index 0 == pad_token)
    shared_embed = layers.Embedding(
        input_dim=vocab_size,
        output_dim=d_model,
        mask_zero=True,
        name="shared_token_embedding",
    )
    pos_enc = PositionalEncoding(d_model=d_model, max_len=max_seq_len, name="pos_encoding")

    # =========================================================================
    # 1. ENCODER
    # =========================================================================
    src_in = layers.Input(shape=(None,), dtype="uint8", name="src_tokens")
    x = layers.Dropout(cfg.dropout_rate)(pos_enc(shared_embed(src_in)))

    for b in range(cfg.n_blocks):
        attn = layers.MultiHeadAttention(
            num_heads=cfg.n_heads,
            key_dim=cfg.key_size,
            output_shape=(d_model,),
            dropout=cfg.dropout_rate,
            name=f"enc_mha_{b}",
        )(query=x, value=x, key=x)
        x = layers.LayerNormalization(epsilon=1e-6, name=f"enc_norm1_{b}")(layers.Add()([x, attn]))

        ff = layers.Conv1D(cfg.ff_hidden_size, 1, activation="relu", name=f"enc_ff1_{b}")(x)
        ff = layers.Dropout(cfg.dropout_rate)(
            layers.Conv1D(d_model, 1, name=f"enc_ff2_{b}")(ff)
        )
        x = layers.LayerNormalization(epsilon=1e-6, name=f"enc_norm2_{b}")(layers.Add()([x, ff]))

    enc_out = ZeroOutPadding(name="encoder_output")(x)
    encoder_model = tf.keras.Model(src_in, enc_out, name="smi2smi_encoder")

    # =========================================================================
    # 2. DECODER (Defined as reusable layers for both training & cached decoding)
    # =========================================================================
    dec_blocks = []
    for b in range(cfg.n_blocks):
        dec_blocks.append(
            {
                "self_mha": layers.MultiHeadAttention(
                    num_heads=cfg.n_heads,
                    key_dim=cfg.key_size,
                    output_shape=(d_model,),
                    dropout=cfg.dropout_rate,
                    name=f"dec_self_mha_{b}",
                ),
                "norm1": layers.LayerNormalization(epsilon=1e-6, name=f"dec_norm1_{b}"),
                "cross_mha": layers.MultiHeadAttention(
                    num_heads=cfg.n_heads,
                    key_dim=cfg.key_size,
                    output_shape=(d_model,),
                    dropout=cfg.dropout_rate,
                    name=f"dec_cross_mha_{b}",
                ),
                "norm2": layers.LayerNormalization(epsilon=1e-6, name=f"dec_norm2_{b}"),
                "ff1": layers.Conv1D(cfg.ff_hidden_size, 1, activation="relu", name=f"dec_ff1_{b}"),
                "ff2": layers.Conv1D(d_model, 1, name=f"dec_ff2_{b}"),
                "norm3": layers.LayerNormalization(epsilon=1e-6, name=f"dec_norm3_{b}"),
            }
        )

    out_proj = layers.Dense(vocab_size, use_bias=False, dtype="float32", name="logits_head" )

    def run_decoder(dec_tokens, encoder_states):
        y = layers.Dropout(cfg.dropout_rate)(pos_enc(shared_embed(dec_tokens)))
        for blk in dec_blocks:
            self_attn = blk["self_mha"](query=y, value=y, key=y, use_causal_mask=True)
            y = blk["norm1"](layers.Add()([y, self_attn]))

            cross_attn = blk["cross_mha"](query=y, value=encoder_states, key=encoder_states)
            y = blk["norm2"](layers.Add()([y, cross_attn]))

            ff = blk["ff1"](y)
            ff = layers.Dropout(cfg.dropout_rate)(blk["ff2"](ff))
            y = blk["norm3"](layers.Add()([y, ff]))
        return out_proj(y)

    # Full training model
    dec_in = layers.Input(shape=(None,), dtype="uint8", name="dec_tokens")
    train_logits = run_decoder(dec_in, enc_out)
    train_model = tf.keras.Model([src_in, dec_in], train_logits, name="smi2smi_transformer")

    lr_schedule = NoamSchedule(warmup_steps=cfg.warmup_steps, lr_factor=cfg.lr_factor)

    train_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.98, epsilon=1e-9, clipnorm=1.0),
        loss=masked_sparse_loss,
        metrics=[masked_sparse_acc],
    )

    # Cached decoder model (takes precomputed encoder states for fast greedy inference)
    cached_enc_in = layers.Input(shape=(None, d_model), name="cached_enc_states")
    cached_dec_in = layers.Input(shape=(None,), dtype="uint8", name="cached_dec_tokens")
    cached_logits = run_decoder(cached_dec_in, cached_enc_in)
    decoder_model = tf.keras.Model(
        [cached_dec_in, cached_enc_in], cached_logits, name="smi2smi_cached_decoder"
    )

    return train_model, encoder_model, decoder_model


from typing import List, Sequence, Tuple

from tensorflow import tf
from tokenizer import BaseTokenizer 

def greedy_canonicalize_batch(
    encoder_model: tf.keras.Model,
    decoder_model: tf.keras.Model,
    src_smiles: Sequence[str],
    tokenizer: BaseTokenizer,
    max_len: int = 256,
) -> List[str]:
    """Vectorized greedy autoregressive decoding with cached encoder states."""
    batch_size = len(src_smiles)
    x = tf.convert_to_tensor(
        tokenizer.encode_batch(src_smiles, add_bos=True, add_eos=True),
        dtype=tf.uint8,
    )

    # 1. Run encoder ONCE for the entire batch
    enc_states = encoder_model(x, training=False)

    # 2. Keep decoder tokens as a tf.Tensor on GPU to match enc_states
    dec_tokens = tf.fill([batch_size, 1], tf.cast(tokenizer.bos_id, tf.uint8))
    finished = tf.zeros([batch_size], dtype=tf.bool)

    for _ in range(max_len + 2):
        logits = decoder_model([dec_tokens, enc_states], training=False)
        next_ids = tf.argmax(logits[:, -1, :], axis=-1, output_type=tf.int32)
        next_ids = tf.cast(next_ids, tf.uint8)

        # Keep finished sequences padded with pad_id (0)
        next_ids = tf.where(finished, tf.cast(tokenizer.pad_id, tf.uint8), next_ids)
        dec_tokens = tf.concat([dec_tokens, next_ids[:, None]], axis=1)

        finished = tf.logical_or(finished, tf.equal(next_ids, tokenizer.eos_id))
        if tf.reduce_all(finished):
            break

    return tokenizer.decode_batch(dec_tokens.numpy(), stop_at_eos=True)



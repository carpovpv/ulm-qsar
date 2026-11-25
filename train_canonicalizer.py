import os
import json
from dataclasses import asdict

from config import UlmConfig
from tf_setup import init_tensorflow

import tensorflow as tf

cfg = UlmConfig()
init_tensorflow(cfg)

from tokenizer import CharSmilesTokenizer
from pipelines.smi2smi import (
        prepare_canon_datasets, 
        CanonTrainSequence, 
        CanonValSequence
)

from models.smi2smi import build_smi2smi_models
from callbacks import (
    CanonicalizationMonitorCallback,
    CheckpointAveragerCallback,
    GpuThermalGuardCallback
)

def main():

    config_path = os.path.join(cfg.output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    tokenizer = CharSmilesTokenizer(cfg)

    print("Preparing dataset and generating SMILES augmentations...")
    train_pairs, val_canon_smiles = prepare_canon_datasets(cfg, tokenizer)
    print(f"Train pairs: {len(train_pairs)} | Validation unique molecules: {len(val_canon_smiles)}")

    train_seq = CanonTrainSequence(train_pairs, tokenizer, cfg, shuffle=True)
    val_seq = CanonValSequence(val_canon_smiles, tokenizer, cfg)

    def make_tf_dataset(seq, is_training: bool = False):
        def gen():
            for i in range(len(seq)):
                yield seq[i]
            if is_training:
                seq.on_epoch_end()

        output_signature = (
            (
                tf.TensorSpec(shape=(None, None), dtype=tf.uint8),
                tf.TensorSpec(shape=(None, None), dtype=tf.uint8),
            ),
            tf.TensorSpec(shape=(None, None), dtype=tf.uint8),
        )
        ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
        if is_training:
            ds = ds.repeat()

        return ds.prefetch(tf.data.AUTOTUNE)

    train_ds = make_tf_dataset(train_seq, is_training=True)
    val_ds = make_tf_dataset(val_seq, is_training=False)

    steps_per_epoch = cfg.steps_per_epoch or len(train_seq)

    train_model, encoder_model, decoder_model = build_smi2smi_models(
        cfg, tokenizer = tokenizer
    )
    train_model.summary()

    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),
        CanonicalizationMonitorCallback(
            val_canon_smiles=val_canon_smiles,
            encoder_model=encoder_model,
            decoder_model=decoder_model,
            tokenizer=tokenizer,
            cfg=cfg,
        ),
        CheckpointAveragerCallback(cfg, encoder_model=encoder_model),
        GpuThermalGuardCallback(cfg, check_every_n_batches=200)
    ]

    train_model.fit(
        train_ds,
        steps_per_epoch = steps_per_epoch,
        validation_data=val_ds,
        validation_steps = len(val_seq),
        epochs=cfg.epochs,
        callbacks=callbacks,
    )

if __name__ == "__main__":
    main()


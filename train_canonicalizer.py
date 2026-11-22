import os
import json
from dataclasses import asdict

from config import UlmConfig
from tf_setup import init_tensorflow

import tensorflow as tf

cfg = UlmConfig(
    smiles_file="data/smiles.smi",
    n_augmentations=10,
    epochs=1000,
    batch_size=64,
    monitor_every_n_epochs=5,
)

init_tensorflow(cfg)

from tokenizer import CharSmilesTokenizer
from smi2smi.pipeline import (
        prepare_canon_datasets, 
        CanonTrainSequence, 
        CanonValSequence
)

from smi2smi.model import build_smi2smi_models
from smi2smi.callbacks import (
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
        return tf.data.Dataset.from_generator(
            gen, output_signature=output_signature
        ).prefetch(tf.data.AUTOTUNE)

    train_ds = make_tf_dataset(train_seq, is_training=True)
    val_ds = make_tf_dataset(val_seq, is_training=False)


    train_model, encoder_model, decoder_model = build_smi2smi_models(
        cfg, tokenizer = tokenizer
    )
    train_model.summary()

    callbacks = [
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
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
    )

if __name__ == "__main__":
    main()


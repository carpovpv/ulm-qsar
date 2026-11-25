import os
import random
from typing import List, Sequence, Tuple

import csv
import numpy as np
from rdkit import Chem

from config import UlmConfig
import tensorflow as tf 

from tokenizer import BaseTokenizer
from pipelines.smi2smi import generate_random_smiles

import subprocess
import time
from typing import Optional


def get_max_gpu_temperature(gpu_ids: str = "0") -> Optional[int]:
    """Query `nvidia-smi` and return the highest temperature (°C) among active GPUs."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        )
        all_temps = [int(line.strip()) for line in output.strip().splitlines() if line.strip()]
        selected_indices = [int(idx.strip()) for idx in str(gpu_ids).split(",") if idx.strip().isdigit()]
        active_temps = [all_temps[i] for i in selected_indices if i < len(all_temps)]
        return max(active_temps) if active_temps else (max(all_temps) if all_temps else None)
    except Exception:
        return None


class GpuThermalGuardCallback(tf.keras.callbacks.Callback):
    """Pauses training when GPU temperature exceeds `max_gpu_temp` until it cools down."""

    def __init__(self, cfg: UlmConfig, check_every_n_batches: int = 0):
        super().__init__()
        self.gpu_ids = cfg.gpu_ids
        self.max_temp = cfg.max_gpu_temp
        self.resume_temp = cfg.resume_gpu_temp
        self.poll_interval = cfg.thermal_check_seconds
        self.check_every_n_batches = check_every_n_batches

    def _check_and_cool(self, context: str = ""):
        temp = get_max_gpu_temperature(self.gpu_ids)
        if temp is None or temp < self.max_temp:
            return

        print(f"\n[Thermal Guard {context}] GPU hit {temp}°C (>= {self.max_temp}°C). Cooling down to <= {self.resume_temp}°C...")
        while True:
            time.sleep(self.poll_interval)
            temp = get_max_gpu_temperature(self.gpu_ids)
            if temp is None or temp <= self.resume_temp:
                print(f"[Thermal Guard] GPU cooled to {temp}°C. Resuming training.\n")
                break
            print(f"  ... current GPU temp: {temp}°C (waiting)")

    def on_epoch_end(self, epoch, logs=None):
        self._check_and_cool(context=f"Epoch {epoch + 1}")

    def on_batch_end(self, batch, logs=None):
        if self.check_every_n_batches > 0 and (batch + 1) % self.check_every_n_batches == 0:
            self._check_and_cool(context=f"Batch {batch + 1}")



class CanonicalizationMonitorCallback(tf.keras.callbacks.Callback):
    """Evaluates autoregressive canonicalization and logs all metrics to CSV."""

    def __init__(
        self,
        val_canon_smiles: Sequence[str],
        encoder_model: tf.keras.Model,
        decoder_model: tf.keras.Model,
        tokenizer: BaseTokenizer,
        cfg: UlmConfig,
    ):
        super().__init__()
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.log_path = os.path.join(cfg.output_dir, "training_log.csv")

        # Initialize CSV header
        with open(self.log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "lr",
                "loss",
                "masked_sparse_acc",
                "val_loss",
                "val_masked_sparse_acc",
                "exact_canon_pct",
                "valid_rdkit_pct",
            ])

        rng = random.Random(cfg.seed)
        sample_pool = list(val_canon_smiles)[: cfg.monitor_n_samples]
        self.eval_pairs: List[Tuple[str, str]] = []
        for canon_smi in sample_pool:
            mol = Chem.MolFromSmiles(canon_smi)
            variants = generate_random_smiles(
                mol, n_augmentations=2, isomeric=cfg.isomeric, rng=rng
            )
            non_canon = next((v for v in variants if v != canon_smi), canon_smi)
            self.eval_pairs.append((non_canon, canon_smi))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_num = epoch + 1
        current_lr = float(self.model.optimizer.learning_rate.numpy())

        exact_pct = ""
        valid_pct = ""

        # Run autoregressive decoding check every N epochs (and on the final epoch)
        if epoch_num % self.cfg.monitor_every_n_epochs == 0 or epoch_num == self.cfg.epochs:
            src_list = [p[0] for p in self.eval_pairs]
            tgt_list = [p[1] for p in self.eval_pairs]

            preds: List[str] = []
            for i in range(0, len(src_list), self.cfg.batch_size):
                batch_src = src_list[i : i + self.cfg.batch_size]
                preds.extend(
                    greedy_canonicalize_batch(
                        self.encoder_model,
                        self.decoder_model,
                        batch_src,
                        self.tokenizer,
                        max_len=self.cfg.max_len,
                    )
                )

            exact_matches = 0
            valid_rdkit_matches = 0
            for pred, target in zip(preds, tgt_list):
                if pred == target:
                    exact_matches += 1
                    valid_rdkit_matches += 1
                else:
                    mol = Chem.MolFromSmiles(pred)
                    if mol is not None:
                        rd_canon = Chem.MolToSmiles(
                            mol, canonical=True, isomericSmiles=self.cfg.isomeric
                        )
                        if rd_canon == target:
                            valid_rdkit_matches += 1

            total = max(1, len(tgt_list))
            exact_pct = round(100.0 * exact_matches / total, 2)
            valid_pct = round(100.0 * valid_rdkit_matches / total, 2)

            print(
                f"\n[Monitor @ Epoch {epoch_num}] "
                f"Exact Canonical Match: {exact_pct:.2f}% ({exact_matches}/{total}) | "
                f"Valid Chemical Match: {valid_pct:.2f}%"
            )
            for src, tgt, pred in list(zip(src_list, tgt_list, preds))[:3]:
                status = "OK" if pred == tgt else "DIFF"
                print(f"  [{status}] In: {src}  -->  Pred: {pred}  (Target: {tgt})")
            print("")

        # Append epoch metrics to gen/training_log.csv
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch_num,
                f"{current_lr:.6e}",
                round(float(logs.get("loss", 0.0)), 6),
                round(float(logs.get("masked_sparse_acc", 0.0)), 6),
                round(float(logs.get("val_loss", 0.0)), 6),
                round(float(logs.get("val_masked_sparse_acc", 0.0)), 6),
                exact_pct,
                valid_pct,
            ])

class CheckpointAveragerCallback(tf.keras.callbacks.Callback):
    """Saves checkpoints at specified epochs and averages weights at training end."""

    def __init__(self, cfg: UlmConfig, encoder_model: tf.keras.Model):
        super().__init__()
        self.epochs_to_save = set(cfg.epochs_to_average)
        self.weights_output_path = cfg.weights_output_path
        self.encoder_output_path = cfg.encoder_output_path
        self.output_dir = cfg.output_dir
        self.encoder_model = encoder_model
        self.saved_files: List[str] = []

    def on_epoch_end(self, epoch, logs=None):
        epoch_num = epoch + 1
        if epoch_num in self.epochs_to_save:
            fname = os.path.join(self.output_dir, f"_ckpt_epoch_{epoch_num}.weights.h5")
            self.model.save_weights(fname)
            self.saved_files.append(fname)
    
    def on_train_end(self, logs=None):
        if len(self.saved_files) > 1:
            print(f"Averaging weights across {len(self.saved_files)} checkpoints...")
            all_weights = []
            for f in self.saved_files:
                self.model.load_weights(f)
                all_weights.append([w.copy() for w in self.model.get_weights()])

            averaged_weights = [
                np.mean(weight_group, axis=0)
                for weight_group in zip(*all_weights)
            ]
            self.model.set_weights(averaged_weights)

        self.model.save_weights(self.weights_output_path)
        self.encoder_model.save_weights(self.encoder_output_path)
        
        print(f"Saved full model weights to:    {self.weights_output_path}")
        print(f"Saved standalone encoder to:    {self.encoder_output_path}")

        for f in self.saved_files:
            if os.path.exists(f):
                os.remove(f)


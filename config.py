import os 

from dataclasses import dataclass, field
from typing import List 

@dataclass
class UlmConfig:
    """Central configuration for ULM mode training and evaluation."""

    # --- Hardware & Reproducibility ---
    gpu_ids: str = "1"
    seed: int = 42
    precision_policy: str = "mixed_bfloat16"  #mixed_float16", "mixed_bfloat16", or "float32"
    memory_growth: bool = True

    max_gpu_temp: int = 80         # Pause if GPU reaches >= 80°C
    resume_gpu_temp: int = 60      # Resume once GPU cools down to <= 60°C
    thermal_check_seconds: int = 5 # Seconds between temperature checks while cooling

    # --- Data & Augmentation ---
    smiles_file: str = "data/chembl-canon-126.smi"
    n_augmentations: int = 10
    val_fraction: float = 0.1
    strip_salts: bool = True
    isomeric: bool = True
    max_len: int = 256
    bucket_size : int = 128

    # --- Transformer Architecture ---
    embedding_size: int = 64
    key_size: int = 64
    n_blocks: int = 3
    n_heads: int = 10
    ff_hidden_size: int = 512
    dropout_rate: float = 0.1

    # --- Training & Schedule ---
    batch_size: int = 64
    steps_per_epoch: int = 25000
    epochs: int = 30
    warmup_steps: int = 16000
    lr_factor: float = 12.0
    epochs_to_average: List[int] = field(default_factory=list)

    # --- Monitoring & Output ---
    output_dir: str = "gen"
    monitor_every_n_epochs: int = 1
    monitor_n_samples: int = 200
    weights_filename: str = "canon_final.weights.h5"
    encoder_filename: str = "canon_encoder.weights.h5"

    @property
    def weights_output_path(self) -> str:
        return os.path.join(self.output_dir, self.weights_filename)

    @property
    def encoder_output_path(self) -> str:
        return os.path.join(self.output_dir, self.encoder_filename)

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        if not self.epochs_to_average and self.epochs >= 5:
            self.epochs_to_average = list(range(self.epochs - 4, self.epochs + 1))

import random
from typing import List, Sequence, Tuple
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.SaltRemover import SaltRemover

from tokenizer import BaseTokenizer
from config import UlmConfig

import tensorflow as tf 

# Silence RDKit parse warnings during bulk augmentation
RDLogger.DisableLog("rdApp.*")

def _fits_max_len(smiles: str, tokenizer: BaseTokenizer, max_len: int) -> bool:
    """Return True if all tokens are in vocab and total encoded length (with BOS/EOS) <= max_len."""
    tokens = tokenizer.tokenize(smiles)
    if (len(tokens) + 2) > max_len:
        return False
    return all(tok in tokenizer._token_to_id for tok in tokens)


def generate_random_smiles(
    mol: Chem.Mol,
    n_augmentations: int = 10,
    isomeric: bool = True,
    rng: random.Random = None,
) -> List[str]:
    """Generate up to `n_augmentations` unique non-canonical SMILES for an RDKit Mol."""

    if rng is None:
        rng = random.Random()

    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return []

    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)
    variants = {canonical}

    # Try up to 3x attempts to collect `n_augmentations` distinct randomizations
    max_attempts = n_augmentations * 3
    for _ in range(max_attempts):
        if len(variants) >= n_augmentations:
            break
        root_atom = rng.randrange(num_atoms)
        rand_smi = Chem.MolToSmiles(
            mol,
            rootedAtAtom=root_atom,
            canonical=False,
            isomericSmiles=isomeric,
        )
        variants.add(rand_smi)

    return list(variants)


def prepare_canon_datasets(
    cfg: UlmConfig,
    tokenizer: BaseTokenizer
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Read a SMILES file, canonicalize, split leak-free, and build augmentation pairs.

    Returns:
        train_pairs: List of `(non_canonical_smiles, canonical_smiles)` pairs.
        val_uniques: List of unique canonical SMILES for validation & generation monitoring.
    """
    rng = random.Random(cfg.seed)
    remover = SaltRemover() if cfg.strip_salts else None

    # 1. Parse, clean, and deduplicate by canonical SMILES first
    unique_mols = {}
    with open(cfg.smiles_file, "r", encoding="utf-8") as f:
        for line in f:
            raw_smi = line.strip().split()[0] if line.strip() else ""
            if not raw_smi:
                continue

            mol = Chem.MolFromSmiles(raw_smi)
            if mol is None:
                continue

            if remover is not None:
                mol = remover.StripMol(mol, dontRemoveEverything=True)
                if mol is None or mol.GetNumAtoms() == 0:
                    continue

            canon_smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=cfg.isomeric)
            if len(canon_smi) > (cfg.max_len - 2) or not tokenizer.is_valid(canon_smi):
                continue

            if not _fits_max_len(canon_smi, tokenizer, cfg.max_len):
                continue

            if canon_smi not in unique_mols:
                unique_mols[canon_smi] = mol

    all_canon_smiles = list(unique_mols.keys())
    rng.shuffle(all_canon_smiles)

    # 2. Split on unique canonical molecules BEFORE augmentation (prevents leakage)
    n_val = max(1, int(len(all_canon_smiles) * cfg.val_fraction))
    val_canon_smiles = all_canon_smiles[:n_val]
    train_canon_smiles = all_canon_smiles[n_val:]

    # 3. Augment training molecules into (non_canonical -> canonical) pairs
    train_pairs: List[Tuple[str, str]] = []
    for canon_smi in train_canon_smiles:
        mol = unique_mols[canon_smi]
        variants = generate_random_smiles(
            mol, n_augmentations=cfg.n_augmentations, isomeric=cfg.isomeric, rng=rng
        )
        for v in variants:
            if len(v) <= (cfg.max_len-2) and tokenizer.is_valid(v):
                train_pairs.append((v, canon_smi))

    rng.shuffle(train_pairs)
    return train_pairs, val_canon_smiles


class CanonTrainSequence(tf.keras.utils.Sequence):
    """Memory-efficient batch generator that shuffles training pairs every epoch."""

    def __init__(
        self,
        pairs: Sequence[Tuple[str, str]],
        tokenizer: BaseTokenizer,
        cfg : UlmConfig,
        shuffle: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pairs = list(pairs)
        self.tokenizer = tokenizer
        self.batch_size = cfg.batch_size
        self.shuffle = shuffle
        self.rng = random.Random(cfg.seed)
        if self.shuffle:
            self.rng.shuffle(self.pairs)

    def __len__(self) -> int:
        return int(np.ceil(len(self.pairs) / self.batch_size))

    def __getitem__(self, idx: int):
        batch_pairs = self.pairs[idx * self.batch_size : (idx + 1) * self.batch_size]
        src_smiles = [p[0] for p in batch_pairs]
        tgt_smiles = [p[1] for p in batch_pairs]

        # Encoder input: ^ + non_canonical + $
        x = self.tokenizer.encode_batch(src_smiles, add_bos=True, add_eos=True)

        # Decoder teacher-forcing sequences:
        # Full sequence is ^ + canonical + $
        full_tgt = self.tokenizer.encode_batch(tgt_smiles, add_bos=True, add_eos=True)
        y_in = full_tgt[:, :-1]   # Decoder input:  ^ + canonical
        z_out = full_tgt[:, 1:]   # Target labels:  canonical + $

        return (x, y_in), z_out

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.pairs)


class CanonValSequence(tf.keras.utils.Sequence):
    """Validation generator creating deterministic randomized inputs for validation loss."""

    def __init__(
        self,
        val_canon_smiles: Sequence[str],
        tokenizer: BaseTokenizer,
        cfg : UlmConfig,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.batch_size = cfg.batch_size
        rng = random.Random(cfg.seed)

        # Build fixed (randomized_input, canonical_target) pairs once for stable val_loss
        self.val_pairs: List[Tuple[str, str]] = []
        for canon_smi in val_canon_smiles:
            mol = Chem.MolFromSmiles(canon_smi)
            variants = generate_random_smiles(mol, n_augmentations=2, isomeric=cfg.isomeric, rng=rng)
            # Pick a non-canonical variant when available
            non_canon = next((v for v in variants if v != canon_smi), canon_smi)
            if tokenizer.is_valid(non_canon):
                self.val_pairs.append((non_canon, canon_smi))

    def __len__(self) -> int:
        return int(np.ceil(len(self.val_pairs) / self.batch_size))

    def __getitem__(self, idx: int):
        batch_pairs = self.val_pairs[idx * self.batch_size : (idx + 1) * self.batch_size]
        src_smiles = [p[0] for p in batch_pairs]
        tgt_smiles = [p[1] for p in batch_pairs]

        x = self.tokenizer.encode_batch(src_smiles, add_bos=True, add_eos=True)
        full_tgt = self.tokenizer.encode_batch(tgt_smiles, add_bos=True, add_eos=True)
        return (x, full_tgt[:, :-1]), full_tgt[:, 1:]



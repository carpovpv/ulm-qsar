import re
from abc import ABC, abstractmethod
from typing import Dict, List, Sequence
import numpy as np

from config import UlmConfig 

class BaseTokenizer(ABC):
    """Abstract base class for SMILES tokenizers.

    Guarantees that index 0 is always the padding token so Keras
    `Embedding(..., mask_zero=True)` works across all implementations.
    """

    def __init__(self, 
                 pad_token: str = " ", 
                 bos_token: str = "^", 
                 eos_token: str = "$",
                 bucket_size : int = 128):

        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.bucket_size = bucket_size

        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        self._build_vocab()

        # Enforce index 0 == pad_token invariant
        if self._token_to_id.get(self.pad_token) != 0:
            raise ValueError(f"Padding token '{self.pad_token}' must have ID 0.")

    @abstractmethod
    def _build_vocab(self) -> None:
        """Populate `self._token_to_id` and `self._id_to_token`."""
        pass

    @abstractmethod
    def tokenize(self, smiles: str) -> List[str]:
        """Split a raw SMILES string into a list of tokens (without BOS/EOS)."""
        pass

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id)

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def bos_id(self) -> int:
        return self._token_to_id[self.bos_token]

    @property
    def eos_id(self) -> int:
        return self._token_to_id[self.eos_token]

    @property
    def dtype(self):
        return np.uint8 if self.vocab_size <= 255 else np.uint16

    def is_valid(self, smiles: str) -> bool:
        """Check whether all tokens in `smiles` exist in the vocabulary."""
        return all(tok in self._token_to_id for tok in self.tokenize(smiles))

    def padded_length(self, raw_len: int) -> int:
        """Round `raw_len` up to the nearest multiple of `self.bucket_size`."""
        return ((raw_len + self.bucket_size - 1) // self.bucket_size) * self.bucket_size

    def encode(self, smiles: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """Convert a single SMILES string into a list of integer token IDs."""
        tokens = self.tokenize(smiles)
        if add_bos:
            tokens = [self.bos_token] + tokens
        if add_eos:
            tokens = tokens + [self.eos_token]
        return [self._token_to_id[t] for t in tokens]

    def decode(self, ids: Sequence[int], stop_at_eos: bool = True) -> str:
        """Convert a sequence of token IDs back into a SMILES string."""
        chars: List[str] = []
        for idx in ids:
            idx = int(idx)
            if idx == self.pad_id or idx == self.bos_id:
                continue
            if idx == self.eos_id:
                if stop_at_eos:
                    break
                continue
            chars.append(self._id_to_token.get(idx, ""))
        return "".join(chars)

    def encode_batch(
        self,
        smiles_list: Sequence[str],
        add_bos: bool = True,
        add_eos: bool = True
    ) -> np.ndarray:
        """Encode and right-pad a batch of SMILES into a 2D NumPy array."""
        encoded = [self.encode(s, add_bos=add_bos, add_eos=add_eos) for s in smiles_list]
        max_len = max(len(seq) for seq in encoded)

        padded_len = self.padded_length(max_len)
        batch = np.zeros((len(encoded), padded_len), dtype=self.dtype)
        for row, seq in enumerate(encoded):
            batch[row, : len(seq)] = seq
        return batch

    def decode_batch(self, batch_ids: np.ndarray, stop_at_eos: bool = True) -> List[str]:
        """Decode a 2D array of token IDs into a list of SMILES strings."""
        return [self.decode(row, stop_at_eos=stop_at_eos) for row in batch_ids]


class CharSmilesTokenizer(BaseTokenizer):
    """Character-level SMILES tokenizer matching the original ulm-qsar vocabulary.
       This vocabulary was used in the original article 
       https://link.springer.com/chapter/10.1007/978-3-030-30493-5_78
    """

    DEFAULT_CHARS = " ^#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVXYZ[\\]abcdefgilmnoprstuy$"

    def __init__(self, cfg: UlmConfig, chars: str = DEFAULT_CHARS):
        self._chars = chars
        super().__init__(pad_token=" ", bos_token="^", eos_token="$", bucket_size = cfg.bucket_size)

    def _build_vocab(self) -> None:
        self._token_to_id = {ch: i for i, ch in enumerate(self._chars)}
        self._id_to_token = {i: ch for i, ch in enumerate(self._chars)}

    def tokenize(self, smiles: str) -> List[str]:
        return list(smiles.strip())


class RegexSmilesTokenizer(BaseTokenizer):
    """Atom-level SMILES tokenizer (keeps multi-char tokens like Cl, Br, [nH] intact)."""

    # Standard Schwaller et al. SMILES regex
    SMILES_REGEX = re.compile(
       r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""
    )

    def __init__(self, cfg: UlmConfig, vocab_tokens: Sequence[str]):
        self._raw_tokens = list(vocab_tokens)
        super().__init__(pad_token="<PAD>", bos_token="^", eos_token="$", bucket_size = cfg.bucket_size)

    def _build_vocab(self) -> None:
        specials = [self.pad_token, self.bos_token, self.eos_token]
        unique_tokens = specials + [t for t in self._raw_tokens if t not in specials]
        self._token_to_id = {tok: i for i, tok in enumerate(unique_tokens)}
        self._id_to_token = {i: tok for i, tok in enumerate(unique_tokens)}

    def tokenize(self, smiles: str) -> List[str]:
        return self.SMILES_REGEX.findall(smiles.strip())


"""
Encoders under test. Each maps text -> a token sequence; the length of that sequence is what a
transformer pays (attention Θ(N²) + KV Θ(N)). We compare:

  * byte            : raw UTF-8 bytes (vocab 256) -- the storage floor & tokenizer-free reference.
  * grapheme        : one token per Unicode grapheme cluster (akshara-preserving), vocab = observed.
  * gpt2 / cl100k / o200k : real English-centric production BPE tokenizers (tiktoken).
  * matched-bpe-V   : a BPE trained ON THE TARGET LANGUAGE ITSELF (held-out: trained on FLORES `dev`,
                      measured on `devtest`) at a fixed vocab V -- the "removable-tax-removed" code.

All are REAL encoders (no estimation). tiktoken and tokenizers run without torch.
"""
from __future__ import annotations
from typing import Callable, Dict, List, Sequence

import regex
import tiktoken
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

from corpora import grapheme_clusters


class Encoder:
    def __init__(self, name: str, count_fn: Callable[[str], int], vocab_size: int):
        self.name = name
        self._count = count_fn
        self.vocab_size = vocab_size

    def count(self, text: str) -> int:
        return self._count(text)

    def total_tokens(self, sentences: Sequence[str]) -> int:
        return sum(self._count(s) for s in sentences)

    def token_counts(self, sentences: Sequence[str]) -> List[int]:
        return [self._count(s) for s in sentences]


# ---- fixed, language-agnostic encoders -------------------------------------
def byte_encoder() -> Encoder:
    return Encoder("byte", lambda t: len(t.encode("utf-8")), 256)


def grapheme_encoder(vocab_size: int = 0) -> Encoder:
    return Encoder("grapheme", lambda t: len(grapheme_clusters(t)), vocab_size)


def tiktoken_encoder(name: str) -> Encoder:
    enc = tiktoken.get_encoding(name)
    return Encoder(name, lambda t: len(enc.encode(t)), enc.n_vocab)


# ---- per-language MATCHED BPE (trained held-out) ---------------------------
def train_matched_bpe(train_texts: Sequence[str], vocab_size: int, name: str) -> Encoder:
    """Train a byte-level BPE on the target language's own dev text. This is the concrete
    'matched code' that removes the vocabulary-mismatch redundancy R. Deterministic."""
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]"],
        show_progress=False,
    )
    tok.train_from_iterator(train_texts, trainer=trainer)
    real_v = tok.get_vocab_size()
    return Encoder(name, lambda t: len(tok.encode(t).ids), real_v)


def build_fixed_encoders() -> Dict[str, Encoder]:
    """Language-agnostic encoders shared across all languages."""
    encs: Dict[str, Encoder] = {}
    encs["byte"] = byte_encoder()
    encs["grapheme"] = grapheme_encoder()
    for tk in ("gpt2", "cl100k_base", "o200k_base"):
        try:
            encs[tk] = tiktoken_encoder(tk)
        except Exception as e:  # network / encoding unavailable -> fail loud later, skip here
            print(f"[encoders] WARN could not load tiktoken {tk}: {e}")
    return encs

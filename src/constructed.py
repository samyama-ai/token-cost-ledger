"""
The "Silicon Vernacular": a CONSTRUCTED matched code for a controlled semantic domain, used to
demonstrate the REMOVABLE redundancy R -> 0 -- the thesis of the paper -- on a source whose floor we
know exactly. It is an upper-bound construction (the discrete-code limit of byte-entropy patching /
BLT, Pagnoni 2024), NOT a practical human language.

Domain: M concepts with a Zipf(s) law (a stand-in for heavy-tailed semantic frequency). The floor is
the source entropy H (bits/concept). We compare bits/concept for:
  * mismatched fixed-width code : ceil(log2 M) bits/concept (a code optimal for the UNIFORM law; it
                                  ignores the Zipf skew) -> pays the full redundancy R = log2(M) - H.
  * matched code, block size k  : Huffman over k-grams -> bits/concept in [H, H + 1/k) -> H as k grows.
A binary symbol code (k=1) already sits within 1 bit of the floor (Lemma 1); block coding drives the
residual to zero. The gap the *matched* code removes vs the mismatched one is the removable tax.
"""
from __future__ import annotations
import math
from collections import Counter
from typing import Dict, List

from entropy import shannon_entropy_from_counts, huffman_code_lengths, expected_code_length
from constructed_source import zipf_concept_source  # split out for reuse/tests


def analyze(M: int = 256, s: float = 1.1, n_tokens: int = 400_000, seed: int = 20260705) -> Dict:
    """Well-sampled symbol-code demonstration only. Block coding (k>1) is intentionally NOT measured
    empirically here: at k>1 the k-gram alphabet is undersampled, so an in-sample Huffman would appear
    to 'beat' the floor purely from finite-sample entropy bias -- a dishonest number. The residual
    <1-bit gap is closed by block/arithmetic coding as a THEOREM (Lemma 1 achievability), which we
    state and cite rather than fake-measure."""
    stream = zipf_concept_source(M, s, n_tokens, seed)
    counts = Counter(stream)
    H = shannon_entropy_from_counts(counts, 2.0)                    # bits/concept -- the floor
    fixed_width = float(math.ceil(math.log2(M)))                    # mismatched (uniform-optimal) code
    lens = huffman_code_lengths(counts, radix=2)                    # matched symbol code
    matched = expected_code_length(counts, lens)                   # bits/concept, k=1
    # sanity: a real code cannot beat the floor (NC1); matched must sit in [H, H+1)
    assert matched >= H - 1e-6, "symbol code below entropy floor -- estimator bug"
    return {
        "domain": {"M_concepts": M, "zipf_s": s, "n_tokens": n_tokens},
        "floor_H_bits_per_concept": H,
        "mismatched_fixed_width_bits_per_concept": fixed_width,
        "matched_symbol_bits_per_concept": matched,
        "matched_gap_to_floor_bits": matched - H,                  # < 1 by Lemma 1
        "redundancy_removed_by_match_bits": fixed_width - matched,
        "redundancy_removed_frac": (fixed_width - matched) / (fixed_width - H) if fixed_width > H else 0.0,
    }

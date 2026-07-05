"""
Information-theoretic primitives for the token-cost ledger.

Everything here is exact or a clearly-labelled estimator. The floor F* = H/log2(V) and the
redundancy R = D_KL(p||q_C)/log2(V) are STANDARD (Shannon source coding + Kraft-McMillan); we
compute, we do not claim them. See NOVELTY.md.

No mocks: entropy estimates come from real symbol counts; content-bit estimates come from a real
universal compressor (lzma). The calibration control (NC2) proves the stack recovers a known H.
"""
from __future__ import annotations
import math
import heapq
import lzma
import bz2
from collections import Counter
from typing import Dict, Iterable, Hashable, Tuple


# ---------------------------------------------------------------------------
# Shannon entropy
# ---------------------------------------------------------------------------
def shannon_entropy_from_counts(counts: Dict[Hashable, int], base: float = 2.0) -> float:
    """Plug-in (MLE) Shannon entropy H = -sum p log p, in `base` units (bits for base 2)."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    logb = math.log(base)
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log(p) / logb
    return h


def miller_madow_correction(counts: Dict[Hashable, int]) -> float:
    """Miller-Madow bias correction term (bits) to ADD to the plug-in entropy:
    (K-1)/(2 N ln2), K = #observed symbols, N = #samples. Honest upward correction for
    finite-sample downward bias of the plug-in estimator."""
    total = sum(counts.values())
    k = sum(1 for c in counts.values() if c > 0)
    if total == 0:
        return 0.0
    return (k - 1) / (2 * total * math.log(2))


def entropy_bits(symbols: Iterable[Hashable], mm_correct: bool = True) -> Tuple[float, int, int]:
    """Return (H_bits_per_symbol, n_symbols, alphabet_size). mm_correct adds Miller-Madow."""
    counts = Counter(symbols)
    h = shannon_entropy_from_counts(counts, 2.0)
    if mm_correct:
        h += miller_madow_correction(counts)
    return h, sum(counts.values()), len([1 for c in counts.values() if c > 0])


# ---------------------------------------------------------------------------
# Huffman coding (binary and V-ary) -- an ACTUAL optimal prefix code, no estimation
# ---------------------------------------------------------------------------
def huffman_code_lengths(counts: Dict[Hashable, int], radix: int = 2) -> Dict[Hashable, int]:
    """Optimal r-ary prefix-code lengths (in radix symbols) via Huffman. For radix r, pad the
    number of leaves so that (K - 1) % (r - 1) == 0 (standard r-ary Huffman dummy-symbol rule)."""
    items = [(c, s) for s, c in counts.items() if c > 0]
    if not items:
        return {}
    if len(items) == 1:
        return {items[0][1]: 1}
    if radix < 2:
        raise ValueError("radix must be >= 2")
    # r-ary Huffman dummy padding
    if radix > 2:
        while (len(items) - 1) % (radix - 1) != 0:
            items.append((0, ("__dummy__", len(items))))
    # heap of (weight, tie, node); node is either symbol-leaf or internal (list of children)
    heap = []
    for i, (c, s) in enumerate(items):
        heap.append((c, i, ("leaf", s)))
    heapq.heapify(heap)
    tie = len(items)
    while len(heap) > 1:
        group = [heapq.heappop(heap) for _ in range(min(radix, len(heap)))]
        w = sum(g[0] for g in group)
        heapq.heappush(heap, (w, tie, ("node", [g[2] for g in group])))
        tie += 1
    lengths: Dict[Hashable, int] = {}

    def walk(node, depth):
        kind, payload = node
        if kind == "leaf":
            if not (isinstance(payload, tuple) and payload and payload[0] == "__dummy__"):
                lengths[payload] = max(depth, 1)
        else:
            for ch in payload:
                walk(ch, depth + 1)

    walk(heap[0][2], 0)
    return lengths


def expected_code_length(counts: Dict[Hashable, int], lengths: Dict[Hashable, int]) -> float:
    """Sum p(s) * len(s): expected code length in radix-symbols per source symbol."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return sum((counts[s] / total) * lengths[s] for s in counts if counts[s] > 0)


def kl_redundancy_bits(counts: Dict[Hashable, int], lengths: Dict[Hashable, int]) -> float:
    """R = E[len]_bits - H(p) = D_KL(p || q_C) where q_C(s) ∝ 2^-len(s). Radix assumed 2 (bits).
    This is the REMOVABLE redundancy of code C relative to the source p. >= 0 by Gibbs."""
    L = expected_code_length(counts, lengths)               # bits/symbol
    H = shannon_entropy_from_counts(counts, 2.0)
    return L - H


# ---------------------------------------------------------------------------
# Content-bit estimate via a real universal compressor (order-agnostic upper bound on H)
# ---------------------------------------------------------------------------
def content_bits_lzma(text: str) -> float:
    """Bits to store `text` under LZMA (a strong universal compressor). Upper bound on the true
    content entropy; consistent across languages when applied identically. Subtract a small
    fixed container overhead measured on the empty string."""
    raw = text.encode("utf-8")
    overhead = len(lzma.compress(b"", preset=9 | lzma.PRESET_EXTREME))
    comp = len(lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME))
    return max(comp - overhead, 0) * 8.0


def content_bits_bz2(text: str) -> float:
    raw = text.encode("utf-8")
    overhead = len(bz2.compress(b"", 9))
    return max(len(bz2.compress(raw, 9)) - overhead, 0) * 8.0


def content_bits_per_char(text: str, atoms) -> float:
    """LZMA content bits normalised per atom (grapheme cluster), a language-comparable density."""
    n = max(len(atoms), 1)
    return content_bits_lzma(text) / n


def content_bits_normalized(atoms) -> float:
    """Script-fair content bits: compress the sequence of grapheme-cluster IDs (2 bytes/id) rather
    than UTF-8 text, so the estimate does NOT charge Indic scripts for their 3-byte-per-codepoint
    UTF-8 assignment. This isolates INTRINSIC content from the encoding-width artifact -> a fairer
    intrinsic floor. Consistent estimator across languages."""
    vocab = {}
    for a in atoms:
        if a not in vocab:
            vocab[a] = len(vocab)
    buf = bytearray()
    for a in atoms:
        i = vocab[a]
        buf += (i & 0xFFFF).to_bytes(2, "little")
        if i > 0xFFFF:  # rare: extend for very large cluster vocabularies
            buf += (i >> 16).to_bytes(2, "little")
    overhead = len(lzma.compress(b"", preset=9 | lzma.PRESET_EXTREME))
    comp = len(lzma.compress(bytes(buf), preset=9 | lzma.PRESET_EXTREME))
    return max(comp - overhead, 0) * 8.0

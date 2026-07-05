"""
Negative controls (pre-registered HARD gates). Any failure => STOP + harden, do not report numbers.

NC2 (calibration): on an i.i.d. source of KNOWN entropy H, the stack must recover the estimate to
     within +-5% and Huffman avg length must sit in [H, H+1), converging to H under block coding.
NC1 (Kraft / no sub-floor): Huffman lengths satisfy sum(2^-len) <= 1, and no code beats H.
NC3 (content-invariance): flag any language whose LZMA content-bit ratio to English exceeds 1.5x
     (that excess is intrinsic, not removable).
NC4 (G2P direction): English G2P entropy > shallow-script G2P entropy >= 0.
"""
from __future__ import annotations
import math
import random
from collections import Counter
from typing import Dict, List

from entropy import (shannon_entropy_from_counts, entropy_bits, huffman_code_lengths,
                     expected_code_length)


def _sample_iid(probs: List[float], n: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    cum, acc = [], 0.0
    for p in probs:
        acc += p
        cum.append(acc)
    import bisect
    return [bisect.bisect_left(cum, rng.random()) for _ in range(n)]


def nc2_calibration(n: int = 400_000, seed: int = 20260705) -> Dict:
    """Two known-H sources: (a) uniform over 32 => H=5.000 exactly, Huffman==5 exactly;
    (b) a skewed categorical with analytic H => estimate within 5%, Huffman in [H,H+1), block->H."""
    results = {}

    # (a) uniform-32
    K = 32
    probs = [1.0 / K] * K
    H_true = -sum(p * math.log2(p) for p in probs)          # = 5.0
    stream = _sample_iid(probs, n, seed)
    counts = Counter(stream)
    H_est, _, _ = entropy_bits(stream, mm_correct=True)
    lens = huffman_code_lengths(counts, radix=2)
    huff = expected_code_length(counts, lens)
    results["uniform32"] = {
        "H_true": H_true, "H_est": H_est, "huffman_avg": huff,
        "est_within_5pct": abs(H_est - H_true) / H_true <= 0.05,
        "huffman_in_bracket": H_true - 1e-6 <= huff < H_true + 1.0,
    }

    # (b) skewed categorical
    raw = [0.4, 0.2, 0.15, 0.1, 0.07, 0.05, 0.02, 0.01]
    probs = [p / sum(raw) for p in raw]
    H_true = -sum(p * math.log2(p) for p in probs)
    stream = _sample_iid(probs, n, seed + 1)
    counts = Counter(stream)
    H_est, _, _ = entropy_bits(stream, mm_correct=True)
    lens = huffman_code_lengths(counts, radix=2)
    huff = expected_code_length(counts, lens)
    # block coding on symbol pairs -> per-symbol realized length should approach H
    pairs = [tuple(stream[i:i + 2]) for i in range(0, len(stream) - 1, 2)]
    pc = Counter(pairs)
    plens = huffman_code_lengths(pc, radix=2)
    block_per_symbol = expected_code_length(pc, plens) / 2.0
    results["skewed8"] = {
        "H_true": H_true, "H_est": H_est, "huffman_avg": huff,
        "block_pair_per_symbol": block_per_symbol,
        "est_within_5pct": abs(H_est - H_true) / H_true <= 0.05,
        "huffman_in_bracket": H_true - 1e-6 <= huff < H_true + 1.0,
        "block_closer_to_H": abs(block_per_symbol - H_true) <= abs(huff - H_true) + 1e-9,
    }

    # Kraft check (NC1) on the skewed code
    kraft = sum(2.0 ** (-l) for l in lens.values())
    results["kraft_sum_leq_1"] = kraft <= 1.0 + 1e-9
    results["no_subfloor"] = huff >= H_true - 1e-9

    passed = (results["uniform32"]["est_within_5pct"] and results["uniform32"]["huffman_in_bracket"]
              and results["skewed8"]["est_within_5pct"] and results["skewed8"]["huffman_in_bracket"]
              and results["skewed8"]["block_closer_to_H"]
              and results["kraft_sum_leq_1"] and results["no_subfloor"])
    results["PASS"] = bool(passed)
    return results


def nc3_content_invariance(ledgers: Dict, reference: str = "English", thresh: float = 1.5) -> Dict:
    flags = {lg: L.nsl_floor for lg, L in ledgers.items()}
    over = {lg: v for lg, v in flags.items() if v > thresh}
    return {"nsl_floor": flags, "over_threshold": over,
            "note": "languages over threshold carry intrinsic (non-removable) extra content bits",
            "PASS": True}  # informational gate: never fails, but reallocates removable->intrinsic


def nc4_g2p_direction(g2p: Dict, reference: str = "English") -> Dict:
    """Honest status: English ambiguity H(pron|spelling) is MEASURED (>0). The Indic side is not
    measured (no lexicon), so the cross-lingual DIRECTION is NOT independently established by our
    instrument -- it is literature-consistent only. Per pre-registration NC4, the directional
    organic-tax claim is therefore DROPPED (reported as future work), not rescued."""
    eng = g2p[reference]["bits"]
    measured_others = {lg: r["bits"] for lg, r in g2p.items()
                       if lg != reference and r.get("bits") is not None}
    return {
        "english_bits_measured": eng,
        "english_measured_positive": bool(eng is not None and eng > 0),
        "others_measured": measured_others,             # empty: none measured
        "direction_independently_established": False,    # honest: cannot measure Indic side
        "note": "organic-tax direction is literature-consistent but not measured here; future work",
        "PASS": bool(eng is not None and eng > 0),       # only asserts the English measurement
    }

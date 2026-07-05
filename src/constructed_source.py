"""Deterministic Zipf concept source for the constructed-encoding demonstration."""
from __future__ import annotations
import bisect
import random
from typing import List


def zipf_concept_source(M: int = 256, s: float = 1.1, n_tokens: int = 400_000,
                        seed: int = 20260705) -> List[int]:
    """Deterministic Zipf(s) sample of concept ids in [0, M) via inverse-CDF sampling."""
    rng = random.Random(seed)
    weights = [1.0 / ((r + 1) ** s) for r in range(M)]
    tot = sum(weights)
    cum, acc = [], 0.0
    for w in weights:
        acc += w / tot
        cum.append(acc)
    return [bisect.bisect_left(cum, rng.random()) for _ in range(n_tokens)]

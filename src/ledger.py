"""
The token-cost ledger. For a fixed parallel content set, decompose each language's cost into:

  NSL_P(L)   normalized sequence length under a PRODUCTION English-centric tokenizer  = T_P(L)/T_P(Eng)
  NSL_M(L)   ... under a per-language MATCHED BPE (same vocab V, each trained on its own dev split)
  NSL_floor  ... the information floor = Hc(L)/Hc(Eng) (LZMA content bits) -- the intrinsic ratio
  rho(L)     removable fraction = 1 - (NSL_M-1)/(NSL_P-1): how much of the production tax a matched
             code removes.  [H1]

All quantities are on the SAME sentences (parallel), so content is held fixed and only the
representation varies. Bootstrap CIs resample sentences jointly across languages.
"""
from __future__ import annotations
import random
import statistics as stats
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from corpora import grapheme_clusters, words, utf8_bytes
from encoders import Encoder
from entropy import content_bits_normalized


def _all_clusters(sents):
    out = []
    for s in sents:
        out.extend(grapheme_clusters(s))
    return out


@dataclass
class LangLedger:
    lang: str
    n_sent: int
    clusters: int
    words: int
    utf8: int
    bpc: float                       # UTF-8 bytes per grapheme cluster
    tokens: Dict[str, int] = field(default_factory=dict)   # encoder -> total tokens
    fert_word: Dict[str, float] = field(default_factory=dict)
    nsl: Dict[str, float] = field(default_factory=dict)     # encoder -> NSL vs English
    content_bits: float = 0.0
    nsl_floor: float = 0.0
    nsl_matched: float = 0.0
    matched_vocab: int = 0
    rho: float = 0.0


def _totals(sents: Sequence[str], enc: Encoder) -> int:
    return enc.total_tokens(sents)


def build_ledgers(
    corpus: Dict[str, List[str]],
    fixed_encoders: Dict[str, Encoder],
    matched_encoders: Dict[str, Encoder],
    production: str = "cl100k_base",
    reference: str = "English",
    compute_floor: bool = True,
) -> Dict[str, LangLedger]:
    ref_sents = corpus[reference]
    # reference token totals per encoder (denominator of NSL)
    ref_tok = {name: _totals(ref_sents, enc) for name, enc in fixed_encoders.items()}
    ref_matched_tok = _totals(ref_sents, matched_encoders[reference])
    ref_bits = content_bits_normalized(_all_clusters(ref_sents)) if compute_floor else 1.0

    out: Dict[str, LangLedger] = {}
    for lang, sents in corpus.items():
        cl = sum(len(grapheme_clusters(s)) for s in sents)
        wd = sum(len(words(s)) for s in sents)
        ub = sum(utf8_bytes(s) for s in sents)
        L = LangLedger(lang=lang, n_sent=len(sents), clusters=cl, words=wd, utf8=ub,
                       bpc=ub / max(cl, 1))
        for name, enc in fixed_encoders.items():
            t = _totals(sents, enc)
            L.tokens[name] = t
            L.fert_word[name] = t / max(wd, 1)
            L.nsl[name] = t / max(ref_tok[name], 1)
        # matched code (this language's own BPE), NSL vs English's own matched BPE
        mt = _totals(sents, matched_encoders[lang])
        L.matched_vocab = matched_encoders[lang].vocab_size
        L.nsl_matched = mt / max(ref_matched_tok, 1)
        L.tokens["matched"] = mt
        L.fert_word["matched"] = mt / max(wd, 1)
        # information floor (script-fair: compress cluster-ID stream, not UTF-8 bytes)
        if compute_floor:
            L.content_bits = content_bits_normalized(_all_clusters(sents))
            L.nsl_floor = L.content_bits / max(ref_bits, 1.0)
        # removable fraction (only meaningful when there is a production tax to remove)
        exc = L.nsl[production] - 1.0
        res = L.nsl_matched - 1.0
        L.rho = (1.0 - res / exc) if exc > 1e-9 else float("nan")
        out[lang] = L
    return out


def bootstrap_median_rho(
    corpus: Dict[str, List[str]],
    fixed_encoders: Dict[str, Encoder],
    matched_encoders: Dict[str, Encoder],
    target_langs: Sequence[str],
    production: str = "cl100k_base",
    reference: str = "English",
    B: int = 300,
    seed: int = 20260705,
) -> Dict[str, float]:
    """Sentence-level bootstrap of the median rho across `target_langs`. Returns point + 95% CI."""
    rng = random.Random(seed)
    n = min(len(v) for v in corpus.values())
    # Precompute per-sentence token counts ONCE (production + each language's matched code).
    prod = fixed_encoders[production]
    prod_counts = {lg: prod.token_counts(corpus[lg][:n]) for lg in corpus}
    matched_counts = {lg: matched_encoders[lg].token_counts(corpus[lg][:n]) for lg in corpus}

    def median_rho(idx: Sequence[int]) -> float:
        sp_ref = sum(prod_counts[reference][i] for i in idx)
        sm_ref = sum(matched_counts[reference][i] for i in idx)
        rs = []
        for lg in target_langs:
            nsl_p = sum(prod_counts[lg][i] for i in idx) / max(sp_ref, 1)
            nsl_m = sum(matched_counts[lg][i] for i in idx) / max(sm_ref, 1)
            exc = nsl_p - 1.0
            if exc > 1e-9:
                rs.append(1.0 - (nsl_m - 1.0) / exc)
        return stats.median(rs) if rs else float("nan")

    point = median_rho(list(range(n)))
    boot = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        boot.append(median_rho(idx))
    boot = sorted(b for b in boot if b == b)
    lo = boot[int(0.025 * len(boot))] if boot else float("nan")
    hi = boot[int(0.975 * len(boot)) - 1] if boot else float("nan")
    return {"median_rho": point, "ci_lo": lo, "ci_hi": hi, "B": len(boot)}

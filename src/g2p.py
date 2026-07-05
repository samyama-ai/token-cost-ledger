"""
The IRREDUCIBLE / organic tax: grapheme-to-phoneme (G2P) ambiguity, a property of a language's
orthography that no re-encoding removes. We proxy it with the Shannon conditional entropy
H(pronunciation | spelling): the bits of pronunciation uncertainty left after seeing the written form.

English (deep orthography) is measured from CMUdict (real, public). Indic abugidas are shallow /
near-isomorphic by construction (cite Torres&Futrell 2025; Sproat 2003): H(P|G) ~ 0 up to documented
exceptions (Indo-Aryan schwa deletion). We label each language's value by its BASIS (measured vs
by-construction) rather than pretending to measure an Indic pronunciation lexicon we do not have.

This is a PROXY (uniform-over-variants upper bound), used only for the direction result (NC4/H3).
"""
from __future__ import annotations
import math
from typing import Dict, Tuple


def english_g2p_entropy() -> Tuple[float, float, int]:
    """H(pron|spelling) proxy over CMUdict = mean over word types of log2(#distinct pronunciations),
    assuming uniform variants (an upper bound). Returns (mean_bits, homograph_rate, n_types)."""
    d = _load_cmudict()
    total, hom, ent = 0, 0, 0.0
    for word, prons in d.items():
        k = len(set(tuple(p) for p in prons))
        if k <= 0:
            continue
        total += 1
        if k > 1:
            hom += 1
        ent += math.log2(k)
    if total == 0:
        raise RuntimeError("CMUdict empty -- real lexicon required for the English G2P proxy")
    return ent / total, hom / total, total


def _load_cmudict() -> Dict[str, list]:
    # prefer the `cmudict` pip package (bundles data); fall back to nltk.
    try:
        import cmudict as _c
        d: Dict[str, list] = {}
        for w, p in _c.entries():
            d.setdefault(w.lower(), []).append(tuple(p))
        if d:
            return d
    except Exception:
        pass
    try:
        import nltk
        try:
            from nltk.corpus import cmudict as _n
            return {w.lower(): v for w, v in _n.dict().items()}
        except LookupError:
            nltk.download("cmudict", quiet=True)
            from nltk.corpus import cmudict as _n
            return {w.lower(): v for w, v in _n.dict().items()}
    except Exception as e:
        raise RuntimeError(f"no CMUdict available (pip install cmudict): {e}")


# We do NOT fabricate bit values for the Indic/shallow side: we have no Indic pronunciation lexicon,
# so H(pron|spelling) is NOT measured there. We record the literature status only. Any cross-lingual
# DIRECTION claim is therefore literature-consistent but NOT independently established by our
# instrument (see PREREGISTRATION.md NC4: dropped, not rescued).
SHALLOW_STATUS: Dict[str, str] = {
    "Hindi":   "not-measured; literature: shallow abugida + Indo-Aryan schwa deletion (small)",
    "Bengali": "not-measured; literature: shallow abugida + Indo-Aryan schwa deletion (small)",
    "Telugu":  "not-measured; literature: near-isomorphic Dravidian abugida (~deterministic)",
    "Tamil":   "not-measured; literature: shallow Dravidian abugida (some allophony)",
    "Kannada": "not-measured; literature: near-isomorphic Dravidian abugida (~deterministic)",
    "Spanish": "not-measured; literature: shallow Latin orthography",
    "German":  "not-measured; literature: fairly shallow Latin orthography",
}


def g2p_table() -> Dict[str, Dict[str, object]]:
    """English: the ambiguity facet H(pron|spelling) MEASURED from CMUdict (real). Others: literature
    status only, bits=None (unmeasured). The mapping-complexity facet (Torres&Futrell 2025,
    compressibility) is cited, not recomputed."""
    mean_bits, hom, n = english_g2p_entropy()
    out: Dict[str, Dict[str, object]] = {
        "English": {"bits": round(mean_bits, 4), "homograph_rate": round(hom, 4),
                    "n_types": n, "basis": "measured (CMUdict homograph entropy)"}
    }
    for lang, status in SHALLOW_STATUS.items():
        out[lang] = {"bits": None, "basis": status}
    return out

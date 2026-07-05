"""
Real parallel multilingual corpora. Primary: FLORES-200 (Meta/NLLB, CC-BY-SA-4.0), a
professionally-translated, sentence-aligned benchmark across 200 languages. Line i of every
language file is the SAME sentence -> a true parallel corpus, which lets us hold semantic
content fixed and vary only the language/encoding.

NO MOCKS: if the real corpus cannot be fetched, the loader raises. It never substitutes
synthetic text for the parallel measurement.
"""
from __future__ import annotations
import os
import io
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, List

import regex  # Unicode-aware; \X = extended grapheme cluster

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FLORES_DIR = DATA_DIR / "flores200_dataset"

# The languages we report on. FLORES codes are <iso639-3>_<ISO15924 script>.
# Chosen to span: Latin control (eng/spa/deu), Devanagari Indo-Aryan (hin), Bengali (ben),
# and Dravidian abugidas (tel/tam/kan) — the agglutinative, high-bytes/char cases.
LANGS: Dict[str, str] = {
    "English":  "eng_Latn",
    "Spanish":  "spa_Latn",
    "German":   "deu_Latn",
    "Hindi":    "hin_Deva",
    "Bengali":  "ben_Beng",
    "Telugu":   "tel_Telu",
    "Tamil":    "tam_Taml",
    "Kannada":  "kan_Knda",
}
REFERENCE_LANG = "English"


def ensure_flores(split: str = "devtest") -> Path:
    """Download+extract FLORES-200 once into data/. Returns the split directory."""
    split_dir = FLORES_DIR / split
    if split_dir.is_dir() and any(split_dir.glob("*." + split)):
        return split_dir
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tgz = DATA_DIR / "flores200_dataset.tar.gz"
    if not tgz.exists():
        print(f"[corpora] downloading FLORES-200 from {FLORES_URL} ...", file=sys.stderr)
        req = urllib.request.Request(FLORES_URL, headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(req, timeout=120) as r, open(tgz, "wb") as f:
            f.write(r.read())
    print("[corpora] extracting ...", file=sys.stderr)
    with tarfile.open(tgz, "r:gz") as t:
        t.extractall(DATA_DIR)
    if not split_dir.is_dir():
        # some releases nest one level; find it
        for cand in DATA_DIR.rglob(split):
            if cand.is_dir() and any(cand.glob("*." + split)):
                return cand
        raise FileNotFoundError(f"FLORES {split} dir not found after extract under {DATA_DIR}")
    return split_dir


def load_parallel(split: str = "devtest", n: int | None = None) -> Dict[str, List[str]]:
    """Return {LangName: [sentences]} with all lists parallel (same length, aligned by index)."""
    split_dir = ensure_flores(split)
    out: Dict[str, List[str]] = {}
    length = None
    for name, code in LANGS.items():
        fp = split_dir / f"{code}.{split}"
        if not fp.exists():
            raise FileNotFoundError(f"missing FLORES file {fp} (real data required, no mock)")
        lines = fp.read_text(encoding="utf-8").splitlines()
        out[name] = lines
        length = len(lines) if length is None else min(length, len(lines))
    for name in out:
        out[name] = out[name][:length]
        if n:
            out[name] = out[name][:n]
    return out


# ---------------------------------------------------------------------------
# Segmentation (atoms)
# ---------------------------------------------------------------------------
_GRAPHEME = regex.compile(r"\X")
_WORD = regex.compile(r"\p{L}[\p{L}\p{M}\p{Nd}]*|\S")


def grapheme_clusters(text: str) -> List[str]:
    """Unicode extended grapheme clusters (\\X). For abugidas this groups a consonant with its
    matras/virama into one visual akshara-like unit -> the fair per-character atom."""
    return _GRAPHEME.findall(text)


def words(text: str) -> List[str]:
    """Whitespace/script-aware word-ish tokens (letters+marks+digits runs)."""
    return _WORD.findall(text)


def utf8_bytes(text: str) -> int:
    return len(text.encode("utf-8"))

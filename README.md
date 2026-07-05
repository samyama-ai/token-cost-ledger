# The multilingual token tax: removable and irreducible

**A reproducible token-cost ledger that measures how much of the non-English "token tax" is a
removable artifact of the code versus intrinsic to the language — a synthesis + measurement study,
not a new tokenizer and not a model-quality claim.**

Large language models cost several times more tokens (and, because attention is O(N²), quadratically
more compute) to process the same content in Indic scripts than in English. This is well documented as
a phenomenon. This repo asks the narrower, useful question: **how much of that tax is removable by
choosing a better code, and how much is intrinsic to the language?** It answers by decomposing the tax,
on real parallel data, into a *removable* coding term and an *irreducible* term.

**Preprint:** *Removable and Irreducible: A Token-Cost Ledger for the Multilingual Tokenization Tax* —
arXiv (cs.CL; cross-list cs.IT), ID pending. Seeded from an exploratory conversation on the digital
"taxes" paid by non-English languages.

> **Scope fence.** This is a **compute-and-memory** accounting. We make **no** claim that fewer tokens
> improve model quality — they need not ([Schmidt et al. 2024](https://arxiv.org/abs/2402.18376)), and
> we agree. Everything here is FLOPs, KV-cache bytes, and context occupancy.

## The claim, honestly stated

On **FLORES-200** (professionally-translated parallel text) across eight languages:

- A production tokenizer (`cl100k_base`) costs up to **8.9× more tokens** for Indic scripts than for
  English (older GPT-2: **16–20× per word**).
- A **script-matched code** trained held-out on **1,012 sentences** removes a **median 64%** of that
  excess (bootstrap 95% CI **[0.638, 0.647]**).
- A **script-fair information floor** (LZMA over the grapheme-cluster stream, not UTF-8 bytes) shows the
  intrinsic content of the same sentences differs by **under 6%** across Indic languages — **the tax is
  representational, not informational.**
- Production tokenizers themselves disagree **4×** on identical Telugu content (`cl100k` 8.29× vs.
  `o200k` 1.93×) — independent evidence the tax is a property of the code.
- A **constructed** matched code removes **98%** of a controlled source's redundancy, landing **0.036
  bits above the entropy floor**; the token tax implies up to **79× attention cost**.

## What we do NOT claim (honest negatives)

- **The orthographic direction is not established.** We pre-registered that English's grapheme-to-phoneme
  ambiguity exceeds shallow Indic scripts'; we measure only the English side (CMUdict homograph entropy
  0.070 bits/type) and have no Indic pronunciation lexicon, so the cross-lingual direction is
  literature-consistent but **unmeasured** — reported as future work.
- The matched code is a **small-data demonstration** (~1k sentences, vocab 2–3k) → ρ = 0.64 is a **lower
  bound** on removability.
- The information floor is an **LZMA upper bound** on intrinsic content.
- **No new theorem:** the fertility floor `H/log₂V` and its KL-redundancy split are prior art
  ([Zouhar 2023](https://arxiv.org/abs/2306.16842); [Erdogan 2026](https://arxiv.org/abs/2601.09039));
  we cite, unify, and measure.

## Reproduce (one command)

```bash
pip install -r requirements.txt      # tiktoken, tokenizers, regex, cmudict, matplotlib (no torch/GPU)
./run.sh                             # downloads FLORES-200 (once), regenerates every number + figure
```

`run.sh` runs the NC2 calibration gate first (recovers a known entropy exactly), trains the per-language
matched BPE held-out on FLORES `dev`, measures on `devtest`, bootstraps ρ, and writes `results/*.json`
plus the three figures. Deterministic seeds; ~15 s after the one-time data download.

## Layout

```
src/entropy.py        Shannon entropy, Huffman, script-fair content bits, calibration primitives
src/corpora.py        real FLORES-200 loader + Unicode grapheme-cluster / word segmentation
src/encoders.py       UTF-8, grapheme, tiktoken (gpt2/cl100k/o200k), per-language matched BPE
src/ledger.py         the token-cost ledger (Eq. 1), rho, bootstrap CIs
src/g2p.py            English G2P ambiguity (CMUdict); Indic = literature status, not fabricated
src/constructed.py    the "Silicon Vernacular" constructed code vs the entropy floor
src/controls.py       NC1–NC4 negative controls (NC2 calibration is a hard gate)
src/run_all.py        orchestrator -> results/*.json
src/figures.py        fig1..fig3
results/              committed reproducible snapshot (JSON + figures)
```

## Citation

See `CITATION.cff`. Code Apache-2.0; the paper is CC-BY-4.0.

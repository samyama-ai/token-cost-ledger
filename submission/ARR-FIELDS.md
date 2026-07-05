# Paste-ready fields — ARR / OpenReview (MRL @ EMNLP)

Upload PDF: `submission/paper20_token_cost_ledger_acl.pdf` (anonymized, 6 pp, review mode).
Everything below is copy-paste for the OpenReview form. **You press submit.**

## Title
Removable and Irreducible: A Token-Cost Ledger for the Multilingual Tokenization Tax

## Abstract (plain text)
Large language models pay a well-documented tax on non-English text: the same content costs several
times more tokens, and because attention is quadratic in sequence length, far more compute. We ask how
much of this tax is removable. Framing the token layer as source coding—transformer compute is monotone
in sequence length, whose per-atom floor is the Shannon rate H/log2(V), an object already applied to
tokenizers in prior work—we assemble a token-cost ledger that splits each language's cost, at fixed
parallel content, into a removable coding redundancy, a residual coding slack, an intrinsic-content
term, and an orthogonal, irreducible grapheme-to-phoneme term that governs the multimodal rather than
the text cost. On FLORES-200 across eight languages, a production tokenizer costs up to 8.9x more tokens
for Indic scripts than for English; a script-matched code trained on 1,012 sentences removes a median
64% of that excess (bootstrap 95% CI [0.638, 0.647]), and a script-fair information floor shows the
intrinsic content differs by under 6%—the tax is representational, not informational. A constructed code
removes 98% of a controlled source's redundancy, and the token tax implies up to 79x attention cost. We
are explicit about scope and failure: this is compute-and-memory accounting, not a model-quality claim;
we neither measure nor claim the cross-lingual direction of the orthographic term; and our matched code
is a conservative small-data demonstration. We contribute the unifying ledger, the
removable-versus-intrinsic attribution, and an open one-command harness.

## TL;DR
We decompose the multilingual LLM tokenization tax into a removable coding term and an irreducible one,
and show on FLORES-200 that a matched code removes ~64% of it — the tax is mostly representational, not
informational.

## Keywords
multilingual tokenization; subword segmentation; tokenizer fairness; information theory; source coding;
Indic languages; compute efficiency; low-resource NLP

## Suggested area / track
Primary: Multilinguality and Language Diversity (or MRL's tokenization/efficiency track, if listed).
Secondary: Efficient methods / Resources and Evaluation.

---

# Responsible NLP Research — checklist answers

**A. General**
- A1 Limitations discussed? **Yes** — §"Limitations and honest negatives" (6 explicit items, incl. a
  pre-registered control we report as NOT established).
- A2 Potential risks? **Yes / low** — the work reduces a cross-language fairness cost; no dual-use or
  harm. Framed positively.
- A3 Do abstract & intro state the claims accurately, no overclaim? **Yes** — scope is explicitly
  compute/memory, not model quality; novelty is conceded to prior work.
- A4 AI writing assistants used? **Yes, disclose** — an AI assistant was used for drafting/engineering
  support; all claims, code, and numbers were authored/verified by the authors. (Adjust to your policy.)

**B. Use of existing artifacts**
- B1 Cited creators? **Yes** — FLORES-200 (NLLB), tiktoken, HuggingFace tokenizers, CMUdict, and all
  method prior art.
- B2 License / terms respected? **Yes** — FLORES-200 CC-BY-SA-4.0; tiktoken (MIT); tokenizers (Apache-2.0);
  CMUdict (BSD-like). Used for research, consistent with intended use.
- B3 Intended-use consistent? **Yes** — standard benchmark/evaluation use.
- B4 PII / offensive content? **No** — FLORES is curated translations of neutral text.
- B5 Documentation of artifacts? **Yes** — languages/splits stated (§3); the released harness documents
  every dependency and version.
- B6 Relevant statistics? **Yes** — 8 languages, FLORES devtest = 1,012 parallel sentences.

**C. Computational experiments**
- C1 Parameters / budget? **No training of large models.** Per-language matched BPE (vocab ≤16k) trained
  on ~1k sentences; total run ~15 s single-thread CPU.
- C2 Compute infrastructure? **CPU only, single thread; no GPU, no model weights.**
- C3 Experimental setup / hyperparameters reported? **Yes** — §3 + the open harness (seeds fixed).
- C4 Descriptive stats & significance? **Yes** — bootstrap 95% CI on the median removable fraction;
  Spearman correlation with p-value; a pre-registered calibration control.
- C5 Packages/versions reported? **Yes** — requirements.txt pins tiktoken, tokenizers, regex, cmudict,
  matplotlib; stdlib lzma.

**D. Human annotators / participants**
- **N/A** — no annotation, no crowdsourcing, no human subjects.

**E. Data / reproducibility**
- Public data (FLORES-200), one-command reproduction, deterministic seeds; harness provided as an
  anonymized supplement (repository URL withheld for review; de-anonymized at camera-ready).

---

# Anonymity double-check (before upload)
- [x] Author block = "Anonymous ACL submission"; no names/affiliation in PDF text.
- [x] No repository URL in the PDF (Reproducibility says "withheld for review").
- [ ] If ARR requires it, confirm the arXiv preprint (non-anon) is acceptable under the current ARR
      anonymity-period policy, OR delay the arXiv post until after the ARR anonymity window — **your call;
      verify the current policy.**

# Submission plan — paper20 (token-cost ledger)

## Target venue (picked)
**MRL — Workshop on Multilingual Representation Learning (co-located with EMNLP), via ACL Rolling
Review (ARR).**

**Why MRL.** The content — the multilingual token tax, cross-language fairness, scripts/orthography — is
squarely this community. The paper's honest, modest-novelty framing (it concedes the fertility floor to
Zouhar/Erdogan, `V*` to Tao, and orthographic-depth-entropy to Torres & Futrell) is a liability at a
top-tier main track but a fit at a workshop that values rigorous, reproducible measurement. ARR lets one
submission be committed to MRL **or** Findings.

**Fallbacks if MRL timing slips:** SIGTYP (typology/scripts), a dedicated Tokenization workshop if one
runs, "Insights from Negative Results in NLP", or Findings of EMNLP/ACL via the same ARR submission.

## What's in this folder
- `paper20_token_cost_ledger_acl.tex` — **anonymized, double-blind** ACL build (`\usepackage[review]{acl}`),
  two-column, **6 pages** (under the typical 8-page workshop limit, refs excluded).
- `acl.sty`, `acl_natbib.bst` — ACL style files (from github.com/acl-org/acl-style-files).
- `paper20_token_cost_ledger_acl.bib`, `fig1..3` — bibliography and figures.
- The authoritative (non-anonymous) arXiv build lives in the samyama-research paper dir
  (`papers/paper20-token-cost-ledger/paper20_token_cost_ledger.{tex,pdf}`) — do not submit that one to a
  double-blind venue. This `submission/` folder lives in the code repo (with the harness) so it stays
  out of the paper-validator suite.

## Build
```
pdflatex paper20_token_cost_ledger_acl && bibtex paper20_token_cost_ledger_acl \
  && pdflatex paper20_token_cost_ledger_acl && pdflatex paper20_token_cost_ledger_acl
```

## Camera-ready (only after acceptance) — de-anonymize
1. `\usepackage[review]{acl}` → `\usepackage[final]{acl}` (removes line numbers, un-blocks authors).
2. Restore the author block: Madhulatha Mandarapu, Sandeep Kunkunuru (VaidhyaMegha Private Limited).
3. Restore the repository URL in the Reproducibility section:
   `https://github.com/samyama-ai/token-cost-ledger`.

## Pre-submission checklist
- [x] Anonymized (author block blocked; no names; repo URL withheld; no self-identifying "our repo" text).
- [x] Under page limit (6 pp content; refs unlimited).
- [x] Current ACL template.
- [x] Limitations section present (ACL requires one) — "Limitations and honest negatives".
- [x] Reproducibility statement (public data, one command, no GPU); harness as anonymized supplement.
- [ ] **Responsible NLP / ARR checklist** — fill at submission time (data license CC-BY-SA for FLORES;
      no human subjects; no new model; compute is single-thread CPU).
- [ ] **VERIFY current ARR + MRL@EMNLP-2026 deadlines** — dates may have shifted since this was written
      (2026-07-05). Submit to the appropriate ARR cycle, then commit to MRL.
- [ ] Optional strengthening before submission (raises ceiling toward Findings): broaden beyond 8
      languages (Arabic/Thai/CJK/African scripts); measure the Indic G2P side to close NC4; add a
      tiny-transformer *cost* (not quality) experiment.

## Notes
- The paper is/also-will-be on arXiv (cs.CL). Most ACL venues permit non-anonymous preprints under the
  ARR anonymity policy, but **confirm the current policy** before/around submission.
- Human presses submit (ARR/OpenReview). This folder produces the submission-ready anonymized PDF only.

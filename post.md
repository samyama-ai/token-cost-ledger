# Announcement draft (you press send)

Invitation-framed, honest, no conquest language. Arms the arXiv link once the ID is live.

---

**Draft (LinkedIn / X):**

How much of the "tax" a language model pays on non-English text is actually *removable*?

It's well known that the same sentence costs several times more tokens in Telugu or Hindi than in
English — and because attention is quadratic, several times more compute. What's less clear is how much
of that is a fixable artifact of the tokenizer versus something intrinsic to the language.

Our new preprint treats the token layer as a source-coding problem and builds a **token-cost ledger**
that separates the two, on real parallel data (FLORES-200):

• A production tokenizer costs up to **8.9× more tokens** for Indic scripts than English.
• A script-matched code trained on just **1,012 sentences** removes a **median 64%** of that excess.
• A script-fair information floor shows the intrinsic content differs by **under 6%** — so the tax is
  mostly *representational, not informational*.

We're deliberately narrow about what we claim: this is a **compute-and-memory** accounting, not a
model-quality claim. We concede the underlying floor and optimal-vocabulary results to prior work, and
we're explicit about one control that *didn't* pan out (the cross-lingual orthographic direction, which
we couldn't measure honestly and so report as open).

Everything reproduces from one command — no GPU, no model weights.

📄 Preprint: [arXiv link — pending]
💻 Code + one-command reproduction: https://github.com/samyama-ai/token-cost-ledger

This is a first brick in a larger question we find fascinating — what *would* a near-optimal language
for foundation models look like? We'd love pushback, especially from folks working on multilingual
tokenizers and byte-level models.

#NLP #Tokenization #Multilingual #InformationTheory #LLM

---

**Tags/people to consider:** authors of Zouhar et al. (noiseless channel), Erdogan et al. (info-theoretic
tokenizers), the IndicSuperTokenizer / Indic-NLP community, byte-latent-transformer folks.

**Note:** post AFTER the arXiv ID is live (submit is a human action). cs.CL primary; a first cs.CL
submission from this account may need an endorsement — flag early.

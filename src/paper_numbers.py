"""Principle 0 for paper20: generate every number the paper quotes, from the artifact.

The paper's headline tax ratios (Telugu 8.29x, Kannada 8.9x, ...) were typed by hand. They are
derivable from results/ledger.json, but a derivable number that lives only in prose is a number
that can go stale silently. This emits results/paper_numbers.json with named keys, which
papers/paper20-token-cost-ledger/claims.yaml resolves.

Recomputes nothing: it only names what run_all.py already measured.
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
ledger = json.load(open(os.path.join(RES, "ledger.json")))
summary = json.load(open(os.path.join(RES, "summary.json")))

prod = summary["config"]["production"]
en_tokens = ledger["English"]["tokens"][prod]

raw_tax = {L: ledger[L]["tokens"][prod] / en_tokens for L in ledger}
boot = summary["H1_removable"]["bootstrap_median_rho"]

# every encoder the paper quotes a ratio for: cl100k (production), o200k, and the matched 16k vocab
tax_by_encoder = {
    enc: {L: ledger[L]["tokens"][enc] / ledger["English"]["tokens"][enc] for L in ledger}
    for enc in ledger["English"]["tokens"]
}

out = {
    "production_tokenizer": prod,
    "english_tokens": en_tokens,
    "raw_tax_per_lang": raw_tax,
    "tax_by_encoder": tax_by_encoder,
    "nsl_floor_per_lang": summary["nc3_content_invariance"]["nsl_floor"],
    "raw_tax_max_lang": max(raw_tax, key=raw_tax.get),
    "raw_tax_max": max(raw_tax.values()),
    "median_rho": boot["median_rho"],
    "rho_ci_lo": boot["ci_lo"],
    "rho_ci_hi": boot["ci_hi"],
    "h2_spearman_rho": summary["H2_bpc_predicts_tax"]["rho"],
    "nc1_kraft_sum_leq_1": summary["nc2_calibration"]["kraft_sum_leq_1"],
    "nc1_no_subfloor": summary["nc2_calibration"]["no_subfloor"],
    "nc2_pass": summary["nc2_calibration"]["PASS"],
    "nc3_pass": summary["nc3_content_invariance"]["PASS"],
    "nc3_languages_over_threshold": list(summary["nc3_content_invariance"]["over_threshold"]),
    "nc4_pass": summary["nc4_g2p_direction"]["PASS"],
}

path = os.path.join(RES, "paper_numbers.json")
json.dump(out, open(path, "w"), indent=1)
print(f"wrote {path}")
for L, v in sorted(raw_tax.items(), key=lambda kv: -kv[1])[:4]:
    print(f"  raw tax {L:9} {v:.3f}x")
print(f"  median rho {out['median_rho']:.4f}  CI [{out['rho_ci_lo']:.4f}, {out['rho_ci_hi']:.4f}]")
print(f"  NC1 kraft={out['nc1_kraft_sum_leq_1']} no_subfloor={out['nc1_no_subfloor']}  "
      f"NC2={out['nc2_pass']} NC3={out['nc3_pass']} NC4={out['nc4_pass']}")

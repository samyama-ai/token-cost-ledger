"""
One-command harness: regenerates every number and figure from REAL data (FLORES-200).

  train matched BPE on FLORES `dev`  ->  measure on FLORES `devtest`  (held out).
  outputs: results/*.json  and  ../paper figures via figures.py (called separately by run.sh).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpora import load_parallel, LANGS, REFERENCE_LANG
from encoders import build_fixed_encoders, train_matched_bpe
from ledger import build_ledgers, bootstrap_median_rho
from controls import nc2_calibration, nc3_content_invariance, nc4_g2p_direction
from constructed import analyze as constructed_analyze
from g2p import g2p_table

RESULTS = Path(__file__).resolve().parent.parent / "results"
MATCHED_VOCAB = 16000
PRODUCTION = "cl100k_base"
INDIC = ["Hindi", "Bengali", "Telugu", "Tamil", "Kannada"]


def main(n_eval: int | None = None):
    RESULTS.mkdir(exist_ok=True)

    print("[run] NC2 calibration (hard gate) ...")
    nc2 = nc2_calibration()
    if not nc2["PASS"]:
        json.dump(nc2, open(RESULTS / "controls.json", "w"), indent=2)
        raise SystemExit("NC2 CALIBRATION FAILED -- measurement stack untrustworthy. STOP.")
    print(f"       NC2 PASS  (uniform32 H_est={nc2['uniform32']['H_est']:.4f}; "
          f"skewed8 huffman={nc2['skewed8']['huffman_avg']:.4f})")

    print("[run] loading FLORES-200 (dev=train, devtest=eval) ...")
    train = load_parallel("dev")
    evalc = load_parallel("devtest", n=n_eval)

    print("[run] building fixed encoders ...")
    fixed = build_fixed_encoders()

    print(f"[run] training per-language matched BPE (V={MATCHED_VOCAB}) on dev ...")
    matched = {}
    for lang in LANGS:
        matched[lang] = train_matched_bpe(train[lang], MATCHED_VOCAB, f"matched-{lang}")
        print(f"       matched-{lang}: vocab={matched[lang].vocab_size}")

    print("[run] building ledgers on devtest ...")
    ledgers = build_ledgers(evalc, fixed, matched, production=PRODUCTION, reference=REFERENCE_LANG)

    # compute Θ(N²) attention proxy per language, vs English, under production AND matched codes
    prod_enc = fixed[PRODUCTION]
    ref_sq = sum(c * c for c in prod_enc.token_counts(evalc[REFERENCE_LANG]))
    attn_proxy = {lg: sum(c * c for c in prod_enc.token_counts(evalc[lg])) / max(ref_sq, 1)
                  for lg in evalc}
    ref_sq_m = sum(c * c for c in matched[REFERENCE_LANG].token_counts(evalc[REFERENCE_LANG]))
    attn_proxy_matched = {lg: sum(c * c for c in matched[lg].token_counts(evalc[lg])) / max(ref_sq_m, 1)
                          for lg in evalc}

    print("[run] bootstrap median rho (Indic) ...")
    boot = bootstrap_median_rho(evalc, fixed, matched, INDIC,
                                production=PRODUCTION, reference=REFERENCE_LANG, B=300)

    print("[run] controls NC3/NC4, constructed encoding, G2P ...")
    g2p = g2p_table()
    nc3 = nc3_content_invariance(ledgers, REFERENCE_LANG)
    nc4 = nc4_g2p_direction(g2p, REFERENCE_LANG)
    constructed = constructed_analyze()

    # serialize
    led_out = {lg: {
        "n_sent": L.n_sent, "clusters": L.clusters, "words": L.words, "utf8_bytes": L.utf8,
        "bytes_per_cluster": L.bpc,
        "tokens": L.tokens, "fertility_per_word": L.fert_word, "nsl": L.nsl,
        "content_bits": L.content_bits, "nsl_floor": L.nsl_floor,
        "nsl_matched": L.nsl_matched, "matched_vocab": L.matched_vocab, "rho": L.rho,
        "attn_proxy_vs_eng": attn_proxy[lg],
        "attn_proxy_matched_vs_eng": attn_proxy_matched[lg],
    } for lg, L in ledgers.items()}

    summary = {
        "config": {"production": PRODUCTION, "matched_vocab": MATCHED_VOCAB,
                   "n_eval_sentences": ledgers[REFERENCE_LANG].n_sent, "indic": INDIC},
        "H1_removable": {
            "rho_per_lang": {lg: ledgers[lg].rho for lg in INDIC},
            "bootstrap_median_rho": boot,
            "PASS_ge_0.5": boot["median_rho"] >= 0.5 and boot["ci_lo"] > 0.25,
        },
        "H2_bpc_predicts_tax": _spearman(
            [ledgers[lg].bpc for lg in evalc],
            [ledgers[lg].nsl[PRODUCTION] for lg in evalc]),
        "nc2_calibration": nc2, "nc3_content_invariance": nc3, "nc4_g2p_direction": nc4,
    }

    json.dump(led_out, open(RESULTS / "ledger.json", "w"), indent=2, ensure_ascii=False)
    json.dump({"nc2": nc2, "nc3": nc3, "nc4": nc4}, open(RESULTS / "controls.json", "w"), indent=2)
    json.dump(constructed, open(RESULTS / "constructed.json", "w"), indent=2)
    json.dump(g2p, open(RESULTS / "g2p.json", "w"), indent=2, ensure_ascii=False)
    json.dump(summary, open(RESULTS / "summary.json", "w"), indent=2, ensure_ascii=False)

    print("\n==== SUMMARY ====")
    print(f"H1 median rho (Indic) = {boot['median_rho']:.3f}  CI[{boot['ci_lo']:.3f},{boot['ci_hi']:.3f}]"
          f"  -> {'PASS' if summary['H1_removable']['PASS_ge_0.5'] else 'NEGATIVE'}")
    print(f"H2 spearman(bpc, NSL_prod) = {summary['H2_bpc_predicts_tax']['rho']:.3f} "
          f"(p={summary['H2_bpc_predicts_tax']['p']:.4f})")
    print(f"NC4 English G2P ambiguity measured = {nc4['english_bits_measured']:.3f} bits; "
          f"direction established = {nc4['direction_independently_established']}")
    for lg in evalc:
        L = ledgers[lg]
        print(f"  {lg:8s} bpc={L.bpc:4.2f}  NSL_prod={L.nsl[PRODUCTION]:.2f}  "
              f"NSL_matched={L.nsl_matched:.2f}  NSL_floor={L.nsl_floor:.2f}  rho={L.rho:.2f}")
    return summary


def _spearman(x, y):
    import math
    n = len(x)
    rx = _rank(x); ry = _rank(y)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    rho = 1 - 6 * d2 / (n * (n * n - 1)) if n > 1 else float("nan")
    # approximate two-sided p via t-approx
    if n > 2 and abs(rho) < 1:
        t = rho * math.sqrt((n - 2) / (1 - rho * rho))
        p = 2 * (1 - _t_cdf(abs(t), n - 2))
    else:
        p = float("nan")
    return {"rho": rho, "p": p, "n": n}


def _rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def _t_cdf(t, df):
    import math
    x = df / (df + t * t)
    return 1 - 0.5 * _betai(df / 2.0, 0.5, x)


def _betai(a, b, x):
    import math
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) / a
    # continued fraction (Lentz)
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 200):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        f *= d * c
        if abs(1 - d * c) < 1e-8:
            break
    return front * (f - 1.0)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=None, help="cap eval sentences (default all)")
    a = ap.parse_args()
    main(a.n_eval)

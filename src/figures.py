"""
Generate the three paper figures from results/*.json. Deterministic; no data recomputation.
Writes fig1..fig3 into the paper directory (passed as argv[1]) AND the local results/ dir.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parent.parent / "results"
ORDER = ["English", "Spanish", "German", "Hindi", "Bengali", "Telugu", "Tamil", "Kannada"]
INDIC = ["Hindi", "Bengali", "Telugu", "Tamil", "Kannada"]
PROD = "cl100k_base"

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 130, "savefig.bbox": "tight"})


def _load():
    led = json.load(open(RES / "ledger.json"))
    summ = json.load(open(RES / "summary.json"))
    con = json.load(open(RES / "constructed.json"))
    g2p = json.load(open(RES / "g2p.json"))
    return led, summ, con, g2p


def fig1_token_tax(led, out):
    langs = [l for l in ORDER if l in led]
    prod = [led[l]["nsl"][PROD] for l in langs]
    matched = [led[l]["nsl_matched"] for l in langs]
    floor = [led[l]["nsl_floor"] for l in langs]
    x = range(len(langs))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar([i - w for i in x], prod, w, label=f"production BPE ({PROD})", color="#c1443c")
    ax.bar([i for i in x], matched, w, label="matched code (per-lang BPE)", color="#e2a33b")
    ax.bar([i + w for i in x], floor, w, label="information floor (LZMA)", color="#3b7ea1")
    ax.axhline(1.0, ls="--", lw=1, color="k", alpha=0.6)
    ax.set_xticks(list(x)); ax.set_xticklabels(langs, rotation=25, ha="right")
    ax.set_ylabel("normalized sequence length\n(English = 1)")
    ax.set_title("The multilingual token tax and how much of it is removable")
    ax.legend(fontsize=8, loc="upper left")
    fig.savefig(out / "fig1_token_tax.png"); fig.savefig(RES / "fig1_token_tax.png")
    plt.close(fig)


def fig2_decomposition(led, summ, out):
    langs = [l for l in INDIC if l in led]
    removable, slack, intrinsic = [], [], []
    for l in langs:
        p = led[l]["nsl"][PROD]; m = led[l]["nsl_matched"]; f = led[l]["nsl_floor"]
        removable.append(max(p - m, 0.0))
        slack.append(max(m - f, 0.0))
        intrinsic.append(max(f - 1.0, 0.0))
    x = range(len(langs))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    b1 = ax.bar(x, removable, 0.6, label="removable (vocabulary mismatch)", color="#c1443c")
    b2 = ax.bar(x, slack, 0.6, bottom=removable, label="residual coding slack", color="#e2a33b")
    bottom2 = [removable[i] + slack[i] for i in x]
    ax.bar(x, intrinsic, 0.6, bottom=bottom2, label="intrinsic content", color="#3b7ea1")
    ax.set_xticks(list(x)); ax.set_xticklabels(langs, rotation=20, ha="right")
    ax.set_ylabel("excess sequence length over English\n(NSL − 1, decomposed)")
    med = summ["H1_removable"]["bootstrap_median_rho"]
    ax.set_title(f"Ledger decomposition of the Indic token tax "
                 f"(median removable ρ={med['median_rho']:.2f})")
    ax.set_ylim(0, max([removable[i] + slack[i] + intrinsic[i] for i in x]) * 1.28)
    ax.legend(fontsize=8, loc="upper left")
    fig.savefig(out / "fig2_decomposition.png"); fig.savefig(RES / "fig2_decomposition.png")
    plt.close(fig)


def fig3_constructed_and_compute(led, con, out):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.6, 3.4))
    # left: mismatched fixed-width vs matched Huffman vs the entropy floor (bits/concept)
    H = con["floor_H_bits_per_concept"]
    fw = con["mismatched_fixed_width_bits_per_concept"]
    matched = con["matched_symbol_bits_per_concept"]
    bars = [fw, matched, H]
    labels = ["mismatched\nfixed-width", "Silicon Vernacular\n(matched Huffman)", "floor\nH=H(p)"]
    cols = ["#c1443c", "#e2a33b", "#3b7ea1"]
    axL.bar(range(3), bars, 0.6, color=cols)
    axL.axhline(H, ls="--", lw=1.2, color="k", alpha=0.6)
    for i, v in enumerate(bars):
        axL.text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=8)
    axL.set_xticks(range(3)); axL.set_xticklabels(labels, fontsize=7.5)
    axL.set_ylabel("bits / concept")
    axL.set_ylim(0, fw * 1.2)
    frac = con["redundancy_removed_frac"]
    axL.set_title(f"Constructed code removes {frac*100:.0f}% of redundancy")
    # right: Θ(N²) attention-compute proxy, production vs matched, vs English=1
    langs = [l for l in ORDER if l in led]
    prod = [led[l]["attn_proxy_vs_eng"] for l in langs]
    matched = [led[l]["attn_proxy_matched_vs_eng"] for l in langs]
    x = range(len(langs)); w = 0.38
    axR.bar([i - w / 2 for i in x], prod, w, label="production BPE", color="#c1443c")
    axR.bar([i + w / 2 for i in x], matched, w, label="matched code", color="#3b7ea1")
    axR.axhline(1.0, ls="--", lw=1, color="k", alpha=0.6)
    axR.set_yscale("log")
    axR.set_xticks(list(x)); axR.set_xticklabels(langs, rotation=30, ha="right", fontsize=8)
    axR.set_ylabel("attention Θ(N²) proxy (English = 1)")
    axR.set_title("Quadratic amplification of the tax")
    axR.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig3_constructed_compute.png"); fig.savefig(RES / "fig3_constructed_compute.png")
    plt.close(fig)


def main(paper_dir: str):
    out = Path(paper_dir)
    out.mkdir(parents=True, exist_ok=True)
    led, summ, con, g2p = _load()
    fig1_token_tax(led, out)
    fig2_decomposition(led, summ, out)
    fig3_constructed_and_compute(led, con, out)
    print(f"[figures] wrote fig1..fig3 to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(RES))

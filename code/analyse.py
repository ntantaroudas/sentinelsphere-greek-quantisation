#!/usr/bin/env python3
"""Aggregate Paper C per item results.

Outputs, all under ../results
  accuracy_summary.json      accuracy with Wilson intervals, speed, memory, per model and language
  tableC1_accuracy.csv       the same as a table, sizes in decimal gigabytes
  tableC2_paired.csv         paired English versus Greek counts with exact McNemar p values,
                             mean prompt tokens per language and the Greek to English token ratio
  tableC3_quant_paired.csv   paired comparisons between Phi-4-mini quantisation levels
  tableC4_fidelity.csv       accuracy on items with high and lower back translation chrF
  letter_distribution.csv    share of each predicted option letter, to expose position bias
  figures/figC1_phi4_quant   Phi-4-mini accuracy versus effective bits per weight
  figures/figC2_q4_models    English and Greek accuracy for the three four bit models
"""
import csv
import glob
import json
import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # figstyle.py sits next to this script
import figstyle                                               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(RES, "figures")
DATA = os.path.join(HERE, "..", "data")
# total parameter count of Phi-4-mini-instruct as reported by llama_model_n_params on the GGUF
PHI_PARAMS = 3836021856
FIDELITY_CUT = 60.0
NICE = {"phi4mini_Q2_K": "Phi-4-mini Q2_K", "phi4mini_Q4_K_M": "Phi-4-mini Q4_K_M",
        "phi4mini_Q8_0": "Phi-4-mini Q8_0", "qwen2.5-3b_Q4_K_M": "Qwen2.5-3B Q4_K_M",
        "krikri-8b_Q4_K_M": "Krikri-8B Q4_K_M",
        "phi4mini_Q4_K_M_static": "Phi-4-mini Q4_K_M static", "phi4mini_Q2_K_static": "Phi-4-mini Q2_K static",
        "phi4mini_Q4_K_M_deployed": "Phi-4-mini Q4_K_M deployed"}


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, c - h, c + h


def mcnemar_exact(b, c):
    """Two sided exact McNemar test on the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def boot_gap(en, el, ids, other=None, b=10000, seed=2026):
    """Paired bootstrap over items of the English minus Greek gap, or, with other=(en2, el2),
    of the difference between two gaps measured on the same items. Returns the 2.5 and 97.5
    percentiles."""
    import numpy as np
    rng = np.random.default_rng(seed)
    e = np.array([en[i]["correct"] for i in ids], float)
    g = np.array([el[i]["correct"] for i in ids], float)
    d = e - g
    if other is not None:
        e2 = np.array([other[0][i]["correct"] for i in ids], float)
        g2 = np.array([other[1][i]["correct"] for i in ids], float)
        d = d - (e2 - g2)
    idx = rng.integers(0, len(ids), size=(b, len(ids)))
    stats = d[idx].mean(axis=1)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def load():
    per, summ = {}, {}
    for path in glob.glob(os.path.join(RES, "mcq_*.jsonl")):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r.get("summary"):
                summ[r["label"]] = r
            elif "item" in r:
                per.setdefault((r["label"], r["lang"]), {})[r["item"]] = r
    return per, summ


def main():
    os.makedirs(FIG, exist_ok=True)
    figstyle.apply()
    per, summ = load()
    labels = sorted({lab for lab, _ in per})

    # accuracy, speed and memory
    rows = []
    for (label, lang), items in sorted(per.items()):
        vals = list(items.values())
        k = sum(it["correct"] for it in vals)
        n = len(vals)
        p, lo, hi = wilson(k, n)
        sec = sorted(it["seconds"] for it in vals)
        s = summ.get(label, {})
        rows.append({"label": label, "lang": lang, "n": n, "correct": k, "acc": p, "lo": lo, "hi": hi,
                     "median_s_per_item": sec[len(sec) // 2],
                     "model_gb": s.get("model_bytes", 0) / 1e9,
                     "peak_rss_gb": s.get("peak_rss_kb", 0) * 1024 / 1e9,
                     "load_s": s.get("load_seconds"),
                     "eff_bpw": (s.get("model_bytes", 0) * 8 / PHI_PARAMS) if label.startswith("phi4mini") else None})
    json.dump(rows, open(os.path.join(RES, "accuracy_summary.json"), "w"), indent=1)
    with open(os.path.join(RES, "tableC1_accuracy.csv"), "w") as fh:
        fh.write("model,lang,n,correct,accuracy,ci_low,ci_high,median_s,model_GB,peak_RSS_GB,effective_bpw\n")
        for r in rows:
            bpw = f"{r['eff_bpw']:.2f}" if r["eff_bpw"] else ""
            fh.write(f"{r['label']},{r['lang']},{r['n']},{r['correct']},{r['acc']:.3f},{r['lo']:.3f},"
                     f"{r['hi']:.3f},{r['median_s_per_item']:.2f},{r['model_gb']:.2f},{r['peak_rss_gb']:.2f},{bpw}\n")

    # paired English versus Greek, token fertility
    paired = []
    for lab in labels:
        en, el = per.get((lab, "en"), {}), per.get((lab, "el"), {})
        ids = sorted(set(en) & set(el))
        both = sum(en[i]["correct"] and el[i]["correct"] for i in ids)
        en_only = sum(en[i]["correct"] and not el[i]["correct"] for i in ids)
        el_only = sum((not en[i]["correct"]) and el[i]["correct"] for i in ids)
        neither = len(ids) - both - en_only - el_only
        tok_en = sum(en[i]["prompt_tokens"] for i in ids) / len(ids)
        tok_el = sum(el[i]["prompt_tokens"] for i in ids) / len(ids)
        glo, ghi = boot_gap(en, el, ids)
        paired.append({"label": lab, "n": len(ids), "both": both, "en_only": en_only, "el_only": el_only,
                       "neither": neither, "gap": (en_only - el_only) / len(ids), "gap_lo": glo, "gap_hi": ghi,
                       "p_mcnemar": mcnemar_exact(en_only, el_only),
                       "tokens_en": tok_en, "tokens_el": tok_el, "token_ratio": tok_el / tok_en,
                       "sec_en": sorted(en[i]["seconds"] for i in ids)[len(ids) // 2],
                       "sec_el": sorted(el[i]["seconds"] for i in ids)[len(ids) // 2]})
    with open(os.path.join(RES, "tableC2_paired.csv"), "w") as fh:
        fh.write("model,n,both_correct,en_only,el_only,neither,gap_en_minus_el,gap_ci_low,gap_ci_high,p_mcnemar,"
                 "mean_prompt_tokens_en,mean_prompt_tokens_el,token_ratio_el_over_en,time_ratio_el_over_en\n")
        for p in paired:
            fh.write(f"{p['label']},{p['n']},{p['both']},{p['en_only']},{p['el_only']},{p['neither']},"
                     f"{p['gap']:.3f},{p['gap_lo']:.3f},{p['gap_hi']:.3f},{p['p_mcnemar']:.2e},"
                     f"{p['tokens_en']:.1f},{p['tokens_el']:.1f},{p['token_ratio']:.2f},"
                     f"{p['sec_el'] / p['sec_en']:.2f}\n")

    # does the gap widen as Phi-4-mini is quantised, paired bootstrap of gap differences
    gapdiff = {}
    for hi_l, lo_l in (("phi4mini_Q8_0", "phi4mini_Q4_K_M"), ("phi4mini_Q4_K_M", "phi4mini_Q2_K"),
                       ("phi4mini_Q8_0", "phi4mini_Q2_K"),
                       ("phi4mini_Q4_K_M_static", "phi4mini_Q4_K_M"), ("phi4mini_Q2_K_static", "phi4mini_Q2_K"),
                       ("phi4mini_Q4_K_M_deployed", "phi4mini_Q4_K_M")):
        if (hi_l, "en") in per and (lo_l, "en") in per:
            a_en, a_el, b_en, b_el = per[(lo_l, "en")], per[(lo_l, "el")], per[(hi_l, "en")], per[(hi_l, "el")]
            ids = sorted(set(a_en) & set(a_el) & set(b_en) & set(b_el))
            d = (sum(a_en[i]["correct"] - a_el[i]["correct"] for i in ids)
                 - sum(b_en[i]["correct"] - b_el[i]["correct"] for i in ids)) / len(ids)
            lo, hi = boot_gap(a_en, a_el, ids, other=(b_en, b_el))
            gapdiff[f"{lo_l}_minus_{hi_l}"] = {"gap_increase": d, "ci_low": lo, "ci_high": hi}
    json.dump(gapdiff, open(os.path.join(RES, "gap_widening.json"), "w"), indent=1)

    # paired comparisons between Phi-4-mini quantisation levels
    qpairs = [("phi4mini_Q8_0", "phi4mini_Q4_K_M"), ("phi4mini_Q4_K_M", "phi4mini_Q2_K"),
              ("phi4mini_Q8_0", "phi4mini_Q2_K"),
              ("phi4mini_Q4_K_M_static", "phi4mini_Q4_K_M"), ("phi4mini_Q2_K_static", "phi4mini_Q2_K"),
              ("phi4mini_Q4_K_M_deployed", "phi4mini_Q4_K_M")]
    with open(os.path.join(RES, "tableC3_quant_paired.csv"), "w") as fh:
        fh.write("lang,higher,lower,acc_higher,acc_higher_only,lower_only,acc_lower,p_mcnemar\n")
        for lang in ("en", "el"):
            for hi_l, lo_l in qpairs:
                a, b = per.get((hi_l, lang), {}), per.get((lo_l, lang), {})
                ids = sorted(set(a) & set(b))
                if not ids:
                    continue
                h_only = sum(a[i]["correct"] and not b[i]["correct"] for i in ids)
                l_only = sum((not a[i]["correct"]) and b[i]["correct"] for i in ids)
                fh.write(f"{lang},{hi_l},{lo_l},{sum(a[i]['correct'] for i in ids)/len(ids):.3f},{h_only},"
                         f"{l_only},{sum(b[i]['correct'] for i in ids)/len(ids):.3f},"
                         f"{mcnemar_exact(h_only, l_only):.2e}\n")

    # robustness to translation fidelity
    chrf = {}
    with open(os.path.join(DATA, "translation_quality.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            chrf[int(r["id"])] = float(r["chrf_item"])
    hi_ids = {i for i, v in chrf.items() if v >= FIDELITY_CUT}
    lo_ids = set(chrf) - hi_ids
    with open(os.path.join(RES, "tableC4_fidelity.csv"), "w") as fh:
        fh.write(f"model,n_high,acc_en_high,acc_el_high,gap_high,n_low,acc_en_low,acc_el_low,gap_low\n")
        for lab in labels:
            en, el = per.get((lab, "en"), {}), per.get((lab, "el"), {})

            def acc(d, ids):
                ids = [i for i in ids if i in d]
                return sum(d[i]["correct"] for i in ids) / len(ids), len(ids)
            aeh, nh = acc(en, hi_ids)
            aelh, _ = acc(el, hi_ids)
            ael, nl = acc(en, lo_ids)
            aell, _ = acc(el, lo_ids)
            fh.write(f"{lab},{nh},{aeh:.3f},{aelh:.3f},{aeh-aelh:.3f},{nl},{ael:.3f},{aell:.3f},{ael-aell:.3f}\n")

    # predicted letter distribution, a check for collapse onto one option
    with open(os.path.join(RES, "letter_distribution.csv"), "w") as fh:
        fh.write("model,lang,A,B,C,D,max_share\n")
        gold = Counter(it["gold"] for it in per[(labels[0], "en")].values())
        n = sum(gold.values())
        fh.write("gold,both," + ",".join(f"{gold[x]/n:.3f}" for x in "ABCD") +
                 f",{max(gold.values())/n:.3f}\n")
        for (lab, lang), items in sorted(per.items()):
            c = Counter(it["pred"] for it in items.values())
            n = sum(c.values())
            fh.write(f"{lab},{lang}," + ",".join(f"{c[x]/n:.3f}" for x in "ABCD") +
                     f",{max(c.values())/n:.3f}\n")

    import matplotlib.pyplot as plt
    # Figure 1, Phi-4-mini accuracy versus effective bits per weight
    phi = [r for r in rows if r["label"].startswith("phi4mini_")
           and not r["label"].endswith(("_static", "_deployed"))]
    static = [r for r in rows if r["label"].endswith("_static")]
    deployed = [r for r in rows if r["label"].endswith("_deployed")]
    fig, ax = plt.subplots(figsize=(figstyle.WIDTH_IN, 3.8))
    for lang, col, mk in (("en", figstyle.OKABE["blue"], "o"), ("el", figstyle.OKABE["orange"], "s")):
        pts = sorted((r["eff_bpw"], r) for r in phi if r["lang"] == lang)
        xs = [b for b, _ in pts]
        ys = [r["acc"] for _, r in pts]
        err = [[r["acc"] - r["lo"] for _, r in pts], [r["hi"] - r["acc"] for _, r in pts]]
        ax.errorbar(xs, ys, yerr=err, marker=mk, color=col, capsize=4,
                    label="English" if lang == "en" else "Greek")
        for b, r in pts:
            ax.annotate(r["label"].split("_", 1)[1], (b, r["acc"]), textcoords="offset points",
                        xytext=(6, -12 if lang == "el" else 6), fontsize=8, color=col)
    ax.axhline(0.25, color=figstyle.OKABE["grey"], lw=0.9, ls="--")
    ax.text(6.8, 0.262, "chance, one in four", ha="center", fontsize=8, color=figstyle.OKABE["grey"])
    ax.set_xlabel("effective bits per weight (file size over parameter count)")
    ax.set_ylabel("accuracy, 120 CyberMetric items")
    ax.set_ylim(0.15, 1.08)
    ax.set_title("Phi-4-mini-instruct, accuracy versus quantisation")
    ax.legend(loc="upper center", ncol=2, fontsize=8)
    figstyle.save(fig, os.path.join(FIG, "figC1_phi4_quant"))

    # Figure 3, files with and without an importance matrix
    byk = {(r["label"], r["lang"]): r for r in rows}
    abl = [("phi4mini_Q4_K_M", "Q4_K_M\ncommunity,\nimportance\nmatrix"),
           ("phi4mini_Q4_K_M_deployed", "Q4_K_M\nplatform file,\nnone\nrecorded"),
           ("phi4mini_Q4_K_M_static", "Q4_K_M\nstatic,\nrequantised\nhere"),
           ("phi4mini_Q2_K", "Q2_K\ncommunity,\nimportance\nmatrix"),
           ("phi4mini_Q2_K_static", "Q2_K\nstatic,\nrequantised\nhere")]
    abl = [(l, t) for l, t in abl if (l, "en") in byk and (l, "el") in byk]
    if len(abl) > 2:
        fig, ax = plt.subplots(figsize=(figstyle.WIDTH_IN, 4.0))
        for j, (lang, col, mk) in enumerate((("en", figstyle.OKABE["blue"], "o"), ("el", figstyle.OKABE["orange"], "s"))):
            xs = [i + (j - 0.5) * 0.22 for i in range(len(abl))]
            ys = [byk[(l, lang)]["acc"] for l, _ in abl]
            err = [[byk[(l, lang)]["acc"] - byk[(l, lang)]["lo"] for l, _ in abl],
                   [byk[(l, lang)]["hi"] - byk[(l, lang)]["acc"] for l, _ in abl]]
            ax.errorbar(xs, ys, yerr=err, fmt=mk, color=col, capsize=4, ls="none",
                        label="English" if lang == "en" else "Greek")
        ax.axvline(2.5, color=figstyle.OKABE["grey"], lw=0.8, ls=":")
        ax.axhline(0.25, color=figstyle.OKABE["grey"], lw=0.9, ls="--")
        ax.set_xticks(range(len(abl)))
        ax.set_xticklabels([t for _, t in abl], fontsize=8)
        ax.set_ylabel("accuracy, 120 CyberMetric items")
        ax.set_ylim(0.15, 1.0)
        ax.set_title("Phi-4-mini files with and without an importance matrix")
        ax.legend(loc="upper right", ncol=2, fontsize=8)
        figstyle.save(fig, os.path.join(FIG, "figC3_importance_matrix"))

    # Figure 2, English and Greek accuracy for the three four bit models
    q4 = [l for l in ("phi4mini_Q4_K_M", "qwen2.5-3b_Q4_K_M", "krikri-8b_Q4_K_M") if (l, "en") in per]
    byk = {(r["label"], r["lang"]): r for r in rows}
    fig, ax = plt.subplots(figsize=(figstyle.WIDTH_IN, 3.8))
    w = 0.36
    for j, (lang, col) in enumerate((("en", figstyle.OKABE["blue"]), ("el", figstyle.OKABE["orange"]))):
        xs = [i + (j - 0.5) * w for i in range(len(q4))]
        ys = [byk[(l, lang)]["acc"] for l in q4]
        err = [[byk[(l, lang)]["acc"] - byk[(l, lang)]["lo"] for l in q4],
               [byk[(l, lang)]["hi"] - byk[(l, lang)]["acc"] for l in q4]]
        ax.bar(xs, ys, width=w, color=col, yerr=err, capsize=3,
               label="English" if lang == "en" else "Greek")
    ax.axhline(0.25, color=figstyle.OKABE["grey"], lw=0.9, ls="--")
    ax.set_xticks(range(len(q4)))
    ax.set_xticklabels([NICE[l] for l in q4])
    ax.set_ylabel("accuracy, 120 CyberMetric items")
    ax.set_ylim(0, 1.0)
    ax.set_title("Three models at the four bit Q4_K_M setting")
    ax.legend(loc="upper left", ncol=2)
    figstyle.save(fig, os.path.join(FIG, "figC2_q4_models"))

    print(f"Paper C, {len(rows)} model and language rows aggregated, high fidelity items {len(hi_ids)}")
    for r in rows:
        print(f"  {r['label']:22s} {r['lang']} acc {r['acc']:.3f} [{r['lo']:.3f},{r['hi']:.3f}] "
              f"{r['median_s_per_item']:.2f}s/item {r['model_gb']:.2f}GB peak {r['peak_rss_gb']:.2f}GB")


if __name__ == "__main__":
    main()

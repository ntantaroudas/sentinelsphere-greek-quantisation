#!/usr/bin/env python3
"""Build the English and Greek multiple choice sets for Paper C.

1. Draw a fixed random sample of CyberMetric-500 items (seed 2026).
2. Translate question and options into Greek with the open Marian model
   Helsinki-NLP/opus-mt-tc-big-en-el (no evaluated model is used for translation).
3. Translate the Greek back into English with Helsinki-NLP/opus-mt-grk-en and score
   each item with chrF against the original, as an automatic fidelity check.
Outputs to ../data:  items_en.json, items_el.json, translation_quality.csv
Needs: torch (CPU), transformers, sentencepiece, sacremoses, sacrebleu.
"""
import csv
import json
import os
import random
import sys

import sacrebleu
from transformers import MarianMTModel, MarianTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data")
SRC = os.environ.get("CYBERMETRIC_500", os.path.join(HERE, "..", "data", "CyberMetric-500-v1.json"))
N_ITEMS = int(os.environ.get("N_ITEMS", "120"))
EN_EL = os.environ.get("EN_EL_MODEL", "Helsinki-NLP/opus-mt-tc-big-en-el")
EL_EN = os.environ.get("EL_EN_MODEL", "Helsinki-NLP/opus-mt-grk-en")


def translator(name):
    tok = MarianTokenizer.from_pretrained(name)
    model = MarianMTModel.from_pretrained(name)
    model.eval()

    def run(texts, batch=16):
        out = []
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], return_tensors="pt", padding=True, truncation=True, max_length=256)
            gen = model.generate(**enc, num_beams=1, max_new_tokens=160)
            out.extend(tok.batch_decode(gen, skip_special_tokens=True))
        return out
    return run


def main():
    os.makedirs(OUT, exist_ok=True)
    items = json.load(open(SRC))["questions"]
    rng = random.Random(2026)
    idx = sorted(rng.sample(range(len(items)), N_ITEMS))
    sample = [{"id": i, "question": items[i]["question"], "answers": items[i]["answers"],
               "solution": items[i]["solution"]} for i in idx]
    json.dump(sample, open(os.path.join(OUT, "items_en.json"), "w"), ensure_ascii=False, indent=1)

    segments = []
    for it in sample:
        segments.append(it["question"])
        segments.extend(it["answers"][k] for k in "ABCD")
    print(f"translating {len(segments)} segments", file=sys.stderr)
    el = translator(EN_EL)(segments)
    back = translator(EL_EN)(el)

    greek, rows = [], []
    for j, it in enumerate(sample):
        q_el, opts_el = el[5 * j], el[5 * j + 1: 5 * j + 5]
        q_back, opts_back = back[5 * j], back[5 * j + 1: 5 * j + 5]
        greek.append({"id": it["id"], "question": q_el, "answers": dict(zip("ABCD", opts_el)),
                      "solution": it["solution"]})
        src = it["question"] + " " + " ".join(it["answers"][k] for k in "ABCD")
        hyp = q_back + " " + " ".join(opts_back)
        rows.append({"id": it["id"], "chrf_item": sacrebleu.sentence_chrf(hyp, [src]).score,
                     "chrf_question": sacrebleu.sentence_chrf(q_back, [it["question"]]).score,
                     "question_en": it["question"], "question_el": q_el, "question_back": q_back})
    json.dump(greek, open(os.path.join(OUT, "items_el.json"), "w"), ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "translation_quality.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    allsrc = [r["question_en"] for r in rows]
    allhyp = [r["question_back"] for r in rows]
    print(json.dumps({"items": N_ITEMS, "corpus_chrf_questions": sacrebleu.corpus_chrf(allhyp, [allsrc]).score,
                      "median_item_chrf": sorted(r["chrf_item"] for r in rows)[N_ITEMS // 2]}))


if __name__ == "__main__":
    main()

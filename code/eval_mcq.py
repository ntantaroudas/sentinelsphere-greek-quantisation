#!/usr/bin/env python3
"""Forced choice multiple choice evaluation of a GGUF model with llama.cpp (Paper C).

For every item the model receives its own chat template (read from the GGUF file), a
system instruction and the question with options A to D, in English or Greek. The first
generated token is restricted to the option letters with a logit bias, so the prediction
is the model's preferred letter (standard forced choice scoring). Greedy decoding.

Usage:  python eval_mcq.py <model.gguf> <label> [--langs en,el] [--threads 2]
Writes ../results/mcq_<label>.jsonl (one line per item and language) and a summary line.
"""
import argparse
import json
import os
import resource
import time

from llama_cpp import Llama

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
RES = os.path.join(HERE, "..", "results")

PROMPTS = {
    "en": {
        "system": "You are a cybersecurity expert. Answer the multiple choice question with the letter of the correct option only.",
        "question": "Question: {q}\n{opts}\nAnswer with A, B, C or D.",
    },
    "el": {
        "system": "Είσαι ειδικός στην κυβερνοασφάλεια. Απάντησε στην ερώτηση πολλαπλής επιλογής μόνο με το γράμμα της σωστής επιλογής.",
        "question": "Ερώτηση: {q}\n{opts}\nΑπάντησε με A, B, C ή D.",
    },
}


def letter_bias(llm):
    bias = {}
    for letter in "ABCD":
        for variant in (letter, " " + letter):
            ids = llm.tokenize(variant.encode(), add_bos=False, special=False)
            if len(ids) == 1:
                bias[ids[0]] = 100.0
    return bias


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("label")
    ap.add_argument("--langs", default="en,el")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    out_path = os.path.join(RES, f"mcq_{a.label}.jsonl")
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path, encoding="utf-8"):
            r = json.loads(line)
            if "item" in r:
                done.add((r["lang"], r["item"]))

    t_load = time.time()
    llm = Llama(model_path=a.model, n_ctx=2048, n_threads=a.threads, n_batch=256, verbose=False)
    load_s = time.time() - t_load
    bias = letter_bias(llm)
    with open(out_path, "a", encoding="utf-8") as fh:
        for lang in a.langs.split(","):
            items = json.load(open(os.path.join(DATA, f"items_{lang}.json"), encoding="utf-8"))
            if a.limit:
                items = items[:a.limit]
            P = PROMPTS[lang]
            for it in items:
                if (lang, it["id"]) in done:
                    continue
                opts = "\n".join(f"{k}. {it['answers'][k]}" for k in "ABCD")
                msgs = [{"role": "system", "content": P["system"]},
                        {"role": "user", "content": P["question"].format(q=it["question"], opts=opts)}]
                t0 = time.time()
                out = llm.create_chat_completion(messages=msgs, max_tokens=1, temperature=0.0,
                                                 logit_bias=bias)
                dt = time.time() - t0
                pred = out["choices"][0]["message"]["content"].strip()[:1]
                fh.write(json.dumps({"label": a.label, "lang": lang, "item": it["id"], "pred": pred,
                                     "gold": it["solution"], "correct": pred == it["solution"],
                                     "prompt_tokens": out["usage"]["prompt_tokens"], "seconds": round(dt, 3)},
                                    ensure_ascii=False) + "\n")
                fh.flush()
        fh.write(json.dumps({"label": a.label, "summary": True, "model_file": os.path.basename(a.model),
                             "model_bytes": os.path.getsize(a.model), "load_seconds": round(load_s, 2),
                             "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                             "threads": a.threads}) + "\n")


if __name__ == "__main__":
    main()

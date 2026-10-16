# How far does quantisation take a local security tutor in Greek and in English, a benchmark of small open models on cybersecurity questions

Code and data for the Open Research Europe article "How far does quantisation take a local security tutor in Greek and in English, a benchmark of small open models on cybersecurity questions" by Nikolaos D. Tantaroudas, Ilias Karachalios and Andrew J. McCracken, a follow on study of the SentinelSphere platform carried out under the Horizon Europe project ResilMesh, Grant Agreement No. 101119681.

Preprint of the short version, https://doi.org/10.20944/preprints202610.1517.v1.

## Contents and how to reproduce

Code and per item results for "Quantisation costs more in Greek than in English, cybersecurity knowledge of small local language models".

`code/make_greek_set.py` draws the 120 items with a fixed seed from the CyberMetric release, translates them with OPUS-MT and scores the back translation, `code/eval_mcq.py` is the forced choice harness, `code/requantise_static.py` produces the static files without an importance matrix, the two shell scripts run all configurations and `code/analyse.py` builds the tables and figures. Every `results/mcq_*.jsonl` file holds one line per item and language with the predicted and the correct letter, the prompt length in tokens and the time, `results/item_fidelity.csv` holds the back translation chrF of every item, and the `table*.csv` files hold the aggregated accuracy, paired test and fidelity tables.

The CyberMetric items and their Greek translations are not included, because they remain under the terms set by the CyberMetric authors. Download `CyberMetric-500-v1.json` from the CyberMetric repository into `data`, then run `python3 code/make_greek_set.py`, `bash code/run_all_models.sh`, `bash code/run_static_ablation.sh` and `python3 code/analyse.py`. The GGUF files are listed in the scripts.

The analysis scripts import `figstyle.py` from the folder `code`, which sets the figure style used in the article, drawn at a display width of at most 150 mm and saved as EPS, PDF and 600 dpi PNG.

## Licence

The code is released under the MIT License, see `LICENSE`. The aggregated results and figures are released under the Creative Commons Zero 1.0 dedication, see `DATA_LICENSE.md`.

## Citation

Please cite the article, and this repository through its Zenodo DOI once a release has been archived, see `CITATION.cff`.

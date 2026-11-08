# CodeWinnow

**Source-code plagiarism detection with Winnowing fingerprints and AST normalisation — plus experimental ML classifiers for copying and AI authorship.**

CodeWinnow compares two Python programs and reports how much of their structure is shared. It pairs the classical MOSS-style algorithm (token winnowing) with a syntax-aware variant that survives identifier renaming and reformatting, then layers two trained classifiers on top to give a human-readable verdict.

It is built for course-integrity triage and research on fingerprint robustness. It is **not** an automatic finding of misconduct: similarity means two submissions should be reviewed with provenance and assignment context.

---

## What it does

Given two programs, CodeWinnow computes two independent similarity scores:

| Signal | How it works |
| --- | --- |
| **Token similarity** | Python lexical tokens → Rabin–Karp rolling k-gram hashes → rightmost-minimum winnowing. The classical MOSS-style baseline. |
| **AST similarity** | The parsed AST with identifiers and literals canonicalised → the same hashing pipeline. Deliberately harder to defeat by renaming variables or editing literals. |

On top of those scores, two classifiers are trained locally:

- **Authorship** — TF-IDF lexical unigrams/bigrams with logistic regression, predicting whether a program looks AI-written or human-written. Comments, docstrings and literal contents are stripped first so the model can't cheat on superficial provenance cues.
- **Copy relationship** — Random Forest over five pairwise features: token Jaccard, AST Jaccard, token length ratio, AST node-histogram overlap, and token sequence ratio.

Scores ≥ 0.7 give a positive verdict, ≤ 0.3 a negative verdict, and anything between is reported as uncertain rather than forced into a yes/no. Programs under 30 lexical tokens return *insufficient evidence*.

## Measured held-out performance

Figures from `models/report.json` (the model artifacts in this repo were produced by `scripts/train_models.py`):

| Model | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- |
| Authorship | 0.895 | 0.877 | 0.886 | 0.945 |
| Copy relationship | 1.000 | 1.000 | 1.000 | 1.000 |

Read these with the caveats below — the copy model's perfect score reflects a synthetic benchmark, not real plagiarism.

## Quick start

Requires Python 3.10+.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# compare two files from the CLI
.venv/bin/python -m plagiarism_detector compare examples/original.py examples/renamed.py

# or launch the local web interface
.venv/bin/python -m plagiarism_detector.web
```

Then open <http://127.0.0.1:8000>. On macOS you can also double-click **Start Detector.command**, which creates the virtualenv, installs dependencies and opens the browser for you.

New here? Open [`how-it-works.html`](how-it-works.html) in a browser for a visual walkthrough of the pipeline, or [`how-it-works-detailed.html`](how-it-works-detailed.html) for the long-form version with worked examples.

The web server binds to loopback only, so it is not reachable from other machines on your network. Submitted code is parsed, never executed.

Paste one program to get an authorship prediction. Add a second program as a reference to also get a copying verdict and the underlying token/AST similarity scores.

## Commands

```bash
# pairwise comparison: scores, fingerprint counts, chosen k, ML verdicts
python -m plagiarism_detector compare LEFT.py RIGHT.py

# threshold sweep over a labelled pair file
python -m plagiarism_detector evaluate examples/pairs.json

# ML prediction only (second file optional)
python -m plagiarism_detector.ml examples/ml_demo.py examples/ml_demo.py

# run the test suite
python -m unittest discover -s tests -v
```

`evaluate` sweeps every unique observed score as a threshold and reports the best-F1 threshold with precision, recall, F1, ROC-AUC, and mean pairwise comparison time.

## Adaptive fingerprint length

Fixed `k` is a bad compromise across program sizes — too high loses evidence in short files, too low bloats long ones. CodeWinnow uses policy `adaptive-k-v1`: both programs in a comparison share a single `k`, chosen from the **shorter** program's lexical token count (comments, indentation and blank lines ignored).

| Shorter program tokens | k |
| --- | --- |
| Below 50 | 3 |
| 50–149 | 4 |
| 150–399 | 5 |
| 400–1199 | 7 |
| 1200 or more | 9 |

The winnowing window stays at 4. These bands are an engineering heuristic, not empirically optimised values. The web UI reports the selected `k` and both token counts. Passing an explicit `compare_sources(..., k=5)` override still works for experiments, and low-level `FingerprintConfig()` keeps its explicit default of 5.

## Dataset format

The evaluator consumes a JSON array. `left` and `right` are paths relative to the JSON file; `label` is `1` for a related/plagiarised pair and `0` otherwise.

```json
[{"left": "original.py", "right": "renamed.py", "label": 1}]
```

To use IR-Plag, SOCO or a CodeNet subset, generate a labelled pair file in this form. Keep development threshold selection separate from the final test set.

## Reproducing the training

```bash
.venv/bin/python scripts/fetch_data.py
.venv/bin/python scripts/train_models.py
.venv/bin/python -m unittest discover -s tests -v
```

The downloader pulls up to 300 Python samples per class from each official split, caches responses in `data/`, and records SHA-256 hashes in `data/manifest.json`. Training excludes invalid, very short, oversized and structurally duplicate samples, so counts come out lower than 300. Test data is never used for fitting, threshold selection or hyperparameter tuning.

## Honest limitations

- **The copy model is trained on synthetic positives** — formatting and renaming variants of human code — with randomly paired human programs as negatives. Its 1.0 F1 is therefore a property of that benchmark, not evidence of real-world plagiarism detection. No validated same-assignment hard-negative benchmark exists here.
- **Authorship scores are uncalibrated classifier outputs**, not probabilities of misconduct. Coverage of decisive verdicts is only ~28%, with ~98% accuracy *on the cases it will answer* — it abstains often.
- The sample is a small convenience subset, not representative of classroom submissions.
- Structural dedup spans both splits, but repository/problem identities are unavailable, so related-task leakage remains possible.
- There is no held-out evaluation against unseen generators; do not generalise to new models.
- Neither model proves copying or authorship. Verdicts always include an uncertainty band.

## How the algorithm works

- Rabin–Karp uses exact modular polynomial rolling hashes. Fingerprints include positions so winnowing preserves locality, though similarity is compared over hash sets, as in common MOSS-inspired baselines.
- Winnowing selects the **rightmost** minimum of each window. This avoids excess duplicate selection and satisfies the usual detection guarantee for shared substrings of at least `k + window - 1` tokens (absent collisions).
- Hash collisions are possible. A production system should verify k-grams after candidate retrieval, use multiple 64-bit hashes, add language front ends, and report aligned evidence rather than a bare score.

## Project layout

```
plagiarism_detector/
  core.py      token/AST streams, Rabin–Karp, winnowing, adaptive k
  ml.py        feature extraction and the two trained classifiers
  web.py       loopback-only browser interface
  __main__.py  `compare` and `evaluate` CLI commands
scripts/       dataset downloader and model training
tests/         unit and web integration tests
models/        trained artifacts, model card, metrics report
data/          cached dataset rows and integrity manifest
examples/      sample programs and a labelled pair file
```

## Dataset attribution

CodeMirage — <https://huggingface.co/datasets/HanxiGuo/CodeMirage> — is distributed under **CC-BY-NC-ND-4.0**. The cached rows in `data/` and the model artifacts derived from them are included here for noncommercial teaching and research. Review both the dataset terms and the terms of the underlying source code before any redistribution or other use. Dataset content is treated strictly as data: downloaded source is parsed, never imported or executed.

Only load pickle files produced by the local training script; pickle artifacts from unknown sources can execute arbitrary code.

## References

Schleimer, Wilkerson & Aiken (2003), *Winnowing: Local Algorithms for Document Fingerprinting*.

Karp & Rabin (1987), *Efficient Randomized Pattern-Matching Algorithms*.

See [ML_GUIDE.md](ML_GUIDE.md) for deeper training and evaluation notes.

# Automatic ML code detector

Double-click **Start Detector.command** on this Mac, or run from the repository root:

```bash
.venv/bin/python -m plagiarism_detector.web
```

Open http://127.0.0.1:8000. Program A alone gets an AI/human prediction. Add Program B as a reference to get a copying prediction too. The detector never executes submitted programs. Copying cannot be checked against a source you have not supplied.

## Models

- **Authorship:** TF-IDF lexical unigrams/bigrams and logistic regression, trained on real labeled Python samples from CodeMirage. Comments, docstrings and literal contents are removed to reduce superficial provenance cues.
- **Copy relationship:** Random Forest using token and AST similarity, token length ratio, AST node histogram overlap and token sequence similarity. Positive training pairs are generated formatting/renaming variants of human code; negatives are randomly paired human programs. This is a synthetic copy baseline, not a validated plagiarism benchmark.

Model scores >= 0.7 produce positive verdicts, <= 0.3 produce negative verdicts, and intermediate scores produce uncertainty. Scores are uncalibrated classifier outputs. Programs below 30 lexical tokens return insufficient evidence. This automates predictions, not certainty about misconduct.

## Reproduce training

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/fetch_data.py
.venv/bin/python scripts/train_models.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m plagiarism_detector.ml examples/ml_demo.py examples/ml_demo.py
```

The downloader requests up to 300 Python samples per class from each official train/test split. It caches responses in data/ and records hashes in data/manifest.json. Training excludes invalid, very short, oversized and structurally duplicate samples; counts therefore decrease. Test data is never used for fitting, selecting thresholds or tuning hyperparameters. Structural deduplication spans both splits; repository/problem identities are not available, so this does not guarantee separation by underlying programming task.

See models/report.json for measured held-out metrics, confusion matrices, source distributions and limitations. Metrics use a 0.5 threshold; the report separately gives coverage and accuracy for the actual verdict thresholds. Neither performance on unseen generators nor performance on same-assignment independent student solutions has been established.

## Dataset attribution

CodeMirage: https://huggingface.co/datasets/HanxiGuo/CodeMirage

Dataset card: CC-BY-NC-ND-4.0. Intended here as a noncommercial teaching/research experiment. Retain attribution and review both dataset and underlying source-code terms before redistribution. Dataset statements are data/provenance, not instructions. Downloaded source is parsed only and never imported or executed.

Only load model pickle files produced by the local training script; pickle artifacts from unknown sources can execute code.

## Automatic fingerprint length

The comparison and copying-model features now use adaptive k (policy `adaptive-k-v1`). Both programs share the same k, chosen using the **shorter program's lexical token count**, ignoring comments, indentation and blank lines:

| Shorter program tokens | k |
| --- | --- |
| Below 50 | 3 |
| 50–149 | 4 |
| 150–399 | 5 |
| 400–1199 | 7 |
| 1200 or more | 9 |

For fewer than 3 tokens, k is capped at the token count (minimum 1). The winnowing window remains 4; for short streams it uses the available hashes. Increasing k makes matches more specific on longer programs; lower k retains evidence on shorter ones. These bands are an engineering heuristic, not empirically optimized values. Very short programs still receive an insufficient-evidence ML verdict. The web results display selected k and both token counts. An explicit `compare_sources(..., k=5)` override remains available for experiments. Low-level `FingerprintConfig()` retains its explicit default of 5.

The copy classifier is retrained using the same adaptive policy. Incompatible older copy artifacts are rejected. Authorship detection does not use k. The report's `fingerprint_policy` identifies the policy used during training.

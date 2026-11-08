# Experimental classifiers

Training source: https://huggingface.co/datasets/HanxiGuo/CodeMirage (CC-BY-NC-ND-4.0). Noncommercial research/teaching baseline; check dataset and underlying-code terms before redistribution.

Authorship: lexical TF-IDF + logistic regression. Copying: five pairwise features + Random Forest. Official test rows are held out; structural duplicates are removed before training. See report.json for actual counts and metrics.

- Small convenience subset, not representative of classroom submissions.
- Authorship scores are uncalibrated model outputs, not probabilities of misconduct.
- Official train/test split plus structural dedup; repository/problem identities unavailable, so related-task leakage remains possible.
- No held-out generator evaluation; do not generalize to unseen models.
- Copy positives are synthetic formatting and renaming transformations; random negatives may contain unknown related code. No validated same-assignment hard-negative benchmark.
- Automatic verdicts include an uncertainty band; neither model proves copying or authorship.

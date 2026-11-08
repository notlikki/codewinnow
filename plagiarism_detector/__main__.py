"""Command line entry points for pairwise comparison and labeled evaluation."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from .core import compare_sources
from .ml import detect


def _metrics(labels: list[int], scores: list[float], threshold: float) -> dict[str, float]:
    predictions = [int(score >= threshold) for score in scores]
    tp = sum(p == 1 and y == 1 for p, y in zip(predictions, labels))
    fp = sum(p == 1 and y == 0 for p, y in zip(predictions, labels))
    fn = sum(p == 0 and y == 1 for p, y in zip(predictions, labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _auc(labels: list[int], scores: list[float]) -> float:
    positives = [s for s, y in zip(scores, labels) if y]
    negatives = [s for s, y in zip(scores, labels) if not y]
    if not positives or not negatives:
        return float("nan")
    wins = sum(1 if p > n else 0.5 if p == n else 0 for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def evaluate(path: Path) -> None:
    pairs = json.loads(path.read_text())
    if not pairs:
        raise ValueError("pair file is empty")
    labels: list[int] = []
    token_scores: list[float] = []
    ast_scores: list[float] = []
    timings: list[float] = []
    for pair in pairs:
        left = (path.parent / pair["left"]).read_text()
        right = (path.parent / pair["right"]).read_text()
        started = time.perf_counter()
        result = compare_sources(left, right)
        timings.append((time.perf_counter() - started) * 1_000)
        labels.append(int(pair["label"]))
        token_scores.append(result.token_similarity)
        ast_scores.append(result.ast_similarity)
    for name, scores in (("token", token_scores), ("ast", ast_scores)):
        thresholds = sorted(set(scores + [0.0, 1.0]))
        threshold, best = max(((t, _metrics(labels, scores, t)) for t in thresholds), key=lambda x: x[1]["f1"])
        print(f"{name}: threshold={threshold:.4f} precision={best['precision']:.4f} "
              f"recall={best['recall']:.4f} f1={best['f1']:.4f} auc={_auc(labels, scores):.4f}")
    print(f"mean_pair_time_ms={statistics.fmean(timings):.3f} pairs={len(pairs)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Python source-code fingerprint comparison")
    sub = parser.add_subparsers(dest="command", required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("pairs", type=Path)
    args = parser.parse_args()
    if args.command == "compare":
        result = compare_sources(args.left.read_text(), args.right.read_text())
        print(f"automatic_k={result.k} token_counts={result.token_counts} window={result.window}")
        print(f"token_similarity={result.token_similarity:.4f} fingerprints={result.token_fingerprints}")
        print(f"ast_similarity={result.ast_similarity:.4f} fingerprints={result.ast_fingerprints}")
        print(json.dumps(detect(args.left.read_text(), args.right.read_text()), indent=2))
    else:
        evaluate(args.pairs)


if __name__ == "__main__":
    main()

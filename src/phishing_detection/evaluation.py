"""Shared classification metrics and clustered bootstrap intervals."""

from __future__ import annotations

from collections import defaultdict
import json
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import StudyConfig

CORE_CLASSIFICATION_METRICS = (
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "false_positive_rate",
)

SUPPLEMENTARY_RANKING_METRICS = (
    "roc_auc",
    "pr_auc",
)

INTERVAL_METRICS = (
    "accuracy",
    *CORE_CLASSIFICATION_METRICS,
    "specificity",
    *SUPPLEMENTARY_RANKING_METRICS,
)

PAIRED_FIELDS = ("sample_id", "source", "corpus_label", "similarity_group_id")


def load_complete_test_comparison(config: StudyConfig) -> dict[str, dict[str, Any]]:
    """Load metrics only when every method covers the frozen test set exactly."""
    expected = sorted(
        (
            row
            for row in pq.read_table(config.split_path).to_pylist()
            if row["split"] == "test"
        ),
        key=lambda row: str(row["sample_id"]),
    )
    if not expected:
        raise ValueError("the split manifest has no frozen held-out test samples")

    runs = (
        (
            "TF–IDF logistic regression",
            config.nlp_dir / "test-predictions.parquet",
            config.nlp_dir / "summary.json",
            "test_metrics",
            False,
        ),
        (
            "GPT-5 Nano zero-shot",
            config.zero_shot_dir / "predictions.parquet",
            config.zero_shot_dir / "summary.json",
            "metrics",
            True,
        ),
        (
            "GPT-5 Nano retrieved four-shot",
            config.retrieval_four_shot_dir / "predictions.parquet",
            config.retrieval_four_shot_dir / "summary.json",
            "metrics",
            True,
        ),
    )
    expected_identity = [
        tuple(row[field] for field in PAIRED_FIELDS) for row in expected
    ]
    comparison: dict[str, dict[str, Any]] = {}
    for name, predictions_path, summary_path, metrics_key, require_complete in runs:
        missing = [
            str(path)
            for path in (predictions_path, summary_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"{name} is missing final artefacts: {', '.join(missing)}"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if require_complete and summary.get("status") != "complete":
            raise ValueError(f"{name} is not complete")
        predictions = sorted(
            pq.read_table(predictions_path).to_pylist(),
            key=lambda row: str(row["sample_id"]),
        )
        identity = [
            tuple(row[field] for field in PAIRED_FIELDS) for row in predictions
        ]
        if identity != expected_identity:
            raise ValueError(f"{name} does not cover the frozen held-out test set")
        metrics = summary.get(metrics_key)
        if not isinstance(metrics, dict) or metrics.get("n") != len(expected):
            raise ValueError(
                f"{name} metrics do not match the frozen held-out test set"
            )
        comparison[name] = metrics
    return comparison


def classification_metrics(
    labels: Sequence[int], predictions: Sequence[int], probabilities: Sequence[float]
) -> dict[str, Any]:
    """Return the fixed metric set used by both NLP and LLM experiments."""
    if not labels or not (len(labels) == len(predictions) == len(probabilities)):
        raise ValueError(
            "labels, predictions and probabilities must have equal non-zero length"
        )
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    both_classes = len(set(labels)) == 2
    return {
        "n": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "roc_auc": (
            float(roc_auc_score(labels, probabilities)) if both_classes else None
        ),
        "pr_auc": (
            float(average_precision_score(labels, probabilities))
            if both_classes
            else None
        ),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def _metric_subset(
    rows: Sequence[Mapping[str, object]], indexes: Iterable[int]
) -> dict[str, float]:
    chosen = [rows[index] for index in indexes]
    labels = np.fromiter((int(row["corpus_label"]) for row in chosen), dtype=np.int8)
    predictions = np.fromiter(
        (int(row["predicted_label"]) for row in chosen), dtype=np.int8
    )
    probabilities = np.fromiter(
        (float(row["phishing_probability"]) for row in chosen), dtype=float
    )
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0

    # Group equal scores so these definitions match threshold-based ROC AUC
    # and average precision, including repeated samples within a bootstrap draw.
    order = np.argsort(-probabilities, kind="stable")
    sorted_scores = probabilities[order]
    sorted_labels = labels[order]
    group_ends = np.r_[
        np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(labels) - 1
    ]
    cumulative_tp = np.cumsum(sorted_labels)[group_ends]
    cumulative_fp = (group_ends + 1) - cumulative_tp
    positives, negatives = int(labels.sum()), int(len(labels) - labels.sum())
    tpr = np.r_[0.0, cumulative_tp / positives]
    fpr = np.r_[0.0, cumulative_fp / negatives]
    roc_auc = float(np.trapezoid(tpr, fpr))
    grouped_precision = cumulative_tp / (cumulative_tp + cumulative_fp)
    grouped_recall = cumulative_tp / positives
    pr_auc = float(np.sum(np.diff(np.r_[0.0, grouped_recall]) * grouped_precision))
    return {
        "accuracy": (tp + tn) / len(labels),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ),
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "specificity": specificity,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def _interval(point: float, samples: Sequence[float]) -> dict[str, object]:
    lower, upper = np.quantile(np.asarray(samples), [0.025, 0.975]).tolist()
    return {
        "point_estimate": point,
        "confidence_interval_95": {"lower": float(lower), "upper": float(upper)},
        "valid_replicates": len(samples),
    }


def clustered_bootstrap(
    runs: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    """Resample paired similarity groups within source/label strata.

    Every run must contain the same samples in the same metadata groups. This
    preserves the paired comparison between detector representations.
    """
    if replicates < 1:
        raise ValueError("replicates must be at least one")
    names = tuple(runs)
    if not names:
        raise ValueError("at least one run is required")
    ordered = {
        name: sorted(rows, key=lambda row: str(row["sample_id"]))
        for name, rows in runs.items()
    }
    reference = ordered[names[0]]
    paired_fields = ("sample_id", "source", "corpus_label", "similarity_group_id")
    for rows in ordered.values():
        if len(rows) != len(reference):
            raise ValueError("runs must contain the same samples")
        for left, right in zip(reference, rows, strict=True):
            if any(left[field] != right[field] for field in paired_fields):
                raise ValueError("runs are not paired by sample and metadata")

    group_indexes: dict[str, list[int]] = defaultdict(list)
    group_strata: dict[str, tuple[str, int]] = {}
    for index, row in enumerate(reference):
        group = str(row["similarity_group_id"])
        stratum = (str(row["source"]), int(row["corpus_label"]))
        if group in group_strata and group_strata[group] != stratum:
            raise ValueError("a similarity group crosses source/label strata")
        group_strata[group] = stratum
        group_indexes[group].append(index)
    strata: dict[tuple[str, int], list[list[int]]] = defaultdict(list)
    for group in sorted(group_indexes):
        strata[group_strata[group]].append(group_indexes[group])
    if {label for _, label in strata} != {0, 1}:
        raise ValueError("bootstrap data must contain both labels")

    full = list(range(len(reference)))
    points = {name: _metric_subset(ordered[name], full) for name in names}
    samples = {name: {metric: [] for metric in INTERVAL_METRICS} for name in names}
    differences = (
        {metric: [] for metric in INTERVAL_METRICS} if len(names) == 2 else None
    )
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        selected: list[int] = []
        for clusters in strata.values():
            for choice in rng.integers(0, len(clusters), size=len(clusters)):
                selected.extend(clusters[int(choice)])
        replicate = {name: _metric_subset(ordered[name], selected) for name in names}
        for name in names:
            for metric, value in replicate[name].items():
                samples[name][metric].append(value)
        if differences is not None:
            for metric in INTERVAL_METRICS:
                differences[metric].append(
                    replicate[names[1]][metric] - replicate[names[0]][metric]
                )

    result: dict[str, Any] = {
        "method": "paired source/label-stratified similarity-group percentile bootstrap",
        "seed": seed,
        "replicates": replicates,
        "samples": len(reference),
        "clusters": len(group_indexes),
        "runs": {
            name: {
                metric: _interval(points[name][metric], values)
                for metric, values in samples[name].items()
            }
            for name in names
        },
    }
    if differences is not None:
        key = f"{names[1]}_minus_{names[0]}"
        result["paired_difference"] = {
            key: {
                metric: _interval(
                    points[names[1]][metric] - points[names[0]][metric], values
                )
                for metric, values in differences.items()
            }
        }
    return result

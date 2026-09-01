"""Train, select and evaluate the TF-IDF logistic-regression experiment."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .config import StudyConfig
from .evaluation import classification_metrics, clustered_bootstrap

PARAMETER_GRID = tuple(
    {"C": c, "ngram_range": ngram, "min_df": min_df}
    for c in (0.1, 1.0, 10.0)
    for ngram in ((1, 1), (1, 2))
    for min_df in (1, 2)
)
REPRESENTATIONS = {"v1.0": "detector_v1_path", "v2.0": "detector_v2_path"}
LOGGER = logging.getLogger(__name__)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _model(
    parameters: dict[str, object], seed: int
) -> tuple[TfidfVectorizer, LogisticRegression]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
        ngram_range=parameters["ngram_range"],  # type: ignore[arg-type]
        min_df=int(parameters["min_df"]),
    )
    classifier = LogisticRegression(
        C=float(parameters["C"]),
        class_weight="balanced",
        max_iter=2_000,
        random_state=seed,
        solver="liblinear",
    )
    return vectorizer, classifier


def _load_rows(
    config: StudyConfig, representation: str
) -> dict[str, list[dict[str, object]]]:
    overlay_path = getattr(config, REPRESENTATIONS[representation])
    overlay = {
        str(row["sample_id"]): row for row in pq.read_table(overlay_path).to_pylist()
    }
    rows = pq.read_table(config.split_path).to_pylist()
    joined: dict[str, list[dict[str, object]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id not in overlay:
            raise ValueError(f"representation is missing sample {sample_id}")
        joined[str(row["split"])].append(
            {**row, "detector_input_text": overlay[sample_id]["detector_input_text"]}
        )
    for values in joined.values():
        values.sort(key=lambda row: str(row["sample_id"]))
    return joined


def _predict(
    vectorizer: TfidfVectorizer,
    classifier: LogisticRegression,
    rows: list[dict[str, object]],
    representation: str,
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    labels = [int(row["corpus_label"]) for row in rows]
    matrix = vectorizer.transform([str(row["detector_input_text"]) for row in rows])
    predictions = classifier.predict(matrix).astype(int).tolist()
    probabilities = classifier.predict_proba(matrix)[:, 1].tolist()
    output = [
        {
            "sample_id": row["sample_id"],
            "source": row["source"],
            "corpus_label": label,
            "similarity_group_id": row["similarity_group_id"],
            "representation": representation,
            "predicted_label": prediction,
            "phishing_probability": probability,
            "correct": label == prediction,
        }
        for row, label, prediction, probability in zip(
            rows, labels, predictions, probabilities, strict=True
        )
    ]
    return classification_metrics(labels, predictions, probabilities), output


def _select_candidate(config: StudyConfig, representation: str) -> dict[str, Any]:
    rows = _load_rows(config, representation)
    train_text = [str(row["detector_input_text"]) for row in rows["train"]]
    train_labels = [int(row["corpus_label"]) for row in rows["train"]]
    candidates: list[dict[str, Any]] = []
    prediction_sets: list[list[dict[str, object]]] = []
    for parameters in PARAMETER_GRID:
        vectorizer, classifier = _model(parameters, config.seed)
        classifier.fit(vectorizer.fit_transform(train_text), train_labels)
        metrics, predictions = _predict(
            vectorizer, classifier, rows["validation"], representation
        )
        candidates.append({"parameters": parameters, "validation_metrics": metrics})
        prediction_sets.append(predictions)
        LOGGER.info(
            "%s candidate %s/%s: balanced accuracy %.4f, F1 %.4f",
            representation,
            len(candidates),
            len(PARAMETER_GRID),
            metrics["balanced_accuracy"],
            metrics["f1"],
        )
    winner_index = max(
        range(len(candidates)),
        key=lambda index: (
            candidates[index]["validation_metrics"]["balanced_accuracy"],
            candidates[index]["validation_metrics"]["f1"],
            -index,
        ),
    )
    return {
        "representation": representation,
        "parameters": candidates[winner_index]["parameters"],
        "validation_metrics": candidates[winner_index]["validation_metrics"],
        "validation_predictions": prediction_sets[winner_index],
        "candidate_count": len(candidates),
        "rows": rows,
    }


def _serialise_parameters(parameters: dict[str, object]) -> dict[str, object]:
    return {
        "C": parameters["C"],
        "ngram_range": list(parameters["ngram_range"]),  # type: ignore[arg-type]
        "min_df": parameters["min_df"],
    }


def _fit_test(
    config: StudyConfig, selected: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, object]], dict[str, object]]:
    rows = selected["rows"]
    fit_rows = rows["train"] + rows["validation"]
    vectorizer, classifier = _model(selected["parameters"], config.seed)
    classifier.fit(
        vectorizer.fit_transform([str(row["detector_input_text"]) for row in fit_rows]),
        [int(row["corpus_label"]) for row in fit_rows],
    )
    metrics, predictions = _predict(
        vectorizer, classifier, rows["test"], selected["representation"]
    )
    names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]
    limit = min(25, len(names))
    features = {
        "phishing": [
            {"feature": str(names[i]), "coefficient": float(coefficients[i])}
            for i in np.argsort(coefficients)[-limit:][::-1]
        ],
        "legitimate": [
            {"feature": str(names[i]), "coefficient": float(coefficients[i])}
            for i in np.argsort(coefficients)[:limit]
        ],
    }
    return metrics, predictions, features


def run_nlp_study(config: StudyConfig) -> dict[str, Any]:
    """Compare v1/v2 on validation, then evaluate only the selected version on test."""
    config.ensure_directories()
    required = (config.split_path, config.detector_v1_path, config.detector_v2_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Run data preparation first: " + ", ".join(missing))
    selected_runs = [_select_candidate(config, name) for name in REPRESENTATIONS]
    winner = max(
        selected_runs,
        key=lambda run: (
            run["validation_metrics"]["balanced_accuracy"],
            run["validation_metrics"]["f1"],
            -list(REPRESENTATIONS).index(run["representation"]),
        ),
    )
    validation_predictions = [
        row for run in selected_runs for row in run["validation_predictions"]
    ]
    pq.write_table(
        pa.Table.from_pylist(validation_predictions),
        config.nlp_dir / "validation-predictions.parquet",
        compression="zstd",
    )
    validation_bootstrap = clustered_bootstrap(
        {run["representation"]: run["validation_predictions"] for run in selected_runs},
        seed=config.seed,
        replicates=config.bootstrap_replicates,
    )
    LOGGER.info(
        "Selected %s on validation; fitting the final test model",
        winner["representation"],
    )
    test_metrics, test_predictions, features = _fit_test(config, winner)
    pq.write_table(
        pa.Table.from_pylist(test_predictions),
        config.nlp_dir / "test-predictions.parquet",
        compression="zstd",
    )
    test_bootstrap = clustered_bootstrap(
        {winner["representation"]: test_predictions},
        seed=config.seed,
        replicates=config.bootstrap_replicates,
    )
    document = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "TF-IDF logistic regression",
        "selection_rule": "validation balanced accuracy, then F1, then simpler v1.0",
        "validation_runs": [
            {
                "representation": run["representation"],
                "selected_parameters": _serialise_parameters(run["parameters"]),
                "candidate_count": run["candidate_count"],
                "metrics": run["validation_metrics"],
            }
            for run in selected_runs
        ],
        "selected_representation": winner["representation"],
        "selected_parameters": _serialise_parameters(winner["parameters"]),
        "test_metrics": test_metrics,
        "validation_bootstrap": validation_bootstrap,
        "test_bootstrap": test_bootstrap,
    }
    _write_json(config.nlp_dir / "summary.json", document)
    _write_json(config.nlp_dir / "top-features.json", features)
    return document

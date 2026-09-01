from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from phishing_detection.config import StudyConfig
from phishing_detection.evaluation import (
    CORE_CLASSIFICATION_METRICS,
    SUPPLEMENTARY_RANKING_METRICS,
    classification_metrics,
    clustered_bootstrap,
    load_complete_test_comparison,
)


def rows(name="v1.0"):
    return [
        {
            "sample_id": "a",
            "source": "ham",
            "corpus_label": 0,
            "similarity_group_id": "g1",
            "predicted_label": 0,
            "phishing_probability": 0.1,
            "representation": name,
        },
        {
            "sample_id": "b",
            "source": "ham",
            "corpus_label": 0,
            "similarity_group_id": "g2",
            "predicted_label": 1,
            "phishing_probability": 0.7,
            "representation": name,
        },
        {
            "sample_id": "c",
            "source": "phish",
            "corpus_label": 1,
            "similarity_group_id": "g3",
            "predicted_label": 1,
            "phishing_probability": 0.9,
            "representation": name,
        },
        {
            "sample_id": "d",
            "source": "phish",
            "corpus_label": 1,
            "similarity_group_id": "g4",
            "predicted_label": 0,
            "phishing_probability": 0.3,
            "representation": name,
        },
    ]


class EvaluationTests(unittest.TestCase):
    def test_report_metric_groups_are_fixed_and_available(self):
        self.assertEqual(
            CORE_CLASSIFICATION_METRICS,
            (
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "false_positive_rate",
            ),
        )
        self.assertEqual(
            SUPPLEMENTARY_RANKING_METRICS,
            ("roc_auc", "pr_auc"),
        )
        result = classification_metrics(
            [0, 0, 1, 1], [0, 1, 1, 0], [0.1, 0.7, 0.9, 0.3]
        )
        self.assertTrue(
            set(CORE_CLASSIFICATION_METRICS + SUPPLEMENTARY_RANKING_METRICS)
            <= result.keys()
        )

    def test_metric_set_includes_confusion_matrix(self):
        result = classification_metrics(
            [0, 0, 1, 1], [0, 1, 1, 0], [0.1, 0.7, 0.9, 0.3]
        )
        self.assertEqual(result["balanced_accuracy"], 0.5)
        self.assertEqual(
            result["confusion_matrix"], {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
        )

    def test_bootstrap_keeps_paired_runs(self):
        result = clustered_bootstrap(
            {"v1.0": rows(), "v2.0": rows("v2.0")}, seed=7, replicates=10
        )
        difference = result["paired_difference"]["v2.0_minus_v1.0"]["f1"][
            "point_estimate"
        ]
        self.assertEqual(difference, 0.0)

    def test_bootstrap_rejects_unpaired_runs(self):
        altered = rows("v2.0")
        altered[0] = {**altered[0], "sample_id": "different"}
        with self.assertRaisesRegex(ValueError, "paired"):
            clustered_bootstrap({"v1.0": rows(), "v2.0": altered}, seed=7, replicates=2)

    def test_final_comparison_rejects_incomplete_strategy_coverage(self):
        with TemporaryDirectory() as directory:
            config = StudyConfig(root=Path(directory))
            config.ensure_directories()
            expected = rows()[:2]
            split = [
                {
                    key: row[key]
                    for key in (
                        "sample_id",
                        "source",
                        "corpus_label",
                        "similarity_group_id",
                    )
                }
                | {"split": "test"}
                for row in expected
            ]
            pq.write_table(pa.Table.from_pylist(split), config.split_path)
            for path, predictions in (
                (config.nlp_dir / "test-predictions.parquet", expected),
                (config.zero_shot_dir / "predictions.parquet", expected),
                (config.retrieval_four_shot_dir / "predictions.parquet", expected[:1]),
            ):
                pq.write_table(pa.Table.from_pylist(predictions), path)
            (config.nlp_dir / "summary.json").write_text(
                json.dumps({"test_metrics": {"n": 2}}), encoding="utf-8"
            )
            for output_dir in (config.zero_shot_dir, config.retrieval_four_shot_dir):
                (output_dir / "summary.json").write_text(
                    json.dumps(
                        {
                            "status": "complete",
                            "accepted_samples": 2,
                            "test_samples": 2,
                            "metrics": {"n": 2},
                        }
                    ),
                    encoding="utf-8",
                )

            with self.assertRaisesRegex(
                ValueError, "retrieved four-shot.*frozen held-out test set"
            ):
                load_complete_test_comparison(config)


if __name__ == "__main__":
    unittest.main()

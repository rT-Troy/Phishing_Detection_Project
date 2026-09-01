from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from phishing_detection.config import StudyConfig
from phishing_detection.nlp import PARAMETER_GRID, run_nlp_study


class NlpTests(unittest.TestCase):
    def test_grid_contains_twelve_frozen_candidates(self):
        self.assertEqual(len(PARAMETER_GRID), 12)

    def test_study_selects_on_validation_and_tests_one_representation(self):
        with TemporaryDirectory() as directory:
            config = StudyConfig(root=Path(directory), bootstrap_replicates=5)
            config.ensure_directories()
            manifest, v1, v2 = [], [], []
            for index in range(40):
                label = index % 2
                split = (
                    "train" if index < 24 else "validation" if index < 32 else "test"
                )
                sample = f"s{index:02d}"
                manifest.append(
                    {
                        "sample_id": sample,
                        "source": "phish" if label else "ham",
                        "corpus_label": label,
                        "similarity_group_id": f"g{index:02d}",
                        "split": split,
                    }
                )
                clean = (
                    "account password danger" if label else "meeting project schedule"
                )
                v1.append(
                    {
                        "sample_id": sample,
                        "detector_input_text": clean,
                        "representation_version": "v1",
                    }
                )
                v2.append(
                    {
                        "sample_id": sample,
                        "detector_input_text": clean + " extra",
                        "representation_version": "v2",
                    }
                )
            pq.write_table(pa.Table.from_pylist(manifest), config.split_path)
            pq.write_table(pa.Table.from_pylist(v1), config.detector_v1_path)
            pq.write_table(pa.Table.from_pylist(v2), config.detector_v2_path)
            result = run_nlp_study(config)
            predictions = pq.read_table(
                config.nlp_dir / "test-predictions.parquet"
            ).to_pylist()
            self.assertEqual(result["selected_representation"], "v1.0")
            self.assertEqual({row["representation"] for row in predictions}, {"v1.0"})
            self.assertTrue((config.nlp_dir / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()

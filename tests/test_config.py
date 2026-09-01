from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from phishing_detection.config import StudyConfig


class ConfigTests(unittest.TestCase):
    def test_paths_are_rooted_under_project(self):
        config = StudyConfig(root=Path("/study"))
        self.assertEqual(
            config.detector_v2_path,
            Path("/study/artifacts/data/detector-input-v2.parquet"),
        )

    def test_ensure_directories_creates_all_output_areas(self):
        with TemporaryDirectory() as directory:
            config = StudyConfig(root=Path(directory))
            config.ensure_directories()
            self.assertTrue(
                all(
                    path.is_dir()
                    for path in (
                        config.data_dir,
                        config.nlp_dir,
                        config.zero_shot_dir,
                        config.retrieval_four_shot_dir,
                    )
                )
            )

    def test_llm_strategies_have_parallel_output_directories(self):
        config = StudyConfig(root=Path("/study"))
        self.assertEqual(config.zero_shot_dir, Path("/study/artifacts/llm/zero-shot"))
        self.assertEqual(
            config.retrieval_four_shot_dir,
            Path("/study/artifacts/llm/retrieval-four-shot"),
        )

    def test_raw_layout_reports_missing_paths(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "Enron"):
                StudyConfig(root=Path(directory)).validate_raw_layout()


if __name__ == "__main__":
    unittest.main()

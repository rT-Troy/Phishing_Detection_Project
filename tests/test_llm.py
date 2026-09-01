from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from phishing_detection.config import StudyConfig
from phishing_detection.llm import TransportResponse, validate_model_output
from phishing_detection.zero_shot import run_zero_shot_batch


class FakeTransport:
    def __init__(self, contents, model="gpt-5-nano-2026-01-01"):
        self.contents = iter(contents)
        self.model = model
        self.calls = 0

    def complete_chat(self, request, *, request_key):
        del request, request_key
        self.calls += 1
        return TransportResponse("request", self.model, next(self.contents), 20, 0, 10)


def valid(label, probability):
    return json.dumps(
        {
            "label": label,
            "phishing_probability": probability,
            "indicators": [
                {
                    "type": (
                        "benign_context"
                        if label == "legitimate"
                        else "credential_request"
                    ),
                    "evidence": "Message evidence",
                }
            ],
            "reason": "Concise reason.",
            "warning": "" if label == "legitimate" else "Do not respond.",
        }
    )


def project(directory):
    config = StudyConfig(root=Path(directory), bootstrap_replicates=4)
    config.ensure_directories()
    (config.root / "prompts").mkdir()
    (config.root / "prompts/phishing-system.txt").write_text("Classify safely.")
    (config.nlp_dir / "summary.json").write_text(
        json.dumps({"selected_representation": "v1.0"})
    )
    rows = [
        {
            "sample_id": "a",
            "source": "ham",
            "corpus_label": 0,
            "similarity_group_id": "g1",
            "split": "test",
        },
        {
            "sample_id": "b",
            "source": "phish",
            "corpus_label": 1,
            "similarity_group_id": "g2",
            "split": "test",
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), config.split_path)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"sample_id": "a", "detector_input_text": "meeting"},
                {"sample_id": "b", "detector_input_text": "password"},
            ]
        ),
        config.detector_v1_path,
    )
    return config


class LlmTests(unittest.TestCase):
    def test_schema_rejects_label_probability_mismatch(self):
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            validate_model_output(json.loads(valid("legitimate", 0.9)))

    def test_external_processing_requires_explicit_authorisation(self):
        with TemporaryDirectory() as directory:
            with self.assertRaises(PermissionError):
                run_zero_shot_batch(
                    project(directory), FakeTransport([]), authorised=False
                )

    def test_invalid_response_gets_one_retry_and_run_completes(self):
        with TemporaryDirectory() as directory:
            config = project(directory)
            transport = FakeTransport(
                ["{}", valid("legitimate", 0.1), valid("phishing", 0.9)]
            )
            result = run_zero_shot_batch(
                config, transport, authorised=True, max_new_cases=2
            )
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["token_usage"]["input"], 60)
            self.assertEqual(
                len((config.zero_shot_dir / "ledger.jsonl").read_text().splitlines()),
                3,
            )
            summary_path = config.zero_shot_dir / "summary.json"
            frozen_summary = summary_path.read_bytes()
            resumed = run_zero_shot_batch(
                config, FakeTransport([]), authorised=True, max_new_cases=2
            )
            self.assertEqual(resumed["status"], "complete")
            self.assertEqual(summary_path.read_bytes(), frozen_summary)

    def test_exhausted_schema_retries_require_explicit_new_round(self):
        with TemporaryDirectory() as directory:
            config = project(directory)
            first = run_zero_shot_batch(
                config,
                FakeTransport(["{}", "{}", valid("phishing", 0.9)]),
                authorised=True,
                max_new_cases=2,
            )
            self.assertEqual(first["status"], "retry_exhausted")
            self.assertEqual(first["accepted_samples"], 1)

            no_retry_transport = FakeTransport([])
            unchanged = run_zero_shot_batch(
                config,
                no_retry_transport,
                authorised=True,
                max_new_cases=2,
            )
            self.assertEqual(unchanged["status"], "retry_exhausted")
            self.assertEqual(no_retry_transport.calls, 0)

            completed = run_zero_shot_batch(
                config,
                FakeTransport([valid("legitimate", 0.1)]),
                authorised=True,
                max_new_cases=1,
                retry_exhausted=True,
            )
            self.assertEqual(completed["status"], "complete")
            records = [
                json.loads(line)
                for line in (config.zero_shot_dir / "ledger.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            retried = records[-1]
            self.assertEqual(retried["retry_round"], 2)
            self.assertEqual(retried["attempt"], 1)
            self.assertIs(retried["authorised_retry"], True)

    def test_resolved_model_mismatch_pauses_run(self):
        with TemporaryDirectory() as directory:
            result = run_zero_shot_batch(
                project(directory),
                FakeTransport([valid("legitimate", 0.1)], model="other-model"),
                authorised=True,
            )
            self.assertEqual(result["status"], "model_mismatch")


if __name__ == "__main__":
    unittest.main()

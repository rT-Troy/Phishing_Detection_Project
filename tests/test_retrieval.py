from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from phishing_detection.config import StudyConfig
from phishing_detection.llm import TransportResponse
from phishing_detection.retrieval_four_shot import (
    MAX_PROMPT_CHARACTERS,
    build_retrieval_four_shot_request,
    build_retrieval_manifest,
    preview_retrieval_four_shot_request,
    run_retrieval_four_shot_batch,
)


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


class FakeTransport:
    def __init__(self, contents):
        self.contents = iter(contents)
        self.requests = []

    def complete_chat(self, request, *, request_key):
        del request_key
        self.requests.append(request)
        return TransportResponse(
            "request",
            "gpt-5-nano-2026-01-01",
            next(self.contents),
            100,
            0,
            10,
        )


def project(directory, representation="v2.0"):
    config = StudyConfig(root=Path(directory), bootstrap_replicates=4)
    config.ensure_directories()
    (config.root / "prompts").mkdir()
    (config.root / "prompts/phishing-system.txt").write_text(
        "Classify safely.", encoding="utf-8"
    )
    (config.nlp_dir / "summary.json").write_text(
        json.dumps({"selected_representation": representation}), encoding="utf-8"
    )
    rows = [
        {
            "sample_id": "train-enron",
            "source": "enron",
            "corpus_label": 0,
            "similarity_group_id": "g1",
            "split": "train",
        },
        {
            "sample_id": "train-nazario",
            "source": "nazario",
            "corpus_label": 1,
            "similarity_group_id": "g2",
            "split": "train",
        },
        {
            "sample_id": "train-spamassassin",
            "source": "spamassassin",
            "corpus_label": 0,
            "similarity_group_id": "g3",
            "split": "train",
        },
        {
            "sample_id": "train-phishingpot",
            "source": "phishingpot",
            "corpus_label": 1,
            "similarity_group_id": "g4",
            "split": "train",
        },
        {
            "sample_id": "test-legitimate",
            "source": "enron",
            "corpus_label": 0,
            "similarity_group_id": "g5",
            "split": "test",
        },
        {
            "sample_id": "test-phishing",
            "source": "nazario",
            "corpus_label": 1,
            "similarity_group_id": "g6",
            "split": "test",
        },
    ]
    text = {
        "train-enron": "project meeting schedule",
        "train-nazario": "verify password account",
        "train-spamassassin": "team lunch agenda",
        "train-phishingpot": "urgent login link",
        "test-legitimate": "project agenda",
        "test-phishing": "urgent password verification",
    }
    pq.write_table(pa.Table.from_pylist(rows), config.split_path)
    representation_path = (
        config.detector_v1_path
        if representation == "v1.0"
        else config.detector_v2_path
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"sample_id": sample_id, "detector_input_text": detector_input}
                for sample_id, detector_input in text.items()
            ]
        ),
        representation_path,
    )
    return config


class RetrievalTests(unittest.TestCase):
    def test_oversized_complete_examples_are_rejected_not_truncated(self):
        examples = [
            ("short", 0),
            ("short", 1),
            ("short", 0),
            ("x" * MAX_PROMPT_CHARACTERS, 1),
        ]
        with self.assertRaisesRegex(ValueError, "not truncated or sent"):
            build_retrieval_four_shot_request("prompt", "target", examples)

    def test_manifest_has_one_train_example_from_each_frozen_source(self):
        with TemporaryDirectory() as directory:
            config = project(directory)
            audit = build_retrieval_manifest(config)
            rows = pq.read_table(config.retrieval_manifest_path).to_pylist()
            self.assertEqual(audit["targets"], 2)
            self.assertEqual(len(rows), 8)
            for target in {row["target_sample_id"] for row in rows}:
                selected = [row for row in rows if row["target_sample_id"] == target]
                self.assertEqual(
                    [row["example_source"] for row in selected],
                    ["enron", "nazario", "spamassassin", "phishingpot"],
                )

    def test_manifest_uses_the_validation_selected_v1_representation(self):
        with TemporaryDirectory() as directory:
            config = project(directory, representation="v1.0")
            audit = build_retrieval_manifest(config)
            metadata = pq.read_schema(config.retrieval_manifest_path).metadata or {}
            self.assertEqual(audit["targets"], 2)
            self.assertEqual(metadata[b"representation"], b"v1.0")

    def test_preview_contains_four_labels_without_adding_source_metadata(self):
        with TemporaryDirectory() as directory:
            request = preview_retrieval_four_shot_request(project(directory))
            system = request["messages"][0]["content"]
            user = request["messages"][1]["content"]
            self.assertEqual(system, "Classify safely.")
            self.assertEqual(user.count("<example index="), 4)
            self.assertEqual(user.count("<label>legitimate</label>"), 2)
            self.assertEqual(user.count("<label>phishing</label>"), 2)
            for source in ("enron", "nazario", "spamassassin", "phishingpot"):
                self.assertNotIn(source, user)

    def test_retrieval_batch_uses_a_separate_resumable_ledger(self):
        with TemporaryDirectory() as directory:
            config = project(directory)
            transport = FakeTransport(
                [valid("legitimate", 0.1), valid("phishing", 0.9)]
            )
            result = run_retrieval_four_shot_batch(
                config, transport, authorised=True, max_new_cases=2
            )
            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(transport.requests), 2)
            self.assertTrue((config.retrieval_four_shot_dir / "ledger.jsonl").exists())
            self.assertFalse((config.zero_shot_dir / "ledger.jsonl").exists())


if __name__ == "__main__":
    unittest.main()

"""Frozen zero-shot GPT-5 Nano strategy."""

from __future__ import annotations

import json
from pathlib import Path

from .config import StudyConfig
from .llm import (
    ChatTransport,
    MAX_COMPLETION_TOKENS,
    MODEL,
    OUTPUT_SCHEMA,
    classification_request,
    hash_file,
    hash_text,
    load_prompt,
    protocol_digest,
    run_protocol_batch,
    selected_representation,
    target_message,
)


def build_zero_shot_request(prompt: str, detector_input: str) -> dict[str, object]:
    """Build the original request without corpus demonstrations."""
    return classification_request(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": target_message(detector_input)},
        ]
    )


def zero_shot_protocol_id(config: StudyConfig, representation_path: Path) -> str:
    """Retain the original protocol identity so the completed ledger remains valid."""
    prompt = load_prompt(config)
    contract = {
        "model": MODEL,
        "reasoning_effort": "minimal",
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "schema": OUTPUT_SCHEMA,
    }
    return protocol_digest(
        {
            "version": "zero-shot-phishing-v1.0",
            "prompt_sha256": hash_text(prompt),
            "contract": contract,
            "representation_sha256": hash_file(representation_path),
            "split_sha256": hash_file(config.split_path),
            "threshold": 0.5,
        }
    )


def run_zero_shot_batch(
    config: StudyConfig,
    transport: ChatTransport,
    *,
    authorised: bool,
    max_new_cases: int = 100,
    retry_exhausted: bool = False,
) -> dict[str, object]:
    """Run or resume the frozen zero-shot strategy."""
    representation, representation_path = selected_representation(config)
    prompt = load_prompt(config)
    frozen_protocol = zero_shot_protocol_id(config, representation_path)
    summary_path = config.zero_shot_dir / "summary.json"
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            if existing.get("protocol_id") != frozen_protocol:
                raise ValueError("completed zero-shot summary has a different protocol")
            return existing
    return run_protocol_batch(
        config,
        transport,
        authorised=authorised,
        max_new_cases=max_new_cases,
        method="gpt-5-nano-zero-shot",
        zero_shot=True,
        representation=representation,
        representation_path=representation_path,
        output_dir=config.zero_shot_dir,
        frozen_protocol=frozen_protocol,
        retry_exhausted=retry_exhausted,
        request_builder=lambda _sample_id, detector_input: build_zero_shot_request(
            prompt, detector_input
        ),
    )

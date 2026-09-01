"""Build the complete source-balanced dataset from the four raw corpora.

The public interface is deliberately small: ``audit_raw_data`` checks the local
source layout and ``build_dataset`` creates every downstream data artefact. The
implementation keeps provenance, duplicate handling and split constraints in
one place so notebook users cannot accidentally run the stages out of order.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import mailbox
import os
from pathlib import Path
import re
import unicodedata
from typing import Any

from importlib.metadata import version
from lingua import LanguageDetectorBuilder
import pyarrow as pa
import pyarrow.parquet as pq

from .config import StudyConfig
from .representation import V1_VERSION, V2_VERSION, deidentify, enrich_v2, parse_email

SOURCES = ("enron", "spamassassin", "nazario", "phishingpot")
_SPAMASSASSIN_NAME = re.compile(r"^\d{5}\.[0-9a-f]{32}$")
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SourceMessage:
    source: str
    source_path: str
    raw_email: bytes
    original_label: str
    corpus_label: int
    corpus_label_name: str
    label_note: str
    subset: str | None = None
    message_index: int | None = None


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_raw_data(config: StudyConfig, *, full: bool = False) -> dict[str, Any]:
    """Check source layout quickly, or hash every source file when ``full`` is set."""
    config.validate_raw_layout()
    roots = {
        "enron": config.raw_root / "Enron" / "maildir",
        "nazario": config.raw_root / "Nazario",
        "phishingpot": config.raw_root / "PhishingPot" / "PhishingPot",
        "spamassassin": config.raw_root / "SpamAssassin",
    }
    details: dict[str, Any] = {}
    for source, root in roots.items():
        files = sorted(path for path in root.rglob("*") if path.is_file())
        item: dict[str, Any] = {
            "path": str(root.relative_to(config.root)),
            "file_count": len(files),
            "bytes": sum(path.stat().st_size for path in files),
        }
        if full:
            aggregate = hashlib.sha256()
            for path in files:
                aggregate.update(path.relative_to(root).as_posix().encode("utf-8"))
                aggregate.update(bytes.fromhex(_sha256_file(path)))
            item["content_manifest_sha256"] = aggregate.hexdigest()
        details[source] = item
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_level": "full" if full else "fast",
        "sources": details,
    }
    _write_json(config.data_dir / "raw-audit.json", result)
    return result


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _seeded(paths: Sequence[Path], root: Path, seed: int) -> list[Path]:
    return sorted(
        paths,
        key=lambda path: hashlib.sha256(
            f"{seed}:{_relative(path, root)}".encode()
        ).digest(),
    )


def _source_messages(root: Path, source: str, seed: int) -> Iterator[SourceMessage]:
    """Yield one deterministic stream per corpus and preserve original label semantics."""
    if source == "enron":
        paths = [
            Path(directory) / name
            for directory, _, names in os.walk(root / "Enron" / "maildir")
            for name in sorted(names)
        ]
        for path in _seeded(paths, root, seed):
            yield SourceMessage(
                "enron",
                _relative(path, root),
                path.read_bytes(),
                "enron",
                0,
                "legitimate",
                "Enron mapped to the experimental legitimate class",
            )
        return
    if source == "spamassassin":
        paths = [
            path
            for subset in ("easy_ham", "easy_ham_2", "hard_ham")
            for path in (root / "SpamAssassin" / subset).iterdir()
            if path.is_file() and _SPAMASSASSIN_NAME.fullmatch(path.name)
        ]
        for path in _seeded(paths, root, seed):
            yield SourceMessage(
                "spamassassin",
                _relative(path, root),
                path.read_bytes(),
                "ham",
                0,
                "legitimate",
                "SpamAssassin ham mapped to the experimental legitimate class",
                path.parent.name,
            )
        return
    if source == "phishingpot":
        for path in _seeded(
            list((root / "PhishingPot" / "PhishingPot").glob("*.eml")), root, seed
        ):
            yield SourceMessage(
                "phishingpot",
                _relative(path, root),
                path.read_bytes(),
                "phishing",
                1,
                "phishing",
                "PhishingPot mapped to the experimental phishing class",
            )
        return
    if source == "nazario":
        metadata = {"LICENSE.txt", "README.txt"}
        paths = [
            path
            for path in (root / "Nazario").iterdir()
            if path.is_file() and path.name not in metadata
        ]
        locators: list[tuple[Path, object, int]] = []
        for path in sorted(paths):
            box = mailbox.mbox(path, factory=lambda handle: handle.read(), create=False)
            try:
                locators.extend(
                    (path, key, index) for index, key in enumerate(box.iterkeys())
                )
            finally:
                box.close()
        locators.sort(
            key=lambda item: hashlib.sha256(
                f"{seed}:{_relative(item[0], root)}:{item[2]}".encode()
            ).digest()
        )
        boxes: dict[Path, mailbox.mbox] = {}
        try:
            for path, key, index in locators:
                boxes.setdefault(
                    path,
                    mailbox.mbox(
                        path, factory=lambda handle: handle.read(), create=False
                    ),
                )
                yield SourceMessage(
                    "nazario",
                    _relative(path, root),
                    boxes[path][key],
                    "phishing",
                    1,
                    "phishing",
                    "Nazario mapped to the experimental phishing class",
                    path.name,
                    index,
                )
        finally:
            for box in boxes.values():
                box.close()
        return
    raise ValueError(f"Unknown source: {source}")


_LANGUAGE_DETECTOR = None


def _language(subject: str, body: str) -> tuple[str, float | None, str, list[str]]:
    global _LANGUAGE_DETECTOR
    text = (
        body.strip()
        if sum(char.isalpha() for char in body) >= 40
        else f"{subject}\n\n{body}".strip()
    )
    if sum(char.isalpha() for char in text) < 40:
        return "unknown", None, "unknown", ["insufficient_alphabetic_text"]
    if _LANGUAGE_DETECTOR is None:
        _LANGUAGE_DETECTOR = (
            LanguageDetectorBuilder.from_all_spoken_languages()
            .with_minimum_relative_distance(0.20)
            .build()
        )
    bounded = text[:20_000]
    detected = _LANGUAGE_DETECTOR.detect_language_of(bounded)
    values = _LANGUAGE_DETECTOR.compute_language_confidence_values(bounded)
    confidence = float(values[0].value) if values else None
    if detected is None:
        return "unknown", confidence, "unknown", ["minimum_relative_distance_not_met"]
    code = detected.iso_code_639_1.name.casefold()
    return (
        code,
        confidence,
        "english_main" if code == "en" else "multilingual_extension",
        [],
    )


def _normalised_content(subject: str, body: str) -> bytes:
    subject = unicodedata.normalize("NFC", subject).strip()
    body = (
        unicodedata.normalize("NFC", body)
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .strip()
    )
    return f"{subject}\n\n{body}".encode()


def _record(message: SourceMessage) -> dict[str, Any]:
    parsed = parse_email(message.raw_email)
    safe_subject, safe_body, detector_v1 = deidentify(
        parsed.subject,
        parsed.body,
        sender_domain=parsed.from_domain,
        recipient_domains=parsed.to_domains,
    )
    language, confidence, cohort, language_warnings = _language(
        parsed.subject, parsed.body
    )
    raw_hash = hashlib.sha256(message.raw_email).hexdigest()
    content_hash = hashlib.sha256(
        _normalised_content(parsed.subject, parsed.body)
    ).hexdigest()
    locator = (
        f"{message.source}:{message.source_path}:{message.message_index}:{raw_hash}"
    )
    return {
        "sample_id": "sample_" + hashlib.sha256(locator.encode()).hexdigest()[:24],
        "source": message.source,
        "source_path": message.source_path,
        "source_message_index": message.message_index,
        "source_subset": message.subset,
        "original_label": message.original_label,
        "corpus_label": message.corpus_label,
        "corpus_label_name": message.corpus_label_name,
        "label_mapping_note": message.label_note,
        "raw_sha256": raw_hash,
        "content_sha256": content_hash,
        "duplicate_group_id": "exact_" + content_hash[:24],
        "parse_status": "ok",
        "parse_warnings": list(parsed.warnings),
        "subject": parsed.subject,
        "body_text": parsed.body,
        "deidentified_subject": safe_subject,
        "deidentified_body_text": safe_body,
        "detector_input_v1": detector_v1,
        "body_source": parsed.body_source,
        "has_plain_part": parsed.has_plain,
        "has_html_part": parsed.has_html,
        "language": language,
        "language_confidence": confidence,
        "language_cohort": cohort,
        "language_detector_name": "lingua",
        "language_detector_version": version("lingua-language-detector"),
        "language_warnings": language_warnings,
        "from_raw": parsed.from_raw,
        "from_address": parsed.from_address,
        "from_domain": parsed.from_domain,
        "to_raw": parsed.to_raw,
        "to_addresses": list(parsed.to_addresses),
        "to_domains": list(parsed.to_domains),
        "recipient_count": len(parsed.to_addresses),
        "date_raw": parsed.date_raw,
        "date_utc": parsed.date_utc,
        "date_parse_status": parsed.date_status,
        "message_id": parsed.message_id,
        "mime_type": parsed.mime_type,
        "attachment_count": len(parsed.attachment_extensions),
        "attachment_extensions": list(parsed.attachment_extensions),
    }


def _candidate_pool(config: StudyConfig) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    by_source: dict[str, list[int]] = defaultdict(list)
    exact: dict[str, list[int]] = defaultdict(list)
    conflicted: set[int] = set()
    scanned: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    failures: list[dict[str, str]] = []
    desired = config.source_cap + config.reserve_per_source
    for source in SOURCES:
        for message in _source_messages(config.raw_root, source, config.seed):
            if len(by_source[source]) >= desired:
                break
            scanned[source] += 1
            try:
                row = _record(message)
            except Exception as error:
                skipped["processing_failure"] += 1
                failures.append(
                    {
                        "source": source,
                        "source_path": message.source_path,
                        "error_type": type(error).__name__,
                        "error": str(error)[:400],
                    }
                )
                continue
            if row["body_source"] == "missing":
                skipped["missing_body"] += 1
                continue
            if row["language_cohort"] != "english_main":
                skipped["not_english_main"] += 1
                continue
            keys = ("raw:" + row["raw_sha256"], "content:" + row["content_sha256"])
            previous = {index for key in keys for index in exact[key]}
            if previous:
                labels = {accepted[index]["corpus_label"] for index in previous}
                if labels == {row["corpus_label"]}:
                    skipped["exact_duplicate"] += 1
                    continue
                conflicted.update(previous)
                skipped["label_conflict"] += 1
            index = len(accepted)
            accepted.append(row)
            by_source[source].append(index)
            for key in keys:
                exact[key].append(index)
            if previous:
                conflicted.add(index)
            if len(by_source[source]) % 500 == 0:
                LOGGER.info(
                    "Accepted %s usable candidates from %s",
                    len(by_source[source]),
                    source,
                )
        LOGGER.info("Finished %s: %s usable candidates", source, len(by_source[source]))
    usable = {
        source: [index for index in by_source[source] if index not in conflicted]
        for source in SOURCES
    }
    selected_per_source = min(
        config.source_cap, *(len(usable[source]) for source in SOURCES)
    )
    output_indices: list[int] = []
    manifest: list[dict[str, Any]] = []
    for source in SOURCES:
        for rank, index in enumerate(usable[source], 1):
            output_indices.append(index)
            manifest.append(
                {
                    "sample_id": accepted[index]["sample_id"],
                    "source": source,
                    "corpus_label": accepted[index]["corpus_label"],
                    "duplicate_group_id": accepted[index]["duplicate_group_id"],
                    "candidate_rank_within_source": rank,
                    "pool_status": (
                        "primary" if rank <= selected_per_source else "reserve"
                    ),
                    "selection_seed": config.seed,
                }
            )
    pq.write_table(
        pa.Table.from_pylist([accepted[index] for index in output_indices]),
        config.canonical_path,
        compression="zstd",
    )
    pq.write_table(
        pa.Table.from_pylist(manifest), config.pool_manifest_path, compression="zstd"
    )
    (config.data_dir / "processing-failures.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in failures),
        encoding="utf-8",
    )
    return {
        "selected_per_source": selected_per_source,
        "primary_total": selected_per_source * 4,
        "pool_total": len(manifest),
        "scanned": dict(scanned),
        "skipped": dict(skipped),
        "failures": len(failures),
    }


def _tokens(subject: str, body: str) -> list[str]:
    return _TOKEN_RE.findall(
        unicodedata.normalize("NFKC", f"{subject}\n{body}").casefold()
    )


def _simhash(tokens: list[str]) -> int:
    features = (
        tokens
        if len(tokens) < 2
        else [left + "\u241f" + right for left, right in zip(tokens, tokens[1:])]
    )
    vector = [0] * 64
    for feature in features:
        value = int.from_bytes(hashlib.sha256(feature.encode()).digest()[:8], "big")
        for bit in range(64):
            vector[bit] += 1 if value & (1 << bit) else -1
    return sum((1 << bit) for bit, value in enumerate(vector) if value >= 0)


def _near_duplicate_edges(config: StudyConfig) -> dict[str, Any]:
    rows = pq.read_table(
        config.canonical_path,
        columns=[
            "sample_id",
            "source",
            "corpus_label",
            "duplicate_group_id",
            "subject",
            "body_text",
        ],
    ).to_pylist()
    records: list[dict[str, Any]] = []
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    emitted: set[tuple[str, str]] = set()
    edges: list[dict[str, Any]] = []
    overflow = 0
    for row in rows:
        tokens = _tokens(str(row["subject"]), str(row["body_text"]))
        if len(tokens) < 20:
            continue
        fingerprint = _simhash(tokens)
        keys = [(band, (fingerprint >> (band * 8)) & 255) for band in range(8)]
        candidates: set[int] = set()
        for key in keys:
            if len(buckets[key]) >= 250:
                overflow += 1
            else:
                candidates.update(buckets[key])
        current = {**row, "fingerprint": fingerprint, "token_count": len(tokens)}
        for index in sorted(candidates):
            previous = records[index]
            pair = tuple(sorted((str(previous["sample_id"]), str(row["sample_id"]))))
            ratio = min(previous["token_count"], len(tokens)) / max(
                previous["token_count"], len(tokens)
            )
            distance = (previous["fingerprint"] ^ fingerprint).bit_count()
            if (
                previous["duplicate_group_id"] != row["duplicate_group_id"]
                and pair not in emitted
                and ratio >= 0.75
                and distance <= 6
            ):
                emitted.add(pair)
                edges.append(
                    {
                        "left_sample_id": pair[0],
                        "right_sample_id": pair[1],
                        "hamming_distance": distance,
                        "token_count_ratio": ratio,
                    }
                )
        index = len(records)
        records.append(current)
        for key in keys:
            if len(buckets[key]) < 250:
                buckets[key].append(index)
    edge_schema = pa.schema(
        [
            pa.field("left_sample_id", pa.string()),
            pa.field("right_sample_id", pa.string()),
            pa.field("hamming_distance", pa.int8()),
            pa.field("token_count_ratio", pa.float64()),
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(edges, schema=edge_schema),
        config.similarity_path,
        compression="zstd",
    )
    audit = {
        "method": "token-bigram-simhash64",
        "candidate_pairs": len(edges),
        "bucket_overflow_events": overflow,
        "candidate_recall_is_exhaustive": overflow == 0,
    }
    _write_json(config.similarity_path.with_suffix(".audit.json"), audit)
    return audit


class _UnionFind:
    def __init__(self, values: Sequence[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def _allocate(count: int, train: float, validation: float) -> dict[str, int]:
    ratios = (train, validation, 1 - train - validation)
    raw = [count * ratio for ratio in ratios]
    values = [int(value) for value in raw]
    for index in sorted(range(3), key=lambda i: (raw[i] - values[i], -i), reverse=True)[
        : count - sum(values)
    ]:
        values[index] += 1
    return dict(zip(("train", "validation", "test"), values, strict=True))


def _split(config: StudyConfig) -> dict[str, Any]:
    pool = pq.read_table(config.pool_manifest_path).to_pylist()
    pool_ids = [str(row["sample_id"]) for row in pool]
    primary_ids = {
        str(row["sample_id"]) for row in pool if row["pool_status"] == "primary"
    }
    canonical = {
        str(row["sample_id"]): row
        for row in pq.read_table(
            config.canonical_path,
            columns=[
                "sample_id",
                "source",
                "corpus_label",
                "corpus_label_name",
                "duplicate_group_id",
            ],
        ).to_pylist()
    }
    union = _UnionFind(pool_ids)
    for edge in pq.read_table(config.similarity_path).to_pylist():
        union.union(str(edge["left_sample_id"]), str(edge["right_sample_id"]))
    full: dict[str, list[str]] = defaultdict(list)
    for sample_id in pool_ids:
        full[union.find(sample_id)].append(sample_id)
    group_by_root = {
        root: "similarity_"
        + hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:24]
        for root, ids in full.items()
    }
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample_id in sorted(primary_ids):
        groups[group_by_root[union.find(sample_id)]].append(canonical[sample_id])
    if any(
        len({row["source"] for row in members}) > 1
        or len({row["corpus_label"] for row in members}) > 1
        for members in groups.values()
    ):
        raise ValueError("A similarity component crosses source or label")
    selected_per_source = len(primary_ids) // 4
    targets = _allocate(
        selected_per_source, config.train_ratio, config.validation_ratio
    )
    assignments: dict[str, str] = {}
    for source in SOURCES:
        counts = {name: 0 for name in targets}
        source_groups = [
            (gid, members)
            for gid, members in groups.items()
            if members[0]["source"] == source
        ]
        source_groups.sort(
            key=lambda item: (
                -len(item[1]),
                hashlib.sha256(f"{config.seed}:{item[0]}".encode()).hexdigest(),
            )
        )
        for gid, members in source_groups:
            split = min(
                targets,
                key=lambda name: (
                    (counts[name] + len(members)) / targets[name],
                    hashlib.sha256(f"{config.seed}:{gid}:{name}".encode()).hexdigest(),
                ),
            )
            assignments[gid] = split
            counts[split] += len(members)
    manifest_id = hashlib.sha256(
        f"{config.seed}:{selected_per_source}:{config.train_ratio}:{config.validation_ratio}".encode()
    ).hexdigest()[:16]
    manifest: list[dict[str, Any]] = []
    for gid, members in groups.items():
        for row in members:
            manifest.append(
                {
                    "sample_id": row["sample_id"],
                    "source": row["source"],
                    "corpus_label": row["corpus_label"],
                    "corpus_label_name": row["corpus_label_name"],
                    "duplicate_group_id": row["duplicate_group_id"],
                    "similarity_group_id": gid,
                    "similarity_group_size": len(members),
                    "split": assignments[gid],
                    "selection_seed": config.seed,
                    "selection_manifest_id": manifest_id,
                }
            )
    manifest.sort(
        key=lambda row: (
            row["source"],
            row["split"],
            row["similarity_group_id"],
            row["sample_id"],
        )
    )
    pq.write_table(
        pa.Table.from_pylist(manifest), config.split_path, compression="zstd"
    )
    counts = Counter(row["split"] for row in manifest)
    audit = {
        "selected_total": len(manifest),
        "counts": dict(counts),
        "selected_per_source": selected_per_source,
        "selection_manifest_id": manifest_id,
        "similarity_groups": len(groups),
    }
    _write_json(config.split_path.with_suffix(".audit.json"), audit)
    return audit


def _raw_bytes(
    config: StudyConfig, row: dict[str, Any], cache: dict[Path, list[bytes]]
) -> bytes:
    path = config.raw_root / str(row["source_path"])
    if row["source"] != "nazario":
        return path.read_bytes()
    if path not in cache:
        box = mailbox.mbox(path, factory=lambda handle: handle.read(), create=False)
        try:
            cache[path] = list(box)
        finally:
            box.close()
    return cache[path][int(row["source_message_index"])]


def _representations(config: StudyConfig) -> dict[str, Any]:
    selected = {
        str(row["sample_id"]): row
        for row in pq.read_table(config.split_path).to_pylist()
    }
    canonical = {
        str(row["sample_id"]): row
        for row in pq.read_table(config.canonical_path).to_pylist()
        if str(row["sample_id"]) in selected
    }
    v1: list[dict[str, Any]] = []
    v2: list[dict[str, Any]] = []
    cache: dict[Path, list[bytes]] = {}
    for sample_id in sorted(selected):
        row = canonical[sample_id]
        v1_text = str(row["detector_input_v1"])
        v2_text = enrich_v2(
            v1_text, _raw_bytes(config, row, cache), tuple(row["attachment_extensions"])
        )
        v1.append(
            {
                "sample_id": sample_id,
                "detector_input_text": v1_text,
                "representation_version": V1_VERSION,
            }
        )
        v2.append(
            {
                "sample_id": sample_id,
                "detector_input_text": v2_text,
                "representation_version": V2_VERSION,
            }
        )
    pq.write_table(
        pa.Table.from_pylist(v1), config.detector_v1_path, compression="zstd"
    )
    pq.write_table(
        pa.Table.from_pylist(v2), config.detector_v2_path, compression="zstd"
    )
    return {"rows": len(v1), "v1": V1_VERSION, "v2": V2_VERSION}


def build_dataset(
    config: StudyConfig, *, force: bool = False, full_raw_audit: bool = False
) -> dict[str, Any]:
    """Build every data artefact in the only valid order, or reuse a verified build."""
    config.ensure_directories()
    raw_audit = audit_raw_data(config, full=full_raw_audit)
    expected = (
        config.canonical_path,
        config.pool_manifest_path,
        config.similarity_path,
        config.split_path,
        config.detector_v1_path,
        config.detector_v2_path,
    )
    if not force and all(path.exists() for path in expected):
        return {
            "status": "reused",
            "raw_audit": raw_audit,
            "split": json.loads(
                config.split_path.with_suffix(".audit.json").read_text()
            ),
            "representations": {
                "rows": pq.read_metadata(config.detector_v2_path).num_rows
            },
        }
    pool = _candidate_pool(config)
    similarity = _near_duplicate_edges(config)
    split = _split(config)
    representations = _representations(config)
    audit = {
        "status": "rebuilt",
        "raw_audit": raw_audit,
        "pool": pool,
        "similarity": similarity,
        "split": split,
        "representations": representations,
    }
    _write_json(config.data_dir / "data-audit.json", audit)
    return audit

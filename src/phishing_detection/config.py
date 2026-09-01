"""Single source of truth for paths and frozen study settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class StudyConfig:
    """Configuration shared by all notebooks.

    Paths are relative to ``root`` so the project remains portable. Seeds and
    split ratios are fixed here rather than repeated across notebook cells.
    """

    root: Path = Path.cwd()
    seed: int = 20260808
    source_cap: int = 4_000
    reserve_per_source: int = 500
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    bootstrap_replicates: int = 10_000

    @property
    def raw_root(self) -> Path:
        return self.root / "ori_data"

    @property
    def artefact_root(self) -> Path:
        return self.root / "artifacts"

    @property
    def data_dir(self) -> Path:
        return self.artefact_root / "data"

    @property
    def nlp_dir(self) -> Path:
        return self.artefact_root / "nlp"

    @property
    def llm_root(self) -> Path:
        return self.artefact_root / "llm"

    @property
    def zero_shot_dir(self) -> Path:
        return self.llm_root / "zero-shot"

    @property
    def retrieval_four_shot_dir(self) -> Path:
        return self.llm_root / "retrieval-four-shot"

    @property
    def retrieval_manifest_path(self) -> Path:
        return self.retrieval_four_shot_dir / "retrieval-manifest.parquet"

    @property
    def canonical_path(self) -> Path:
        return self.data_dir / "canonical.parquet"

    @property
    def pool_manifest_path(self) -> Path:
        return self.data_dir / "candidate-pool.parquet"

    @property
    def similarity_path(self) -> Path:
        return self.data_dir / "similarity-edges.parquet"

    @property
    def split_path(self) -> Path:
        return self.data_dir / "split-manifest.parquet"

    @property
    def detector_v1_path(self) -> Path:
        return self.data_dir / "detector-input-v1.parquet"

    @property
    def detector_v2_path(self) -> Path:
        return self.data_dir / "detector-input-v2.parquet"

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir,
            self.nlp_dir,
            self.zero_shot_dir,
            self.retrieval_four_shot_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def validate_raw_layout(self) -> None:
        required = (
            self.raw_root / "Enron" / "maildir",
            self.raw_root / "Nazario",
            self.raw_root / "PhishingPot" / "PhishingPot",
            self.raw_root / "SpamAssassin" / "easy_ham",
            self.raw_root / "SpamAssassin" / "easy_ham_2",
            self.raw_root / "SpamAssassin" / "hard_ham",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing raw corpus paths: " + ", ".join(missing))

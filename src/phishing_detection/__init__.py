"""Reproducible phishing-email detection study."""

from .config import StudyConfig
from .data_pipeline import build_dataset as prepare_data
from .evaluation import (
    CORE_CLASSIFICATION_METRICS,
    SUPPLEMENTARY_RANKING_METRICS,
    load_complete_test_comparison,
)
from .nlp import run_nlp_study
from .retrieval_four_shot import (
    build_retrieval_manifest,
    preview_retrieval_four_shot_request,
    run_retrieval_four_shot_batch,
)
from .zero_shot import run_zero_shot_batch

__all__ = [
    "StudyConfig",
    "prepare_data",
    "run_nlp_study",
    "CORE_CLASSIFICATION_METRICS",
    "SUPPLEMENTARY_RANKING_METRICS",
    "load_complete_test_comparison",
    "run_zero_shot_batch",
    "build_retrieval_manifest",
    "preview_retrieval_four_shot_request",
    "run_retrieval_four_shot_batch",
]

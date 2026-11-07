"""Tests for the macOS evaluation helpers, with stub encoders and stores (no models, no network)."""

from datasets import Dataset
import numpy as np
import pytest
import torch

from evaluate_mac import ENCODER_CHECK_ROWS, attention_for, check_encoder
from memorilla.data import QUESTION_COLUMN

EMBEDDING_DIM = 4


class StubEncoder:
    """Returns fixed vectors in place of a transformers encoder."""

    def __init__(self, vectors: np.ndarray) -> None:
        self._vectors = vectors

    def vectors(self, texts: list[str]) -> np.ndarray:
        return self._vectors[: len(texts)]


class StubStore:
    """Returns fixed question embeddings in place of an embedding store."""

    def __init__(self, questions: torch.Tensor) -> None:
        self._questions = questions

    def question(self, row: int) -> torch.Tensor:
        return self._questions[row]


def unit_rows(count: int) -> np.ndarray:
    """Distinct unit vectors, one per row."""
    rng = np.random.default_rng(0)
    rows = rng.normal(size=(count, EMBEDDING_DIM)).astype(np.float32)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


@pytest.fixture
def dataset() -> Dataset:
    return Dataset.from_dict({QUESTION_COLUMN: [f"question {row}" for row in range(ENCODER_CHECK_ROWS)]})


def test_attention_is_eager_on_mps_and_sdpa_elsewhere() -> None:
    assert attention_for(torch.device("mps")) == "eager"
    assert attention_for(torch.device("cpu")) == "sdpa"
    assert attention_for(torch.device("cuda")) == "sdpa"


def test_check_encoder_accepts_matching_embeddings(dataset: Dataset) -> None:
    vectors = unit_rows(ENCODER_CHECK_ROWS)
    lowest = check_encoder(StubEncoder(vectors), dataset, StubStore(torch.from_numpy(vectors).half()))
    assert lowest == pytest.approx(1.0, abs=1e-3)


def test_check_encoder_rejects_different_embeddings(dataset: Dataset) -> None:
    vectors = unit_rows(ENCODER_CHECK_ROWS)
    with pytest.raises(RuntimeError):
        check_encoder(StubEncoder(vectors), dataset, StubStore(torch.from_numpy(-vectors)))


def test_check_encoder_rejects_nan(dataset: Dataset) -> None:
    vectors = unit_rows(ENCODER_CHECK_ROWS)
    computed = vectors.copy()
    computed[0] = np.nan
    with pytest.raises(RuntimeError):
        check_encoder(StubEncoder(computed), dataset, StubStore(torch.from_numpy(vectors)))

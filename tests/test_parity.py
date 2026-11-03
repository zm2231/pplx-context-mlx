"""FP32 parity against vectors recorded from upstream's PyTorch model (scripts/make_reference.py)."""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from pplx_context_mlx import load_model, text

FIXTURES = Path(__file__).parent / "fixtures"
CASES = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
QUANTIZED_MAX_ABS_ERROR = 1
MIN_COSINE = 0.9999


@pytest.fixture(scope="module")
def engine():
    return load_model()


@pytest.fixture(scope="module")
def reference():
    return np.load(FIXTURES / "reference.npz")


def _cosines(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    live = np.linalg.norm(right, axis=-1) > 0
    left, right = left[live], right[live]
    return (left * right).sum(-1) / (np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1))


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_quantized_vectors_match_reference(case, engine, reference):
    encode = engine.encode_queries if case["task"] == "query" else engine.encode
    rows = encode(case["documents"], batch_size=len(case["documents"]))
    for index, row in enumerate(rows):
        expected = reference[f"{case['id']}/encoded/{index}"].astype(np.float32)
        assert row.shape == expected.shape
        if row.size == 0:
            continue
        assert np.abs(row - expected).max() <= QUANTIZED_MAX_ABS_ERROR
        cosines = _cosines(row, expected)
        assert cosines.size == 0 or cosines.min() >= MIN_COSINE


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_batching_matches_one_at_a_time(case, engine):
    encode = engine.encode_queries if case["task"] == "query" else engine.encode
    batched = encode(case["documents"], batch_size=len(case["documents"]))
    single = encode(case["documents"], batch_size=1)
    for left, right in zip(batched, single, strict=True):
        assert left.shape == right.shape
        if left.size:
            assert np.abs(left - right).max() <= QUANTIZED_MAX_ABS_ERROR


def test_layer_states_match_reference(engine, reference):
    case = next(case for case in CASES if case["id"] == "contextual_three_chunks")
    features = engine.tokenizer.encode_documents(case["documents"])
    trace: list[mx.array] = []
    engine.model.hidden_states(mx.array(features["input_ids"]), mx.array(features["attention_mask"]), trace)
    for index, state in enumerate(trace[:-1], start=1):
        expected = reference[f"trace/{index}"]
        error = np.abs(np.array(state) - expected).max() / np.abs(expected).max()
        assert error < 1e-3, f"layer {index} relative error {error}"


def test_sliced_attention_matches_whole(monkeypatch):
    query = mx.random.normal((2, 8, 40, 16))
    key = mx.random.normal((2, 2, 40, 16))
    value = mx.random.normal((2, 2, 40, 16))
    mask = mx.array([[True] * 40, [True] * 25 + [False] * 15])[:, None, None, :]
    whole = text._sliced_attention(query, key, value, 0.25, mask)
    monkeypatch.setattr(text, "QUERY_SLICE_TOKENS", 7)
    sliced = text._sliced_attention(query, key, value, 0.25, mask)
    assert np.allclose(np.array(whole), np.array(sliced), atol=1e-6)


def test_context_changes_the_vector_for_identical_chunk_text(engine):
    first = engine.encode([["Our quarterly revenue grew by a third.", "It was the best result in company history."]])[0][1]
    second = engine.encode([["The marathon runner broke the national record.", "It was the best result in company history."]])[0][1]
    assert not np.array_equal(first, second)

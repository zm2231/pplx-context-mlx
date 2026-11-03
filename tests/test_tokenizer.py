import json
from pathlib import Path

import numpy as np
import pytest

from pplx_context_mlx import resolve_checkpoint
from pplx_context_mlx.config import ModelConfig
from pplx_context_mlx.tokenizer import ContextualTokenizer

FIXTURES = Path(__file__).parent / "fixtures"
CASES = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tokenizer():
    path = resolve_checkpoint("perplexity-ai/pplx-embed-v2-context-0.6b")
    return ContextualTokenizer(path, ModelConfig.from_dict(json.loads((path / "config.json").read_text())))


@pytest.fixture(scope="module")
def reference():
    return np.load(FIXTURES / "reference.npz")


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_input_mapping_matches_reference(case, tokenizer, reference):
    encode = tokenizer.encode_queries if case["task"] == "query" else tokenizer.encode_documents
    features = encode(case["documents"])
    assert np.array_equal(features["input_ids"], reference[f"{case['id']}/input_ids"])
    if case["task"] == "document":
        assert np.array_equal(features["chunk_token_spans"], reference[f"{case['id']}/spans"])


def test_document_prefix_is_literal_text_and_query_prefix_is_reserved(tokenizer):
    assert tokenizer.encode_documents([["x"]])["input_ids"][0, 0] != 248078
    assert tokenizer.encode_queries([["x"]])["input_ids"][0, 0] == 248077


def test_query_row_must_have_one_string(tokenizer):
    with pytest.raises(ValueError, match="exactly one"):
        tokenizer.encode_queries([["a", "b"]])

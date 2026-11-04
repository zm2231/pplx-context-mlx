import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("datasets")
spec = importlib.util.spec_from_file_location("vidore_eval", Path(__file__).resolve().parent.parent / "evals" / "vidore.py")
vidore = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vidore)


class FakeTokenizer:
    def encode(self, text):
        words = text.split()
        offsets, cursor = [], 0
        for word in words:
            start = text.index(word, cursor)
            cursor = start + len(word)
            offsets.append((start, cursor))
        return SimpleNamespace(ids=list(range(len(words))), offsets=offsets)


class FakeEngine:
    def __init__(self):
        self.tokenizer = SimpleNamespace(_tokenizer=FakeTokenizer())
        self.calls = []

    def encode(self, rows, batch_size, normalize_embeddings):
        self.calls.append(rows[0])
        return [np.ones((len(rows[0]), 2), dtype=np.float32)]


CORPUS = [
    {"doc_id": "a", "page_number_in_doc": 2, "corpus_id": 11, "markdown": "three four"},
    {"doc_id": "b", "page_number_in_doc": 1, "corpus_id": 20, "markdown": "other"},
    {"doc_id": "a", "page_number_in_doc": 1, "corpus_id": 10, "markdown": "one two"},
    {"doc_id": "a", "page_number_in_doc": 3, "corpus_id": 12, "markdown": None},
    {"doc_id": "a", "page_number_in_doc": 4, "corpus_id": 13, "markdown": "five"},
]


def test_context_mode_keeps_page_order_and_aligned_owners():
    engine = FakeEngine()
    vectors, owners = vidore._encode_by_document(engine, CORPUS, chunk_tokens=1, context_tokens=4)

    assert sum(len(row) for row in vectors) == len(owners)
    assert owners == [10, 10, 11, 11, 12, 13, 20]
    assert [chunk.strip() for window in engine.calls for chunk in window] == ["one", "two", "three", "four", "", "five", "other"]


def test_context_and_page_modes_cover_the_same_pages():
    page_owners = vidore._encode_by_page(FakeEngine(), CORPUS, 1)[1]
    context_owners = vidore._encode_by_document(FakeEngine(), CORPUS, 1, 4)[1]

    assert sorted(page_owners) == sorted(context_owners)


def test_windows_partition_chunks_within_the_limit():
    engine = FakeEngine()
    chunks = ["a b", "c d", "e", "f g h"]

    windows = vidore._windows(engine, chunks, 4)

    assert [chunk for window in windows for chunk in window] == chunks
    assert windows == [["a b"], ["c d"], ["e"], ["f g h"]]

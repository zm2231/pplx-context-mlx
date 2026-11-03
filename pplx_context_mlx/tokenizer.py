from __future__ import annotations

from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from .config import ModelConfig


class ContextualTokenizer:
    """Upstream's text input mapping.

    ``encode_special_tokens`` matches the reference's ``split_special_tokens=True``: chunk text
    can't inject control tokens, and the literal document prefix tokenizes as ordinary text
    rather than as its reserved id. Queries do use the reserved query prefix id.
    """

    def __init__(self, path: str | Path, config: ModelConfig, pad_token: str = "<|endoftext|>"):
        self._tokenizer = Tokenizer.from_file(str(Path(path) / "tokenizer.json"))
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        self._tokenizer.encode_special_tokens = True
        self.config = config
        self.pad_id = self.reserved_id(pad_token)
        self.query_prefix_id = self.reserved_id(config.query_prefix)

    def reserved_id(self, token: str) -> int:
        value = self._tokenizer.token_to_id(token)
        if value is None:
            raise ValueError(f"Tokenizer has no token {token!r}")
        return value

    def pad(self, rows: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
        width = max(len(row) for row in rows)
        input_ids = np.full((len(rows), width), self.pad_id, dtype=np.int32)
        attention_mask = np.zeros((len(rows), width), dtype=np.bool_)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = row
            attention_mask[index, : len(row)] = True
        return input_ids, attention_mask

    def encode_queries(self, queries: list[list[str]]) -> dict[str, np.ndarray]:
        if any(len(row) != 1 for row in queries):
            raise ValueError("Each query must contain exactly one string")
        encoded = self._tokenizer.encode_batch([row[0] for row in queries])
        input_ids, attention_mask = self.pad([[self.query_prefix_id, *item.ids] for item in encoded])
        if input_ids.shape[1] > self.config.query_length:
            raise ValueError(f"Input exceeds the {self.config.query_length}-token limit")
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def encode_documents(self, documents: list[list[str]]) -> dict[str, np.ndarray]:
        prefix, marker = self.config.document_prefix, self.config.boundary_marker
        texts, char_spans = [], []
        for chunks in documents:
            parts, spans, cursor = [prefix], [], len(prefix)
            for index, chunk in enumerate(chunks):
                if index:
                    parts.append(marker)
                    cursor += len(marker)
                spans.append((cursor, cursor + len(chunk)))
                parts.append(chunk)
                cursor += len(chunk)
            texts.append("".join(parts))
            char_spans.append(spans)
        encoded = self._tokenizer.encode_batch(texts)
        input_ids, attention_mask = self.pad([item.ids for item in encoded])
        if input_ids.shape[1] > self.config.document_length:
            raise ValueError(
                f"Input exceeds the {self.config.document_length}-token limit; split the document into context windows"
            )
        token_spans = np.zeros((len(documents), max(1, *map(len, documents)), 2), dtype=np.int32)
        for row, spans in enumerate(char_spans):
            offsets = np.array(encoded[row].offsets, dtype=np.int64).reshape(-1, 2)
            starts, ends = offsets[:, 0], offsets[:, 1]
            valid = ends > starts
            for column, (start, end) in enumerate(spans):
                indices = np.flatnonzero(valid & (starts < end) & (ends > start))
                if indices.size:
                    token_spans[row, column] = (indices[0], indices[-1] + 1)
                elif documents[row][column]:
                    raise ValueError(f"Chunk {column} in document {row} produced no tokens")
        return {"input_ids": input_ids, "attention_mask": attention_mask, "chunk_token_spans": token_spans}

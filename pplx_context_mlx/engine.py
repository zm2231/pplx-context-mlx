from __future__ import annotations

import mlx.core as mx
import numpy as np
from PIL import Image

from .checkpoint import (
    CHECKPOINT_0_6B,
    load_checkpoint,
    resolve_checkpoint,
    validate_dtype,
)
from .config import ModelConfig
from .images import prepare_images
from .model import PplxContextualModel, quantize_int8
from .tokenizer import ContextualTokenizer


def _validate(documents: list[list[str]], batch_size: int) -> None:
    if not isinstance(documents, list) or any(
        not isinstance(row, list) or any(not isinstance(text, str) for text in row) for row in documents
    ):
        raise TypeError("Expected a list of documents, each a list of text chunks")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")


def _normalize(vectors: mx.array) -> mx.array:
    return vectors / mx.maximum(mx.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)


class ContextualEngine:
    """Same call shape as upstream: one (chunk_count, embedding_dim) array per document."""

    def __init__(self, model: PplxContextualModel, tokenizer: ContextualTokenizer, config: ModelConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def project_documents(self, documents: list[list[str]]) -> mx.array:
        features = self.tokenizer.encode_documents(documents)
        return self.model.embed_documents(
            mx.array(features["input_ids"]),
            mx.array(features["attention_mask"]),
            mx.array(features["chunk_token_spans"]),
        )

    def project_queries(self, queries: list[list[str]]) -> mx.array:
        features = self.tokenizer.encode_queries(queries)
        return self.model.embed_queries(mx.array(features["input_ids"]), mx.array(features["attention_mask"]))

    def _encode(self, rows: list[list[str]], project, batch_size: int, normalize_embeddings: bool) -> list[np.ndarray]:
        _validate(rows, batch_size)
        output: list[np.ndarray] = []
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            vectors = quantize_int8(project(batch))
            if normalize_embeddings:
                vectors = _normalize(vectors)
            mx.eval(vectors)
            matrix = np.array(vectors, dtype=np.float32, copy=True)
            output.extend(matrix[index, : len(chunks)].copy() for index, chunks in enumerate(batch))
        return output

    def encode(
        self, documents: list[list[str]], batch_size: int = 32, *, normalize_embeddings: bool = False
    ) -> list[np.ndarray]:
        return self._encode(documents, self.project_documents, batch_size, normalize_embeddings)

    def encode_queries(
        self, queries: list[list[str]], batch_size: int = 32, *, normalize_embeddings: bool = False
    ) -> list[np.ndarray]:
        return self._encode(queries, self.project_queries, batch_size, normalize_embeddings)

    def prepare_images(
        self, images: list[Image.Image], *, chunks: int = 8, min_pixels: int = 16384, max_pixels: int = 1800000
    ) -> dict[str, np.ndarray]:
        return prepare_images(
            images, self.tokenizer, self.config, chunks=chunks, min_pixels=min_pixels, max_pixels=max_pixels
        )

    def encode_images(
        self,
        images: list[Image.Image],
        batch_size: int = 1,
        *,
        normalize_embeddings: bool = False,
        chunks: int = 8,
        min_pixels: int = 16384,
        max_pixels: int = 1800000,
    ) -> list[np.ndarray]:
        """Return one (band_count, embedding_dim) array per image."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        output: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            features = self.prepare_images(
                images[start : start + batch_size], chunks=chunks, min_pixels=min_pixels, max_pixels=max_pixels
            )
            vectors = quantize_int8(self.model.embed_images(features))
            if normalize_embeddings:
                vectors = _normalize(vectors)
            mx.eval(vectors)
            matrix = np.array(vectors, dtype=np.float32, copy=True)
            output.extend(row[mask] for row, mask in zip(matrix, features["chunk_mask"], strict=True))
        return output


def load_model(
    path_or_repo: str = CHECKPOINT_0_6B,
    revision: str | None = None,
    dtype: str = "float32",
    cache_limit_mb: int = 512,
) -> ContextualEngine:
    validate_dtype(dtype)
    mx.set_cache_limit(cache_limit_mb * 1024 * 1024)
    path = resolve_checkpoint(path_or_repo, revision)
    model, config = load_checkpoint(path, dtype)
    return ContextualEngine(model, ContextualTokenizer(path, config), config)

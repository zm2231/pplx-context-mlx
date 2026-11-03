from __future__ import annotations

import mlx.core as mx
from mlx import nn

from .config import ModelConfig
from .text import TextModel


def mean_pool(hidden: mx.array, attention_mask: mx.array) -> mx.array:
    mask = attention_mask[..., None].astype(hidden.dtype)
    return ((hidden * mask).sum(axis=1) / mx.maximum(mask.sum(axis=1), 1.0))[:, None]


def span_pool(hidden: mx.array, spans: mx.array) -> mx.array:
    """Mean of each chunk's tokens; spans are (batch, chunks, 2) half-open token ranges."""
    batch, _, width = hidden.shape
    prefix = mx.concatenate([mx.zeros((batch, 1, width), hidden.dtype), mx.cumsum(hidden, axis=1)], axis=1)
    rows = mx.arange(batch)[:, None]
    starts, ends = spans[..., 0], spans[..., 1]
    counts = mx.maximum(ends - starts, 1)[..., None].astype(hidden.dtype)
    return (prefix[rows, ends] - prefix[rows, starts]) / counts


def quantize_int8(projected: mx.array) -> mx.array:
    return mx.clip(mx.round(mx.tanh(projected) * 127), -128, 127)


class PplxContextualModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.language_model = TextModel(config.text)
        self.contextual_projection = nn.Linear(config.text.hidden_size, config.embedding_dim, bias=False)

    def hidden_states(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
        trace: list[mx.array] | None = None,
    ) -> mx.array:
        batch, length = input_ids.shape
        position_ids = mx.broadcast_to(mx.arange(length), (3, batch, length))
        embeds = self.language_model.embed_tokens(input_ids)
        return self.language_model(embeds, attention_mask, position_ids, trace).astype(mx.float32)

    def project(self, pooled: mx.array) -> mx.array:
        return pooled @ self.contextual_projection.weight.astype(mx.float32).T

    def embed_queries(self, input_ids: mx.array, attention_mask: mx.array) -> mx.array:
        return self.project(mean_pool(self.hidden_states(input_ids, attention_mask), attention_mask))

    def embed_documents(self, input_ids: mx.array, attention_mask: mx.array, spans: mx.array) -> mx.array:
        return self.project(span_pool(self.hidden_states(input_ids, attention_mask), spans))

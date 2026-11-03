"""The Qwen3.5 vision encoder, with parameter names matching the upstream checkpoint."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx import nn

from .config import VisionConfig


class PatchProjection(nn.Module):
    """Upstream's Conv3d with kernel equal to stride: a linear map over flattened patches."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        patch = config.patch_size
        self.weight = mx.zeros((config.hidden_size, config.in_channels, config.temporal_patch_size, patch, patch))
        self.bias = mx.zeros((config.hidden_size,))

    def __call__(self, pixels: mx.array) -> mx.array:
        weight = self.weight.reshape(self.weight.shape[0], -1)
        return pixels.astype(weight.dtype) @ weight.T + self.bias


class PatchEmbed(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.proj = PatchProjection(config)

    def __call__(self, pixels: mx.array) -> mx.array:
        return self.proj(pixels)


def _rotate(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    y = x.astype(mx.float32)
    turned = mx.concatenate([-y[..., half:], y[..., :half]], axis=-1)
    return (y * cos + turned * sin).astype(x.dtype)


class VisionAttention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=True)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array, lengths: list[int]) -> mx.array:
        total = x.shape[0]
        qkv = self.qkv(x).reshape(total, 3, self.num_heads, self.head_dim)
        query, key, value = _rotate(qkv[:, 0], cos, sin), _rotate(qkv[:, 1], cos, sin), qkv[:, 2]
        outputs = []
        offset = 0
        for length in lengths:
            q, k, v = (item[offset : offset + length].transpose(1, 0, 2)[None] for item in (query, key, value))
            attended = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
            outputs.append(attended.transpose(0, 2, 1, 3).reshape(length, -1))
            offset += length
        return self.proj(mx.concatenate(outputs, axis=0))


class VisionMLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_fc2(nn.gelu_approx(self.linear_fc1(x)))


class VisionBlock(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = VisionAttention(config)
        self.mlp = VisionMLP(config)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array, lengths: list[int]) -> mx.array:
        x = x + self.attn(self.norm1(x), cos, sin, lengths)
        return x + self.mlp(self.norm2(x))


class PatchMerger(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        merged = config.hidden_size * config.spatial_merge_size**2
        self.merged = merged
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(merged, merged, bias=True)
        self.linear_fc2 = nn.Linear(merged, config.out_hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_fc2(nn.gelu(self.linear_fc1(self.norm(x).reshape(-1, self.merged))))


def _merge_order(values: np.ndarray, height: int, width: int, merge: int) -> np.ndarray:
    tail = values.shape[2:]
    blocks = values.reshape(height // merge, merge, width // merge, merge, *tail)
    return blocks.transpose(0, 2, 1, 3, *range(4, blocks.ndim)).reshape(height * width, *tail)


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.grid_side = int(config.num_position_embeddings**0.5)
        self.patch_embed = PatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.blocks = [VisionBlock(config) for _ in range(config.depth)]
        self.merger = PatchMerger(config)
        spatial = (config.hidden_size // config.num_heads) // 2
        self._inv_freq = 1.0 / (10000.0 ** (np.arange(0, spatial, 2, dtype=np.float32) / spatial))

    def _position_embeddings(self, grids: np.ndarray) -> mx.array:
        merge = self.config.spatial_merge_size
        side = self.grid_side
        parts = []
        for _, height, width in grids.tolist():
            rows = np.linspace(0, side - 1, height, dtype=np.float32)
            cols = np.linspace(0, side - 1, width, dtype=np.float32)
            row_low, col_low = rows.astype(np.int64), cols.astype(np.int64)
            row_high, col_high = np.minimum(row_low + 1, side - 1), np.minimum(col_low + 1, side - 1)
            row_frac, col_frac = (rows - row_low)[:, None], (cols - col_low)[None]
            corners = [
                (row_low, col_low, (1 - row_frac) * (1 - col_frac)),
                (row_low, col_high, (1 - row_frac) * col_frac),
                (row_high, col_low, row_frac * (1 - col_frac)),
                (row_high, col_high, row_frac * col_frac),
            ]
            dtype = self.pos_embed.weight.dtype
            terms = [
                self.pos_embed(mx.array((row[:, None] * side + col[None]).reshape(-1)))
                * mx.array(weight.reshape(-1, 1), dtype=dtype)
                for row, col, weight in corners
            ]
            mixed = terms[0] + terms[1] + terms[2] + terms[3]
            flat = np.arange(height * width).reshape(height, width)
            parts.append(mixed[mx.array(_merge_order(flat, height, width, merge))])
        return mx.concatenate(parts, axis=0)

    def _rotary(self, grids: np.ndarray) -> tuple[mx.array, mx.array]:
        merge = self.config.spatial_merge_size
        parts = []
        for _, height, width in grids.tolist():
            rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
            coords = np.stack([rows, cols], axis=-1)
            parts.append(_merge_order(coords, height, width, merge))
        freqs = np.concatenate(parts, axis=0)[..., None].astype(np.float32) * self._inv_freq
        flat = mx.array(freqs.reshape(freqs.shape[0], -1))
        doubled = mx.concatenate([flat, flat], axis=-1)[:, None]
        return mx.cos(doubled), mx.sin(doubled)

    def __call__(self, pixels: mx.array, grids: np.ndarray) -> mx.array:
        if (grids[:, 0] != 1).any():
            raise ValueError("Only still images are supported")
        lengths = [int(height * width) for _, height, width in grids.tolist()]
        x = self.patch_embed(pixels) + self._position_embeddings(grids).astype(self.pos_embed.weight.dtype)
        cos, sin = self._rotary(grids)
        for block in self.blocks:
            x = block(x, cos, sin, lengths)
            mx.eval(x)
        return self.merger(x)

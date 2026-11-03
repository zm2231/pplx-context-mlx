"""The Qwen3.5 text backbone, with parameter names matching the upstream checkpoint."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx import nn

from .config import TextConfig
from .gated_delta import gated_delta_update, normalize_qk


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = mx.zeros((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        y = x.astype(mx.float32)
        y = y * mx.rsqrt(mx.mean(y * y, axis=-1, keepdims=True) + self.eps)
        return (y * (1.0 + self.weight.astype(mx.float32))).astype(x.dtype)


class GatedRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        y = x.astype(mx.float32)
        y = y * mx.rsqrt(mx.mean(y * y, axis=-1, keepdims=True) + self.eps)
        y = self.weight * y.astype(x.dtype)
        return (y * nn.silu(gate.astype(mx.float32))).astype(x.dtype)


class RotaryEmbedding(nn.Module):
    """Interleaved T/H/W rotary frequencies. Text positions make all three axes equal."""

    def __init__(self, config: TextConfig):
        super().__init__()
        dim = int(config.head_dim * config.partial_rotary_factor)
        self._inv_freq = 1.0 / (config.rope_theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        axis = np.zeros(self._inv_freq.shape[0], dtype=np.int32)
        for index, offset in enumerate((1, 2), start=1):
            axis[offset : config.mrope_section[index] * 3 : 3] = index
        self._axis_masks = [mx.array((axis == index).astype(np.float32)) for index in range(3)]

    def _recompose(self, values: mx.array) -> mx.array:
        masks = self._axis_masks
        mixed = values[0] * masks[0] + values[1] * masks[1] + values[2] * masks[2]
        return mx.concatenate([mixed, mixed], axis=-1)

    def __call__(self, position_ids: mx.array, dtype: mx.Dtype) -> tuple[mx.array, mx.array]:
        freqs = position_ids[..., None].astype(mx.float32) * self._inv_freq
        return self._recompose(mx.cos(freqs)).astype(dtype), self._recompose(mx.sin(freqs)).astype(dtype)


def _rotate(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    dim = cos.shape[-1]
    head = x[..., :dim]
    half = dim // 2
    turned = mx.concatenate([-head[..., half:], head[..., :half]], axis=-1)
    return mx.concatenate([head * cos + turned * sin, x[..., dim:]], axis=-1)


class Attention(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.scale = config.head_dim**-0.5
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array, mask: mx.array | None) -> mx.array:
        batch, length, _ = x.shape
        query, gate = mx.split(self.q_proj(x).reshape(batch, length, self.num_heads, self.head_dim * 2), 2, axis=-1)
        query = self.q_norm(query).transpose(0, 2, 1, 3)
        key = self.k_norm(self.k_proj(x).reshape(batch, length, self.num_kv_heads, self.head_dim)).transpose(0, 2, 1, 3)
        value = self.v_proj(x).reshape(batch, length, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        cos, sin = cos[:, None], sin[:, None]
        out = mx.fast.scaled_dot_product_attention(
            _rotate(query, cos, sin),
            _rotate(key, cos, sin),
            value,
            scale=self.scale,
            mask=None if mask is None else mask[:, None, None, :],
        )
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(out * mx.sigmoid(gate.reshape(batch, length, -1)))


class CausalConv(nn.Module):
    """Per-channel causal convolution; the checkpoint stores weights as (channels, 1, kernel)."""

    def __init__(self, channels: int, kernel: int):
        super().__init__()
        self.weight = mx.zeros((channels, 1, kernel))

    def __call__(self, x: mx.array) -> mx.array:
        kernel = self.weight.shape[-1]
        length = x.shape[1]
        padded = mx.pad(x, [(0, 0), (kernel - 1, 0), (0, 0)])
        out = padded[:, :length] * self.weight[:, 0, 0]
        for tap in range(1, kernel):
            out = out + padded[:, tap : tap + length] * self.weight[:, 0, tap]
        return out


class GatedDeltaNet(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.k_dim = config.linear_key_head_dim
        self.v_dim = config.linear_value_head_dim
        key_width = self.num_k_heads * self.k_dim
        value_width = self.num_v_heads * self.v_dim
        self.splits = (key_width, key_width * 2)
        self.in_proj_qkv = nn.Linear(config.hidden_size, key_width * 2 + value_width, bias=False)
        self.in_proj_z = nn.Linear(config.hidden_size, value_width, bias=False)
        self.in_proj_b = nn.Linear(config.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(config.hidden_size, self.num_v_heads, bias=False)
        self.conv1d = CausalConv(key_width * 2 + value_width, config.linear_conv_kernel_dim)
        self.dt_bias = mx.ones((self.num_v_heads,))
        self.A_log = mx.zeros((self.num_v_heads,))
        self.norm = GatedRMSNorm(self.v_dim, config.rms_norm_eps)
        self.out_proj = nn.Linear(value_width, config.hidden_size, bias=False)

    def __call__(self, x: mx.array, mask: mx.array | None) -> mx.array:
        if mask is not None:
            x = x * mask[..., None].astype(x.dtype)
        batch, length, _ = x.shape
        query, key, value = mx.split(nn.silu(self.conv1d(self.in_proj_qkv(x))), self.splits, axis=-1)
        query = query.reshape(batch, length, self.num_k_heads, self.k_dim).astype(mx.float32)
        key = key.reshape(batch, length, self.num_k_heads, self.k_dim).astype(mx.float32)
        value = value.reshape(batch, length, self.num_v_heads, self.v_dim).astype(mx.float32)
        query, key = normalize_qk(query, key, inv_scale=self.k_dim**-0.5, eps=1e-6)
        out, _ = gated_delta_update(
            query,
            key,
            value,
            self.in_proj_a(x).astype(mx.float32),
            self.in_proj_b(x).astype(mx.float32),
            self.A_log.astype(mx.float32),
            self.dt_bias.astype(mx.float32),
        )
        gate = self.in_proj_z(x).reshape(batch, length, self.num_v_heads, self.v_dim)
        out = self.norm(out.astype(x.dtype), gate)
        return self.out_proj(out.reshape(batch, length, -1))


class MLP(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, kind: str):
        super().__init__()
        if kind == "linear_attention":
            self.linear_attn = GatedDeltaNet(config)
        else:
            self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array, mask: mx.array | None) -> mx.array:
        normed = self.input_layernorm(x)
        if "linear_attn" in self:
            x = x + self.linear_attn(normed, mask)
        else:
            x = x + self.self_attn(normed, cos, sin, mask)
        return x + self.mlp(self.post_attention_layernorm(x))


class TextModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DecoderLayer(config, kind) for kind in config.layer_types]
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary = RotaryEmbedding(config)

    def __call__(
        self,
        inputs_embeds: mx.array,
        attention_mask: mx.array,
        position_ids: mx.array,
        trace: list[mx.array] | None = None,
    ) -> mx.array:
        mask = None if bool(attention_mask.all()) else attention_mask
        cos, sin = self.rotary(position_ids, inputs_embeds.dtype)
        x = inputs_embeds
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
            mx.eval(x)
            if trace is not None:
                trace.append(x)
        return self.norm(x)

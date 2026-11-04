from types import SimpleNamespace

import mlx.core as mx
import pytest

from pplx_context_mlx import text

CONFIG = SimpleNamespace(
    linear_num_key_heads=2,
    linear_num_value_heads=2,
    linear_key_head_dim=128,
    linear_value_head_dim=64,
    hidden_size=32,
    linear_conv_kernel_dim=4,
    rms_norm_eps=1e-6,
)


def _network() -> text.GatedDeltaNet:
    net = text.GatedDeltaNet(CONFIG)
    net.conv1d.weight = mx.random.normal(net.conv1d.weight.shape)
    mx.eval(net.parameters())
    return net


@pytest.mark.parametrize(("length", "slice_tokens"), [(1, 2), (3, 2), (8, 2), (64, 7), (64, 16), (250, 100), (2048, 1024), (2049, 1024)])
def test_streaming_linear_attention_is_bit_identical(monkeypatch, length, slice_tokens):
    mx.random.seed(length)
    net = _network()
    x = mx.random.normal((2, length, CONFIG.hidden_size))
    monkeypatch.setattr(text, "SLICE_ABOVE_TOKENS", 10_000)
    whole = net(x, None)
    monkeypatch.setattr(text, "SLICE_ABOVE_TOKENS", 0)
    monkeypatch.setattr(text, "LINEAR_SLICE_TOKENS", slice_tokens)
    sliced = net(x, None)

    assert sliced.shape == whole.shape
    assert mx.abs(whole).max().item() > 0
    assert mx.array_equal(whole, sliced).item()


def test_streaming_applies_the_padding_mask(monkeypatch):
    mx.random.seed(9)
    net = _network()
    x = mx.random.normal((2, 40, CONFIG.hidden_size))
    mask = mx.array([[True] * 40, [True] * 25 + [False] * 15])
    monkeypatch.setattr(text, "SLICE_ABOVE_TOKENS", 10_000)
    whole = net(x, mask)
    monkeypatch.setattr(text, "SLICE_ABOVE_TOKENS", 0)
    monkeypatch.setattr(text, "LINEAR_SLICE_TOKENS", 9)

    assert mx.abs(whole).max().item() > 0
    assert mx.array_equal(whole, net(x, mask)).item()

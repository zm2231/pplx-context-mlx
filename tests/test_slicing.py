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


@pytest.mark.parametrize(("length", "slice_tokens"), [(8, 2), (64, 7), (64, 16), (250, 100), (2049, 1024), (3073, 1024)])
def test_streaming_linear_attention_matches_whole(monkeypatch, length, slice_tokens):
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
    assert mx.allclose(whole, sliced, atol=2e-4, rtol=0).item()


def test_state_carry_is_what_makes_slices_match(monkeypatch):
    mx.random.seed(3)
    net = _network()
    x = mx.random.normal((1, 64, CONFIG.hidden_size))
    monkeypatch.setattr(text, "SLICE_ABOVE_TOKENS", 10_000)
    whole = net(x, None)
    a = net.in_proj_a(x).astype(mx.float32)
    b = net.in_proj_b(x).astype(mx.float32)
    reset = mx.concatenate([net._step(x, a, b, start, start + 16, None)[0] for start in range(0, 64, 16)], axis=1)

    assert not mx.allclose(whole, reset, atol=2e-4, rtol=0).item()


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
    assert mx.allclose(whole, net(x, mask), atol=2e-4, rtol=0).item()

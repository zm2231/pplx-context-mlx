import mlx.core as mx
import pytest

from pplx_context_mlx import text


def test_sliced_convolution_is_bit_identical(monkeypatch):
    mx.random.seed(0)
    conv = text.CausalConv(6, 4)
    conv.weight = mx.random.normal((6, 1, 4))
    x = mx.random.normal((2, 2500, 6))
    monkeypatch.setattr(text, "CONV_SLICE_TOKENS", 10_000)
    whole = conv(x)
    monkeypatch.setattr(text, "CONV_SLICE_TOKENS", 7)
    sliced = conv(x)

    assert mx.array_equal(whole, sliced).item()
    assert sliced.shape == x.shape


@pytest.mark.parametrize("length", [1, 3, 8])
def test_short_input_matches_slice_boundaries(monkeypatch, length):
    mx.random.seed(1)
    conv = text.CausalConv(2, 4)
    conv.weight = mx.random.normal((2, 1, 4))
    x = mx.random.normal((1, length, 2))
    monkeypatch.setattr(text, "CONV_SLICE_TOKENS", 10_000)
    whole = conv(x)
    monkeypatch.setattr(text, "CONV_SLICE_TOKENS", 2)

    assert mx.array_equal(whole, conv(x)).item()

import mlx.core as mx
import numpy as np

from pplx_context_mlx.model import mean_pool, quantize_int8, span_pool


def test_span_pool_means_each_half_open_span():
    hidden = mx.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [9.0, 9.0]]])
    spans = mx.array([[[0, 2], [2, 3]]])
    assert np.allclose(np.array(span_pool(hidden, spans)), [[[2.0, 3.0], [5.0, 6.0]]])


def test_empty_span_pools_to_zero():
    hidden = mx.array([[[1.0, 2.0], [3.0, 4.0]]])
    assert np.allclose(np.array(span_pool(hidden, mx.array([[[0, 0]]]))), [[[0.0, 0.0]]])


def test_mean_pool_ignores_padding():
    hidden = mx.array([[[2.0], [4.0], [100.0]]])
    assert np.allclose(np.array(mean_pool(hidden, mx.array([[True, True, False]]))), [[[3.0]]])


def test_quantize_saturates_at_127():
    values = np.array(quantize_int8(mx.array([-100.0, 0.0, 100.0])))
    assert values.tolist() == [-127.0, 0.0, 127.0]

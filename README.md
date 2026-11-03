# pplx-context-mlx

Perplexity's [pplx-embed-v2-context](https://huggingface.co/perplexity-ai/pplx-embed-v2-context-0.6b) contextual embeddings on Apple silicon with MLX. Loads the upstream checkpoint unchanged, with no conversion step, PyTorch, or Transformers.

Independent project; not affiliated with or endorsed by Perplexity.

```python
from pplx_context_mlx import load_model

engine = load_model()
chunks = engine.encode([["First chunk of a document.", "Second chunk, which sees the first."]], normalize_embeddings=True)
query = engine.encode_queries([["What does the second chunk see?"]], normalize_embeddings=True)
scores = chunks[0] @ query[0][0]
```

Page images embed as horizontal bands, with one vector per band:

```python
from PIL import Image

bands = engine.encode_images([Image.open("page.png")], chunks=8, normalize_embeddings=True)[0]
page_score = (bands @ query[0][0]).max()
```

## Parity

Measured against the upstream PyTorch model at revision `6d1fddab`, FP32 on CPU, over 16 text cases and 8 image cases, including padded batches, multilingual text, and a 13,871-token document.

| MLX dtype | Largest int8 difference | Lowest cosine | Weights |
|---|---:|---:|---:|
| `float32` (default) | 1 step | 0.999998 | 2.38 GB |
| `float16` | 1 step | 0.99978 | 1.19 GB |

Both rows compare against upstream at FP32. Image pixels match upstream byte for byte on every fixture. Details are in `reports/parity-*.json`; regenerate with `scripts/make_reference.py` and `scripts/parity_report.py`.

## Speed

On an M4 Pro with 64 GB. MLX values are medians of three runs. Upstream ran on CPU at its best thread count for each workload, because 1 thread was fastest for short inputs and 4 to 6 for long ones. No CUDA comparison was run.

| Workload | MLX `float16` | MLX `float32` | Upstream CPU |
|---|---:|---:|---:|
| One query | 7.4 ms | 11.3 ms | 72.1 ms |
| Queries per second, batches of 32 | 598 | 516 | 51 |
| 224-token document | 0.029 s | 0.033 s | 0.161 s |
| 1,785-token document | 0.21 s | 0.25 s | 1.13 s |
| 7,137-token document | 1.16 s | 1.45 s | 4.47 s |
| A4 page, 8 bands | 0.91 s | 1.10 s | 4.11 s |

## Long documents

Attention scores queries in slices, so peak memory stays far below the square of the document length. Every chunk sees the whole document, so very long inputs are slow rather than impossible.

| Tokens | `float16` time | `float16` peak | `float32` time | `float32` peak |
|---:|---:|---:|---:|---:|
| 13,681 | 3.0 s | 2.7 GB | 3.7 s | 4.9 GB |
| 29,521 | 10.4 s | 4.0 GB | 14.7 s | 7.6 GB |
| 59,041 | 36.3 s | 6.7 GB | | |
| 119,041 | 144 s | 12.2 GB | | |

Accuracy is verified against upstream up to 13,871 tokens. Longer runs are measured for time and memory only. Reproduce with `scripts/benchmark.py` and `scripts/context_limit.py`; results are in `reports/`.

## Scope

Tested with `pplx-embed-v2-context-0.6b` only. This package contains no model weights and downloads them from Perplexity's repository; check that repository's terms before use, since its model card does not state a license. Video input is not supported.

Third-party notices are in `NOTICE`.

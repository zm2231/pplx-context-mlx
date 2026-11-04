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

## Retrieval

[ConTEB](https://github.com/illuin-tech/conteb) nDCG@10, on the official task splits, against Table 5 of the [paper](https://arxiv.org/abs/2610.38854). Reproduce with `evals/conteb.py`.

| Task | Paper | MLX `float32` | MLX `float16` |
|---|---:|---:|---:|
| Geography | 94.1 | 94.10 | 94.10 |
| Football | 79.2 | 79.17 | 79.16 |
| SQuAD | 87.3 | 87.27 | 87.24 |
| NarrativeQA | 83.4 | 83.39 | 83.40 |
| MLDR | 89.8 | 89.65 | 89.78 |
| COVID-QA | 56.1 | 56.32 | 56.21 |
| Insurance | 88.0 | 88.93 | 88.93 |
| ESG | 67.3 | 53.54 | 52.74 |
| Average | 80.7 | 79.05 | 78.94 |

ESG is the one large gap, and it is not specific to this port: upstream's PyTorch model scored 52.74 under the same harness. Insurance also matches upstream exactly at 88.93. ESG has 36 queries and documents of up to 60,217 tokens, which were embedded whole here. Its score changes with how much context each chunk sees:

| Context per pass | ESG nDCG@10 |
|---:|---:|
| Non-contextual | 49.8 |
| 512 tokens | 54.3 |
| 2,048 | 62.4 |
| 4,096 | 66.4 |
| 8,192 | 62.1 |
| 32,768 | 57.3 |
| Whole document | 53.5 |

So more context is not automatically better on this task. Which setting the paper used for ESG is not stated, and one query is worth about 2.8 points.

## Memory

MLX's buffer cache is process-wide and otherwise grows far past the model's needs, reaching about 50 GB on a 64 GB Mac in a run that was then killed with SIGKILL; with the cap, the same workload completes. `load_model` caps it at 1 GB by default; set `cache_limit_mb` to change it. Peak footprints below are measured from the OS and include that cache.

| Tokens | `float16` peak | `float32` peak |
|---:|---:|---:|
| 13,681 | 4.1 GB | 6.1 GB |
| 29,521 | 5.3 GB | 9.0 GB |
| 59,041 | 8.0 GB | |
| 119,041 | 13.2 GB | |

Attention scores queries in slices, so memory stays far below the square of the document length. Vector accuracy against upstream is verified up to 13,871 tokens. Results are in `reports/`.

## Scope

Tested with `pplx-embed-v2-context-0.6b` only. This package contains no model weights and downloads them from Perplexity's repository. Perplexity's model cards for `pplx-embed-v2-context-9b` and `pplx-embed-v2-late-0.6b` declare MIT; the `0.6b` context card does not currently carry a license tag. Video input is not supported.

Third-party notices are in `NOTICE`.

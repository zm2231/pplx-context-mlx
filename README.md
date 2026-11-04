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

Both rows compare against upstream at FP32. Image pixels match upstream byte for byte on every fixture. On 48 real ViDoRe-V3 page images (384 band vectors), MLX `float32` stays within one int8 step with a lowest cosine of 0.9999986 (`reports/real-pages/summary.json`). Upstream on transformers 5.22.1 stays within one int8 step of the recorded 5.4.0 fixtures, once its missing `config._commit_hash` is set by hand; without that, its code raises `AttributeError` on that version (`reports/compat/transformers-5.22.1-drift.json`). The reference ran on CPU, with upstream's PyTorch fallback rather than its CUDA fast path.

`scripts/fault_injection.py` breaks nine parts of the model on purpose and confirms the vectors catch each one (`reports/fault-injection.json`). Details are in `reports/parity-*.json`; regenerate with `scripts/make_reference.py` and `scripts/parity_report.py`.

## Retrieval

Scored against the tables in the [paper](https://arxiv.org/abs/2610.38854). Reproduce with `evals/`.

### ViDoRe-V3, text to Markdown

Whole pages, one chunk per page, nDCG@10 averaged over six languages, 18,252 pages. This setting has no chunker to choose, so it is the cleanest comparison (paper Table 12).

| Task | Paper | MLX `float32` |
|---|---:|---:|
| Finance (EN) | 59.14 | 59.19 |
| Finance (FR) | 43.51 | 43.54 |
| Industrial | 43.47 | 43.45 |
| Pharmaceuticals | 63.37 | 63.46 |
| Computer science | 70.69 | 70.66 |
| HR | 56.22 | 56.25 |
| Energy | 63.62 | 63.64 |
| Physics | 46.47 | 46.35 |
| Average | 55.81 | 55.82 |

With 128-token chunks (paper Table 13) the average is 58.33 against 58.51. The paper's chunker is not public, so this run uses fixed token slices.

### ConTEB

nDCG@10 on the official task splits (paper Table 5).

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

ESG is the one large gap, and it is not specific to this port: upstream's PyTorch model scored 52.74 under the same harness, and 88.93 on Insurance, the same as MLX. On those two tasks every query has the same top result in both runtimes, with 99.7% to 100% top-10 overlap (`reports/conteb/upstream-insurance-esg.json`, `reports/conteb/mlx-vs-upstream-rankings.json`). ESG has 36 queries and documents of up to 60,217 tokens, which were embedded whole here. Its score changes with how much context each chunk sees:

| Context per pass | ESG nDCG@10 |
|---:|---:|
| Non-contextual | 49.8 |
| 512 tokens | 54.3 |
| 2,048 | 62.4 |
| 4,096 | 66.4 |
| 8,192 | 62.1 |
| 32,768 | 57.3 |
| Whole document | 53.5 |

So more context is not automatically better on this task. Which setting the paper used for ESG is not stated, and one query is worth about 2.8 points. Windows split documents between chunks (`--window-tokens`); results are in `reports/conteb/esg-windows/`, and the whole-document row is in `reports/conteb/mlx-float32.json`.

## Speed

On an M4 Pro with 64 GB. Each run started with 1-minute and 5-minute load averages under 2.7, as recorded in its report. Background services were running. MLX values are medians of three runs. Upstream ran on CPU at its best thread count for each workload, chosen from 1, 2, 4 and 6, because 1 thread was fastest for short inputs and 4 to 6 for long ones. No CUDA comparison was run.

| Workload | MLX `float16` | MLX `float32` | Upstream CPU |
|---|---:|---:|---:|
| One query | 7.4 ms | 11.4 ms | 71.9 ms |
| Queries per second, batches of 32 | 600 | 522 | 50 |
| 224-token document | 0.029 s | 0.032 s | 0.160 s |
| 1,785-token document | 0.20 s | 0.24 s | 1.13 s |
| 7,137-token document | 1.16 s | 1.43 s | 4.55 s |
| A4 page, 8 bands | 0.87 s | 1.03 s | 4.11 s |

Results are in `reports/benchmark/`. Reproduce with `scripts/benchmark.py`.

## Memory

`load_model` caps MLX's buffer cache at 1 GB by default; set `cache_limit_mb` to change it. The cap applies to the whole process, because MLX's cache is process-wide, and a later `load_model` call replaces it.

Without the cap, a traced run over ESG documents in 32,768-token windows reached 50.47 GB of cache with 2.38 GB of active memory, then ended at window 10 of 38 without a traceback (`reports/memory/uncapped-esg-trace.log`). An earlier run of that task exited with code 137. What killed them was not identified. The same traced workload with the cap holds the cache near 1 GB and completes all 38 windows (`reports/memory/capped-esg-trace.log`).

Peak footprints below are measured from the OS and include the cache. Times are indicative: the document-length runs were not held to the load limits above.

| Tokens | `float16` time | `float16` peak | `float32` time | `float32` peak |
|---:|---:|---:|---:|---:|
| 13,681 | 3.0 s | 4.1 GB | 3.8 s | 6.0 GB |
| 29,521 | 10.8 s | 5.3 GB | 17.9 s | 9.1 GB |
| 59,041 | 42.0 s | 8.0 GB | | |
| 119,041 | 171 s | 13.2 GB | | |

Upstream on CPU measured a lower footprint on the speed workloads, 2.6 GB against 5.5 GB for MLX `float32`. No memory advantage over upstream is claimed.

Attention scores queries in slices, so memory stays far below the square of the document length. Vector accuracy against upstream is verified up to 13,871 tokens. Very long inputs are possible, but they are not automatically better; see ESG above. Results are in `reports/`.

## Scope

Tested with `pplx-embed-v2-context-0.6b` only. This package contains no model weights and downloads them from Perplexity's repository. Perplexity's model cards for `pplx-embed-v2-context-9b` and `pplx-embed-v2-late-0.6b` declare MIT; the `0.6b` context card does not currently carry a license tag. Video input is not supported.

Third-party notices are in `NOTICE`.

# pplx-context-mlx

Perplexity's [pplx-embed-v2-context-0.6b](https://huggingface.co/perplexity-ai/pplx-embed-v2-context-0.6b) contextual embeddings on Apple silicon. Each chunk's vector sees the rest of its document, and the upstream checkpoint loads unchanged, with no conversion step, PyTorch, or Transformers.

Independent project; not affiliated with or endorsed by Perplexity.

## Quick start

```bash
pip install pplx-context-mlx
```

```python
from pplx_context_mlx import load_model

engine = load_model()  # the first call downloads about 2.4 GB of weights
chunks = engine.encode(
    [["First chunk of a document.", "Second chunk, which sees the first."]],
    normalize_embeddings=True,
)
query = engine.encode_queries([["What does the second chunk see?"]], normalize_embeddings=True)
scores = chunks[0] @ query[0][0]
```

Each document is a list of chunks, and you get one 1,024-dimensional vector per chunk. Pass `dtype="float16"` to `load_model` for half the weight memory and a slightly faster model.

Page images embed as horizontal bands, with one vector per band:

```python
from PIL import Image

bands = engine.encode_images([Image.open("page.png")], chunks=8, normalize_embeddings=True)[0]
page_score = (bands @ query[0][0]).max()
```

From the command line, input is a JSON list of documents:

```bash
pplx-context-mlx docs.json --dtype float16 --normalize > vectors.json
```

## At a glance

Measured on an M4 Pro with 64 GB, against upstream PyTorch on CPU. Sections below have the details and the report behind each figure.

| Question | Answer |
|---|---|
| Same vectors as upstream? | Within 1 int8 step; lowest cosine 0.999998 at `float32`, 0.99978 at `float16` |
| Same retrieval scores as the paper? | ViDoRe-V3 55.82 against 55.81; ConTEB within 0.22 on six of eight tasks, with [two gaps upstream shares](#conteb) |
| How fast? | One query in 7.5 ms; a 7,137-token document in 1.21 s (`float16`), 3 to 12 times upstream CPU |
| How much memory? | 3.1 GB (`float16`) or 4.5 GB (`float32`) on the speed suite; 11.8 GB at 113,281 tokens (`float16`) |

Not tested: a 16 GB Mac, the 9B model, or a comparison with upstream on a GPU. Vector accuracy against upstream is verified up to 13,871 tokens.

## Parity

Measured against the upstream PyTorch model at revision `6d1fddab`, FP32 on CPU (its PyTorch fallback, not its CUDA fast path), over 16 text cases and 8 image cases, including padded batches, multilingual text, and a 13,871-token document.

| MLX dtype | Largest int8 difference | Lowest cosine | Weights |
|---|---:|---:|---:|
| `float32` (default) | 1 step | 0.999998 | 2.38 GB |
| `float16` | 1 step | 0.99978 | 1.19 GB |

Both rows compare against upstream at FP32. Other checks:

| Check | Result | Report |
|---|---|---|
| Image pixels, every fixture | Byte-for-byte match | `tests/test_images.py` |
| 48 real ViDoRe-V3 pages, 384 band vectors (`float32`) | Within 1 int8 step, lowest cosine 0.9999986 | `reports/real-pages/summary.json` |
| Upstream on transformers 5.22.1 against the recorded 5.4.0 fixtures | Within 1 int8 step, once its missing `config._commit_hash` is set by hand; otherwise upstream raises `AttributeError` | `reports/compat/transformers-5.22.1-drift.json` |
| Nine parts of the model broken on purpose | The vectors catch all nine | `reports/fault-injection.json` |

Regenerate with `scripts/make_reference.py` and `scripts/parity_report.py`. Results are in `reports/parity-*.json`.

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
| One query | 7.5 ms | 11.3 ms | 71.9 ms |
| Queries per second, batches of 32 | 599 | 520 | 51 |
| 224-token document | 0.029 s | 0.032 s | 0.160 s |
| 1,785-token document | 0.20 s | 0.25 s | 1.13 s |
| 7,137-token document | 1.21 s | 1.53 s | 4.53 s |
| A4 page, 8 bands | 0.90 s | 1.10 s | 4.12 s |

Results are in `reports/benchmark/`. Reproduce with `scripts/benchmark.py`.

## Memory

Peak memory on the speed suite:

| Runtime | Footprint | Resident |
|---|---:|---:|
| MLX `float16` | 3.1 GB | 1.3 GB |
| MLX `float32` | 4.5 GB | 2.9 GB |
| Upstream CPU, FP32 | 2.6 GB | 4.3 GB |

Each measure flatters one side: MLX's Metal buffers count toward footprint more than resident memory, and upstream's resident figure is 1.7 GB above its footprint. Taking each runtime's larger figure, `float32` is slightly above upstream, 4.5 GB against 4.3 GB, and `float16` is about 29% below it. Results are in `reports/benchmark/`.

Long documents, all on the 0.6B model. Times are indicative, because these runs were not held to the load limits above (`reports/context-limit.jsonl`).

| Tokens | `float16` time | `float16` peak | `float32` time | `float32` peak |
|---:|---:|---:|---:|---:|
| 12,961 | 2.8 s | 3.6 GB | 3.4 s | 5.0 GB |
| 28,081 | 9.4 s | 4.4 GB | 12.0 s | 6.4 GB |
| 56,161 | 31.7 s | 6.9 GB | | |
| 113,281 | 141 s | 11.8 GB | | |

Peaks here are OS footprint, including the cache. Very long inputs are possible, but they are not automatically better; see [ESG](#conteb).

**What keeps it low.** Above 2,048 tokens, attention and the causal convolution run in slices. Weights are cast one tensor at a time, and `load_model` caps MLX's buffer cache at 512 MB. Set `cache_limit_mb` to change the cap. MLX's cache is process-wide, so the cap covers the whole process, and a later `load_model` call replaces it.

**Changes since 0.1.0.** Version 0.1.1 lowered these peaks from 5.5 GB (`float32`) and 3.8 GB (`float16`) on the speed suite. Long documents got slightly slower in exchange: 7,137 tokens went from 1.43 s to 1.53 s at `float32`. Projected vectors are bit-identical to 0.1.0 on long, padded, query, and image samples at both dtypes (`reports/memory/bit-identity-0.1.0-vs-0.1.1.json`). The retrieval scores above were measured on 0.1.0 and were not rerun.

**Why the cache is capped.** Without a cap, a traced run over ESG documents in 32,768-token windows reached 50.47 GB of cache with 2.38 GB of active memory, then ended at window 10 of 38 without a traceback (`reports/memory/uncapped-esg-trace.log`). An earlier run of that task exited with code 137. What killed them was not identified. With a 1 GB cap, the same workload holds the cache near 1 GB and completes all 38 windows (`reports/memory/capped-esg-trace.log`).

## Scope

- **Models:** tested with `pplx-embed-v2-context-0.6b` only.
- **Inputs:** text and page images. Video is not supported.
- **Weights:** this package contains none and downloads them from Perplexity's repository.
- **License:** the code is MIT. Perplexity's cards for `pplx-embed-v2-context-9b` and `pplx-embed-v2-late-0.6b` declare MIT; the `0.6b` context card does not currently carry a license tag.

Third-party notices are in `NOTICE`.

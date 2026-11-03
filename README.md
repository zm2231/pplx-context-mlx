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

Parity against the upstream PyTorch model is recorded in `reports/`; regenerate with `scripts/make_reference.py` and `scripts/parity_report.py`.

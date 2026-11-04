from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset

from pplx_context_mlx import load_model

LANGUAGES = ("english", "french", "german", "italian", "portuguese", "spanish")
TASKS = {
    "finance_en": (59.14, 62.52),
    "finance_fr": (43.51, 47.56),
    "industrial": (43.47, 45.26),
    "pharmaceuticals": (63.37, 63.96),
    "computer_science": (70.69, 73.40),
    "hr": (56.22, 60.73),
    "energy": (63.62, 67.83),
    "physics": (46.47, 46.79),
}
TOP_K = 10


def _chunks(engine, markdown: str, size: int | None) -> list[str]:
    if size is None:
        return [markdown]
    offsets = engine.tokenizer._tokenizer.encode(markdown).offsets
    if not offsets:
        return [markdown]
    starts = [offsets[index][0] for index in range(0, len(offsets), size)]
    bounds = [*starts[1:], len(markdown)]
    return [markdown[:bounds[0]], *(markdown[left:right] for left, right in zip(starts[1:], bounds[1:], strict=True))]


def _ndcg(ranked: list[int], gains: dict[int, int]) -> float:
    dcg = sum(gains.get(item, 0) / math.log2(rank + 2) for rank, item in enumerate(ranked[:TOP_K]))
    ideal = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(sorted(gains.values(), reverse=True)[:TOP_K]))
    return dcg / ideal if ideal else 0.0


def _windows(engine, chunks: list[str], limit: int) -> list[list[str]]:
    windows: list[list[str]] = [[]]
    used = 0
    for chunk in chunks:
        cost = len(engine.tokenizer._tokenizer.encode(chunk).ids) + 1
        if windows[-1] and used + cost > limit:
            windows.append([])
            used = 0
        windows[-1].append(chunk)
        used += cost
    return windows


def _encode_by_page(engine, corpus: Any, chunk_tokens: int | None) -> tuple[list[np.ndarray], list[int]]:
    vectors, owners = [], []
    for row in corpus:
        chunks = _chunks(engine, row["markdown"] or "", chunk_tokens)
        vectors.append(engine.encode([chunks], batch_size=1, normalize_embeddings=True)[0])
        owners.extend([int(row["corpus_id"])] * len(chunks))
    return vectors, owners


def _encode_by_document(engine, corpus: Any, chunk_tokens: int, context_tokens: int) -> tuple[list[np.ndarray], list[int]]:
    documents: dict[str, list[tuple[int, int, str]]] = {}
    for row in corpus:
        documents.setdefault(row["doc_id"], []).append((int(row["page_number_in_doc"]), int(row["corpus_id"]), row["markdown"] or ""))
    vectors, owners = [], []
    for pages in documents.values():
        chunks, chunk_owners = [], []
        for _, corpus_id, markdown in sorted(pages):
            page_chunks = [chunk for chunk in _chunks(engine, markdown, chunk_tokens) if chunk]
            chunks.extend(page_chunks)
            chunk_owners.extend([corpus_id] * len(page_chunks))
        for window in _windows(engine, chunks, context_tokens):
            vectors.append(engine.encode([window], batch_size=1, normalize_embeddings=True)[0])
        owners.extend(chunk_owners)
    return vectors, owners


def run_task(engine, task: str, chunk_tokens: int | None, context_tokens: int | None = None) -> dict:
    repo = f"vidore/vidore_v3_{task}"
    corpus: Any = load_dataset(repo, "corpus", split="test").remove_columns(["image"])
    queries: Any = load_dataset(repo, "queries", split="test")
    qrels: Any = load_dataset(repo, "qrels", split="test")
    gains: dict[str, dict[int, int]] = {}
    for row in qrels:
        gains.setdefault(str(row["query_id"]), {})[int(row["corpus_id"])] = int(row["score"])
    started = time.perf_counter()
    if context_tokens is None:
        vectors, owners = _encode_by_page(engine, corpus, chunk_tokens)
    else:
        vectors, owners = _encode_by_document(engine, corpus, chunk_tokens or 128, context_tokens)
    matrix = np.concatenate(vectors).astype(np.float32)
    owner = np.array(owners)
    document_seconds = time.perf_counter() - started
    scores_by_language = {}
    for language in LANGUAGES:
        rows = [row for row in queries if row["language"] == language]
        encoded = engine.encode_queries([[row["query"]] for row in rows], batch_size=32, normalize_embeddings=True)
        results = []
        for row, vector in zip(rows, encoded, strict=True):
            scores = matrix @ vector[0]
            best: dict[int, float] = {}
            for index in np.argsort(-scores, kind="stable"):
                best.setdefault(int(owner[index]), float(scores[index]))
                if len(best) >= TOP_K:
                    break
            results.append(_ndcg(list(best), gains.get(str(row["query_id"]), {})))
        scores_by_language[language] = float(np.mean(results)) * 100
    return {
        "pages": len(corpus),
        "chunks": int(matrix.shape[0]),
        "queries": len(queries),
        "ndcg_at_10": float(np.mean(list(scores_by_language.values()))),
        "languages": scores_by_language,
        "document_seconds": document_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score text-to-markdown retrieval on ViDoRe-V3, one chunk per page by default.")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--chunk-tokens", type=int, help="Split each page into chunks of this many tokens")
    parser.add_argument("--context-tokens", type=int, help="Share context across a document's pages, in windows of this many tokens")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    engine = load_model(dtype=args.dtype)
    results = {}
    for task in args.tasks:
        paper = TASKS[task][0 if args.chunk_tokens is None else 1]
        results[task] = {"paper_ndcg_at_10": paper, **run_task(engine, task, args.chunk_tokens, args.context_tokens)}
        print(task, json.dumps({key: value for key, value in results[task].items() if key != "languages"}), flush=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    summary = {
        "dtype": args.dtype,
        "chunk_tokens": args.chunk_tokens,
        "context_tokens": args.context_tokens,
        "commit": commit,
        "platform": platform.platform(),
        "average_ndcg_at_10": float(np.mean([item["ndcg_at_10"] for item in results.values()])),
        "paper_average_ndcg_at_10": float(np.mean([TASKS[task][0 if args.chunk_tokens is None else 1] for task in results])),
        "tasks": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print({key: summary[key] for key in ("dtype", "chunk_tokens", "average_ndcg_at_10", "paper_average_ndcg_at_10")}, flush=True)


if __name__ == "__main__":
    main()

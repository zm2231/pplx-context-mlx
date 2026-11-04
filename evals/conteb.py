from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset

TASKS = {
    "geography": ("illuin-conteb/geography", "train", 94.1),
    "esg": ("illuin-conteb/esg-reports", "test", 67.3),
    "insurance": ("illuin-conteb/insurance", "train", 88.0),
    "football": ("illuin-conteb/football", "train", 79.2),
    "squad": ("illuin-conteb/squad-conteb-eval", "validation", 87.3),
    "narrativeqa": ("illuin-conteb/narrative-qa", "test", 83.4),
    "mldr": ("illuin-conteb/mldr-conteb-eval", "test", 89.8),
    "covid": ("illuin-conteb/covid-qa", "train", 56.1),
}
TOP_K = 10


def _engine(runtime: str, threads: int):
    if runtime.startswith("mlx-"):
        from pplx_context_mlx import load_model

        return load_model(dtype=runtime.removeprefix("mlx-"))
    import torch
    from transformers import AutoModel

    torch.set_num_threads(threads)
    return AutoModel.from_pretrained(
        "perplexity-ai/pplx-embed-v2-context-0.6b",
        revision="6d1fddab0a18478acc0a51edd1e81333fe441095",
        trust_remote_code=True,
        dtype=torch.float32,
        attn_implementation="sdpa",
    ).eval()


def _load(path: str, split: str) -> tuple[list[list[str]], list[str], list[str], list[list[str]]]:
    documents: Any = load_dataset(path, "documents", split=split)
    queries: Any = load_dataset(path, "queries", split=split)
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in documents:
        grouped[row["chunk_id"].split("_")[0]].append((row["chunk"], row["chunk_id"]))
    docs, ids = [], []
    for key in sorted(grouped):
        docs.append([chunk for chunk, _ in grouped[key]])
        ids.extend(chunk_id for _, chunk_id in grouped[key])
    gold = [row["chunk_ids"] if "chunk_ids" in row else [row["chunk_id"]] for row in queries]
    return docs, ids, [row["query"] for row in queries], gold


def _windows(engine, document: list[str], limit: int | None) -> list[list[str]]:
    if limit is None:
        return [document]
    windows: list[list[str]] = [[]]
    used = 0
    for chunk in document:
        cost = _token_count(engine, chunk) + 1
        if windows[-1] and used + cost > limit:
            windows.append([])
            used = 0
        windows[-1].append(chunk)
        used += cost
    return windows


def _token_count(engine, chunk: str) -> int:
    if hasattr(engine, "tokenizer") and hasattr(engine.tokenizer, "encode_documents"):
        return int(engine.tokenizer.encode_documents([[chunk]])["input_ids"].shape[1])
    return int(engine.prepare_inputs([[chunk]], "document")["input_ids"].shape[1])


def _ndcg(ranked: list[str], relevant: set[str]) -> float:
    gain = sum(1 / math.log2(rank + 2) for rank, item in enumerate(ranked[:TOP_K]) if item in relevant)
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(len(relevant), TOP_K)))
    return gain / ideal


def run_task(engine, name: str, path: str, split: str, output: Path, window_tokens: int | None) -> dict:
    documents, ids, queries, gold = _load(path, split)
    documents = [window for document in documents for window in _windows(engine, document, window_tokens)]
    started = time.perf_counter()
    rows = []
    for start in range(len(documents)):
        rows.extend(engine.encode(documents[start : start + 1], batch_size=1, normalize_embeddings=True))
    corpus = np.concatenate(rows).astype(np.float32)
    doc_seconds = time.perf_counter() - started
    if corpus.shape[0] != len(ids):
        raise ValueError(f"{name}: {corpus.shape[0]} vectors for {len(ids)} chunks")
    started = time.perf_counter()
    order = sorted(range(len(queries)), key=lambda index: len(queries[index]))
    query_rows = engine.encode_queries([[queries[index]] for index in order], batch_size=32, normalize_embeddings=True)
    query_vectors = np.zeros((len(queries), corpus.shape[1]), dtype=np.float32)
    for position, index in enumerate(order):
        query_vectors[index] = query_rows[position][0]
    query_seconds = time.perf_counter() - started
    top = np.zeros((len(queries), TOP_K), dtype=np.int32)
    for start in range(0, len(queries), 512):
        scores = query_vectors[start : start + 512] @ corpus.T
        top[start : start + 512] = np.argsort(-scores, axis=1, kind="stable")[:, :TOP_K]
    per_query = np.array(
        [_ndcg([ids[item] for item in top[index]], set(gold[index])) for index in range(len(queries))], dtype=np.float32
    )
    np.savez_compressed(output / f"{name}.npz", top=top, ndcg=per_query, corpus=corpus.astype(np.float16), queries=query_vectors.astype(np.float16))
    return {
        "documents": len(documents),
        "chunks": len(ids),
        "queries": len(queries),
        "ndcg_at_10": float(per_query.mean()) * 100,
        "document_seconds": doc_seconds,
        "query_seconds": query_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a runtime on ConTEB with the official task splits.")
    parser.add_argument("runtime", choices=["mlx-float32", "mlx-float16", "torch"])
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--window-tokens", type=int, help="Split longer documents between chunks into windows of this many tokens")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    engine = _engine(args.runtime, args.threads)
    results = {}
    for name in args.tasks:
        path, split, paper = TASKS[name]
        results[name] = {"paper_ndcg_at_10": paper, **run_task(engine, name, path, split, args.output, args.window_tokens)}
        print(name, json.dumps(results[name]), flush=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    summary = {
        "runtime": args.runtime,
        "window_tokens": args.window_tokens,
        "commit": commit,
        "platform": platform.platform(),
        "average_ndcg_at_10": float(np.mean([item["ndcg_at_10"] for item in results.values()])),
        "paper_average_ndcg_at_10": float(np.mean([item["paper_ndcg_at_10"] for item in results.values()])),
        "tasks": results,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print({key: summary[key] for key in ("runtime", "average_ndcg_at_10", "paper_average_ndcg_at_10")}, flush=True)


if __name__ == "__main__":
    main()

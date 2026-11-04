from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

from footprint import peak_footprint_gb
from PIL import Image

SENTENCE = "Researchers measured the effect across many sites, and it remained stable over several long seasons. "
PAGE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "images" / "page_a4.png"
QUERY = [["What did the researchers measure across the sites?"]]
QUERIES = [[f"Question {index}: what did the researchers measure across the sites?"] for index in range(32)]


def _document(chunks: int) -> list[list[str]]:
    return [[SENTENCE * 13] * chunks]


def _median(run, repeats: int) -> float:
    run()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        run()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def _workloads(count_tokens, encode_queries, encode_documents, encode_images) -> dict[str, float]:
    page = [Image.open(PAGE).convert("RGB")]
    results = {
        "query_ms": _median(lambda: encode_queries(QUERY), 20) * 1000,
        "queries_per_s_batch32": 32 / _median(lambda: encode_queries(QUERIES), 5),
        "image_a4_8_bands_s": _median(lambda: encode_images(page), 3),
    }
    for chunks in (1, 8, 32):
        document = _document(chunks)
        tokens = count_tokens(document)
        seconds = _median(lambda rows=document: encode_documents(rows), 3)
        results[f"document_{tokens}_tokens_s"] = seconds
        results[f"document_{tokens}_tokens_per_s"] = tokens / seconds
    return results


def _mlx(dtype: str) -> dict:
    import mlx.core as mx

    from pplx_context_mlx import load_model

    started = time.perf_counter()
    engine = load_model(dtype=dtype)
    load_seconds = time.perf_counter() - started
    mx.reset_peak_memory()
    results = _workloads(
        lambda rows: int(engine.tokenizer.encode_documents(rows)["input_ids"].shape[1]),
        lambda rows: engine.encode_queries(rows, batch_size=len(rows)),
        lambda rows: engine.encode(rows, batch_size=len(rows)),
        lambda images: engine.encode_images(images),
    )
    return {"runtime": f"mlx-{dtype}", "load_s": load_seconds, "peak_active_gb": mx.get_peak_memory() / 1e9, **results}


def _torch(threads: list[int]) -> dict:
    import torch
    from transformers import AutoModel

    started = time.perf_counter()
    model = AutoModel.from_pretrained(
        "perplexity-ai/pplx-embed-v2-context-0.6b",
        revision="6d1fddab0a18478acc0a51edd1e81333fe441095",
        trust_remote_code=True,
        dtype=torch.float32,
        attn_implementation="sdpa",
    ).eval()
    load_seconds = time.perf_counter() - started
    by_threads = {}
    for count in threads:
        torch.set_num_threads(count)
        by_threads[count] = _workloads(
            lambda rows: int(model.prepare_inputs(rows, "document")["input_ids"].shape[1]),
            lambda rows: model.encode_queries(rows, batch_size=len(rows)),
            lambda rows: model.encode(rows, batch_size=len(rows)),
            lambda images: model.encode_images(images),
        )
    best = {}
    for key in by_threads[threads[0]]:
        choose = max if key.endswith(("_per_s", "per_s_batch32")) else min
        best[key] = choose(run[key] for run in by_threads.values())
    return {"runtime": "torch-cpu-fp32-best-thread-count", "load_s": load_seconds, **best, "by_threads": by_threads}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime", choices=["mlx-float32", "mlx-float16", "torch"])
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 6])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = os.getloadavg()
    result = _torch(args.threads) if args.runtime == "torch" else _mlx(args.runtime.removeprefix("mlx-"))
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True).stdout
    memory = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True).stdout)
    result.update(
        {
            "peak_footprint_gb": peak_footprint_gb(),
            "chip": chip.strip(),
            "memory_gb": memory // 2**30,
            "platform": platform.platform(),
            "load_average_before": before,
            "load_average_after": os.getloadavg(),
        }
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

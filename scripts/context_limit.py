from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx
from footprint import peak_footprint_gb

from pplx_context_mlx import load_model

SENTENCE = "Researchers measured the effect across many sites, and it remained stable over several long seasons. "


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed one document of a given token length and report peak memory.")
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--chunk-tokens", type=int, default=256)
    args = parser.parse_args()
    engine = load_model(dtype=args.dtype)
    mx.eval(engine.model.parameters())
    resident = mx.get_active_memory()
    per_sentence = len(engine.tokenizer.encode_queries([[SENTENCE]])["input_ids"][0]) - 1
    sentences = args.chunk_tokens // per_sentence
    chunks = [SENTENCE * sentences for _ in range(max(1, args.tokens // (sentences * per_sentence)))]
    features = engine.tokenizer.encode_documents([chunks])
    mx.reset_peak_memory()
    started = time.perf_counter()
    vectors = engine.encode([chunks], batch_size=1)[0]
    seconds = time.perf_counter() - started
    print(
        json.dumps(
            {
                "tokens": int(features["input_ids"].shape[1]),
                "chunks": len(chunks),
                "dtype": args.dtype,
                "vectors": int(vectors.shape[0]),
                "seconds": round(seconds, 2),
                "tokens_per_second": round(features["input_ids"].shape[1] / seconds),
                "weights_gb": round(resident / 1e9, 2),
                "peak_active_gb": round(mx.get_peak_memory() / 1e9, 2),
                "peak_footprint_gb": round(peak_footprint_gb(), 2),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

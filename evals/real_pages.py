from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset

REPOS = {
    "finance_en": 12,
    "industrial": 12,
    "pharmaceuticals": 12,
    "physics": 12,
}


def _pages() -> list[tuple[str, Any]]:
    pages = []
    for task, count in REPOS.items():
        corpus: Any = load_dataset(f"vidore/vidore_v3_{task}", "corpus", split="test")
        step = len(corpus) // count
        pages.extend((f"{task}/{index}", corpus[index]["image"].convert("RGB")) for index in range(0, step * count, step))
    return pages


def _encode(runtime: str, pages: list[tuple[str, Any]], chunks: int) -> dict[str, Any]:
    if runtime == "mlx":
        from pplx_context_mlx import load_model

        engine = load_model()
        return {name: engine.encode_images([image], chunks=chunks)[0] for name, image in pages}
    import torch
    from transformers import AutoModel

    torch.set_num_threads(6)
    model = AutoModel.from_pretrained(
        "perplexity-ai/pplx-embed-v2-context-0.6b",
        revision="6d1fddab0a18478acc0a51edd1e81333fe441095",
        trust_remote_code=True,
        dtype=torch.float32,
        attn_implementation="sdpa",
    ).eval()
    return {name: np.asarray(model.encode_images([image], chunks=chunks)[0]) for name, image in pages}


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed real ViDoRe-V3 page images, then compare runtimes.")
    parser.add_argument("runtime", choices=["mlx", "torch", "compare"])
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--folder", type=Path, required=True)
    args = parser.parse_args()
    args.folder.mkdir(parents=True, exist_ok=True)
    if args.runtime != "compare":
        np.savez_compressed(args.folder / f"{args.runtime}.npz", **_encode(args.runtime, _pages(), args.chunks))
        return
    left, right = np.load(args.folder / "mlx.npz"), np.load(args.folder / "torch.npz")
    errors, cosines, exact, sizes = [], [], [], []
    for name in left.files:
        a, b = left[name].astype(np.float32), right[name].astype(np.float32)
        if a.shape != b.shape:
            raise ValueError(f"{name}: {a.shape} vs {b.shape}")
        errors.append(float(np.abs(a - b).max()))
        cosines.append(float(((a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))).min()))
        exact.append(float((a == b).mean()))
        sizes.append(a.shape[0])
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    report = {
        "commit": commit,
        "reference": "upstream PyTorch on CPU, FP32",
        "chunks_per_page": args.chunks,
        "pages": len(errors),
        "vectors": sum(sizes),
        "max_int8_error": max(errors),
        "min_cosine": min(cosines),
        "min_exact_fraction": min(exact),
    }
    (args.folder / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

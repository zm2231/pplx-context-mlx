from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from PIL import Image

from pplx_context_mlx import images, load_model, model, text, vision

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
TEXT = {case["id"]: case for case in json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))}
IMAGES = {case["id"]: case for case in json.loads((FIXTURES / "image_cases.json").read_text(encoding="utf-8"))}
REFERENCE = np.load(FIXTURES / "reference.npz")
OPTIONS = ("chunks", "min_pixels", "max_pixels")


def _text_error(engine, case_id: str) -> float:
    case = TEXT[case_id]
    encode = engine.encode_queries if case["task"] == "query" else engine.encode
    rows = encode(case["documents"], batch_size=len(case["documents"]))
    return max(
        float(np.abs(row - REFERENCE[f"{case_id}/encoded/{index}"].astype(np.float32)).max())
        for index, row in enumerate(rows)
        if row.size
    )


def _image_error(engine, case_id: str) -> float:
    case = IMAGES[case_id]
    loaded = [Image.open(FIXTURES / "images" / name).convert("RGB") for name in case["images"]]
    options = {key: case[key] for key in OPTIONS if key in case}
    rows = engine.encode_images(loaded, batch_size=len(loaded), **options)
    return max(
        float(np.abs(row - REFERENCE[f"image/{case_id}/encoded/{index}"].astype(np.float32)).max())
        for index, row in enumerate(rows)
    )


def _patched(owner: Any, name: str, replacement: Any, measure: Callable[[], float]) -> float:
    original = getattr(owner, name)
    setattr(owner, name, replacement)
    try:
        return measure()
    finally:
        setattr(owner, name, original)


def _misplaced_separators(original):
    def replacement(grid, chunks, merge, ids):
        tokens, positions, spans = original(grid, chunks, merge, ids)
        positions = positions.copy()
        marks = [index for index, token in enumerate(tokens) if token == ids["separator"]]
        positions[:, marks] = positions[:, [mark + 1 for mark in marks]]
        return tokens, positions, spans

    return replacement


def main() -> None:
    parser = argparse.ArgumentParser(description="Break parts of the model on purpose and confirm the vectors catch it.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    engine = load_model()
    document = "contextual_three_chunks"
    batch = "contextual_batch_unequal"
    page = "page_small_many_bands"
    attention = mx.fast.scaled_dot_product_attention

    def causal(query, key, value, scale, mask=None):
        return attention(query, key, value, scale=scale, mask="causal")

    def unmasked(query, key, value, scale, mask=None):
        return attention(query, key, value, scale=scale)

    tokenizer = engine.tokenizer._tokenizer
    tokenizer.encode_special_tokens = False
    results = {"reserved document prefix id": _text_error(engine, document)}
    tokenizer.encode_special_tokens = True
    results["no text rotary"] = _patched(text, "_rotate", lambda x, _cos, _sin: x, lambda: _text_error(engine, document))
    results["causal attention"] = _patched(mx.fast, "scaled_dot_product_attention", causal, lambda: _text_error(engine, document))
    results["no padding mask"] = _patched(mx.fast, "scaled_dot_product_attention", unmasked, lambda: _text_error(engine, batch))
    results["no vision rotary"] = _patched(vision, "_rotate", lambda x, _cos, _sin: x, lambda: _image_error(engine, page))
    results["no patch merge order"] = _patched(
        vision, "_merge_order", lambda values, height, width, _merge: values.reshape(height * width, *values.shape[2:]), lambda: _image_error(engine, page)
    )
    results["no position embeddings"] = _patched(
        vision.VisionModel,
        "_position_embeddings",
        lambda self, grids: mx.zeros((int(grids[:, 1:].prod(-1).sum()), self.config.hidden_size)),
        lambda: _image_error(engine, page),
    )
    results["misplaced separators"] = _patched(images, "_image_row", _misplaced_separators(images._image_row), lambda: _image_error(engine, page))
    results["mean instead of band pooling"] = _patched(
        model, "span_pool", lambda hidden, spans: model.mean_pool(hidden, mx.ones(hidden.shape[:2], dtype=mx.bool_)) * mx.ones((1, spans.shape[1], 1)), lambda: _image_error(engine, page)
    )
    baseline = max(_text_error(engine, document), _image_error(engine, page))
    missed = [name for name, error in results.items() if error <= 1]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    report = {"commit": commit, "mlx": version("mlx"), "baseline_int8_error": baseline, "injected_int8_error": results, "missed": missed}
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if missed:
        raise SystemExit(f"Faults not caught: {missed}")


if __name__ == "__main__":
    main()

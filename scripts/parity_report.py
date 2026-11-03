from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from pplx_context_mlx import load_model

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float((left * right).sum() / (np.linalg.norm(left) * np.linalg.norm(right)))


def _measure(engine, cases, reference) -> list[dict]:
    results = []
    for case in cases:
        rows, task = case["documents"], case["task"]
        project = engine.project_queries if task == "query" else engine.project_documents
        encode = engine.encode_queries if task == "query" else engine.encode
        started = time.perf_counter()
        projected = project(rows)
        mx.eval(projected)
        seconds = time.perf_counter() - started
        projected = np.array(projected)
        expected = reference[f"{case['id']}/projected"]
        encoded = encode(rows, batch_size=len(rows))
        quantized, truth = [], []
        for index, row in enumerate(encoded):
            quantized.append(row)
            truth.append(reference[f"{case['id']}/encoded/{index}"].astype(np.float32))
        got, want = np.concatenate(quantized), np.concatenate(truth)
        live = np.linalg.norm(want, axis=-1) > 0
        cosines = [_cosine(a, b) for a, b in zip(got[live], want[live], strict=True)]
        results.append(
            {
                "id": case["id"],
                "tokens": int(reference[f"{case['id']}/input_ids"].shape[1]),
                "vectors": int(got.shape[0]),
                "projected_max_abs_error": float(np.abs(projected - expected).max()),
                "projected_max_abs": float(np.abs(expected).max()),
                "int8_max_abs_error": float(np.abs(got - want).max()) if got.size else 0.0,
                "int8_exact_fraction": float((got == want).mean()) if got.size else 1.0,
                "min_cosine": min(cosines) if cosines else None,
                "seconds": seconds,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
    reference = np.load(FIXTURES / "reference.npz")
    engine = load_model(dtype=args.dtype)
    results = _measure(engine, cases, reference)
    report = {
        "dtype": args.dtype,
        "mlx": mx.__version__,
        "platform": platform.platform(),
        "reference": json.loads((FIXTURES / "reference.json").read_text(encoding="utf-8")),
        "worst_int8_max_abs_error": max(item["int8_max_abs_error"] for item in results),
        "worst_min_cosine": min(item["min_cosine"] for item in results if item["min_cosine"] is not None),
        "min_int8_exact_fraction": min(item["int8_exact_fraction"] for item in results),
        "cases": results,
    }
    for item in results:
        print(
            f"{item['id']:28s} t={item['tokens']:5d} v={item['vectors']:3d} "
            f"proj_err={item['projected_max_abs_error']:.2e} int8_err={item['int8_max_abs_error']:.0f} "
            f"exact={item['int8_exact_fraction']:.4f} cos={item['min_cosine'] if item['min_cosine'] is None else round(item['min_cosine'], 7)}"
        )
    print({key: report[key] for key in ("dtype", "worst_int8_max_abs_error", "worst_min_cosine", "min_int8_exact_fraction")})
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

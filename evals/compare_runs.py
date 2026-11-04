from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare top-10 rankings from two evals/conteb.py runs.")
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {}
    for task in args.tasks:
        left, right = np.load(args.left / f"{task}.npz"), np.load(args.right / f"{task}.npz")
        report[task] = {
            "queries": int(left["top"].shape[0]),
            "top1_agreement": float((left["top"][:, 0] == right["top"][:, 0]).mean()),
            "top10_overlap": float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(left["top"], right["top"], strict=True)])),
            "identical_top10_order": float((left["top"] == right["top"]).all(axis=1).mean()),
            "left_ndcg_at_10": float(left["ndcg"].mean()) * 100,
            "right_ndcg_at_10": float(right["ndcg"].mean()) * 100,
        }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

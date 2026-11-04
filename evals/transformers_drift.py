from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AutoModel

REPO = "perplexity-ai/pplx-embed-v2-context-0.6b"
REVISION = "6d1fddab0a18478acc0a51edd1e81333fe441095"
FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare upstream on the installed transformers with the recorded fixtures.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(6)
    model = AutoModel.from_pretrained(
        REPO,
        revision=REVISION,
        trust_remote_code=True,
        dtype=torch.float32,
        attn_implementation="sdpa",
    ).eval()
    model.config._commit_hash = REVISION
    reference = np.load(FIXTURES / "reference.npz")
    results = {}
    for case in json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8")):
        encode = model.encode_queries if case["task"] == "query" else model.encode
        rows = encode(case["documents"], batch_size=len(case["documents"]))
        pairs = [(np.asarray(row), reference[f"{case['id']}/encoded/{index}"]) for index, row in enumerate(rows)]
        results[case["id"]] = max((float(np.abs(a - b).max()) for a, b in pairs if a.size), default=0.0)
    for case in json.loads((FIXTURES / "image_cases.json").read_text(encoding="utf-8")):
        images = [Image.open(FIXTURES / "images" / name).convert("RGB") for name in case["images"]]
        options = {key: case[key] for key in ("chunks", "min_pixels", "max_pixels") if key in case}
        rows = model.encode_images(images, batch_size=len(images), **options)
        pairs = [(np.asarray(row), reference[f"image/{case['id']}/encoded/{index}"]) for index, row in enumerate(rows)]
        results[f"image/{case['id']}"] = max(float(np.abs(a - b).max()) for a, b in pairs)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    report = {
        "commit": commit,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "reference_transformers": json.loads((FIXTURES / "reference.json").read_text(encoding="utf-8"))["transformers"],
        "note": "config._commit_hash set by hand; upstream's code otherwise raises AttributeError on this version",
        "worst_int8_error": max(results.values()),
        "cases": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

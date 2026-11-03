"""Record the upstream PyTorch model's output for tests/fixtures/cases.json (CPU, FP32)."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModel

REPO = "perplexity-ai/pplx-embed-v2-context-0.6b"
REVISION = "6d1fddab0a18478acc0a51edd1e81333fe441095"
FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model = AutoModel.from_pretrained(
        REPO, revision=REVISION, trust_remote_code=True, dtype=torch.float32, attn_implementation="sdpa"
    ).eval()
    cases = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
    arrays: dict[str, np.ndarray] = {}
    manifest = []
    with torch.inference_mode():
        for case in cases:
            task, rows = case["task"], case["documents"]
            features = model.prepare_inputs(rows, task)
            projected = model._pool(features, task)
            encoded = model.encode(rows, task=task, batch_size=len(rows))
            arrays[f"{case['id']}/input_ids"] = features["input_ids"].numpy().astype(np.int32)
            arrays[f"{case['id']}/projected"] = projected.numpy().astype(np.float32)
            if task == "document":
                arrays[f"{case['id']}/spans"] = features["chunk_token_spans"].numpy().astype(np.int32)
            for index, row in enumerate(encoded):
                arrays[f"{case['id']}/encoded/{index}"] = np.asarray(row, dtype=np.int8)
            manifest.append({"id": case["id"], "tokens": int(features["input_ids"].shape[1])})
            print(case["id"], features["input_ids"].shape, flush=True)
        trace_case = next(case for case in cases if case["id"] == "contextual_three_chunks")
        features = model.prepare_inputs(trace_case["documents"], "document")
        output = model(
            input_ids=features["input_ids"],
            attention_mask=features["attention_mask"],
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        for index, state in enumerate(output.hidden_states):
            arrays[f"trace/{index}"] = state.numpy().astype(np.float32)
    np.savez_compressed(FIXTURES / "reference.npz", **arrays)
    (FIXTURES / "reference.json").write_text(
        json.dumps(
            {
                "repo": REPO,
                "revision": REVISION,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "device": "cpu",
                "dtype": "float32",
                "platform": platform.platform(),
                "cases": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .checkpoint import CHECKPOINT_0_6B
from .engine import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed a JSON list of documents, each a list of chunk strings.")
    parser.add_argument("input", help="JSON file, or - for stdin")
    parser.add_argument("--task", choices=["document", "query"], default="document")
    parser.add_argument("--model", default=CHECKPOINT_0_6B)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()
    rows = json.loads(sys.stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8"))
    engine = load_model(args.model, dtype=args.dtype)
    encode = engine.encode_queries if args.task == "query" else engine.encode
    json.dump([row.tolist() for row in encode(rows, normalize_embeddings=args.normalize)], sys.stdout)

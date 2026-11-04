from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from pplx_context_mlx import load_model

SENTENCE = "Researchers measured the effect across many sites, and it remained stable over several long seasons. "
PAGE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "images" / "page_a4.png"
DOCUMENTS = {"long": [[SENTENCE * 13] * 32], "padded": [[SENTENCE * 13] * 8, [SENTENCE]], "medium": [[SENTENCE * 13] * 12]}
QUERIES = [["short query"], ["a much longer question about what researchers measured"]]


def dump(path: Path) -> None:
    arrays = {}
    for dtype in ("float32", "float16"):
        engine = load_model(dtype=dtype)
        for name, rows in DOCUMENTS.items():
            vectors = engine.project_documents(rows)
            mx.eval(vectors)
            arrays[f"{dtype}-{name}"] = np.array(vectors, dtype=np.float32)
        vectors = engine.project_queries(QUERIES)
        mx.eval(vectors)
        arrays[f"{dtype}-queries"] = np.array(vectors, dtype=np.float32)
        arrays[f"{dtype}-image"] = engine.encode_images([Image.open(PAGE).convert("RGB")])[0]
    np.savez(path, **arrays)


def compare(first: Path, second: Path, output: Path | None) -> None:
    left, right = np.load(first), np.load(second)
    result = {name: bool(np.array_equal(left[name], right[name])) for name in left.files}
    text = json.dumps({"bit_identical": result}, indent=2) + "\n"
    print(text, end="")
    if output:
        output.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump projected vectors, or compare two dumps bit for bit.")
    parser.add_argument("files", type=Path, nargs="+", help="one file to dump, or two to compare")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if len(args.files) == 1:
        dump(args.files[0])
    else:
        compare(args.files[0], args.files[1], args.output)


if __name__ == "__main__":
    main()

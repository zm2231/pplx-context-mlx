import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pplx_context_mlx import engine as engine_module
from pplx_context_mlx import load_model

FIXTURES = Path(__file__).parent / "fixtures"
TEXT = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))
IMAGES = json.loads((FIXTURES / "image_cases.json").read_text(encoding="utf-8"))
MAX_ERROR = 1
MIN_COSINE = 0.9997


@pytest.fixture(scope="module")
def engine():
    return load_model(dtype="float16")


@pytest.fixture(scope="module")
def reference():
    return np.load(FIXTURES / "reference.npz")


def _check(rows, key, reference):
    for index, row in enumerate(rows):
        expected = reference[f"{key}/encoded/{index}"].astype(np.float32)
        assert row.shape == expected.shape
        if row.size == 0:
            continue
        assert np.abs(row - expected).max() <= MAX_ERROR
        live = np.linalg.norm(expected, axis=-1) > 0
        cosine = (row[live] * expected[live]).sum(-1) / (
            np.linalg.norm(row[live], axis=-1) * np.linalg.norm(expected[live], axis=-1)
        )
        assert cosine.size == 0 or cosine.min() >= MIN_COSINE


@pytest.mark.parametrize("case", TEXT, ids=[case["id"] for case in TEXT])
def test_text_within_float16_envelope(case, engine, reference):
    encode = engine.encode_queries if case["task"] == "query" else engine.encode
    _check(encode(case["documents"], batch_size=len(case["documents"])), case["id"], reference)


@pytest.mark.parametrize("case", IMAGES, ids=[case["id"] for case in IMAGES])
def test_images_within_float16_envelope(case, engine, reference):
    images = [Image.open(FIXTURES / "images" / name).convert("RGB") for name in case["images"]]
    options = {key: case[key] for key in ("chunks", "min_pixels", "max_pixels") if key in case}
    _check(engine.encode_images(images, batch_size=len(images), **options), f"image/{case['id']}", reference)


def test_bfloat16_is_rejected_before_any_download(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("resolved a checkpoint for an unsupported dtype")

    monkeypatch.setattr(engine_module, "resolve_checkpoint", fail)
    with pytest.raises(ValueError, match="Unsupported dtype"):
        load_model(dtype="bfloat16")

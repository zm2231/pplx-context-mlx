import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pplx_context_mlx import load_model

FIXTURES = Path(__file__).parent / "fixtures"
CASES = json.loads((FIXTURES / "image_cases.json").read_text(encoding="utf-8"))
OPTIONS = ("chunks", "min_pixels", "max_pixels")


def _load(case):
    images = [Image.open(FIXTURES / "images" / name).convert("RGB") for name in case["images"]]
    return images, {key: case[key] for key in OPTIONS if key in case}


@pytest.fixture(scope="module")
def engine():
    return load_model()


@pytest.fixture(scope="module")
def reference():
    return np.load(FIXTURES / "reference.npz")


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_input_mapping_matches_reference(case, engine, reference):
    images, options = _load(case)
    features = engine.prepare_images(images, **options)
    key = f"image/{case['id']}"
    for name in ("input_ids", "attention_mask", "chunk_token_spans", "chunk_mask", "image_grid_thw"):
        assert np.array_equal(features[name].astype(np.int32), reference[f"{key}/{name}"]), name
    assert np.array_equal(features["position_ids"], reference[f"{key}/position_ids"])


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_pixels_match_reference_exactly(case, engine, reference):
    images, options = _load(case)
    frame = engine.prepare_images(images, **options)["pixel_values"][:, :768]
    assert np.array_equal(np.round((frame * 0.5 + 0.5) * 255).astype(np.uint8), reference[f"image/{case['id']}/pixels_uint8"])


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_vectors_match_reference(case, engine, reference):
    images, options = _load(case)
    rows = engine.encode_images(images, batch_size=len(images), **options)
    for index, row in enumerate(rows):
        expected = reference[f"image/{case['id']}/encoded/{index}"].astype(np.float32)
        assert row.shape == expected.shape
        assert np.abs(row - expected).max() <= 1
        cosine = (row * expected).sum(-1) / (np.linalg.norm(row, axis=-1) * np.linalg.norm(expected, axis=-1))
        assert cosine.min() >= 0.9999


def test_batching_matches_one_at_a_time(engine):
    case = next(case for case in CASES if case["id"] == "batch_unequal")
    images, options = _load(case)
    batched = engine.encode_images(images, batch_size=2, **options)
    single = engine.encode_images(images, batch_size=1, **options)
    for left, right in zip(batched, single, strict=True):
        assert left.shape == right.shape
        assert np.abs(left - right).max() <= 1


def test_normalize_is_keyword_only(engine):
    with pytest.raises(TypeError):
        engine.encode_images([Image.new("RGB", (64, 64))], 1, True)


def test_rejects_non_images(engine):
    with pytest.raises(ValueError, match="PIL images"):
        engine.prepare_images(["not an image"])

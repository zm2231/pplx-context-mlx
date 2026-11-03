from __future__ import annotations

import math

import numpy as np
from PIL import Image

from .config import ModelConfig
from .tokenizer import ContextualTokenizer


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    if max(height, width) / min(height, width) > 200:
        raise ValueError("Image aspect ratio must be smaller than 200")
    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    if resized_height * resized_width > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        resized_height = max(factor, math.floor(height / beta / factor) * factor)
        resized_width = max(factor, math.floor(width / beta / factor) * factor)
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * beta / factor) * factor
        resized_width = math.ceil(width * beta / factor) * factor
    return resized_height, resized_width


def _cubic(x: np.ndarray) -> np.ndarray:
    x = np.abs(x)
    near = ((1.5 * x - 2.5) * x) * x + 1
    far = ((-0.5 * x + 2.5) * x - 4) * x + 2
    return np.where(x < 1, near, np.where(x < 2, far, 0.0))


def _resample_axis(value: np.ndarray, axis: int, target: int) -> np.ndarray:
    original = value.shape[axis]
    scale = original / target
    stretch = max(scale, 1.0)
    support = 2 * stretch
    kernels = []
    for index in range(target):
        center = (index + 0.5) * scale
        low = max(int(center - support + 0.5), 0)
        high = min(int(center + support + 0.5), original)
        positions = np.arange(low, high)
        weights = _cubic((positions - center + 0.5) / stretch)
        kernels.append((positions, weights / weights.sum()))
    largest = max(float(weights.max()) for _, weights in kernels)
    precision = 0
    while precision < 22 and int(0.5 + largest * (1 << (precision + 1))) < 32768:
        precision += 1
    rows = []
    for positions, weights in kernels:
        fixed = (weights * (1 << precision) + np.where(weights < 0, -0.5, 0.5)).astype(np.int64)
        taken = np.take(value, positions, axis=axis).astype(np.int64)
        shape = [1] * taken.ndim
        shape[axis] = fixed.size
        total = (taken * fixed.reshape(shape)).sum(axis=axis) + (1 << (precision - 1))
        rows.append(np.clip(total >> precision, 0, 255).astype(np.uint8))
    return np.stack(rows, axis=axis)


def bicubic_resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Antialiased bicubic matching torch's CPU uint8 arithmetic byte for byte; Pillow differs. See NOTICE."""
    if image.shape[1] != width:
        image = _resample_axis(image, 1, width)
    if image.shape[0] != height:
        image = _resample_axis(image, 0, height)
    return image


def image_patches(
    image: Image.Image, *, patch: int, temporal: int, merge: int, min_pixels: int, max_pixels: int
) -> tuple[np.ndarray, tuple[int, int, int]]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = smart_resize(rgb.shape[0], rgb.shape[1], patch * merge, min_pixels, max_pixels)
    scaled = (bicubic_resize(rgb, height, width).astype(np.float32) * (1 / 255) - 0.5) / 0.5
    grid_h, grid_w = height // patch, width // patch
    blocks = scaled.transpose(2, 0, 1).reshape(3, grid_h // merge, merge, patch, grid_w // merge, merge, patch)
    blocks = blocks.transpose(1, 4, 2, 5, 0, 3, 6)
    frames = np.repeat(blocks[..., None, :, :], temporal, axis=-3)
    return frames.reshape(grid_h * grid_w, -1), (1, grid_h, grid_w)


def _image_row(
    grid: tuple[int, int, int], chunks: int, merge: int, ids: dict[str, int]
) -> tuple[list[int], np.ndarray, list[tuple[int, int]]]:
    height, width = grid[1] // merge, grid[2] // merge
    count = height * width
    start = 2
    end = start + count
    native = np.tile(np.arange(end + 1, dtype=np.float32), (3, 1))
    native[0, start:end] = start
    native[1, start:end] = np.repeat(np.arange(height), width) + start
    native[2, start:end] = np.tile(np.arange(width), height) + start
    native[:, end:] = np.arange(native.shape[1] - end) + start + max(height, width)
    center = (native[2, start:end].min() + native[2, start:end].max()) / 2
    bands = np.array_split(np.arange(height), min(chunks, height))
    tokens = [ids["prefix"], ids["vision_start"]]
    positions = [native[:, :start]]
    spans = []
    cursor = start
    for index, band in enumerate(bands):
        band_start = start + int(band[0]) * width
        band_end = start + (int(band[-1]) + 1) * width
        if index:
            left = native[:, band_start - 1]
            right = native[:, band_start]
            tokens.append(ids["separator"])
            positions.append(np.array([[left[0]], [(left[1] + right[1]) / 2], [center]], dtype=np.float32))
        offset = len(tokens)
        tokens.extend([ids["image"]] * (band_end - band_start))
        positions.append(native[:, band_start:band_end])
        spans.append((offset, offset + band_end - band_start))
        cursor = band_end
    tokens.append(ids["vision_end"])
    positions.append(native[:, cursor:])
    return tokens, np.concatenate(positions, axis=1), spans


def prepare_images(
    images: list[Image.Image],
    tokenizer: ContextualTokenizer,
    config: ModelConfig,
    *,
    chunks: int = 8,
    min_pixels: int = 16384,
    max_pixels: int = 1800000,
) -> dict[str, np.ndarray]:
    if not images or any(not isinstance(image, Image.Image) for image in images):
        raise ValueError("Expected a nonempty list of PIL images")
    if chunks < 1 or not 0 < min_pixels <= max_pixels:
        raise ValueError("Expected chunks >= 1 and 0 < min_pixels <= max_pixels")
    vision = config.vision
    ids = {
        "prefix": tokenizer.reserved_id(config.document_prefix),
        "vision_start": config.vision_start_token_id,
        "vision_end": config.vision_end_token_id,
        "image": config.image_token_id,
        "separator": tokenizer.reserved_id(config.boundary_marker),
    }
    pixels, grids, rows, positions, spans = [], [], [], [], []
    for image in images:
        patches, grid = image_patches(
            image,
            patch=vision.patch_size,
            temporal=vision.temporal_patch_size,
            merge=vision.spatial_merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        tokens, position, row_spans = _image_row(grid, chunks, vision.spatial_merge_size, ids)
        pixels.append(patches)
        grids.append(grid)
        rows.append(tokens)
        positions.append(position)
        spans.append(row_spans)
    input_ids, attention_mask = tokenizer.pad(rows)
    if input_ids.shape[1] > config.document_length:
        raise ValueError("Image input exceeds document_length; reduce max_pixels")
    position_ids = np.zeros((3, *input_ids.shape), dtype=np.float32)
    chunk_spans = np.zeros((len(images), max(len(row) for row in spans), 2), dtype=np.int32)
    chunk_mask = np.zeros(chunk_spans.shape[:2], dtype=np.bool_)
    for row in range(len(images)):
        position_ids[:, row, : positions[row].shape[1]] = positions[row]
        chunk_spans[row, : len(spans[row])] = spans[row]
        chunk_mask[row, : len(spans[row])] = True
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "pixel_values": np.concatenate(pixels, axis=0),
        "image_grid_thw": np.array(grids, dtype=np.int32),
        "chunk_token_spans": chunk_spans,
        "chunk_mask": chunk_mask,
    }

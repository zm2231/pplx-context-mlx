from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def page(width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for line in range(height // 28):
        draw.text((30, 10 + line * 28), f"Line {line}: quarterly revenue row {line * 37}, the quick brown fox", fill="black")
    draw.rectangle((width // 2, height // 6, width - 30, height // 2), fill=(30, 90, 200))
    draw.ellipse((30, height // 2, width // 3, height - 30), fill=(220, 40, 40))
    return image


def photo(width: int, height: int) -> Image.Image:
    rng = np.random.default_rng(width * height)
    across, down = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    value = (np.sin(across * 17) + np.cos(down * 23) + rng.normal(0, 0.08, (height, width))) * 60 + 128
    return Image.fromarray(np.clip(np.stack([value, value * 0.7 + 30, 255 - value], -1), 0, 255).astype(np.uint8))


def main() -> None:
    folder = FIXTURES / "images"
    folder.mkdir(exist_ok=True)
    page(1240, 1754).save(folder / "page_a4.png")
    page(612, 792).save(folder / "page_small.png")
    page(1023, 769).save(folder / "page_mixed_rounding.png")
    photo(800, 600).save(folder / "photo.png")
    photo(90, 120).save(folder / "tiny.png")
    page(2000, 300).save(folder / "wide.png")
    cases = [
        {"id": "page_a4_default", "images": ["page_a4.png"]},
        {"id": "page_small_one_band", "images": ["page_small.png"], "chunks": 1},
        {"id": "page_small_many_bands", "images": ["page_small.png"], "chunks": 16},
        {"id": "mixed_rounding", "images": ["page_mixed_rounding.png"], "max_pixels": 800000},
        {"id": "photo", "images": ["photo.png"], "chunks": 4},
        {"id": "tiny_upscaled", "images": ["tiny.png"]},
        {"id": "wide_bands_capped", "images": ["wide.png"], "chunks": 64},
        {"id": "batch_unequal", "images": ["page_small.png", "tiny.png"], "chunks": 3},
    ]
    (FIXTURES / "image_cases.json").write_text(json.dumps(cases, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download

from .config import ModelConfig
from .model import PplxContextualModel
from .text import MappedEmbedding
from .vision import VisionModel

CHECKPOINT_0_6B = "perplexity-ai/pplx-embed-v2-context-0.6b"
REVISION_0_6B = "6d1fddab0a18478acc0a51edd1e81333fe441095"
_FILES = ["config.json", "tokenizer.json", "model.safetensors.index.json", "*.safetensors"]
_DTYPES = {"float32": mx.float32, "float16": mx.float16}
_KEEP_FP32 = ("contextual_projection.",)
_VISION_PREFIX = "visual."
_EMBEDDING_TENSOR = "language_model.embed_tokens.weight"
_FILE_DTYPES = {"F32": np.dtype("<f4"), "F16": np.dtype("<f2")}


def resolve_checkpoint(path_or_repo: str, revision: str | None = None) -> Path:
    local = Path(path_or_repo).expanduser()
    if local.is_dir():
        return local
    if path_or_repo == CHECKPOINT_0_6B and revision is None:
        revision = REVISION_0_6B
    return Path(snapshot_download(path_or_repo, revision=revision, allow_patterns=_FILES))


def _weight_map(path: Path) -> dict[str, str]:
    return json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]


def _shard_tensors(path: Path, names: Iterable[str]) -> dict[str, list[str]]:
    weight_map = _weight_map(path)
    shards: dict[str, list[str]] = {}
    for name in names:
        shards.setdefault(weight_map[name], []).append(name)
    return shards


def load_weights(path: Path, dtype: mx.Dtype, keep: Callable[[str], bool]) -> dict[str, mx.array]:
    weights: dict[str, mx.array] = {}
    for filename, names in _shard_tensors(path, filter(keep, _weight_map(path))).items():
        tensors = mx.load(str(path / filename))
        if not isinstance(tensors, dict):
            raise TypeError(f"Expected named tensors in {filename}")
        for name in names:
            if name not in tensors:
                raise ValueError(f"Checkpoint is missing tensor: {name}")
            value = tensors.pop(name)
            weights[name] = value if name.startswith(_KEEP_FP32) else value.astype(dtype)
            mx.eval(weights[name])
    return weights


def _mapped_tensor(path: Path, name: str) -> np.ndarray:
    filename = _weight_map(path)[name]
    with (path / filename).open("rb") as handle:
        header_size = int.from_bytes(handle.read(8), "little")
        entry = json.loads(handle.read(header_size))[name]
    if entry["dtype"] not in _FILE_DTYPES:
        raise ValueError(f"{name} is stored as {entry['dtype']}; expected one of {sorted(_FILE_DTYPES)}")
    return np.memmap(
        path / filename,
        dtype=_FILE_DTYPES[entry["dtype"]],
        mode="r",
        offset=8 + header_size + entry["data_offsets"][0],
        shape=tuple(entry["shape"]),
    )


def validate_dtype(dtype: str) -> None:
    if dtype not in _DTYPES:
        raise ValueError(f"Unsupported dtype {dtype!r}; choose from {sorted(_DTYPES)}")


def _load_vision(path: Path, config: ModelConfig, dtype: mx.Dtype) -> VisionModel:
    vision = VisionModel(config.vision)
    weights = load_weights(path, dtype, lambda name: name.startswith(_VISION_PREFIX))
    vision.load_weights([(name.removeprefix(_VISION_PREFIX), value) for name, value in weights.items()], strict=True)
    vision.eval()
    mx.eval(vision.parameters())
    return vision


def load_checkpoint(path: Path, dtype: str = "float32") -> tuple[PplxContextualModel, ModelConfig]:
    validate_dtype(dtype)
    config = ModelConfig.from_dict(json.loads((path / "config.json").read_text()))
    model = PplxContextualModel(config)
    weights = load_weights(path, _DTYPES[dtype], lambda name: not name.startswith(_VISION_PREFIX) and name != _EMBEDDING_TENSOR)
    model.load_weights(list(weights.items()), strict=True)
    model.language_model.embed_tokens = MappedEmbedding(_mapped_tensor(path, _EMBEDDING_TENSOR), _DTYPES[dtype])
    model.vision_loader = lambda: _load_vision(path, config, _DTYPES[dtype])
    model.eval()
    mx.eval(model.parameters())
    return model, config

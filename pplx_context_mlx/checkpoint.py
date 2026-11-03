from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download

from .config import ModelConfig
from .model import PplxContextualModel

CHECKPOINT_0_6B = "perplexity-ai/pplx-embed-v2-context-0.6b"
REVISION_0_6B = "6d1fddab0a18478acc0a51edd1e81333fe441095"
_FILES = ["config.json", "tokenizer.json", "model.safetensors.index.json", "*.safetensors"]
_DTYPES = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}
_KEEP_FP32 = ("contextual_projection.",)


def resolve_checkpoint(path_or_repo: str, revision: str | None = None) -> Path:
    local = Path(path_or_repo).expanduser()
    if local.is_dir():
        return local
    if path_or_repo == CHECKPOINT_0_6B and revision is None:
        revision = REVISION_0_6B
    return Path(snapshot_download(path_or_repo, revision=revision, allow_patterns=_FILES))


def load_weights(path: Path, dtype: mx.Dtype) -> dict[str, mx.array]:
    weight_map = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    weights: dict[str, mx.array] = {}
    for filename in sorted(set(weight_map.values())):
        tensors = mx.load(str(path / filename))
        if not isinstance(tensors, dict):
            raise TypeError(f"Expected named tensors in {filename}")
        for key, value in tensors.items():
            weights[key] = value if key.startswith(_KEEP_FP32) else value.astype(dtype)
    missing = set(weight_map) - set(weights)
    if missing:
        raise ValueError(f"Checkpoint is missing tensors: {sorted(missing)[:5]}")
    return weights


def load_checkpoint(path: Path, dtype: str = "float32") -> tuple[PplxContextualModel, ModelConfig]:
    if dtype not in _DTYPES:
        raise ValueError(f"Unsupported dtype {dtype!r}; choose from {sorted(_DTYPES)}")
    config = ModelConfig.from_dict(json.loads((path / "config.json").read_text()))
    model = PplxContextualModel(config)
    model.load_weights(list(load_weights(path, _DTYPES[dtype]).items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model, config

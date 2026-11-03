from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TextConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    layer_types: tuple[str, ...]
    head_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_num_key_heads: int
    linear_value_head_dim: int
    linear_num_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    partial_rotary_factor: float
    mrope_section: tuple[int, ...]


@dataclass(frozen=True)
class VisionConfig:
    depth: int
    hidden_size: int
    intermediate_size: int
    in_channels: int
    num_heads: int
    num_position_embeddings: int
    out_hidden_size: int
    patch_size: int
    spatial_merge_size: int
    temporal_patch_size: int


@dataclass(frozen=True)
class ModelConfig:
    text: TextConfig
    vision: VisionConfig
    image_token_id: int
    vision_start_token_id: int
    vision_end_token_id: int
    embedding_dim: int
    query_prefix: str
    document_prefix: str
    boundary_marker: str
    query_length: int
    document_length: int

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ModelConfig:
        if raw.get("model_type") != "pplx_contextual_qwen3_5":
            raise ValueError(f"Expected a pplx_contextual_qwen3_5 checkpoint, got {raw.get('model_type')!r}")
        text = raw["text_config"]
        if text.get("is_causal", True):
            raise ValueError("Expected a bidirectional checkpoint (text_config.is_causal = false)")
        kinds = set(text["layer_types"])
        if not kinds <= {"linear_attention", "full_attention"}:
            raise ValueError(f"Unsupported layer types: {sorted(kinds)}")
        if text.get("mlp_only_layers"):
            raise ValueError("MLP-only layers are not supported")
        rope = text["rope_parameters"]
        if rope.get("rope_type") != "default" or not rope.get("mrope_interleaved"):
            raise ValueError(f"Unsupported rope parameters: {rope}")
        vision = raw["vision_config"]
        if vision.get("deepstack_visual_indexes"):
            raise ValueError("Deepstack visual features are not supported")
        return cls(
            vision=VisionConfig(
                depth=vision["depth"],
                hidden_size=vision["hidden_size"],
                intermediate_size=vision["intermediate_size"],
                in_channels=vision["in_channels"],
                num_heads=vision["num_heads"],
                num_position_embeddings=vision["num_position_embeddings"],
                out_hidden_size=vision["out_hidden_size"],
                patch_size=vision["patch_size"],
                spatial_merge_size=vision["spatial_merge_size"],
                temporal_patch_size=vision["temporal_patch_size"],
            ),
            image_token_id=raw["image_token_id"],
            vision_start_token_id=raw["vision_start_token_id"],
            vision_end_token_id=raw["vision_end_token_id"],
            text=TextConfig(
                vocab_size=text["vocab_size"],
                hidden_size=text["hidden_size"],
                intermediate_size=text["intermediate_size"],
                layer_types=tuple(text["layer_types"]),
                head_dim=text["head_dim"],
                num_attention_heads=text["num_attention_heads"],
                num_key_value_heads=text["num_key_value_heads"],
                linear_conv_kernel_dim=text["linear_conv_kernel_dim"],
                linear_key_head_dim=text["linear_key_head_dim"],
                linear_num_key_heads=text["linear_num_key_heads"],
                linear_value_head_dim=text["linear_value_head_dim"],
                linear_num_value_heads=text["linear_num_value_heads"],
                rms_norm_eps=text["rms_norm_eps"],
                rope_theta=rope["rope_theta"],
                partial_rotary_factor=rope["partial_rotary_factor"],
                mrope_section=tuple(rope["mrope_section"]),
            ),
            embedding_dim=raw["embedding_dim"],
            query_prefix=raw["query_prefix"],
            document_prefix=raw["document_prefix"],
            boundary_marker=raw["boundary_marker"],
            query_length=raw["query_length"],
            document_length=raw["document_length"],
        )

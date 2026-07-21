"""Shared VLM model discovery and dropdown resolution.

The actual model architecture and inference stay in the consuming node pack.
This module only centralizes the bounded local scan, compatible-config filter,
curated Hugging Face choices, and custom-model fallback used by VLM selectors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from .model_resolve import (
    CUSTOM_MODEL_CHOICE,
    discover_model_dirs,
    model_dropdown_choices,
    resolve_choice_or_custom,
    resolve_model_dir,
)

KNOWN_REMOTE_MODELS: tuple[str, ...] = (
    "mlx-community/gemma-4-e4b-it-mxfp8",
    "mlx-community/gemma-3-12b-it-4bit",
    "mlx-community/gemma-3-12b-it-8bit",
    "mlx-community/gemma-3-4b-it-4bit",
    "mlx-community/gemma-3-27b-it-4bit",
    "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
    "mlx-community/Qwen2.5-VL-7B-Instruct-8bit",
    "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
    "mlx-community/Qwen2.5-VL-32B-Instruct-4bit",
    "mlx-community/Qwen2.5-VL-72B-Instruct-4bit",
    "mlx-community/Qwen2-VL-2B-Instruct-4bit",
    "mlx-community/Qwen2-VL-7B-Instruct-4bit",
    "mlx-community/InternVL3-8B-4bit",
    "mlx-community/pixtral-12b-4bit",
    "mlx-community/llava-v1.6-mistral-7b-4bit",
    "mlx-community/llava-1.5-7b-4bit",
    "mlx-community/SmolVLM-Instruct-4bit",
    "mlx-community/SmolVLM2-2.2B-Instruct-4bit",
    "mlx-community/SmolVLM2-2.2B-Instruct-mlx",
    "mlx-community/idefics2-8b-4bit",
    "mlx-community/Phi-3.5-vision-instruct-4bit",
    "mlx-community/Phi-3-vision-128k-instruct-4bit",
    "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit",
)

VISION_CONFIG_KEYS: tuple[str, ...] = (
    "vision_config",
    "vision_tower",
    "visual",
    "image_token_index",
)
EXCLUDED_MODEL_TYPES = frozenset({"llava_qwen2"})
DEFAULT_DISCOVERY_PATTERNS: tuple[str, ...] = (
    "*/config.json",
    "*/*/config.json",
    "*/*/*/config.json",
)


def is_vlm_config(config: dict) -> bool:
    """Return whether a config describes an ``mlx-vlm`` compatible model."""
    if not isinstance(config, dict):
        return False
    if str(config.get("model_type", "")).lower() in EXCLUDED_MODEL_TYPES:
        return False
    return any(key in config for key in VISION_CONFIG_KEYS)


def discover_vlm_models(*, roots: Sequence[Path] | None = None) -> list[str]:
    """Discover portable ids for compatible local VLM directories."""
    return discover_model_dirs(
        marker_file="config.json",
        patterns=DEFAULT_DISCOVERY_PATTERNS,
        predicate=is_vlm_config,
        roots=roots,
        exclude_parts=("text_encoder", "FastVLM"),
    )


def vlm_model_dropdown(
    *,
    default_contains: str = "gemma-3-12b-it-4bit",
    roots: Sequence[Path] | None = None,
    remote_choices: Sequence[str] = KNOWN_REMOTE_MODELS,
) -> tuple[list[str], str]:
    """Return local-first VLM dropdown choices and the requested default."""
    return model_dropdown_choices(
        discover_vlm_models(roots=roots),
        remote_choices,
        default_contains=default_contains,
    )


def resolve_vlm_choice(
    choice: str,
    custom_model_id: str = "",
    *,
    status: Callable[[str], None] = print,
) -> Path:
    """Resolve a VLM dropdown/custom selection to an installed model path."""
    source = resolve_choice_or_custom(choice, custom_model_id)
    if not source:
        raise ValueError("Select a VLM or enter a custom model path / Hugging Face repo id")
    return resolve_model_dir(source, status=status)


__all__ = [
    "CUSTOM_MODEL_CHOICE",
    "DEFAULT_DISCOVERY_PATTERNS",
    "EXCLUDED_MODEL_TYPES",
    "KNOWN_REMOTE_MODELS",
    "VISION_CONFIG_KEYS",
    "discover_vlm_models",
    "is_vlm_config",
    "resolve_vlm_choice",
    "vlm_model_dropdown",
]

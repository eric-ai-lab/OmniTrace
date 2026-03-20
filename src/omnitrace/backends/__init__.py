from __future__ import annotations

from typing import Any, Dict, List

from omnitrace.core import ModelBundle, GenResult, GenerationConfig, DEFAULT_GENERATION_CONFIG

from .qwen import (
    load_model_qwen,
    prepare_inputs_qwen,
    generate_with_attn_qwen,
)
from .minicpm import (
    load_model_minicpm,
    prepare_inputs_minicpm,
    generate_with_attn_minicpm,
)


def load_model(model_name: str = "qwen", model_path: str | None = None) -> ModelBundle:
    """
    Factory: load a backend model by name.
    """
    model_name = model_name.lower().strip()

    if model_name == "qwen":
        return load_model_qwen(model_path=model_path)
    if model_name == "minicpm":
        return load_model_minicpm(model_path=model_path)

    raise ValueError(f"Unknown model_name: {model_name}")


def prepare_inputs(
    bundle: ModelBundle,
    messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Dispatch to backend-specific input preparation.
    """
    if bundle.model_type == "qwen":
        return prepare_inputs_qwen(bundle, messages)
    if bundle.model_type == "minicpm":
        return prepare_inputs_minicpm(bundle, messages)

    raise ValueError(f"Unknown bundle.model_type: {bundle.model_type}")


def generate_with_attn(
    bundle: ModelBundle,
    inputs: Dict[str, Any],
    gen_cfg: GenerationConfig = DEFAULT_GENERATION_CONFIG,
) -> GenResult:
    """
    Dispatch to backend-specific generation + attribution signal extraction.
    """
    if bundle.model_type == "qwen":
        return generate_with_attn_qwen(bundle, inputs, gen_cfg)
    if bundle.model_type == "minicpm":
        return generate_with_attn_minicpm(bundle, inputs, gen_cfg)

    raise ValueError(f"Unknown bundle.model_type: {bundle.model_type}")


__all__ = [
    "load_model",
    "prepare_inputs",
    "generate_with_attn",
    "load_model_qwen",
    "prepare_inputs_qwen",
    "generate_with_attn_qwen",
    "load_model_minicpm",
    "prepare_inputs_minicpm",
    "generate_with_attn_minicpm",
]
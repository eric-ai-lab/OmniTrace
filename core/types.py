from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Set, Dict

import numpy as np
import torch


@dataclass
class ModelBundle:
    """
    Loaded model bundle plus normalized backend metadata.
    """
    model: Any
    processor: Any
    tokenizer: Any
    model_type: str
    special_token_ids: Set[int] = field(default_factory=set)

    # Optional modality token ids exposed by a backend
    audio_token_id: int = -1
    vision_token_id: int = -1
    image_token_id: int = -1


@dataclass(frozen=True)
class GenerationConfig:
    """
    Generic generation config shared across all modalities.
    """
    max_new_tokens: int = 256
    min_new_tokens: int = 0
    no_repeat_ngram_size: int = 3
    method: str = "attmean"   # attmean | attraw | attgrads


DEFAULT_GENERATION_CONFIG = GenerationConfig()


@dataclass
class GenResult:
    """
    Unified output from backend generation + attribution signal extraction.
    """
    text: str
    generated_ids: torch.Tensor
    input_ids: torch.Tensor
    input_length: int

    # Attention-based path
    all_attentions: List[np.ndarray]

    # Source-region metadata
    source_start: int
    source_end: int
    text_positions: np.ndarray
    text_region_start: int

    # Gradient-based path
    grad_scores_by_step: Optional[List[torch.Tensor]] = None

    # Optional modality-specific source positions
    image_positions: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.int64)
    )
    audio_positions: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.int64)
    )
    vision_positions: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.int64)
    )
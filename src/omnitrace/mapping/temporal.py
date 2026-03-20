from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

from omnitrace.audio_processing.semantic_chunking import SemanticChunk
from .types import TokenMapping


class AudioMapper(Protocol):
    """
    Maps an argmax source token position in the audio region to a structured source dict.
    """

    def __call__(
        self,
        max_idx: int,
        max_weight: float,
        audio_positions: np.ndarray,
        audio_duration: float,
    ) -> Dict[str, Any]:
        ...


class TimeBinMapper:
    """
    Map temporal token positions to fixed-size time bins.
    """

    def __init__(self, bin_size: float = 1.0):
        self.bin_size = float(bin_size)

    def __call__(
        self,
        max_idx: int,
        max_weight: float,
        audio_positions: np.ndarray,
        audio_duration: float,
    ) -> Dict[str, Any]:
        num_audio = len(audio_positions)
        if num_audio == 0:
            return {
                "idx": -1,
                "type": "AUDIO",
                "token": -1,
                "weight": 0.0,
            }

        rel_idx = int(np.searchsorted(audio_positions, max_idx))
        if rel_idx >= num_audio:
            rel_idx = num_audio - 1

        time_point = rel_idx / num_audio * audio_duration
        time_bin = int(time_point // self.bin_size)

        return {
            "idx": max_idx,
            "type": "AUDIO",
            "token": time_bin,
            "weight": float(max_weight),
        }


class SemanticChunkMapper:
    """
    Map temporal token positions to semantic ASR chunk ids.
    """

    def __init__(self, semantic_chunks: List[SemanticChunk]):
        self.chunks = semantic_chunks

    def _find_by_time(self, time_point: float) -> int:
        for ch in self.chunks:
            if ch.start <= time_point < ch.end:
                return int(ch.index)
        return -1

    def __call__(
        self,
        max_idx: int,
        max_weight: float,
        audio_positions: np.ndarray,
        audio_duration: float,
    ) -> Dict[str, Any]:
        num_audio = len(audio_positions)
        if num_audio == 0:
            return {
                "idx": -1,
                "type": "AUDIO",
                "token": -1,
                "weight": 0.0,
            }

        rel_idx = int(np.searchsorted(audio_positions, max_idx))
        if rel_idx >= num_audio:
            rel_idx = num_audio - 1

        time_point = rel_idx / num_audio * audio_duration
        chunk_id = self._find_by_time(time_point)

        return {
            "idx": max_idx,
            "type": "AUDIO",
            "token": chunk_id,
            "weight": float(max_weight),
        }


class VisionFrameMapper:
    """
    Map visual token positions in video to temporal-group bins.
    """

    def __init__(self, bin_size: float = 1.0):
        self.bin_size = float(bin_size)

    def __call__(
        self,
        max_idx: int,
        max_weight: float,
        vision_positions: np.ndarray,
        duration: float,
    ) -> Dict[str, Any]:
        num_vision = len(vision_positions)
        if num_vision == 0:
            return {
                "idx": -1,
                "type": "VISION",
                "token": -1,
                "weight": 0.0,
            }

        rel_idx = int(np.searchsorted(vision_positions, max_idx))
        if rel_idx >= num_vision:
            rel_idx = num_vision - 1

        num_groups = max(1, round(duration / self.bin_size))
        tokens_per_group = max(1, num_vision // num_groups)
        group_idx = min(rel_idx // tokens_per_group, num_groups - 1)

        return {
            "idx": max_idx,
            "type": "VISION",
            "token": group_idx,
            "weight": float(max_weight),
        }


def _scores_to_numpy(scores: Any, input_length: int) -> np.ndarray:
    """
    Convert one grad-score vector to a fixed-length numpy array.
    """
    if isinstance(scores, np.ndarray):
        arr = scores
    else:
        arr = scores.detach().float().cpu().numpy()

    if len(arr) >= input_length:
        return arr[:input_length]

    padded = np.zeros(input_length, dtype=np.float32)
    padded[:len(arr)] = arr
    return padded


def _modality_argmax(
    scores: np.ndarray,
    pos_set: set[int],
) -> Tuple[int, float]:
    """
    Argmax restricted to a modality-specific position set.
    """
    if not pos_set:
        return -1, 0.0

    masked = scores.copy()
    for p in range(len(masked)):
        if p not in pos_set:
            masked[p] = 0.0

    idx = int(np.argmax(masked))
    return idx, float(masked[idx])


def build_token_mappings(
    gen_result,
    audio_mapper: AudioMapper,
    audio_duration: float,
    tokenizer: Any,
    special_token_ids: Optional[set[int]] = None,
    vision_mapper: Optional[VisionFrameMapper] = None,
) -> List[TokenMapping]:
    """
    Build per-generated-token source mappings from generation-time attribution signals.

    Audio-only mode:
        - one argmax over audio positions

    Video mode:
        - separate argmax over vision and audio positions
        - keep both internal sources
    """
    all_attentions = gen_result.all_attentions
    grad_scores = getattr(gen_result, "grad_scores_by_step", None)

    audio_positions = gen_result.audio_positions
    vision_positions = getattr(gen_result, "vision_positions", np.array([], dtype=np.int64))

    generated_ids = gen_result.generated_ids
    input_ids = gen_result.input_ids
    input_length = gen_result.input_length
    source_start = gen_result.source_start
    source_end = gen_result.source_end

    use_grad = grad_scores is not None and len(grad_scores) > 0
    use_attn = all_attentions is not None and len(all_attentions) > 0

    if not use_grad and not use_attn:
        return []

    num_steps = len(grad_scores) if use_grad else len(all_attentions)

    audio_pos_set = set(audio_positions.tolist())
    has_vision = vision_mapper is not None and len(vision_positions) > 0
    vision_pos_set = set(vision_positions.tolist()) if has_vision else set()

    special_ids = special_token_ids if special_token_ids is not None else set()
    special_pos_mask = np.array(
        [int(input_ids[p].item()) in special_ids for p in range(len(input_ids))],
        dtype=bool,
    )

    mappings: List[TokenMapping] = []

    for i in range(num_steps):
        if use_grad:
            scores = _scores_to_numpy(grad_scores[i], input_length)
        else:
            scores = all_attentions[i].copy()

        scores[special_pos_mask[:len(scores)]] = 0.0
        scores[:source_start] = 0.0
        if source_end < len(scores):
            scores[source_end:] = 0.0

        if has_vision:
            v_idx, v_w = _modality_argmax(scores, vision_pos_set)
            vision_src = (
                vision_mapper(v_idx, v_w, vision_positions, audio_duration)
                if v_w > 0 else None
            )

            a_idx, a_w = _modality_argmax(scores, audio_pos_set)
            audio_src = (
                audio_mapper(a_idx, a_w, audio_positions, audio_duration)
                if a_w > 0 else None
            )

            src = vision_src or audio_src or {
                "idx": -1,
                "type": "NONE",
                "token": -1,
                "weight": 0.0,
            }

        else:
            a_idx, a_w = _modality_argmax(scores, audio_pos_set)
            if a_w > 0:
                src = audio_mapper(a_idx, a_w, audio_positions, audio_duration)
            else:
                src = {
                    "idx": -1,
                    "type": "NONE",
                    "token": -1,
                    "weight": 0.0,
                }

            vision_src = None
            audio_src = None

        gen_idx = input_length + i
        if gen_idx < len(generated_ids):
            gid = int(generated_ids[gen_idx].item())
            gstr = tokenizer.decode([gid], clean_up_tokenization_spaces=True)
        else:
            gid, gstr = 0, ""

        mappings.append(
            TokenMapping(
                gen_token=(gid, gstr),
                max_src_token=src,
                max_vision_src=vision_src,
                max_audio_src=audio_src,
            )
        )

    return mappings
from __future__ import annotations

import logging
import os
import subprocess
from typing import Any, Dict, List, Optional
import cv2

from backends import prepare_inputs, generate_with_attn
from constants import TEMPORAL_PATCH_SIZE, DEFAULT_VIDEO_FPS
from core import (
    GenerationConfig,
    DEFAULT_GENERATION_CONFIG,
    find_prompt_span,
    chunk_prompt_text,
    chunks_to_token_spans,
    curate_sources_with_conf,
    validate_method,
)
from mapping.temporal import (
    TimeBinMapper,
    VisionFrameMapper,
    build_token_mappings,
)

logger = logging.getLogger(__name__)


# ======================================================================
# Message building
# ======================================================================

def build_video_messages(
    video: str,
    prompt: str,
    *,
    video_max_pixels: Optional[int] = None,
    video_fps: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Standard OmniTrace video message format.
    """
    video_item: Dict[str, Any] = {
        "type": "video",
        "video": video,
    }
    if video_max_pixels is not None:
        video_item["total_pixels"] = video_max_pixels
    if video_fps is not None:
        video_item["fps"] = video_fps

    return [
        {
            "role": "user",
            "content": [
                video_item,
                {"type": "text", "text": prompt},
            ],
        }
    ]


# ======================================================================
# Helpers
# ======================================================================

def get_video_duration(video_path: str) -> float:
    """
    Read video duration in seconds using ffprobe.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()

        if fps <= 0:
            raise RuntimeError("Invalid FPS, cannot compute duration")

        return frame_count / fps


def merge_consecutive_time_bins(
    curated_bins: List[int],
    bin_size: float,
    duration: float,
) -> List[List[float]]:
    """
    Convert curated time-bin ids into merged [start, end] ranges.
    """
    if not curated_bins:
        return []

    sorted_bins = sorted(set(curated_bins))
    merged = []

    start_bin = end_bin = sorted_bins[0]
    for b in sorted_bins[1:]:
        if b == end_bin + 1:
            end_bin = b
        else:
            merged.append((start_bin, end_bin))
            start_bin = end_bin = b
    merged.append((start_bin, end_bin))

    return [
        [
            round(s * bin_size, 2),
            round(min((e + 1) * bin_size, duration), 2),
        ]
        for s, e in merged
    ]


# ======================================================================
# Attribution
# ======================================================================

def attribute_video_chunks(
    output_text: str,
    token_mappings: List[Any],
    tokenizer: Any,
    *,
    bin_size: float,
    duration: float,
    curate_cfg=None,
) -> Dict[str, Any]:
    """
    Aggregate token-level video mappings into generation chunks.

    Final output only exposes:
      - video_sources_curated

    Internally:
      - curate visual bins
      - curate audio bins
      - take the union
      - merge into final time ranges
    """
    if not token_mappings:
        return {"gen_chunks": [], "per_chunk_attribution": []}

    tokens = [tm.gen_token[1] for tm in token_mappings]

    gen_chunks, chunk_to_token_spans, chunk_to_pos_spans = chunks_to_token_spans(
        text=output_text,
        token=tokens,
        tokenizer=tokenizer,
        return_pos=True,
        strict_token_check=False,
    )

    per_chunk = []

    for i, chunk_span in enumerate(gen_chunks):
        token_span = chunk_to_token_spans[i]
        pos_spans = chunk_to_pos_spans[i] if chunk_to_pos_spans else []
        all_pos = [tag for span in pos_spans for tag in span] if pos_spans else []

        if token_span and token_span[0]:
            cs = token_span[0][0]
            ce = token_span[-1][-1] if token_span[-1] else cs
        else:
            cs, ce = 0, 0

        vision_srcs, vision_confs, vision_pos = [], [], []
        audio_srcs, audio_confs, audio_pos = [], [], []

        for j, tm in enumerate(token_mappings[cs:ce + 1]):
            pos_tag = all_pos[j] if j < len(all_pos) else "X"

            v_src = tm.max_vision_src
            if v_src is not None:
                v_tok = int(v_src.get("token", -1))
                v_w = float(v_src.get("weight", 0.0))
                if v_tok >= 0 and v_w > 0:
                    vision_srcs.append(v_tok)
                    vision_confs.append(v_w)
                    vision_pos.append(pos_tag)

            a_src = tm.max_audio_src
            if a_src is not None:
                a_tok = int(a_src.get("token", -1))
                a_w = float(a_src.get("weight", 0.0))
                if a_tok >= 0 and a_w > 0:
                    audio_srcs.append(a_tok)
                    audio_confs.append(a_w)
                    audio_pos.append(pos_tag)

        if curate_cfg is None:
            curated_vision = curate_sources_with_conf(
                source_ids=vision_srcs,
                pos=vision_pos,
                conf=vision_confs,
            )
            curated_audio = curate_sources_with_conf(
                source_ids=audio_srcs,
                pos=audio_pos,
                conf=audio_confs,
            )
        else:
            curated_vision = curate_sources_with_conf(
                source_ids=vision_srcs,
                pos=vision_pos,
                conf=vision_confs,
                cfg=curate_cfg,
            )
            curated_audio = curate_sources_with_conf(
                source_ids=audio_srcs,
                pos=audio_pos,
                conf=audio_confs,
                cfg=curate_cfg,
            )

        combined_bins = list(set(curated_vision) | set(curated_audio))
        video_formatted = merge_consecutive_time_bins(
            curated_bins=combined_bins,
            bin_size=bin_size,
            duration=duration,
        )

        per_chunk.append(
            {
                "chunk_id": i,
                "chunk_text": str(chunk_span.text) if hasattr(chunk_span, "text") else str(chunk_span),
                "video_sources_curated": video_formatted,
            }
        )

    return {
        "gen_chunks": [str(x.text) if hasattr(x, "text") else str(x) for x in gen_chunks],
        "per_chunk_attribution": per_chunk,
    }


# ======================================================================
# Main public API
# ======================================================================

def trace_video(
    bundle,
    method: str,
    prompt: str,
    video: str,
    video_fps: Optional[float] = None,
    video_max_pixels: Optional[int] = None,
    generation_config: Optional[GenerationConfig] = None,
) -> Dict[str, Any]:
    """
    Unified video tracing API.

    Args:
        bundle:
            Loaded backend model bundle.
        method:
            One of {"attmean", "attraw", "attgrads"}.
        prompt:
            User prompt.
        video:
            Path to video file.
        video_fps:
            Optional frame sampling rate.
        video_max_pixels:
            Optional total frame pixel budget.
        generation_config:
            Optional generation config override.

    Returns:
        A unified OmniTrace video result dict.
    """
    method = validate_method(method)

    if generation_config is None:
        generation_config = DEFAULT_GENERATION_CONFIG
    generation_config = GenerationConfig(
        max_new_tokens=generation_config.max_new_tokens,
        min_new_tokens=generation_config.min_new_tokens,
        no_repeat_ngram_size=generation_config.no_repeat_ngram_size,
        method=method,
    )

    video_duration = get_video_duration(video)
    fps = video_fps or DEFAULT_VIDEO_FPS
    bin_size = TEMPORAL_PATCH_SIZE / fps

    logger.info(
        f"Video temporal bin size: {bin_size:.4f}s "
        f"(temporal_patch={TEMPORAL_PATCH_SIZE}, fps={fps})"
    )

    messages = build_video_messages(
        video=video,
        prompt=prompt,
        video_max_pixels=video_max_pixels,
        video_fps=video_fps,
    )

    # ------------------------------------------------------------------
    # Prepare inputs + generate
    # ------------------------------------------------------------------
    inputs = prepare_inputs(bundle, messages)
    gen_result = generate_with_attn(bundle, inputs, gen_cfg=generation_config)

    # ------------------------------------------------------------------
    # Prompt alignment kept internally for consistency/debugging
    # ------------------------------------------------------------------
    prompt_ids = bundle.tokenizer(prompt, add_special_tokens=False)["input_ids"]
    prompt_start, _ = find_prompt_span(
        gen_result.input_ids,
        prompt_ids,
        gen_result.source_start,
        gen_result.source_end,
    )
    if prompt_start >= 0:
        gen_result.text_region_start = prompt_start
    else:
        prompt_start = gen_result.text_region_start

    # Build prompt text chunks internally if needed later.
    # Final video output will not expose text sources.
    _ = chunk_prompt_text(prompt, bundle.tokenizer)

    # ------------------------------------------------------------------
    # Token-level attribution mapping
    # ------------------------------------------------------------------
    vision_mapper = VisionFrameMapper(bin_size=bin_size)
    audio_mapper = TimeBinMapper(bin_size=bin_size)

    token_mappings = build_token_mappings(
        gen_result=gen_result,
        audio_mapper=audio_mapper,
        audio_duration=video_duration,
        tokenizer=bundle.tokenizer,
        special_token_ids=bundle.special_token_ids,
        vision_mapper=vision_mapper,
    )
    logger.info(f"Built {len(token_mappings)} video token mappings")

    # ------------------------------------------------------------------
    # Chunk-level attribution
    # ------------------------------------------------------------------
    attribution = attribute_video_chunks(
        output_text=gen_result.text,
        token_mappings=token_mappings,
        tokenizer=bundle.tokenizer,
        bin_size=bin_size,
        duration=video_duration,
    )

    return {
        "prompt": prompt,
        "response": gen_result.text,
        "attribution": attribution,
    }
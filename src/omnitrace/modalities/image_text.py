from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from omnitrace.backends import prepare_inputs, generate_with_attn
from omnitrace.core import (
    GenerationConfig,
    DEFAULT_GENERATION_CONFIG,
    find_prompt_span,
    chunks_to_token_spans,
    curate_sources_with_conf,
    validate_method,
)
from omnitrace.mapping import build_image_text_token_mappings

logger = logging.getLogger(__name__)


# ======================================================================
# Message building
# ======================================================================

def _sanitize_text_for_minicpm(s: str) -> str:
    """
    Prevent MiniCPM processor from interpreting raw dataset placeholders.
    """
    return s.replace("<image>", "[image]").replace("</image>", "[/image]")


def build_image_text_messages(
    prompt: str,
    content: List[Dict[str, Any]],
    *,
    model_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Build standard OmniTrace image-text messages.

    Convention:
      - prompt is prepended as the first text block
      - then user-provided interleaved content follows

    Example content:
      [
          {"type": "text", "text": "Look at the first image."},
          {"type": "image", "image": "a.jpg"},
          {"type": "text", "text": "Now compare with the second image."},
          {"type": "image", "image": "b.jpg"},
      ]
    """
    normalized_content: List[Dict[str, Any]] = []

    prompt_text = prompt
    if model_type == "minicpm":
        prompt_text = _sanitize_text_for_minicpm(prompt_text)

    normalized_content.append({"type": "text", "text": prompt_text})

    for item in content:
        item_type = item.get("type")

        if item_type == "text":
            text = item["text"]
            if model_type == "minicpm":
                text = _sanitize_text_for_minicpm(text)
            normalized_content.append({"type": "text", "text": text})

        elif item_type == "image":
            normalized_content.append({"type": "image", "image": item["image"]})

        else:
            raise ValueError(f"Unsupported image_text content type: {item_type}")

    return [
        {
            "role": "user",
            "content": normalized_content,
        }
    ]


# ======================================================================
# Source span extraction
# ======================================================================

def _extract_text_segments_from_message(
    message_content: List[Dict[str, Any]],
) -> List[str]:
    """
    Group consecutive text blocks into source text segments separated by images.

    Example:
      text, text, image, text, image, text
    ->
      ["text+text", "text", "text"]
    """
    source_text: List[str] = []
    current_text = ""

    for item in message_content:
        item_type = item.get("type")

        if item_type == "image":
            if current_text:
                source_text.append(current_text)
            current_text = ""

        elif item_type == "text":
            current_text += item["text"]

        else:
            raise ValueError(f"Unsupported item type in image-text message: {item_type}")

    if current_text:
        source_text.append(current_text)

    return source_text


def _get_qwen_image_text_spans(
    input_ids: List[int],
    prompt_start: int,
) -> Tuple[List[int], List[Tuple[str, Tuple[int, int]]], Dict[str, List[Tuple[int, int]]]]:
    """
    Recover source modality spans for Qwen image-text inputs.

    Assumes image regions are delimited by:
      <|vision_bos|> ... <|vision_eos|>

    Returns:
      trimmed_input_ids,
      ordered token spans,
      modality spans dict
    """
    # Qwen image-text old logic used this slice.
    start_idx = prompt_start
    end_idx = len(input_ids) - 5
    trimmed = input_ids[start_idx:end_idx]

    # hardcoded Qwen vision bos/eos ids, consistent with old pipeline
    VISION_BOS_TOKEN = 151652
    VISION_EOS_TOKEN = 151653

    token_spans: List[Tuple[str, Tuple[int, int]]] = []
    modality_spans: Dict[str, List[Tuple[int, int]]] = {
        "text": [],
        "image": [],
    }

    image_spans: List[Tuple[int, int]] = []
    curr_img_start = None

    for i, tok in enumerate(trimmed):
        if tok == VISION_BOS_TOKEN:
            curr_img_start = i
        elif tok == VISION_EOS_TOKEN and curr_img_start is not None:
            image_spans.append((curr_img_start, i))
            curr_img_start = None

    cur = 0
    for img_start, img_end in image_spans:
        if cur < img_start:
            text_span = (cur, img_start - 1)
            token_spans.append(("text", text_span))
            modality_spans["text"].append(text_span)

        img_span = (img_start, img_end)
        token_spans.append(("image", img_span))
        modality_spans["image"].append(img_span)
        cur = img_end + 1

    if cur < len(trimmed):
        text_span = (cur, len(trimmed) - 1)
        token_spans.append(("text", text_span))
        modality_spans["text"].append(text_span)

    return trimmed, token_spans, modality_spans


def _get_minicpm_image_text_spans(
    input_ids: List[int],
    prompt_start: int,
    image_bounds: Any,
) -> Tuple[List[int], List[Tuple[str, Tuple[int, int]]], Dict[str, List[Tuple[int, int]]]]:
    """
    Recover source modality spans for MiniCPM image-text inputs.

    MiniCPM exposes image bounds through processor output `image_bound`.

    Returns:
      trimmed_input_ids,
      ordered token spans,
      modality spans dict
    """
    start_idx = prompt_start
    end_idx = len(input_ids) - 9
    trimmed = input_ids[start_idx:end_idx]

    if image_bounds is None:
        raise ValueError("MiniCPM image-text requires image_bounds/image_bound.")

    if len(image_bounds) != 1:
        raise ValueError("Expected exactly one sample in MiniCPM image_bound.")

    image_spans = image_bounds[0].detach().cpu().tolist()

    # keep compatibility with your old image-text implementation
    adjusted_image_spans = []
    for start, end in image_spans:
        adjusted_image_spans.append((start - 5, end))
    image_spans = adjusted_image_spans

    token_spans: List[Tuple[str, Tuple[int, int]]] = []
    modality_spans: Dict[str, List[Tuple[int, int]]] = {
        "text": [],
        "image": [],
    }

    cur = 0
    for img_start, img_end in image_spans:
        if cur < img_start - 1:
            text_span = (cur, img_start - 1)
            token_spans.append(("text", text_span))
            modality_spans["text"].append(text_span)

        img_span = (img_start, img_end)
        token_spans.append(("image", img_span))
        modality_spans["image"].append(img_span)
        cur = img_end + 1

    if cur < len(trimmed):
        text_span = (cur, len(trimmed) - 1)
        token_spans.append(("text", text_span))
        modality_spans["text"].append(text_span)

    return trimmed, token_spans, modality_spans


def extract_image_text_source_spans(
    *,
    bundle,
    gen_result,
    prepared_inputs: Dict[str, Any],
) -> Tuple[List[int], List[Tuple[str, Tuple[int, int]]], Dict[str, List[Tuple[int, int]]]]:
    """
    Dispatch source span extraction by backend.
    """
    input_ids = gen_result.input_ids.tolist()
    prompt_start = gen_result.text_region_start

    if bundle.model_type == "qwen":
        return _get_qwen_image_text_spans(
            input_ids=input_ids,
            prompt_start=prompt_start,
        )

    if bundle.model_type == "minicpm":
        return _get_minicpm_image_text_spans(
            input_ids=input_ids,
            prompt_start=prompt_start,
            image_bounds=prepared_inputs.get("image_bound", None),
        )

    raise ValueError(f"Unsupported model_type for image_text: {bundle.model_type}")


# ======================================================================
# Source chunk building
# ======================================================================

def build_image_text_source_chunks(
    *,
    message_content: List[Dict[str, Any]],
    source_input_ids: List[int],
    modality_spans: Dict[str, List[Tuple[int, int]]],
    tokenizer: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build text chunks and image chunks for image-text attribution.

    Returns:
      input_text_chunks,
      input_image_chunks
    """
    input_text_chunks: List[Dict[str, Any]] = []
    input_image_chunks: List[Dict[str, Any]] = []

    # image chunks are simple one-unit-per-image spans
    for span in modality_spans["image"]:
        input_image_chunks.append(
            {
                "chunk_id": len(input_image_chunks),
                "input_id_span": (int(span[0]), int(span[1])),
            }
        )

    source_text_segments = _extract_text_segments_from_message(message_content)

    if len(source_text_segments) != len(modality_spans["text"]):
        raise ValueError(
            "Mismatch between recovered text spans and message text segments: "
            f"{len(source_text_segments)} vs {len(modality_spans['text'])}"
        )

    for text_segment, token_span in zip(source_text_segments, modality_spans["text"]):
        span_ids = source_input_ids[token_span[0]: token_span[1] + 1]
        tokens = tokenizer.convert_ids_to_tokens(span_ids)

        chunk_spans, chunk_to_token_spans = chunks_to_token_spans(
            text=text_segment,
            token=tokens,
            tokenizer=tokenizer,
            return_pos=False,
            strict_token_check=False,
        )

        default_chunk_idx = token_span[0]

        for chunk_span, chunk_to_token_span in zip(chunk_spans, chunk_to_token_spans):
            if not chunk_to_token_span or not chunk_to_token_span[0] or not chunk_to_token_span[-1]:
                continue

            input_text_chunks.append(
                {
                    "chunk_id": len(input_text_chunks),
                    "input_id_span": (
                        default_chunk_idx + chunk_to_token_span[0][0],
                        default_chunk_idx + chunk_to_token_span[-1][-1],
                    ),
                    "text": str(chunk_span.text),
                }
            )

    return input_text_chunks, input_image_chunks


# ======================================================================
# Attribution
# ======================================================================

def attribute_image_text_chunks(
    *,
    output_text: str,
    token_mappings: List[Dict[str, Any]],
    input_text_chunks: List[Dict[str, Any]],
    input_image_chunks: List[Dict[str, Any]],
    tokenizer: Any,
    curate_cfg=None,
) -> Dict[str, Any]:
    """
    Aggregate image-text token mappings into generation chunks.

    Final output includes:
      - image_sources_curated
      - text_sources_curated
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

        img_srcs, img_confs, img_pos = [], [], []
        txt_srcs, txt_confs, txt_pos = [], [], []

        for j, tm in enumerate(token_mappings[cs:ce + 1]):
            pos_tag = all_pos[j] if j < len(all_pos) else "X"
            src = tm.max_src_token
            src_type = src.get("type", "NONE")

            if src_type == "IMG":
                source_id = int(src.get("token", -1))
                weight = float(src.get("weight", 0.0))
                if source_id >= 0:
                    img_srcs.append(source_id)
                    img_confs.append(weight)
                    img_pos.append(pos_tag)

            elif src_type == "TXT":
                source_id = int(src.get("token", -1))
                weight = float(src.get("weight", 0.0))
                if source_id >= 0:
                    txt_srcs.append(source_id)
                    txt_confs.append(weight)
                    txt_pos.append(pos_tag)

        if curate_cfg is None:
            curated_img = curate_sources_with_conf(
                source_ids=img_srcs,
                pos=img_pos,
                conf=img_confs,
            )
            curated_txt = curate_sources_with_conf(
                source_ids=txt_srcs,
                pos=txt_pos,
                conf=txt_confs,
            )
        else:
            curated_img = curate_sources_with_conf(
                source_ids=img_srcs,
                pos=img_pos,
                conf=img_confs,
                cfg=curate_cfg,
            )
            curated_txt = curate_sources_with_conf(
                source_ids=txt_srcs,
                pos=txt_pos,
                conf=txt_confs,
                cfg=curate_cfg,
            )

        image_formatted = [
            {"source_chunk_id": sid}
            for sid in curated_img
            if 0 <= sid < len(input_image_chunks)
        ]

        text_formatted = [
            {
                "source_chunk_id": sid,
                "source_text": input_text_chunks[sid]["text"],
            }
            for sid in curated_txt
            if 0 <= sid < len(input_text_chunks)
        ]

        per_chunk.append(
            {
                "chunk_id": i,
                "chunk_text": str(chunk_span.text) if hasattr(chunk_span, "text") else str(chunk_span),
                "image_sources_curated": image_formatted,
                "text_sources_curated": text_formatted,
            }
        )

    return {
        "gen_chunks": [str(x.text) if hasattr(x, "text") else str(x) for x in gen_chunks],
        "per_chunk_attribution": per_chunk,
    }


# ======================================================================
# Main public API
# ======================================================================

def trace_image_text(
    bundle,
    method: str,
    prompt: str,
    content: List[Dict[str, Any]],
    generation_config: Optional[GenerationConfig] = None,
) -> Dict[str, Any]:
    """
    Unified image-text tracing API.

    Args:
        bundle:
            Loaded backend model bundle.
        method:
            One of {"attmean", "attraw", "attgrads"}.
        prompt:
            Leading prompt text.
        content:
            Interleaved image-text content list.
        generation_config:
            Optional generation config override.

    Returns:
        A unified OmniTrace image-text result dict.
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

    messages = build_image_text_messages(
        prompt=prompt,
        content=content,
        model_type=bundle.model_type,
    )

    prepared_inputs = prepare_inputs(bundle, messages)
    gen_result = generate_with_attn(bundle, prepared_inputs, gen_cfg=generation_config)

    # align the explicit prompt text inside the prompt-side source region
    prompt_ids = bundle.tokenizer(
        messages[0]["content"][0]["text"],
        add_special_tokens=False,
    )["input_ids"]

    prompt_start, _ = find_prompt_span(
        gen_result.input_ids,
        prompt_ids,
        gen_result.source_start,
        gen_result.source_end,
    )
    if prompt_start >= 0:
        gen_result.text_region_start = prompt_start

    source_input_ids, _, modality_spans = extract_image_text_source_spans(
        bundle=bundle,
        gen_result=gen_result,
        prepared_inputs=prepared_inputs,
    )

    input_text_chunks, input_image_chunks = build_image_text_source_chunks(
        message_content=messages[0]["content"],
        source_input_ids=source_input_ids,
        modality_spans=modality_spans,
        tokenizer=bundle.tokenizer,
    )

    token_mappings = build_image_text_token_mappings(
        bundle=bundle,
        gen_result=gen_result,
        source_input_ids=source_input_ids,
        input_text_chunks=input_text_chunks,
        input_image_chunks=input_image_chunks,
    )
    logger.info(f"Built {len(token_mappings)} image-text token mappings")

    attribution = attribute_image_text_chunks(
        output_text=gen_result.text,
        token_mappings=token_mappings,
        input_text_chunks=input_text_chunks,
        input_image_chunks=input_image_chunks,
        tokenizer=bundle.tokenizer,
    )

    return {
        "prompt": prompt,
        "response": gen_result.text,
        "input_text_chunks": input_text_chunks,
        "input_image_chunks": input_image_chunks,
        "attribution": attribution,
    }
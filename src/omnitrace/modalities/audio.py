from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import librosa

from omnitrace.backends import prepare_inputs, generate_with_attn
from omnitrace.constants import DEFAULT_BIN_SIZE
from omnitrace.core import (
    GenerationConfig,
    DEFAULT_GENERATION_CONFIG,
    find_prompt_span,
    chunk_prompt_text,
    chunks_to_token_spans,
    curate_sources_with_conf,
    validate_method,
)
from omnitrace.mapping.temporal import (
    AudioMapper,
    TimeBinMapper,
    SemanticChunkMapper,
    build_token_mappings,
)
from omnitrace.audio_processing import (
    SemanticChunk,
    ChunkerConfig,
    chunk_audio,
)

logger = logging.getLogger(__name__)


# ======================================================================
# Message building
# ======================================================================

def build_audio_messages(audio: str, prompt: str) -> List[Dict[str, Any]]:
    """
    Standard OmniTrace audio message format.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": prompt},
            ],
        }
    ]


# ======================================================================
# Audio source formatters
# ======================================================================

def merge_consecutive_time_bins(
    curated_bins: List[int],
    bin_size: float,
    audio_duration: float,
) -> List[List[float]]:
    """
    Convert curated time-bin ids into merged [start, end] time ranges.
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
            round(min((e + 1) * bin_size, audio_duration), 2),
        ]
        for s, e in merged
    ]


def format_audio_sources_time_bins(
    curated_ids: List[int],
    input_audio_chunks: List[Dict[str, Any]],
    *,
    bin_size: float,
    audio_duration: float,
) -> List[List[float]]:
    """
    Format curated fixed-bin ids as merged time spans.
    """
    return merge_consecutive_time_bins(
        curated_bins=curated_ids,
        bin_size=bin_size,
        audio_duration=audio_duration,
    )


def format_audio_sources_semantic(
    curated_ids: List[int],
    input_audio_chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Format curated semantic chunk ids as compact chunk records.
    Consecutive semantic chunks are merged.
    """
    valid = sorted(
        set(
            sid for sid in curated_ids
            if 0 <= sid < len(input_audio_chunks)
        )
    )
    if not valid:
        return []

    groups: List[List[int]] = []
    start = prev = valid[0]

    for sid in valid[1:]:
        if sid == prev + 1:
            prev = sid
        else:
            groups.append(list(range(start, prev + 1)))
            start = prev = sid
    groups.append(list(range(start, prev + 1)))

    out: List[Dict[str, Any]] = []
    for grp in groups:
        texts = []
        starts = []
        ends = []

        for sid in grp:
            ch = input_audio_chunks[sid]
            texts.append(ch.get("transcript", ch.get("text", "")))

            tr = ch.get("time_range", [])
            if len(tr) >= 2:
                starts.append(float(tr[0]))
                ends.append(float(tr[1]))

        out.append(
            {
                "source_chunk_id": grp if len(grp) > 1 else grp[0],
                "source_text": " ".join(texts).strip(),
                "time_range": [min(starts), max(ends)] if starts else [],
            }
        )

    return out


# ======================================================================
# Attribution
# ======================================================================

def attribute_audio_chunks(
    output_text: str,
    token_mappings: List[Any],
    input_audio_chunks: List[Dict[str, Any]],
    tokenizer: Any,
    *,
    input_text_chunks: Optional[List[Dict[str, Any]]] = None,
    curate_cfg=None,
    audio_chunk_mode: str = "time_bins",
    bin_size: float = DEFAULT_BIN_SIZE,
    audio_duration: float = 0.0,
) -> Dict[str, Any]:
    """
    Aggregate token-level mappings into generation chunks and produce
    final audio attribution output.

    Final output only contains:
      - audio_sources_curated
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

        audio_srcs, audio_confs, audio_pos = [], [], []

        for j, tm in enumerate(token_mappings[cs:ce + 1]):
            pos_tag = all_pos[j] if j < len(all_pos) else "X"

            src_type = tm.max_src_token.get("type", "NONE")
            if src_type == "AUDIO":
                source_id = tm.max_src_token.get("token", -1)
                weight = float(tm.max_src_token.get("weight", 0.0))
                if source_id >= 0:
                    audio_srcs.append(source_id)
                    audio_confs.append(weight)
                    audio_pos.append(pos_tag)

        curated_audio = curate_sources_with_conf(
            source_ids=audio_srcs,
            pos=audio_pos,
            conf=audio_confs,
            cfg=curate_cfg,
        ) if curate_cfg is not None else curate_sources_with_conf(
            source_ids=audio_srcs,
            pos=audio_pos,
            conf=audio_confs,
        )

        if audio_chunk_mode == "time_bins":
            audio_formatted = format_audio_sources_time_bins(
                curated_ids=curated_audio,
                input_audio_chunks=input_audio_chunks,
                bin_size=bin_size,
                audio_duration=audio_duration,
            )
        elif audio_chunk_mode == "semantic":
            audio_formatted = format_audio_sources_semantic(
                curated_ids=curated_audio,
                input_audio_chunks=input_audio_chunks,
            )
        else:
            raise ValueError(f"Unsupported audio_chunk_mode: {audio_chunk_mode}")

        per_chunk.append(
            {
                "chunk_id": i,
                "chunk_text": str(chunk_span.text) if hasattr(chunk_span, "text") else str(chunk_span),
                "audio_sources_curated": audio_formatted,
            }
        )

    return {
        "gen_chunks": [str(x.text) if hasattr(x, "text") else str(x) for x in gen_chunks],
        "per_chunk_attribution": per_chunk,
    }


# ======================================================================
# Chunk builders
# ======================================================================

def _build_audio_chunks_time_bins(audio_duration: float) -> List[Dict[str, Any]]:
    """
    Placeholder audio chunk list for time-bin mode.
    We only need a single global chunk descriptor because final formatting
    happens by bin ids.
    """
    return [
        {
            "chunk_id": 0,
            "audio_span": [0.0, float(audio_duration)],
            "input_id_span": None,
        }
    ]


def _build_audio_chunks_semantic(
    semantic_chunks: List[SemanticChunk],
) -> List[Dict[str, Any]]:
    """
    Convert semantic chunk dataclasses to output-friendly dicts.
    """
    return [
        {
            "chunk_id": int(ch.index),
            "transcript": ch.text,
            "time_range": [float(ch.start), float(ch.end)],
        }
        for ch in semantic_chunks
    ]


def _resolve_semantic_chunks(
    *,
    audio: str,
    semantic_chunks: Optional[List[SemanticChunk]] = None,
    asr_model: str = "paraformer",
    chunker_config: Optional[ChunkerConfig] = None,
) -> List[SemanticChunk]:
    """
    Resolve semantic audio chunks either from provided chunks or by running ASR chunking.
    """
    if semantic_chunks is not None:
        return semantic_chunks

    if asr_model == "no_asr":
        return []

    if asr_model in {"whisper", "scribe_v2"}:
        # keep compatibility with your existing ablation backends
        if asr_model == "whisper":
            from asr_ablation.whisper_asr import chunk_audio_whisper
            return chunk_audio_whisper(audio_path=audio)
        if asr_model == "scribe_v2": # ElevenLabs Scribe-based semantic chunking
            return chunk_audio(
                audio_path=audio,
                language_code=chunker_config.language_code if chunker_config else None,
                embedding_model=chunker_config.embedding_model if chunker_config else "BAAI/bge-m3",
                min_duration=chunker_config.min_chunk_duration if chunker_config else 5.0,
                max_duration=chunker_config.max_chunk_duration if chunker_config else 60.0,
                diarize=chunker_config.diarize if chunker_config else False,
            )

    # default: paraformer
    from asr_ablation.parafomer_asr import chunk_audio_paraformer
    return chunk_audio_paraformer(audio_path=audio)
     


# ======================================================================
# Main public API
# ======================================================================

def trace_audio(
    bundle,
    method: str,
    prompt: str,
    audio: str,
    audio_chunk_mode: str = "time_bins",   # "time_bins" | "semantic"
    bin_size: float = DEFAULT_BIN_SIZE,
    asr_model: str = "paraformer",
    semantic_chunks: Optional[List[SemanticChunk]] = None,
    chunker_config: Optional[ChunkerConfig] = None,
    generation_config: Optional[GenerationConfig] = None,
) -> Dict[str, Any]:
    """
    Unified audio tracing API.

    Args:
        bundle:
            Loaded backend model bundle.
        method:
            One of {"attmean", "attraw", "attgrads"}.
        prompt:
            User prompt.
        audio:
            Path to audio file.
        audio_chunk_mode:
            "time_bins" or "semantic".
        bin_size:
            Bin size in seconds for time-bin mode.
        asr_model:
            Semantic chunking backend, e.g. "scribe_v2", "whisper", "paraformer", "no_asr".
        semantic_chunks:
            Optional precomputed semantic chunks.
        chunker_config:
            Optional semantic chunking config.
        generation_config:
            Optional generation config override.

    Returns:
        A unified OmniTrace audio result dict.
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

    audio_duration = float(librosa.get_duration(path=audio))
    messages = build_audio_messages(audio=audio, prompt=prompt)

    # ------------------------------------------------------------------
    # Build source chunks + mapper
    # ------------------------------------------------------------------
    if audio_chunk_mode == "time_bins":
        input_audio_chunks = _build_audio_chunks_time_bins(audio_duration)
        audio_mapper: AudioMapper = TimeBinMapper(bin_size=bin_size)

    elif audio_chunk_mode == "semantic":
        resolved_chunks = _resolve_semantic_chunks(
            audio=audio,
            semantic_chunks=semantic_chunks,
            asr_model=asr_model,
            chunker_config=chunker_config,
        )
        input_audio_chunks = _build_audio_chunks_semantic(resolved_chunks)
        audio_mapper = SemanticChunkMapper(resolved_chunks)

    else:
        raise ValueError(
            f"Unsupported audio_chunk_mode: {audio_chunk_mode}. "
            f"Use 'time_bins' or 'semantic'."
        )

    # ------------------------------------------------------------------
    # Prepare inputs + generate
    # ------------------------------------------------------------------
    inputs = prepare_inputs(bundle, messages)
    gen_result = generate_with_attn(bundle, inputs, gen_cfg=generation_config)

    # ------------------------------------------------------------------
    # Locate prompt span and build text chunks
    # (kept internally for prompt alignment / future debugging)
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

    input_text_chunks_rel = chunk_prompt_text(prompt, bundle.tokenizer)
    input_text_chunks = []

    for chunk in input_text_chunks_rel:
        chunk_abs = dict(chunk)
        s, e = chunk_abs["input_id_span"]
        chunk_abs["input_id_span"] = (int(s) + prompt_start, int(e) + prompt_start)
        input_text_chunks.append(chunk_abs)

    # ------------------------------------------------------------------
    # Token-level attribution mapping
    # ------------------------------------------------------------------
    token_mappings = build_token_mappings(
        gen_result=gen_result,
        audio_mapper=audio_mapper,
        audio_duration=audio_duration,
        tokenizer=bundle.tokenizer,
        special_token_ids=bundle.special_token_ids,
        vision_mapper=None,
    )
    logger.info(f"Built {len(token_mappings)} audio token mappings")

    # patch source prompt span into placeholder audio chunk in time-bin mode
    if audio_chunk_mode == "time_bins" and len(gen_result.audio_positions) > 0:
        input_audio_chunks[0]["input_id_span"] = [
            int(gen_result.audio_positions[0]),
            int(gen_result.audio_positions[-1]),
        ]

    # ------------------------------------------------------------------
    # Chunk-level attribution
    # ------------------------------------------------------------------
    attribution = attribute_audio_chunks(
        output_text=gen_result.text,
        token_mappings=token_mappings,
        input_audio_chunks=input_audio_chunks,
        input_text_chunks=input_text_chunks,
        tokenizer=bundle.tokenizer,
        audio_chunk_mode=audio_chunk_mode,
        bin_size=bin_size,
        audio_duration=audio_duration,
    )

    return {
        "prompt": prompt,
        "response": gen_result.text,
        "input_audio_chunks": input_audio_chunks,
        "attribution": attribution,
    }
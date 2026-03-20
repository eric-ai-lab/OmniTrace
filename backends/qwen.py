from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import torch

from constants import QWEN_MODEL_PATH, QWEN_SPECIAL_TOKENS
from core import (
    ModelBundle,
    GenResult,
    GenerationConfig,
    DEFAULT_GENERATION_CONFIG,
    aggregate_step_attention,
    find_source_boundaries,
    GRADIENT_METHODS,
    validate_method,
)
from gradients import compute_grad_scores_by_step

logger = logging.getLogger(__name__)


def load_model_qwen(model_path: str | None = None) -> ModelBundle:
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        Qwen2_5OmniVisionFlashAttention2,
    )

    path = model_path or QWEN_MODEL_PATH
    logger.info(f"Loading Qwen2.5-Omni from {path} (eager LLM + flash vision)...")

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        path,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="eager",
    )
    model.disable_talker()

    # Keep LLM eager so attentions are available; patch vision encoder for efficiency.
    for block in model.thinker.visual.blocks:
        block.attn.__class__ = QWen2_5OmniVisionFlashAttention2 if False else Qwen2_5OmniVisionFlashAttention2
    logger.info("Patched Qwen vision encoder attention to flash_attention_2")

    processor = Qwen2_5OmniProcessor.from_pretrained(path)
    tokenizer = processor.tokenizer

    image_token_id = model.config.thinker_config.audio_token_index
    audio_token_id = model.config.thinker_config.audio_token_index
    vision_token_id = model.config.thinker_config.video_token_index

    logger.info(f"Image token id: {image_token_id}")
    logger.info(f"Audio token id: {audio_token_id}")
    logger.info(f"Vision token id: {vision_token_id}")
    logger.info("Qwen model loaded successfully")

    return ModelBundle(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        model_type="qwen",
        special_token_ids=set(QWEN_SPECIAL_TOKENS),
        audio_token_id=audio_token_id,
        vision_token_id=vision_token_id,
        image_token_id=image_token_id,
    )


def prepare_inputs_qwen(
    bundle: ModelBundle,
    messages: List[Dict[str, Any]],
) -> Dict[str, torch.Tensor]:
    """
    Standard Qwen multimodal input preparation from chat messages.
    Supports text / image / audio / video depending on message content.
    """
    from qwen_omni_utils import process_mm_info

    processor = bundle.processor
    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    audios, images, videos = process_mm_info(messages, use_audio_in_video=True)

    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=True,
    )
    inputs = inputs.to(bundle.model.device).to(bundle.model.dtype)
    return inputs


def generate_with_attn_qwen(
    bundle: ModelBundle,
    inputs: Dict[str, torch.Tensor],
    gen_cfg: GenerationConfig = DEFAULT_GENERATION_CONFIG,
) -> GenResult:
    """
    Generate output text and attribution signals for Qwen.
    """
    method = validate_method(gen_cfg.method)

    input_length = int(inputs["input_ids"].shape[1])
    input_ids_1d = inputs["input_ids"][0].detach().cpu()

    (
        source_start,
        source_end,
        image_positions,
        vision_positions,
        audio_positions,
        text_positions,
        text_region_start,
    ) = find_source_boundaries(
        input_ids=input_ids_1d,
        audio_token_id=bundle.audio_token_id,
        tokenizer=bundle.tokenizer,
        special_token_ids=bundle.special_token_ids,
        vision_token_id=bundle.vision_token_id,
        image_token_id=bundle.image_token_id,
    )

    logger.info(f"Input length: {input_length}")
    logger.info(f"Source region: [{source_start}, {source_end})")
    if len(vision_positions) > 0:
        logger.info(f"Vision tokens: {len(vision_positions)}")
    if len(audio_positions) > 0:
        logger.info(
            f"Audio tokens: {len(audio_positions)} "
            f"(positions: {audio_positions[0]} - {audio_positions[-1]})"
        )
    logger.info(f"Text region start: {text_region_start}")
    logger.info(f"Text tokens in source region: {len(text_positions)}")

    is_grad_method = method in GRADIENT_METHODS
    want_attn = not is_grad_method

    gen_kwargs: Dict[str, Any] = dict(
        thinker_max_new_tokens=gen_cfg.max_new_tokens,
        no_repeat_ngram_size=gen_cfg.no_repeat_ngram_size,
        output_attentions=want_attn,
        return_dict_in_generate=True,
        do_sample=False,
    )
    if gen_cfg.min_new_tokens > 0:
        gen_kwargs["thinker_min_new_tokens"] = gen_cfg.min_new_tokens

    logger.info(f"Generating with method={method} ...")
    with torch.inference_mode():
        outputs = bundle.model.generate(**inputs, **gen_kwargs)

    generated_ids = outputs.sequences[0].detach().cpu()
    text = bundle.tokenizer.decode(
        generated_ids[input_length:],
        skip_special_tokens=True,
    )

    all_attentions: List[np.ndarray] = []
    grad_scores_by_step = None

    if is_grad_method:
        sequences = outputs.sequences.detach()
        logger.info(
            f"Computing gradient-based scores for "
            f"{int(sequences.shape[1]) - input_length} generated steps..."
        )
        grad_scores_by_step = compute_grad_scores_by_step(
            omni_model=bundle.model,
            model_type=bundle.model_type,
            inputs=inputs,
            sequences=sequences,
            prompt_len=input_length,
            score_mode="grad_x_attn",
        )
        logger.info(f"Computed {len(grad_scores_by_step)} grad score vectors")
    else:
        if hasattr(outputs, "attentions") and outputs.attentions is not None:
            for step_attn_tuple in outputs.attentions:
                step_attn = aggregate_step_attention(
                    step_attentions=step_attn_tuple,
                    input_length=input_length,
                    method=method,
                )
                if step_attn is not None:
                    all_attentions.append(step_attn)
        else:
            logger.warning("No attentions returned from Qwen generate()")

    num_generated = len(generated_ids) - input_length
    logger.info(f"Generated {num_generated} tokens")
    if not is_grad_method:
        logger.info(f"Collected {len(all_attentions)} attention maps")
    logger.info(f"Response: {text}")

    return GenResult(
        text=text,
        generated_ids=generated_ids,
        input_ids=input_ids_1d,
        input_length=input_length,
        all_attentions=all_attentions,
        source_start=source_start,
        source_end=source_end,
        text_positions=text_positions,
        text_region_start=text_region_start,
        grad_scores_by_step=grad_scores_by_step,
        image_positions=image_positions,
        audio_positions=audio_positions,
        vision_positions=vision_positions,
    )
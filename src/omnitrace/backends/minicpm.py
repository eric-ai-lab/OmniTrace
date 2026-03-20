from __future__ import annotations

import importlib
import logging
from copy import deepcopy
from typing import Any, Dict, List

import librosa
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel

from omnitrace.constants import MINICPM_MODEL_PATH, MINICPM_SPECIAL_TOKENS
from omnitrace.core import (
    ModelBundle,
    GenResult,
    GenerationConfig,
    DEFAULT_GENERATION_CONFIG,
    aggregate_step_attention,
    GRADIENT_METHODS,
    validate_method,
)
from omnitrace.gradients import compute_grad_scores_by_step

logger = logging.getLogger(__name__)

_SIGLIP_PATCH_SIZE = 14


def _resize_frame(img: Image.Image, max_pixels: int) -> Image.Image:
    """
    Resize one frame so that w*h <= max_pixels and dimensions align to patch size.
    """
    w, h = img.size
    if w * h <= max_pixels:
        return img

    scale = (max_pixels / (w * h)) ** 0.5
    new_w = max(_SIGLIP_PATCH_SIZE, (int(w * scale) // _SIGLIP_PATCH_SIZE) * _SIGLIP_PATCH_SIZE)
    new_h = max(_SIGLIP_PATCH_SIZE, (int(h * scale) // _SIGLIP_PATCH_SIZE) * _SIGLIP_PATCH_SIZE)
    return img.resize((new_w, new_h), Image.BICUBIC)


def _extract_video_total_pixels(messages: List[Dict[str, Any]]) -> int:
    """
    Extract optional total_pixels budget from a video content item.
    """
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "video":
                    return int(item.get("total_pixels", 0) or 0)
    return 0


def _build_minicpm_user_content(messages: List[Dict[str, Any]]) -> List[Any]:
    user_content: List[Any] = []

    for msg in messages:
        if msg.get("role") != "user":
            continue

        content = msg.get("content", [])
        if not isinstance(content, list):
            continue

        for item in content:
            if not isinstance(item, dict):
                continue

            item_type = item.get("type")

            if item_type == "text":
                user_content.append(item["text"])

            elif item_type == "image":
                img = item["image"]
                if isinstance(img, Image.Image):
                    user_content.append(img.convert("RGB"))
                else:
                    user_content.append(Image.open(img).convert("RGB"))

            elif item_type == "audio":
                audio_input, _ = librosa.load(item["audio"], sr=16000, mono=True)
                user_content.append(audio_input)

            elif item_type == "video":
                user_content.append(
                    {"type": "video_url", "video_url": {"url": item["video"], "use_audio": True}}
                )

            else:
                raise ValueError(f"Unsupported content type for MiniCPM: {item_type}")

    return user_content


def load_model_minicpm(model_path: str | None = None) -> ModelBundle:
    path = model_path or MINICPM_MODEL_PATH
    logger.info(f"Loading MiniCPM-o from {path} (eager attention)...")

    model = AutoModel.from_pretrained(
        path,
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        init_vision=True,
        init_audio=True,
        init_tts=False,
    )
    model.eval().cuda()

    # Patch vision encoder to flash attention if available, while keeping LLM eager.
    if hasattr(model, "vpm") and model.vpm is not None:
        try:
            from transformers.utils import is_flash_attn_2_available

            if is_flash_attn_2_available():
                import sys

                module_name = model.vpm.encoder.layers[0].self_attn.__class__.__module__
                siglip_mod = sys.modules[module_name]
                SiglipFlashAttention2 = siglip_mod.SiglipFlashAttention2

                model.vpm._use_flash_attention_2 = True
                for layer in model.vpm.encoder.layers:
                    layer.self_attn.__class__ = SiglipFlashAttention2
                    layer.self_attn.is_causal = False
                    layer._use_flash_attention_2 = True

                logger.info("Patched MiniCPM vision encoder attention to flash_attention_2")
            else:
                logger.info("flash_attn not available; MiniCPM vision encoder stays eager")
        except Exception as e:
            logger.warning(f"Failed to patch MiniCPM vision encoder flash attention: {e}")

    model.prepare_processor(processor=None, tokenizer=None)
    tokenizer = model.processor.tokenizer

    logger.info("MiniCPM model loaded successfully")

    return ModelBundle(
        model=model,
        processor=model.processor,
        tokenizer=tokenizer,
        model_type="minicpm",
        special_token_ids=set(MINICPM_SPECIAL_TOKENS),
        audio_token_id=-1,
        vision_token_id=-1,
        image_token_id=-1,
    )


def prepare_inputs_minicpm(
    bundle: ModelBundle,
    messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build MiniCPM inputs from standard OmniTrace messages.

    Supports:
      - audio + text
      - video + text (with use_audio=True)
      - later, image-text can also be added through the same path if needed
    """
    user_content = _build_minicpm_user_content(messages)

    video_total_pixels = _extract_video_total_pixels(messages)

    model = bundle.model
    tokenizer = bundle.tokenizer

    _pkg = type(bundle.model).__module__.rsplit(".", 1)[0]
    _utils = importlib.import_module(f"{_pkg}.utils")
    normalize_content_item = _utils.normalize_content_item

    msgs = [{"role": "user", "content": user_content}]
    copy_msgs = deepcopy(msgs)

    images: List[Any] = []
    audios: List[Any] = []
    audio_parts: List[int] = []

    for i, msg in enumerate(copy_msgs):
        content = msg["content"]
        if isinstance(content, str):
            content = [content]

        cur_msgs = []
        for c in content:
            if isinstance(c, np.ndarray):
                audios.append(c)
                audio_parts.append(i)
                cur_msgs.append("<audio>./</audio>")
            elif isinstance(c, str):
                cur_msgs.append(c)
            elif isinstance(c, dict) and c.get("type") == "video_url":
                _, omni_contents = normalize_content_item(c)
                for item in omni_contents:
                    if hasattr(item, "mode"):  # PIL image-like video frame
                        images.append(item)
                        cur_msgs.append("<image>./</image>")
                    elif isinstance(item, np.ndarray):
                        audios.append(item)
                        audio_parts.append(i)
                        cur_msgs.append("<audio>./</audio>")
            elif isinstance(c, Image.Image):
                images.append(c.convert("RGB"))
                cur_msgs.append("<image>./</image>")

        msg["content"] = "\n".join(cur_msgs)

    if video_total_pixels > 0 and images:
        max_pixels_per_frame = max(1, video_total_pixels // len(images))
        orig_sizes = [img.size for img in images]
        images = [_resize_frame(img, max_pixels_per_frame) for img in images]
        logger.info(
            f"Video pixel budget: {video_total_pixels} total, "
            f"{max_pixels_per_frame}/frame, {len(images)} frames, "
            f"{orig_sizes[0]} -> {images[0].size}"
        )

    prompt = tokenizer.apply_chat_template(
        copy_msgs,
        tokenize=False,
        add_generation_prompt=True,
        use_tts_template=False,
        enable_thinking=False,
    )

    inputs = model.processor(
        [prompt],
        [images],
        [audios],
        [audio_parts],
        return_tensors="pt",
        max_length=8192,
    ).to(model.device)

    n_audio = len(inputs.get("audio_bounds", [[]])[0])
    n_image = len(inputs.get("image_bound", [[]])[0])
    logger.info(
        f"MiniCPM input length: {inputs['input_ids'].shape[1]}, "
        f"image_bounds: {n_image}, audio_bounds: {n_audio}"
    )

    return dict(inputs)


def generate_with_attn_minicpm(
    bundle: ModelBundle,
    inputs: Dict[str, Any],
    gen_cfg: GenerationConfig = DEFAULT_GENERATION_CONFIG,
) -> GenResult:
    """
    Generate output text and attribution signals for MiniCPM.
    """
    method = validate_method(gen_cfg.method)

    model = bundle.model
    tokenizer = bundle.tokenizer

    input_ids = inputs["input_ids"]
    input_length = int(input_ids.shape[1])
    input_ids_1d = input_ids[0].detach().cpu()

    audio_bounds = inputs.get("audio_bounds", [[]])
    image_bounds = inputs.get("image_bound", [[]])

    audio_positions_list: List[int] = []
    for bound in audio_bounds[0]:
        audio_positions_list.extend(range(int(bound[0]), int(bound[1])))
    audio_positions = np.array(audio_positions_list, dtype=np.int64)

    vision_positions_list: List[int] = []
    for bound in image_bounds[0]:
        vision_positions_list.extend(range(int(bound[0]), int(bound[1])))
    vision_positions = np.array(vision_positions_list, dtype=np.int64)

    has_video_or_image = len(vision_positions) > 0
    all_source = np.sort(np.concatenate([vision_positions, audio_positions])) if (
        len(vision_positions) > 0 or len(audio_positions) > 0
    ) else np.array([], dtype=np.int64)

    source_start = int(all_source[0]) if len(all_source) > 0 else 0

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    im_end_positions = torch.where(input_ids_1d == im_end_id)[0]
    source_end = int(im_end_positions[-1].item()) if len(im_end_positions) > 0 else input_length

    last_source_pos = int(all_source[-1]) if len(all_source) > 0 else source_start
    text_region_start = last_source_pos + 2

    special_ids_tensor = torch.tensor(sorted(list(bundle.special_token_ids)), dtype=torch.long)
    special_mask = torch.isin(input_ids_1d.cpu(), special_ids_tensor)

    source_pos_set = set(all_source.tolist())
    text_positions_list = [
        i for i in range(source_start, source_end)
        if i not in source_pos_set and not bool(special_mask[i].item())
    ]
    text_positions = np.array(text_positions_list, dtype=np.int64)

    logger.info(f"Input length: {input_length}")
    logger.info(f"Source region: [{source_start}, {source_end})")
    if has_video_or_image:
        logger.info(f"Vision/image tokens: {len(vision_positions)}")
    if len(audio_positions) > 0:
        logger.info(f"Audio tokens: {len(audio_positions)}")
    logger.info(f"Text region start: {text_region_start}")

    is_grad_method = method in GRADIENT_METHODS
    want_attn = not is_grad_method

    model_inputs = {
        "input_ids": input_ids,
        "audio_features": inputs.get("audio_features", []),
        "audio_feature_lens": inputs.get("audio_feature_lens", []),
        "image_bound": inputs.get("image_bound", [[]]),
        "audio_bounds": audio_bounds,
        "spk_bounds": inputs.get("spk_bounds", []),
        "pixel_values": inputs.get("pixel_values", [[]]),
        "tgt_sizes": inputs.get("tgt_sizes", None),
    }

    model_inputs["inputs_embeds"], _ = model.get_vllm_embedding(model_inputs)
    model_inputs["inputs_embeds"] = model.get_omni_embedding(
        model_inputs,
        input_embeddings=model_inputs["inputs_embeds"],
        chunk_length=model.config.audio_chunk_length,
    )

    terminators = [tokenizer.convert_tokens_to_ids(t) for t in model.terminators]

    gen_kwargs: Dict[str, Any] = dict(
        inputs_embeds=model_inputs["inputs_embeds"],
        attention_mask=inputs.get("attention_mask", None),
        pad_token_id=0,
        eos_token_id=terminators,
        output_hidden_states=True,
        output_attentions=want_attn,
        return_dict_in_generate=True,
        do_sample=False,
        num_beams=1,
        max_new_tokens=gen_cfg.max_new_tokens,
        no_repeat_ngram_size=gen_cfg.no_repeat_ngram_size,
    )
    if gen_cfg.min_new_tokens > 0:
        gen_kwargs["min_new_tokens"] = gen_cfg.min_new_tokens

    logger.info(f"Generating with method={method} ...")
    with torch.inference_mode():
        outputs = model.llm.generate(**gen_kwargs)

    raw_ids = outputs.sequences[0].detach().cpu()
    result_ids = raw_ids[raw_ids != 0]
    if len(result_ids) > 0 and hasattr(tokenizer, "bos_id") and result_ids[0] == tokenizer.bos_id:
        result_ids = result_ids[1:]
    if len(result_ids) > 0 and int(result_ids[-1].item()) in terminators:
        result_ids = result_ids[:-1]

    text = tokenizer.decode(result_ids)
    text = text.split("<|tts_eos|>")[0].strip()

    # For MiniCPM, downstream mapping expects prompt + generated sequence together.
    generated_ids = torch.cat([input_ids_1d, raw_ids], dim=0)

    all_attentions: List[np.ndarray] = []
    grad_scores_by_step = None

    if is_grad_method:
        sequences = outputs.sequences.detach()
        full_sequences = torch.cat([input_ids.detach(), sequences], dim=1)

        prompt_embeds = model_inputs["inputs_embeds"].detach()
        gen_token_ids = full_sequences[:, input_length:].to(model.llm.device)

        with torch.no_grad():
            gen_embeds = model.llm.get_input_embeddings()(gen_token_ids)

        full_embeds = torch.cat([prompt_embeds, gen_embeds], dim=1)

        logger.info(
            f"Computing gradient-based scores for {int(sequences.shape[1])} generated steps..."
        )
        grad_scores_by_step = compute_grad_scores_by_step(
            omni_model=model,
            model_type=bundle.model_type,
            inputs={"inputs_embeds": full_embeds},
            sequences=full_sequences,
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
            logger.warning("No attentions returned from MiniCPM generate()")

    num_generated = len(raw_ids)
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
        image_positions=np.array([], dtype=np.int64),
        audio_positions=audio_positions,
        vision_positions=vision_positions,
    )
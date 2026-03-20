from __future__ import annotations

import os
import random
from typing import Any, List, Optional, Set, Tuple

import numpy as np
import torch


ATTENTION_METHODS = {"attmean", "attraw"}
GRADIENT_METHODS = {"attgrads"}
SUPPORTED_METHODS = ATTENTION_METHODS | GRADIENT_METHODS


def validate_method(method: str) -> str:
    method = method.lower().strip()
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported attribution method: {method}. "
            f"Supported methods: {sorted(SUPPORTED_METHODS)}"
        )
    return method


def set_determinism(seed: int = 42) -> None:
    """
    Set global random seeds for reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def _attmean(step_attentions: Tuple) -> Optional[torch.Tensor]:
    """
    Mean over (batch, heads, q_len) for each layer, then mean across layers.
    Returns [K].
    """
    per_layer: List[torch.Tensor] = []
    for layer_attn in step_attentions:
        if layer_attn is None:
            continue
        per_layer.append(layer_attn.mean(dim=(0, 1, 2)).cpu())

    if not per_layer:
        return None
    return torch.stack(per_layer, dim=0).mean(dim=0)


def _attraw(step_attentions: Tuple) -> Optional[torch.Tensor]:
    """
    Last layer only, mean over (batch, heads, q_len).
    Returns [K].
    """
    if not step_attentions:
        return None

    last_layer = step_attentions[-1]
    if last_layer is None:
        return None
    return last_layer.mean(dim=(0, 1, 2)).cpu()


def aggregate_step_attention(
    step_attentions: Tuple,
    input_length: int,
    method: str = "attmean",
) -> Optional[np.ndarray]:
    """
    Aggregate one generation step's attention into a [input_length] numpy vector.

    Only valid for attention-based methods.
    Gradient-based methods are handled separately in gradients.py.
    """
    method = validate_method(method)

    if method in GRADIENT_METHODS:
        return None
    if step_attentions is None or len(step_attentions) == 0:
        return None

    if method == "attmean":
        step_attn = _attmean(step_attentions)
    elif method == "attraw":
        step_attn = _attraw(step_attentions)
    else:
        return None

    if step_attn is None or len(step_attn) < input_length:
        return None

    return step_attn[:input_length].float().numpy()


def find_prompt_span(
    input_ids: torch.Tensor,
    prompt_ids: List[int],
    search_start: int,
    search_end: int,
) -> Tuple[int, int]:
    """
    Locate prompt_ids as a contiguous span within input_ids.
    Returns (start_idx, end_idx), or (-1, -1) if not found.
    """
    if not prompt_ids:
        return -1, -1

    input_list = input_ids.tolist()
    end_limit = min(search_end, len(input_list))
    max_start = end_limit - len(prompt_ids)

    for i in range(search_start, max_start + 1):
        if input_list[i:i + len(prompt_ids)] == prompt_ids:
            return i, i + len(prompt_ids) - 1
    return -1, -1


def find_source_boundaries(
    input_ids: torch.Tensor,
    audio_token_id: int,
    tokenizer: Any,
    special_token_ids: Set[int],
    vision_token_id: int = -1,
    image_token_id: int = -1,
) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Detect source-region boundaries by searching for placeholder token IDs.

    Returns:
        (
            source_start,
            source_end,
            image_positions,
            vision_positions,
            audio_positions,
            text_positions,
            text_region_start,
        )

    Notes:
    - For Qwen audio/video, audio/vision token ids are usually enough.
    - For image-text paths, image_positions may remain empty and modality-
      specific span extraction can still happen later in the image_text module.
    - source_end is currently inferred via the last <|im_end|> token.
    """
    if audio_token_id >= 0:
        audio_positions = torch.where(input_ids == audio_token_id)[0].cpu().numpy()
    else:
        audio_positions = np.array([], dtype=np.int64)

    if vision_token_id >= 0:
        vision_positions = torch.where(input_ids == vision_token_id)[0].cpu().numpy()
    else:
        vision_positions = np.array([], dtype=np.int64)

    if image_token_id >= 0:
        image_positions = torch.where(input_ids == image_token_id)[0].cpu().numpy()
    else:
        image_positions = np.array([], dtype=np.int64)

    all_source = np.sort(
        np.concatenate([image_positions, vision_positions, audio_positions])
    )

    source_start = int(all_source[0]) if len(all_source) > 0 else 0

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    im_end_positions = torch.where(input_ids == im_end_id)[0]
    source_end = int(im_end_positions[-1].item()) if len(im_end_positions) > 0 else len(input_ids)

    last_source = int(all_source[-1]) if len(all_source) > 0 else source_start
    text_region_start = last_source + 2

    special_ids_tensor = torch.tensor(
        sorted(list(special_token_ids)),
        dtype=torch.long,
        device=input_ids.device,
    )
    special_mask = torch.isin(input_ids, special_ids_tensor)

    source_pos_set = set(all_source.tolist())
    text_positions = np.array(
        [
            i for i in range(source_start, source_end)
            if i not in source_pos_set and not bool(special_mask[i].item())
        ],
        dtype=np.int64,
    )

    return (
        source_start,
        source_end,
        image_positions,
        vision_positions,
        audio_positions,
        text_positions,
        text_region_start,
    )
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

from .types import TokenMapping


def _build_special_mask(all_ids: torch.Tensor, special_token_ids: set[int]) -> torch.Tensor:
    special_tensor = torch.tensor(
        sorted(list(special_token_ids)),
        dtype=torch.long,
        device=all_ids.device,
    )
    return torch.isin(all_ids, special_tensor)


def _find_text_chunk_id(
    text_chunk_spans: List[Tuple[int, int]],
    token_idx: int,
) -> int:
    for i, (s, e) in enumerate(text_chunk_spans):
        if s <= token_idx <= e:
            return i
    return -1


def build_image_text_token_mappings(
    *,
    bundle,
    gen_result,
    source_input_ids: List[int],
    input_text_chunks: List[Dict[str, Any]],
    input_image_chunks: List[Dict[str, Any]],
) -> List[TokenMapping]:
    """
    Build per-generated-token mappings for image-text attribution.

    Source types:
      - IMG
      - TXT
    """
    input_ids_1d = gen_result.input_ids
    generated_ids_1d = gen_result.generated_ids

    all_ids = generated_ids_1d.detach()
    special_mask_all = _build_special_mask(all_ids, bundle.special_token_ids)

    prompt_start = gen_result.text_region_start
    source_len = len(source_input_ids)

    step_mask = special_mask_all[prompt_start:prompt_start + source_len]

    span_map = torch.full((source_len,), -1, dtype=torch.long)
    for chunk in input_image_chunks:
        img_id = int(chunk["chunk_id"])
        s, e = chunk["input_id_span"]
        span_map[s:e + 1] = img_id

    input_len = int(input_ids_1d.numel())

    all_attentions = gen_result.all_attentions
    grad_scores = gen_result.grad_scores_by_step

    use_grad = grad_scores is not None and len(grad_scores) > 0
    num_steps = len(grad_scores) if use_grad else len(all_attentions)

    text_chunk_spans = [x["input_id_span"] for x in input_text_chunks]

    token_mappings: List[TokenMapping] = []

    for i in range(num_steps):
        if use_grad:
            scores = grad_scores[i].detach().cpu()
        else:
            scores = torch.tensor(all_attentions[i])

        scores = scores[prompt_start:prompt_start + source_len].detach().clone()
        scores[step_mask] = 0.0

        max_token_idx = int(scores.argmax().item())
        max_weight = float(scores[max_token_idx].item())

        image_id = int(span_map[max_token_idx].item())
        if image_id != -1:
            max_src_token = {
                "idx": max_token_idx,
                "type": "IMG",
                "token": image_id,
                "weight": max_weight,
            }
        else:
            src_chunk_id = _find_text_chunk_id(text_chunk_spans, max_token_idx)
            max_src_token = {
                "idx": max_token_idx,
                "type": "TXT",
                "token": src_chunk_id,
                "weight": max_weight,
            }

        gen_token_id = int(generated_ids_1d[input_len + i].item())
        gen_token = bundle.tokenizer.decode(
            [gen_token_id],
            clean_up_tokenization_spaces=True,
        )

        token_mappings.append(
            TokenMapping(
                gen_token=(gen_token_id, gen_token),
                max_src_token=max_src_token,
            )
        )

    return token_mappings
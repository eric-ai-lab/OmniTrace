import torch
from typing import Any, Dict, List, Optional

def _deinference_clone(x: Any) -> Any:
    """
    Convert 'inference tensors' into normal tensors usable by autograd by cloning.
    Keeps dtype/device.
    """
    if isinstance(x, torch.Tensor):
            return x.detach().clone()
    return x

def clone_inputs_for_grad(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clone all tensor fields to avoid 'Inference tensors cannot be saved for backward'.
    """
    return {k: _deinference_clone(v) for k, v in inputs.items()}

def _get_text_model(omni_model, model_type):
    if model_type == "qwen":
        if hasattr(omni_model, "thinker") and omni_model.thinker is not None:
            return omni_model.thinker
        if hasattr(omni_model, "model") and omni_model.model is not None:
            return omni_model.model
        return omni_model
    elif model_type == "minicpm":
        if hasattr(omni_model, "llm") and omni_model.llm is not None:
            return omni_model.llm
        if hasattr(omni_model, "model") and omni_model.model is not None:
            return omni_model.model
        return omni_model

def _slice_inputs_for_prefix(inputs_grad: Dict[str, Any], prefix_ids: torch.Tensor) -> Dict[str, Any]:
    """
    Slice inputs to match prefix length.
    """
    out: Dict[str, Any] = dict(inputs_grad)
    prefix_len = prefix_ids.shape[1]

    if "inputs_embeds" in out and out["inputs_embeds"] is not None:
        out["inputs_embeds"] = out["inputs_embeds"][:, :prefix_len, :]
        device = out["inputs_embeds"].device
    else:
        prefix_ids = prefix_ids.detach().clone()
        out["input_ids"] = prefix_ids
        device = prefix_ids.device

    out["attention_mask"] = torch.ones(1, prefix_len, dtype=torch.long, device=device)
    return out


def compute_grad_scores_by_step(
    *,
    omni_model,                          # bundle.model (wrapper) OR thinker
    model_type: str,                    # "qwen" or "minicpm" (for text model extraction)
    inputs: Dict[str, Any],              # processor outputs used for generate()
    sequences: torch.Tensor,             # [1, prompt_len + gen_len] (may be inference tensor)
    num_steps: Optional[int] = None,
    prompt_len: Optional[int] = None,
    layer: int = -1,                     # last layer by default
    score_mode: str = "grad_x_attn",     # "grad_x_attn" or "grad_only"
    clamp_positive: bool = True,
    max_steps_cap: int = 1000,           
    use_amp: bool = True,                # bf16 autocast to reduce activations
    empty_cache_each_step: bool = False, # enable only if fragmentation is bad
) -> List[torch.Tensor]:
    """
    Memory-optimized Grad-based scores over prefix key positions for next-token generation.

    Per step i:
      - teacher-forced prefix forward
      - target = logit(last_pos, y_i)
      - grad wrt ONE attention layer probs
      - score over keys from last query row (optionally Grad×Attn)
      - store scores on CPU immediately
    """
    score_mode = score_mode.lower()
    if score_mode not in {"grad_x_attn", "grad_only"}:
        raise ValueError(f"Unknown score_mode: {score_mode}")

    text_model = _get_text_model(omni_model, model_type)
    text_model.train(False)

    # Clone once to avoid inference-tensor backward error
    inputs_grad = clone_inputs_for_grad(inputs)
    seq = sequences.clone()
    seq.requires_grad_(False)

    if prompt_len is None:
        if "input_ids" not in inputs_grad:
            raise ValueError("prompt_len not provided and inputs has no input_ids.")
        prompt_len = int(inputs_grad["input_ids"].shape[1])

    gen_len = int(seq.shape[1]) - prompt_len
    if gen_len <= 0:
        return []

    if num_steps is None:
        num_steps = gen_len
    num_steps = int(min(num_steps, gen_len, max_steps_cap))

    scores_by_step: List[torch.Tensor] = []

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_amp and torch.cuda.is_available()
        else torch.enable_grad()
    )

    with torch.enable_grad():
        for i in range(num_steps):
            y_i = int(seq[0, prompt_len + i].item())
            prefix_len = prompt_len + i
            prefix_ids = seq[:, :prefix_len]

            step_inputs = _slice_inputs_for_prefix(inputs_grad, prefix_ids)

            text_model.zero_grad(set_to_none=True)

            with autocast_ctx:
                out = text_model(
                    **step_inputs,
                    output_attentions=True,
                    use_cache=False,
                    return_dict=True,
                )
                logits = out.logits

                # pick exactly one layer index
                L = len(out.attentions)
                l = layer if layer >= 0 else (L + layer)
                if not (0 <= l < L):
                    raise IndexError(f"layer index {layer} out of range for {L} layers")

                attn = out.attentions[l]  # [1, H, K, K] typically

                # drop other layers early
                out.attentions = None

                target = logits[0, -1, y_i]

            grad_attn = torch.autograd.grad(
                outputs=target,
                inputs=attn,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]

            dA_q = grad_attn[0, :, -1, :]  # [H, K]

            if score_mode == "grad_only":
                scores = dA_q.mean(dim=0)  # [K]
            else:
                A_q = attn[0, :, -1, :]    # [H, K]
                scores = (dA_q * A_q).mean(dim=0)

            if clamp_positive:
                scores = scores.clamp(min=0)

            # move to CPU immediately to bound GPU memory
            scores_by_step.append(scores.detach().cpu())

            # aggressively free tensors
            del out, logits, attn, target, grad_attn, dA_q, scores
            if score_mode == "grad_x_attn":
                try:
                    del A_q
                except UnboundLocalError:
                    pass

            if empty_cache_each_step:
                torch.cuda.empty_cache()

    return scores_by_step
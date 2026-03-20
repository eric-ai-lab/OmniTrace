from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

from omnitrace.utils import *
from omnitrace.tracer import OmniTracer

# =========================================================
# Sample conversion
# =========================================================

def sample_to_trace_request(sample: Dict[str, Any], repo_root: Path) -> Dict[str, Any]:
    """
    Convert one dataset sample into a unified OmniTracer request dict.

    Returns a dict like:
        {
            "modality": "audio" | "video" | "image_text",
            "trace_kwargs": {...}
        }
    """
    prompt = sample.get("prompt", "")
    question_blocks = sample.get("question", [])

    if not isinstance(question_blocks, list):
        raise ValueError("Sample field `question` must be a list.")

    modality = detect_modality(question_blocks)

    # -----------------------------------------------------
    # image_text
    # -----------------------------------------------------
    if modality == "image_text":
        content: List[Dict[str, Any]] = []

        for block in question_blocks:
            if "text" in block:
                content.append({"type": "text", "text": block["text"]})
            elif "image" in block:
                content.append(
                    {
                        "type": "image",
                        "image": resolve_media_path(block["image"], repo_root),
                    }
                )
            else:
                raise ValueError(f"Unsupported image_text block: {block}")

        return {
            "modality": "image_text",
            "trace_kwargs": {
                "prompt": prompt,
                "content": content,
            },
        }

    # -----------------------------------------------------
    # audio
    # -----------------------------------------------------
    if modality == "audio":
        audio_path = None
        text_block = None

        for block in question_blocks:
            if "audio" in block:
                audio_path = resolve_media_path(block["audio"], repo_root)
            elif "text" in block:
                text_block = block["text"]

        if audio_path is None:
            raise ValueError("Audio sample missing audio path.")

        full_prompt = prompt
        if text_block:
            full_prompt = f"{prompt}{text_block}" if prompt else text_block

        trace_kwargs = {
            "prompt": full_prompt,
            "audio": audio_path,
            "audio_chunk_mode": "time_bins",
        }

        return {
            "modality": "audio",
            "trace_kwargs": trace_kwargs,
        }

    # -----------------------------------------------------
    # video
    # -----------------------------------------------------
    if modality == "video":
        video_path = None
        text_block = None

        for block in question_blocks:
            if "video" in block:
                video_path = resolve_media_path(block["video"], repo_root)
            elif "text" in block:
                text_block = block["text"]

        if video_path is None:
            raise ValueError("Video sample missing video path.")

        full_prompt = prompt
        if text_block:
            full_prompt = f"{prompt}{text_block}" if prompt else text_block

        return {
            "modality": "video",
            "trace_kwargs": {
                "prompt": full_prompt,
                "video": video_path,
            },
        }

    raise ValueError(f"Unsupported modality: {modality}")


# =========================================================
# Output formatting
# =========================================================

def build_output_record(
    sample: Dict[str, Any],
    modality: str,
    trace_kwargs: Dict[str, Any],
    result: Dict[str, Any],
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "id": sample.get("id"),
        "task": sample.get("task"),
        "source_dataset": sample.get("source_dataset"),
        "modality": modality,
        "prompt": trace_kwargs.get("prompt"),
        "question": sample.get("question"),
        "answer": sample.get("answer"),
        "response": result.get("response"),
        "attribution": result.get("attribution", {}),
    }

    if modality == "image_text":
        record["input_text_chunks"] = result.get("input_text_chunks", [])
        record["input_image_chunks"] = result.get("input_image_chunks", [])

    elif modality == "audio":
        record["input_audio_chunks"] = result.get("input_audio_chunks", [])

    elif modality == "video":
        # current video API only returns attribution + response
        pass

    return record


# =========================================================
# Main runner
# =========================================================

def run_demo(
    questions_path: str,
    model_name: str,
    method: str,
    output_path: str | None,
    limit: int | None,
    use_asr_for_audio: bool = False,
) -> List[Dict[str, Any]]:
    repo_root = Path(".").resolve()
    items = load_json(questions_path)

    if not isinstance(items, list):
        raise ValueError("Expected question file to contain a list of samples.")

    if limit is not None:
        items = items[:limit]

    tracer = OmniTracer(model_name=model_name, method=method)

    print("Loaded backend:")
    print(tracer.get_backend_info())
    print("-" * 80)

    outputs: List[Dict[str, Any]] = []

    for i, sample in enumerate(items):
        sample_id = sample.get("id", i)
        print(f"[{i+1}/{len(items)}] Running sample id={sample_id}")

        request = sample_to_trace_request(sample, repo_root=repo_root)
        modality = request["modality"]
        trace_kwargs = request["trace_kwargs"]
        if use_asr_for_audio: trace_kwargs["audio_chunk_mode"] = "semantic"

        print(f"Detected modality: {modality}")

        result = tracer.trace(
            modality=modality,
            **trace_kwargs,
        )

        record = build_output_record(
            sample=sample,
            modality=modality,
            trace_kwargs=trace_kwargs,
            result=result,
        )
        outputs.append(record)

        print("Response:")
        print(record["response"])
        print("-" * 80)

    if output_path:
        if output_path.endswith(".jsonl"):
            save_jsonl(output_path, outputs)
        else:
            save_json(output_path, outputs)
        print(f"Saved results to: {output_path}")

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Universal OmniTrace demo runner")
    parser.add_argument(
        "--questions_path",
        type=str,
        required=True,
        help="Path to question JSON file",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="qwen",
        choices=["qwen", "minicpm"],
        help="Backend model",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="attmean",
        choices=["attmean", "attraw", "attgrads"],
        help="Attribution method",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional output file (.json or .jsonl)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N samples",
    )

    args = parser.parse_args()

    run_demo(
        questions_path=args.questions_path,
        model_name=args.model_name,
        method=args.method,
        output_path=args.output_path,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
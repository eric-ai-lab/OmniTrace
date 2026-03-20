from __future__ import annotations

from typing import Any, Dict, List, Optional

from backends import load_model
from core import validate_method
from modalities import trace_audio, trace_video, trace_image_text


class OmniTracer:
    """
    Unified public entry point for OmniTrace.

    Example:
        tracer = OmniTracer(model_name="qwen", method="attmean")

        result = tracer.trace(
            modality="audio",
            prompt="Summarize the audio recording in English.",
            audio="sample.wav",
            audio_chunk_mode="semantic",
        )
    """

    def __init__(
        self,
        model_name: str = "qwen",
        method: str = "attmean",
        model_path: Optional[str] = None,
    ) -> None:
        self.model_name = model_name.lower().strip()
        self.method = validate_method(method)
        self.model_path = model_path

        self.bundle = load_model(
            model_name=self.model_name,
            model_path=self.model_path,
        )

    def trace(
        self,
        modality: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run OmniTrace on a specific modality.

        Supported modalities:
          - image_text
          - audio
          - video
        """
        modality = modality.lower().strip()

        if modality == "image_text":
            return self.trace_image_text(**kwargs)
        if modality == "audio":
            return self.trace_audio(**kwargs)
        if modality == "video":
            return self.trace_video(**kwargs)

        raise ValueError(
            f"Unsupported modality: {modality}. "
            f"Supported modalities: ['image_text', 'audio', 'video']"
        )

    def trace_image_text(
        self,
        *,
        prompt: str,
        content: List[Dict[str, Any]],
        generation_config: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Trace attribution for interleaved image-text inputs.

        Args:
            prompt:
                Leading prompt text.
            content:
                Interleaved content list, e.g.
                [
                    {"type": "text", "text": "..."},
                    {"type": "image", "image": "a.jpg"},
                    {"type": "text", "text": "..."},
                ]
            generation_config:
                Optional GenerationConfig override.
        """
        return trace_image_text(
            bundle=self.bundle,
            method=self.method,
            prompt=prompt,
            content=content,
            generation_config=generation_config,
        )

    def trace_audio(
        self,
        *,
        prompt: str,
        audio: str,
        audio_chunk_mode: str = "time_bins",
        bin_size: float = 1.0,
        asr_model: str = "scribe_v2",
        semantic_chunks: Optional[List[Any]] = None,
        chunker_config: Optional[Any] = None,
        generation_config: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Trace attribution for audio input.

        Args:
            prompt:
                User prompt.
            audio:
                Path to audio file.
            audio_chunk_mode:
                "time_bins" or "semantic".
            bin_size:
                Time bin size in seconds when using time_bins mode.
            asr_model:
                Semantic chunking backend. Examples:
                "scribe_v2", "whisper", "paraformer", "no_asr".
            semantic_chunks:
                Optional precomputed semantic chunks.
            chunker_config:
                Optional ChunkerConfig for semantic chunking.
            generation_config:
                Optional GenerationConfig override.
        """
        return trace_audio(
            bundle=self.bundle,
            method=self.method,
            prompt=prompt,
            audio=audio,
            audio_chunk_mode=audio_chunk_mode,
            bin_size=bin_size,
            asr_model=asr_model,
            semantic_chunks=semantic_chunks,
            chunker_config=chunker_config,
            generation_config=generation_config,
        )

    def trace_video(
        self,
        *,
        prompt: str,
        video: str,
        video_fps: Optional[float] = None,
        video_max_pixels: Optional[int] = None,
        generation_config: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Trace attribution for video input.

        Args:
            prompt:
                User prompt.
            video:
                Path to video file.
            video_fps:
                Optional frame sampling rate.
            video_max_pixels:
                Optional total pixel budget for sampled frames.
            generation_config:
                Optional GenerationConfig override.
        """
        return trace_video(
            bundle=self.bundle,
            method=self.method,
            prompt=prompt,
            video=video,
            video_fps=video_fps,
            video_max_pixels=video_max_pixels,
            generation_config=generation_config,
        )

    def get_backend_info(self) -> Dict[str, Any]:
        """
        Return lightweight information about the loaded backend.
        Useful for debugging or demo display.
        """
        return {
            "model_name": self.model_name,
            "method": self.method,
            "model_type": self.bundle.model_type,
            "special_token_count": len(self.bundle.special_token_ids),
            "audio_token_id": self.bundle.audio_token_id,
            "vision_token_id": self.bundle.vision_token_id,
            "image_token_id": self.bundle.image_token_id,
        }
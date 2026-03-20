from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ======================================================================
# Dataclasses
# ======================================================================

@dataclass
class TranscriptSegment:
    """
    One transcript segment with timestamps.
    """
    text: str
    start: float
    end: float
    speaker_id: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class SemanticChunk:
    """
    One semantically coherent audio chunk.
    """
    index: int
    start: float
    end: float
    text: str
    sentences: List[str]
    sentence_timestamps: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "text": self.text,
            "sentence_count": len(self.sentences),
            "sentence_timestamps": self.sentence_timestamps,
        }


@dataclass
class ChunkerConfig:
    """
    Semantic audio chunker configuration.
    """
    # ElevenLabs ASR
    elevenlabs_api_key: Optional[str] = None
    asr_model: str = "scribe_v2"
    language_code: Optional[str] = None
    diarize: bool = False
    num_speakers: Optional[int] = None
    tag_audio_events: bool = False
    timestamps_granularity: str = "word"

    # Embeddings
    embedding_model: str = "BAAI/bge-m3"

    # Chunking controls
    min_chunk_duration: float = 5.0
    max_chunk_duration: float = 60.0
    similarity_threshold: Optional[float] = None

    # Runtime
    device: str = "cuda:0"


# ======================================================================
# Embedding wrapper
# ======================================================================

class EmbedderWrapper:
    """
    Embedding wrapper supporting:
      - BGE-M3 via FlagEmbedding
      - generic SentenceTransformer models
    """

    def __init__(self, model_name: str, device: str = "cuda:0"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._backend = None

    def _load_model(self):
        if self._model is not None:
            return

        if self.model_name.lower() in ("baai/bge-m3", "bge-m3"):
            try:
                from FlagEmbedding import BGEM3FlagModel

                logger.info(f"Loading BGE-M3 model: {self.model_name}")
                self._model = BGEM3FlagModel(
                    self.model_name,
                    use_fp16=True,
                    device=self.device,
                )
                self._backend = "bge-m3"
                return
            except ImportError:
                logger.warning(
                    "FlagEmbedding not installed; falling back to sentence-transformers."
                )

        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading sentence-transformers model: {self.model_name}")
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self._backend = "sentence-transformer"

    @staticmethod
    def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        return embeddings / norms

    def encode(
        self,
        texts: List[str],
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        self._load_model()

        if self._backend == "bge-m3":
            output = self._model.encode(texts, batch_size=32, max_length=8192)
            embeddings = output["dense_vecs"]
        else:
            embeddings = self._model.encode(
                texts,
                show_progress_bar=show_progress_bar,
                normalize_embeddings=normalize_embeddings,
            )

        if normalize_embeddings and self._backend == "bge-m3":
            embeddings = self._l2_normalize(embeddings)

        return embeddings


# ======================================================================
# Shared utilities
# ======================================================================

def _is_cjk_text(text: str) -> bool:
    """
    Check if a string is predominantly CJK text.
    """
    if not text:
        return False

    cjk_count = sum(
        1
        for c in text
        if '\u4e00' <= c <= '\u9fff'
        or '\u3040' <= c <= '\u309f'
        or '\u30a0' <= c <= '\u30ff'
        or '\uac00' <= c <= '\ud7af'
    )
    return cjk_count / len(text) > 0.3


def _compute_similarities(embeddings: np.ndarray) -> List[float]:
    """
    Cosine similarity between consecutive normalized embeddings.
    """
    similarities: List[float] = []
    for i in range(1, len(embeddings)):
        sim = float(np.dot(embeddings[i - 1], embeddings[i]))
        similarities.append(sim)
    logger.debug(f"Computed {len(similarities)} similarity scores")
    return similarities


def _auto_threshold(similarities: List[float]) -> float:
    """
    Automatic threshold:
        mean - 0.8 * std
    then clamped to [0.3, 0.7].
    """
    if not similarities:
        return 0.5

    mean_sim = float(np.mean(similarities))
    std_sim = float(np.std(similarities))
    threshold = mean_sim - 0.8 * std_sim
    threshold = max(0.3, min(0.7, threshold))

    logger.debug(
        f"Similarity stats: mean={mean_sim:.3f}, std={std_sim:.3f}, "
        f"threshold={threshold:.3f}"
    )
    return threshold


def _find_boundaries(
    similarities: List[float],
    segments: List[TranscriptSegment],
    threshold: float,
    min_duration: float,
    max_duration: float,
) -> List[int]:
    """
    Find semantic chunk boundaries using similarity drops and duration constraints.
    """
    boundaries = [0]
    chunk_start_idx = 0

    for i, sim in enumerate(similarities):
        current_duration = segments[i].end - segments[chunk_start_idx].start

        if current_duration >= max_duration:
            boundaries.append(i + 1)
            chunk_start_idx = i + 1
            logger.debug(
                f"Forced split at {i + 1} (max duration exceeded: {current_duration:.1f}s)"
            )
            continue

        if sim < threshold and current_duration >= min_duration:
            boundaries.append(i + 1)
            chunk_start_idx = i + 1
            logger.debug(f"Semantic split at {i + 1} (sim={sim:.3f} < {threshold:.3f})")

    boundaries.append(len(segments))
    return sorted(set(boundaries))


def _build_chunks(
    segments: List[TranscriptSegment],
    boundaries: List[int],
) -> List[SemanticChunk]:
    """
    Convert boundary indices into SemanticChunk objects.
    """
    chunks: List[SemanticChunk] = []

    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        chunk_segs = segments[start_idx:end_idx]

        if not chunk_segs:
            continue

        sentences = [s.text for s in chunk_segs]
        timestamps = [(s.start, s.end) for s in chunk_segs]

        chunks.append(
            SemanticChunk(
                index=i,
                start=chunk_segs[0].start,
                end=chunk_segs[-1].end,
                text=" ".join(sentences),
                sentences=sentences,
                sentence_timestamps=timestamps,
            )
        )

    return chunks


def _single_chunk(segments: List[TranscriptSegment]) -> List[SemanticChunk]:
    """
    Fallback for very short transcripts.
    """
    if not segments:
        return []

    sentences = [s.text for s in segments]
    timestamps = [(s.start, s.end) for s in segments]

    return [
        SemanticChunk(
            index=0,
            start=segments[0].start,
            end=segments[-1].end,
            text=" ".join(sentences),
            sentences=sentences,
            sentence_timestamps=timestamps,
        )
    ]


def _log_chunks_summary(chunks: List[SemanticChunk]) -> None:
    for chunk in chunks:
        logger.info(
            f"  Chunk {chunk.index}: [{chunk.start:.1f}s - {chunk.end:.1f}s] "
            f"({chunk.duration:.1f}s, {len(chunk.sentences)} sentences)"
        )


def _semantic_chunk_pipeline(
    segments: List[TranscriptSegment],
    embedder: EmbedderWrapper,
    min_dur: float,
    max_dur: float,
    threshold: Optional[float],
) -> List[SemanticChunk]:
    """
    Shared semantic chunking pipeline:
      1. embed transcript segments
      2. compute adjacent similarities
      3. pick threshold
      4. find boundaries
      5. build chunks
    """
    if not segments:
        logger.warning("No transcript segments available")
        return []

    if len(segments) < 2:
        return _single_chunk(segments)

    texts = [s.text for s in segments]
    logger.info(f"Computing embeddings for {len(texts)} transcript segments...")

    embeddings = embedder.encode(
        texts,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    similarities = _compute_similarities(embeddings)

    if threshold is None:
        threshold = _auto_threshold(similarities)

    logger.info(f"Using similarity threshold: {threshold:.3f}")

    boundaries = _find_boundaries(
        similarities=similarities,
        segments=segments,
        threshold=threshold,
        min_duration=min_dur,
        max_duration=max_dur,
    )
    chunks = _build_chunks(segments, boundaries)

    logger.info(f"Created {len(chunks)} semantic chunks")
    _log_chunks_summary(chunks)

    return chunks


# ======================================================================
# ElevenLabs Scribe chunker
# ======================================================================

class SemanticAudioChunker:
    """
    Semantic audio chunker using:
      1. ElevenLabs Scribe transcription
      2. embedding similarity boundary detection
    """

    def __init__(self, config: Optional[ChunkerConfig] = None):
        self.config = config or ChunkerConfig()
        self._client = None
        self._embedder = None

        self._api_key = (
            self.config.elevenlabs_api_key
            or os.getenv("ELEVENLABS_API_KEY")
        )
        if not self._api_key:
            raise ValueError(
                "ElevenLabs API key not found. "
                "Set ELEVENLABS_API_KEY or pass elevenlabs_api_key in ChunkerConfig."
            )

    @property
    def client(self):
        if self._client is None:
            from elevenlabs import ElevenLabs

            logger.info("Initializing ElevenLabs client")
            logger.info(f"  Model: {self.config.asr_model}")
            logger.info(f"  Language: {self.config.language_code or 'auto-detect'}")
            logger.info(f"  Diarize: {self.config.diarize}")
            logger.info(f"  Timestamps: {self.config.timestamps_granularity}")

            self._client = ElevenLabs(api_key=self._api_key)
            logger.info("ElevenLabs client initialized successfully")
        return self._client

    @property
    def embedder(self):
        if self._embedder is None:
            logger.info(f"Loading embedding model: {self.config.embedding_model}")
            self._embedder = EmbedderWrapper(
                model_name=self.config.embedding_model,
                device=self.config.device,
            )
            logger.info("Embedding model loaded successfully")
        return self._embedder

    def transcribe(self, audio_path: str) -> List[TranscriptSegment]:
        logger.info(f"Transcribing audio with ElevenLabs Scribe: {audio_path}")

        with open(audio_path, "rb") as audio_file:
            result = self.client.speech_to_text.convert(
                file=audio_file,
                model_id=self.config.asr_model,
                language_code=self.config.language_code,
                tag_audio_events=self.config.tag_audio_events,
                diarize=self.config.diarize,
                num_speakers=self.config.num_speakers,
                timestamps_granularity=self.config.timestamps_granularity,
            )

        logger.info("ElevenLabs transcription complete")

        if not result or not result.words:
            logger.warning("Empty transcription result")
            return []

        segments = self._words_to_segments(result.words)

        for seg in segments:
            seg.text = re.sub(r"\s*-\s*", "", seg.text)
            seg.text = seg.text.strip()

        segments = [s for s in segments if s.text.strip()]
        logger.info(f"ElevenLabs returned {len(segments)} transcript segments")
        return segments

    def _words_to_segments(self, words: List[Any]) -> List[TranscriptSegment]:
        if not words:
            return []

        segments: List[TranscriptSegment] = []
        current_words: List[str] = []
        current_start = None
        current_speaker = None

        sentence_endings = {
            ".", "!", "?", "\u3002", "\uff01", "\uff1f", "\uff1b", "\u2026"
        }

        for word_info in words:
            word_type = getattr(word_info, "type", "word")
            if word_type in ("spacing", "audio_event"):
                continue

            word_text = getattr(word_info, "text", "")
            if not word_text:
                continue

            word_start = getattr(word_info, "start", 0)
            word_end = getattr(word_info, "end", 0)
            speaker_id = getattr(word_info, "speaker_id", None)

            if current_start is None:
                current_start = word_start
                current_speaker = speaker_id

            current_words.append(word_text)

            is_sentence_end = any(word_text.rstrip().endswith(p) for p in sentence_endings)
            speaker_changed = self.config.diarize and speaker_id != current_speaker and current_words

            if is_sentence_end or speaker_changed:
                if current_words:
                    if _is_cjk_text("".join(current_words)):
                        sentence_text = "".join(current_words)
                    else:
                        sentence_text = " ".join(current_words)

                    segments.append(
                        TranscriptSegment(
                            text=sentence_text.strip(),
                            start=current_start,
                            end=word_end,
                            speaker_id=current_speaker,
                        )
                    )

                current_words = []
                current_start = None

                if speaker_changed and word_text:
                    current_words = [word_text]
                    current_start = word_start
                    current_speaker = speaker_id

        if current_words:
            last_word = words[-1]
            if _is_cjk_text("".join(current_words)):
                sentence_text = "".join(current_words)
            else:
                sentence_text = " ".join(current_words)

            segments.append(
                TranscriptSegment(
                    text=sentence_text.strip(),
                    start=current_start,
                    end=getattr(last_word, "end", 0),
                    speaker_id=current_speaker,
                )
            )

        return segments

    def chunk(
        self,
        audio_path: str,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[SemanticChunk]:
        min_dur = min_duration or self.config.min_chunk_duration
        max_dur = max_duration or self.config.max_chunk_duration
        threshold = similarity_threshold or self.config.similarity_threshold

        segments = self.transcribe(audio_path)
        return _semantic_chunk_pipeline(
            segments=segments,
            embedder=self.embedder,
            min_dur=min_dur,
            max_dur=max_dur,
            threshold=threshold,
        )

    def get_chunk_for_time(
        self,
        chunks: List[SemanticChunk],
        time_point: float,
    ) -> Optional[SemanticChunk]:
        for chunk in chunks:
            if chunk.start <= time_point <= chunk.end:
                return chunk
        return None

    def save_chunks(
        self,
        chunks: List[SemanticChunk],
        output_path: str,
    ) -> None:
        data = {
            "config": {
                "asr_model": self.config.asr_model,
                "language_code": self.config.language_code,
                "embedding_model": self.config.embedding_model,
                "min_chunk_duration": self.config.min_chunk_duration,
                "max_chunk_duration": self.config.max_chunk_duration,
                "diarize": self.config.diarize,
            },
            "num_chunks": len(chunks),
            "chunks": [chunk.to_dict() for chunk in chunks],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Chunks saved to {output_path}")


# ======================================================================
# Convenience functions
# ======================================================================

def create_chunker(
    language_code: Optional[str] = None,
    embedding_model: str = "BAAI/bge-m3",
    min_duration: float = 5.0,
    max_duration: float = 60.0,
    diarize: bool = False,
    device: str = "cuda:0",
    api_key: Optional[str] = None,
) -> SemanticAudioChunker:
    config = ChunkerConfig(
        elevenlabs_api_key=api_key,
        language_code=language_code,
        embedding_model=embedding_model,
        min_chunk_duration=min_duration,
        max_chunk_duration=max_duration,
        diarize=diarize,
        device=device,
    )
    return SemanticAudioChunker(config)


def chunk_audio(
    audio_path: str,
    language_code: Optional[str] = None,
    embedding_model: str = "BAAI/bge-m3",
    min_duration: float = 5.0,
    max_duration: float = 60.0,
    diarize: bool = False,
) -> List[SemanticChunk]:
    chunker = create_chunker(
        language_code=language_code,
        embedding_model=embedding_model,
        min_duration=min_duration,
        max_duration=max_duration,
        diarize=diarize,
    )
    return chunker.chunk(audio_path)
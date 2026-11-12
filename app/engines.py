"""Speech-to-text engines behind one small protocol.

Adding another backend (OpenAI API, Deepgram, ...) means writing one class with
``name``, ``model`` and ``transcribe()``; nothing else in the pipeline changes.
"""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from app.audio import wav_duration
from app.config import Settings

if TYPE_CHECKING:
    import numpy as np
    from faster_whisper import WhisperModel


@dataclass(frozen=True)
class RawSegment:
    """Engine output before post-processing. Times are relative to the input file."""

    start: float
    end: float
    text: str
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0


@dataclass(frozen=True)
class EngineResult:
    segments: list[RawSegment]
    language: str | None


class TranscriptionEngine(Protocol):
    name: str
    model: str

    def transcribe(self, wav_path: Path, language: str | None = None) -> EngineResult:
        """Transcribe a canonical 16 kHz mono WAV file."""
        ...

    def is_loaded(self) -> bool: ...


@lru_cache(maxsize=4)
def _load_whisper(model_size: str, device: str, compute_type: str) -> WhisperModel:
    # Imported lazily so tests and the mock engine never import CTranslate2.
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device=device, compute_type=compute_type)


def _read_pcm(wav_path: Path) -> np.ndarray:
    """Canonical WAV -> float32 samples in [-1, 1].

    Input is already 16 kHz mono PCM, so we hand Whisper the samples directly
    instead of letting it decode the file again through PyAV.
    """
    import numpy as np

    with wave.open(str(wav_path), "rb") as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


class FasterWhisperEngine:
    """Open-source Whisper on CTranslate2. The model loads once per process."""

    name = "faster-whisper"

    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        self.model = model_size
        self._device = device
        self._compute_type = compute_type

    def _whisper(self) -> WhisperModel:
        return _load_whisper(self.model, self._device, self._compute_type)

    def is_loaded(self) -> bool:
        return _load_whisper.cache_info().currsize > 0

    def transcribe(self, wav_path: Path, language: str | None = None) -> EngineResult:
        segments, info = self._whisper().transcribe(
            _read_pcm(wav_path),  # already 16 kHz mono PCM: no second decode
            language=language,
            vad_filter=True,  # skip non-speech: faster and fewer hallucinations
            beam_size=5,
            condition_on_previous_text=False,  # stops one bad window poisoning the rest
        )
        # ``segments`` is a lazy generator: transcription happens while iterating.
        raw = [
            RawSegment(s.start, s.end, s.text, s.avg_logprob, s.no_speech_prob) for s in segments
        ]
        return EngineResult(segments=raw, language=info.language)


class MockEngine:
    """Deterministic fake: one segment per 2.5 s of audio. For tests and offline demos."""

    name = "mock"
    model = "mock"
    segment_seconds = 2.5

    def is_loaded(self) -> bool:
        return True

    def transcribe(self, wav_path: Path, language: str | None = None) -> EngineResult:
        duration = wav_duration(wav_path)
        count = math.ceil(duration / self.segment_seconds)
        segments = [
            RawSegment(
                start=i * self.segment_seconds,
                end=min(duration, (i + 1) * self.segment_seconds),
                text=f"mock segment {i + 1}",
                avg_logprob=-0.1,
                no_speech_prob=0.01,
            )
            for i in range(count)
        ]
        return EngineResult(segments=segments, language=language or "en")


def build_engine(cfg: Settings) -> TranscriptionEngine:
    if cfg.engine == "mock":
        return MockEngine()
    if cfg.engine == "faster-whisper":
        return FasterWhisperEngine(cfg.model_size, cfg.device, cfg.compute_type)
    raise ValueError(f"Unknown engine {cfg.engine!r} (expected 'faster-whisper' or 'mock')")

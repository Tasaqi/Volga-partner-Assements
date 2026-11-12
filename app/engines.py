"""Speech-to-text engines behind one small protocol.

Adding another backend (Deepgram, AssemblyAI, ...) means writing one class with
``name``, ``model``, ``transcribe()`` and ``is_loaded()``; nothing else changes.
``FallbackEngine`` composes any two engines, e.g. OpenAI API -> local Whisper.
"""

from __future__ import annotations

import logging
import math
import os
import time
import wave
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from app import audio
from app.audio import wav_duration
from app.config import Settings
from app.languages import TO_LANGUAGE_CODE

if TYPE_CHECKING:
    import numpy as np
    from faster_whisper import WhisperModel
    from openai import OpenAI

log = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """The engine failed on input that passed validation (not the caller's fault)."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code  # 502: upstream API failed; 500: local engine failed


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
    fallback: bool = False  # True when produced by FallbackEngine's secondary engine


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
        try:
            segments, info = self._whisper().transcribe(
                _read_pcm(wav_path),  # already 16 kHz mono PCM: no second decode
                language=language,
                vad_filter=True,  # skip non-speech: faster and fewer hallucinations
                beam_size=5,
                condition_on_previous_text=False,  # stops one bad window poisoning the rest
            )
            # ``segments`` is a lazy generator: transcription happens while iterating.
            raw = [
                RawSegment(s.start, s.end, s.text, s.avg_logprob, s.no_speech_prob)
                for s in segments
            ]
        except RuntimeError as exc:  # CTranslate2 failures, e.g. out of memory
            raise TranscriptionError(f"Local transcription failed: {exc}", 500) from exc
        return EngineResult(segments=raw, language=info.language)


class OpenAIEngine:
    """Hosted OpenAI transcription API. Needs ``OPENAI_API_KEY``."""

    name = "openai"

    def __init__(self, model: str = "whisper-1", client: OpenAI | None = None):
        if client is None:
            from openai import OpenAI

            # Raises at startup if OPENAI_API_KEY is missing. The SDK already retries
            # 429/5xx and connection errors with backoff; only then does fallback start.
            client = OpenAI(timeout=120, max_retries=2)
        self.model = model
        self._client = client

    def is_loaded(self) -> bool:
        return True

    def transcribe(self, wav_path: Path, language: str | None = None) -> EngineResult:
        import openai

        # Lossless FLAC is ~half the bytes of WAV: faster uploads, and a 10-minute chunk
        # stays far below the API's 25 MB limit.
        flac = wav_path.with_suffix(".flac")
        try:
            audio.encode_flac(wav_path, flac)
            with flac.open("rb") as f:
                result = self._client.audio.transcriptions.create(
                    model=self.model,
                    file=f,
                    response_format="verbose_json",  # the only format that returns segments
                    timestamp_granularities=["segment"],
                    **({"language": language} if language else {}),
                )
        except (openai.OpenAIError, audio.AudioError) as exc:
            raise TranscriptionError(f"OpenAI transcription failed: {exc}") from exc
        finally:
            flac.unlink(missing_ok=True)

        raw = [
            RawSegment(s.start, s.end, s.text, s.avg_logprob, s.no_speech_prob)
            for s in result.segments or []
        ]
        # The API reports "english"; faster-whisper reports "en". Normalise to codes.
        name = (getattr(result, "language", None) or "").lower()
        return EngineResult(segments=raw, language=TO_LANGUAGE_CODE.get(name, name or None))


class FallbackEngine:
    """Try ``primary``; if it fails, transcribe the same chunk with ``fallback``.

    A failure opens a simple circuit breaker: for ``cooldown_s`` the primary is
    skipped, so during an outage each chunk does not first wait out the primary's
    retries and timeout.
    """

    def __init__(
        self, primary: TranscriptionEngine, fallback: TranscriptionEngine, cooldown_s: float = 60.0
    ):
        self.primary, self.fallback, self.cooldown_s = primary, fallback, cooldown_s
        self.name = f"{primary.name}+{fallback.name}"
        self.model = f"{primary.model}+{fallback.model}"
        self._skip_primary_until = 0.0  # shared by chunk threads; a racy read costs one call

    def is_loaded(self) -> bool:
        return self.primary.is_loaded() and self.fallback.is_loaded()

    def transcribe(self, wav_path: Path, language: str | None = None) -> EngineResult:
        primary_error = "skipped: failed recently"
        if time.monotonic() >= self._skip_primary_until:  # circuit breaker after a failure
            try:
                return self.primary.transcribe(wav_path, language)
            except TranscriptionError as exc:
                self._skip_primary_until = time.monotonic() + self.cooldown_s
                primary_error = str(exc)
                log.warning("%s failed, using %s: %s", self.primary.name, self.fallback.name, exc)
        try:
            return replace(self.fallback.transcribe(wav_path, language), fallback=True)
        except TranscriptionError as exc:
            raise TranscriptionError(
                f"{self.primary.name}: {primary_error}; {self.fallback.name}: {exc}",
                exc.status_code,
            ) from exc


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


def _single_engine(name: str, cfg: Settings) -> TranscriptionEngine:
    if name == "mock":
        return MockEngine()
    if name == "faster-whisper":
        return FasterWhisperEngine(cfg.model_size, cfg.device, cfg.compute_type)
    if name == "openai":
        return OpenAIEngine(cfg.openai_model)
    raise ValueError(f"Unknown engine {name!r} (expected 'faster-whisper', 'openai' or 'mock')")


def build_engine(cfg: Settings) -> TranscriptionEngine:
    if cfg.engine == "openai" and not os.environ.get("OPENAI_API_KEY"):
        if cfg.fallback_engine in ("", "none", "openai"):
            raise ValueError("TP_ENGINE=openai needs OPENAI_API_KEY (or a TP_FALLBACK_ENGINE)")
        # Degrade instead of refusing to start, so the service still runs keyless/offline.
        log.warning("OPENAI_API_KEY not set: using %s only", cfg.fallback_engine)
        return _single_engine(cfg.fallback_engine, cfg)
    primary = _single_engine(cfg.engine, cfg)
    # The mock engine is for offline tests and demos, so it is never wrapped.
    if cfg.engine == "mock" or cfg.fallback_engine in ("", "none", cfg.engine):
        return primary
    fallback = _single_engine(cfg.fallback_engine, cfg)
    if isinstance(fallback, FasterWhisperEngine):
        fallback._whisper()  # load now: a broken fallback should fail at startup, not mid-outage
    return FallbackEngine(primary, fallback, cfg.fallback_cooldown_s)

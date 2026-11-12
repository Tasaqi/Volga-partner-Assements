"""Speech-to-text engines behind one interface.

The API layer depends only on the `Transcriber` protocol, so engines are swappable via
configuration and tests can inject a fake without touching a real model or network.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol

import numpy as np

from transcriber.audio import WHISPER_SAMPLE_RATE, encode_audio
from transcriber.config import Settings, settings


log = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """The engine failed on input that passed validation."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code  # 502: upstream API failed; 500: local engine failed


@dataclass
class Segment:
    start: float  # seconds from the start of the audio
    end: float
    text: str


@dataclass
class Transcription:
    text: str
    language: str | None
    duration: float
    segments: list[Segment] = field(default_factory=list)


def _segment(start: float, end: float, text: str) -> Segment:
    # Millisecond precision is what subtitle formats use; more digits are noise.
    return Segment(start=round(start, 3), end=round(end, 3), text=text.strip())


def _language_code(name: str | None) -> str | None:
    """The hosted API returns 'english'; local Whisper returns 'en'. Normalise to ISO 639-1."""
    from whisper.tokenizer import TO_LANGUAGE_CODE

    return TO_LANGUAGE_CODE.get(name.lower(), name) if name else None


class Transcriber(Protocol):
    name: str

    def transcribe(
        self, samples: np.ndarray, *, prompt: str | None = None, language: str | None = None
    ) -> Transcription:
        """Transcribe 16 kHz mono float32 PCM.

        ``prompt`` is preceding text used as context; ``language`` skips auto-detection.
        """
        ...


class LocalWhisperTranscriber:
    """Open-source OpenAI Whisper running in-process (CPU or GPU)."""

    name = "local"

    def __init__(self, model_name: str):
        import torch
        import whisper  # heavy import (torch); keep it off the module import path

        self._model = whisper.load_model(model_name)
        self._fp16 = torch.cuda.is_available()  # fp16 is only faster (and supported) on GPU
        # One model instance is shared by all request threads; PyTorch inference on a shared
        # module is not guaranteed thread-safe, and parallel runs would multiply memory use.
        self._lock = threading.Lock()

    def transcribe(self, samples, *, prompt=None, language=None) -> Transcription:
        try:
            with self._lock:
                result = self._model.transcribe(
                    samples, fp16=self._fp16, initial_prompt=prompt, language=language
                )
        except RuntimeError as e:  # PyTorch failures, e.g. CUDA out of memory
            raise TranscriptionError(f"Local transcription failed: {e}", status_code=500) from e
        return Transcription(
            text=result["text"].strip(),
            language=result.get("language"),
            duration=round(len(samples) / WHISPER_SAMPLE_RATE, 3),
            segments=[_segment(s["start"], s["end"], s["text"]) for s in result["segments"]],
        )


class OpenAIWhisperTranscriber:
    """Hosted OpenAI transcription API. Needs OPENAI_API_KEY."""

    name = "openai"

    def __init__(self, model_name: str, client=None):
        if client is None:
            from openai import OpenAI

            # Reads OPENAI_API_KEY (raises at startup if missing). The SDK retries 429/5xx and
            # connection errors with exponential backoff; only after that do we fall back.
            client = OpenAI(timeout=120, max_retries=2)
        self._client = client
        self._model = model_name

    def transcribe(self, samples, *, prompt=None, language=None) -> Transcription:
        import openai

        # Send normalised 16 kHz mono FLAC rather than the original file: every input format
        # works (the API accepts only ~10), and uploads shrink (Whisper resamples to 16 kHz
        # mono server-side anyway, so nothing is lost).
        payload = ("audio.flac", encode_audio(samples), "audio/flac")
        optional = {k: v for k, v in {"prompt": prompt, "language": language}.items() if v}
        try:
            result = self._client.audio.transcriptions.create(
                model=self._model,
                file=payload,
                response_format="verbose_json",  # the only format that returns segments
                timestamp_granularities=["segment"],
                **optional,
            )
        except openai.OpenAIError as e:
            raise TranscriptionError(f"Upstream transcription failed: {e}") from e
        return Transcription(
            text=result.text.strip(),
            language=_language_code(getattr(result, "language", None)),
            duration=round(result.duration, 3),
            segments=[_segment(s.start, s.end, s.text) for s in result.segments or []],
        )


class FallbackTranscriber:
    """Try ``primary``; if it fails, transcribe the same chunk with ``fallback``.

    A failure opens a simple circuit breaker: for ``cooldown`` seconds the primary is skipped,
    so during an outage each chunk does not first wait out the primary's retries and timeout.
    """

    def __init__(self, primary: Transcriber, fallback: Transcriber, cooldown: float = 60.0):
        self.name = f"{primary.name}+{fallback.name}"
        self._primary, self._fallback, self._cooldown = primary, fallback, cooldown
        self._skip_primary_until = 0.0  # shared across requests; a racy read only costs one call

    def transcribe(self, samples, *, prompt=None, language=None) -> Transcription:
        kwargs = {"prompt": prompt, "language": language}
        primary_error = "skipped: primary recently failed"
        if time.monotonic() >= self._skip_primary_until:
            try:
                return self._primary.transcribe(samples, **kwargs)
            except TranscriptionError as e:
                self._skip_primary_until = time.monotonic() + self._cooldown
                primary_error = str(e)
                log.warning("%s failed, using %s: %s", self._primary.name, self._fallback.name, e)
        try:
            return self._fallback.transcribe(samples, **kwargs)
        except TranscriptionError as e:
            raise TranscriptionError(
                f"{self._primary.name}: {primary_error}; {self._fallback.name}: {e}", e.status_code
            ) from e


def _engine(name: str, cfg: Settings) -> Transcriber:
    if name == "local":
        return LocalWhisperTranscriber(cfg.whisper_model)
    if name == "openai":
        return OpenAIWhisperTranscriber(cfg.openai_model)
    raise ValueError(f"Unknown transcription engine '{name}' (expected 'local' or 'openai')")


def build_transcriber(cfg: Settings) -> Transcriber:
    primary = _engine(cfg.engine, cfg)
    if cfg.fallback_engine in ("", "none", cfg.engine):
        return primary
    # Built eagerly: a broken fallback should fail the deploy, not surface mid-outage.
    return FallbackTranscriber(primary, _engine(cfg.fallback_engine, cfg))


@lru_cache(maxsize=1)
def get_transcriber() -> Transcriber:
    """One engine per process: loading a model takes seconds and hundreds of MB."""
    return build_transcriber(settings)

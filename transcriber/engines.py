"""Speech-to-text engines behind one interface.

The API layer depends only on the `Transcriber` protocol, so engines are swappable via
configuration and tests can inject a fake without touching a real model or network.
"""

import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from transcriber.audio import load_audio
from transcriber.config import Settings, settings


class TranscriptionError(Exception):
    """The engine failed on input that passed validation."""

    status_code = 502


@dataclass
class Transcription:
    text: str
    language: str | None


class Transcriber(Protocol):
    name: str

    def transcribe(self, audio_path: Path) -> Transcription: ...


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

    def transcribe(self, audio_path: Path) -> Transcription:
        audio = load_audio(audio_path)
        with self._lock:
            result = self._model.transcribe(audio, fp16=self._fp16)
        return Transcription(text=result["text"].strip(), language=result.get("language"))


class OpenAIWhisperTranscriber:
    """Hosted OpenAI transcription API. Needs OPENAI_API_KEY."""

    name = "openai"

    def __init__(self, model_name: str, client=None):
        if client is None:
            from openai import OpenAI

            client = OpenAI()  # reads OPENAI_API_KEY; raises at startup if missing
        self._client = client
        self._model = model_name

    def transcribe(self, audio_path: Path) -> Transcription:
        import openai

        try:
            with open(audio_path, "rb") as f:
                result = self._client.audio.transcriptions.create(
                    model=self._model, file=f, response_format="verbose_json"
                )
        except openai.OpenAIError as e:
            raise TranscriptionError(f"Upstream transcription failed: {e}") from e
        return Transcription(text=result.text.strip(), language=getattr(result, "language", None))


def build_transcriber(cfg: Settings) -> Transcriber:
    if cfg.engine == "local":
        return LocalWhisperTranscriber(cfg.whisper_model)
    if cfg.engine == "openai":
        return OpenAIWhisperTranscriber(cfg.openai_model)
    raise ValueError(f"Unknown TRANSCRIBER_ENGINE '{cfg.engine}' (expected 'local' or 'openai')")


@lru_cache(maxsize=1)
def get_transcriber() -> Transcriber:
    """One engine per process: loading a model takes seconds and hundreds of MB."""
    return build_transcriber(settings)

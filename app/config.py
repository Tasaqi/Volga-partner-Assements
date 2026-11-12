"""Runtime settings, read from environment variables prefixed with ``TP_``.

Example: ``TP_MODEL_SIZE=small TP_CHUNK_WORKERS=2 uvicorn ...``
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Engine
    # Primary: hosted OpenAI Whisper (needs OPENAI_API_KEY). "faster-whisper" | "openai" | "mock"
    engine: str = "openai"
    # Used when the primary fails, so the hosted API degrades to open-source Whisper
    # instead of erroring. "none" (or the same name as ``engine``) disables it.
    fallback_engine: str = "faster-whisper"
    fallback_cooldown_s: float = 60.0  # skip a failing primary for this long
    openai_model: str = "whisper-1"
    model_size: str = "base"  # tiny | base | small | medium | large-v3
    device: str = "cpu"  # cpu | cuda | auto
    compute_type: str = "int8"

    # Auth: when set, /transcriptions requires header "X-API-Key: <value>". Empty = open.
    api_key: str = ""

    # Input limits
    max_upload_mb: int = 500
    max_duration_s: float = 4 * 3600

    # Files longer than this become a background job (HTTP 202)
    sync_max_duration_s: float = 120.0
    job_workers: int = 2

    # Long-audio chunking
    chunk_seconds: float = 600.0
    silence_search_s: float = 30.0  # look for a silence within +/- this of each boundary
    chunk_overlap_s: float = 2.0  # added on both sides of a hard cut
    silence_noise_db: float = -35.0
    silence_min_s: float = 0.5
    chunk_workers: int = 1

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @classmethod
    def from_env(cls, **overrides: object) -> Settings:
        """Build settings from ``TP_<FIELD_NAME>`` env vars; ``overrides`` win."""
        values: dict[str, object] = {}
        for f in dataclasses.fields(cls):
            raw = os.environ.get(f"TP_{f.name.upper()}")
            if raw is not None:
                cast = type(f.default)  # every field has a typed default
                values[f.name] = cast(raw)
        values.update(overrides)
        return cls(**values)  # type: ignore[arg-type]

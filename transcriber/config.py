"""Runtime configuration, read from environment variables (12-factor style)."""

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    max_upload_mb: int = field(default_factory=lambda: int(_env("MAX_UPLOAD_MB", "500")))
    # "local" runs open-source Whisper in-process; "openai" calls the hosted Whisper API.
    engine: str = field(default_factory=lambda: _env("TRANSCRIBER_ENGINE", "local"))
    whisper_model: str = field(default_factory=lambda: _env("WHISPER_MODEL", "base"))
    openai_model: str = field(default_factory=lambda: _env("OPENAI_TRANSCRIBE_MODEL", "whisper-1"))
    # Long audio is processed in chunks of about this length. 10 min keeps memory at ~40 MB
    # per chunk and the FLAC upload (~1 MB/min) well under the hosted API's 25 MB cap.
    chunk_seconds: float = field(default_factory=lambda: float(_env("CHUNK_SECONDS", "600")))

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


settings = Settings()

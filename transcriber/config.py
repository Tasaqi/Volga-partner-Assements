"""Runtime configuration, read from environment variables (12-factor style)."""

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    max_upload_mb: int = field(default_factory=lambda: int(os.getenv("MAX_UPLOAD_MB", "500")))
    # "local" runs open-source Whisper in-process; "openai" calls the hosted Whisper API.
    engine: str = field(default_factory=lambda: os.getenv("TRANSCRIBER_ENGINE", "local"))
    # Used when the primary engine fails. Defaults to local Whisper, so the hosted engine
    # degrades to open source instead of erroring. "none" disables; same as engine = no-op.
    fallback_engine: str = field(default_factory=lambda: os.getenv("TRANSCRIBER_FALLBACK", "local"))
    whisper_model: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "base"))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_TRANSCRIBE_MODEL", "whisper-1"))
    # Long audio is processed in chunks of about this length. 10 min keeps memory at ~40 MB
    # per chunk and the FLAC upload (~1 MB/min) well under the hosted API's 25 MB cap.
    chunk_seconds: float = field(default_factory=lambda: float(os.getenv("CHUNK_SECONDS", "600")))
    # Cost/capacity guard: 500 MB of low-bitrate Opus is ~180 hours of audio.
    max_audio_minutes: float = field(default_factory=lambda: float(os.getenv("MAX_AUDIO_MINUTES", "180")))

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


settings = Settings()

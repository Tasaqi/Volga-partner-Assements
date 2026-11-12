"""Runtime configuration, read from environment variables (12-factor style)."""

import os
from dataclasses import dataclass, field


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    max_upload_mb: int = field(default_factory=lambda: _env_int("MAX_UPLOAD_MB", 500))

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


settings = Settings()

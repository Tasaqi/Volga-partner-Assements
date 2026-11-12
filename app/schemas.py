"""Public data models returned by the API."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, computed_field


class OutputFormat(str, Enum):
    json = "json"
    srt = "srt"
    vtt = "vtt"
    txt = "txt"


def format_timestamp(seconds: float, sep: str = ".") -> str:
    """``3661.25`` -> ``01:01:01.250`` (``sep=","`` gives the SRT style)."""
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


class Segment(BaseModel):
    id: int
    start: float = Field(description="Seconds from the start of the original file")
    end: float
    text: str
    confidence: float = Field(ge=0.0, le=1.0)

    # Numbers stay for machines (seek, maths); this readable copy is for people.
    @computed_field  # type: ignore[prop-decorator]
    @property
    def timeframe(self) -> str:
        return f"{format_timestamp(self.start)} --> {format_timestamp(self.end)}"


class Metadata(BaseModel):
    engine: str
    model: str
    chunks: int
    chunks_cut_in_silence: int = Field(description="Chunk boundaries snapped to a silence")
    fallback_chunks: int = Field(0, description="Chunks transcribed by the fallback engine")
    processing_seconds: float
    source_format: dict[str, str | int | float | None] = Field(
        description="codec, container, sample_rate, channels of the uploaded file"
    )


class TranscriptionResult(BaseModel):
    filename: str
    language: str | None
    duration: float
    text: str
    segments: list[Segment]
    metadata: Metadata


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class Job(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.queued
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    filename: str
    format: OutputFormat = OutputFormat.json
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

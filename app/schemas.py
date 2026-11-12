"""Public data models returned by the API and CLI."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class OutputFormat(str, Enum):
    json = "json"
    srt = "srt"
    vtt = "vtt"
    txt = "txt"


class Segment(BaseModel):
    id: int
    start: float = Field(description="Seconds from the start of the original file")
    end: float
    text: str
    confidence: float = Field(ge=0.0, le=1.0)


class Metadata(BaseModel):
    engine: str
    model: str
    chunks: int
    chunks_cut_in_silence: int = Field(description="Chunk boundaries snapped to a silence")
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

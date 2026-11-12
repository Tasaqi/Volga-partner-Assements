"""Chunked transcription: arbitrarily long audio with bounded memory and per-chunk progress."""

from pathlib import Path
from typing import Iterator

from transcriber.audio import iter_chunks
from transcriber.engines import Segment, Transcriber, Transcription

# Whisper's prompt window is 224 tokens; the tail of the previous chunk is enough context.
PROMPT_CHARS = 200


def transcribe_chunks(
    path: Path, transcriber: Transcriber, chunk_seconds: float
) -> Iterator[Transcription]:
    """Yield one Transcription per chunk, with timestamps shifted to absolute file time."""
    language = None
    previous_text = ""
    for offset, samples in iter_chunks(path, chunk_seconds):
        result = transcriber.transcribe(
            samples,
            # Continuity across the cut: names, spelling and punctuation style carry over.
            prompt=previous_text[-PROMPT_CHARS:] or None,
            # Detect language once. A chunk of music or silence can otherwise be "detected"
            # as a different language and come back garbled.
            language=language,
        )
        language = language or result.language
        previous_text = result.text or previous_text
        chunk_end = offset + result.duration
        yield Transcription(
            text=result.text,
            language=language,
            duration=result.duration,
            segments=[
                # Whisper occasionally predicts an end past the audio; clamp to the chunk.
                Segment(round(offset + s.start, 3), round(min(offset + s.end, chunk_end), 3), s.text)
                for s in result.segments
            ],
        )


def merge(chunks: Iterator[Transcription]) -> Transcription:
    parts = list(chunks)
    return Transcription(
        text=" ".join(p.text for p in parts if p.text),
        language=next((p.language for p in parts if p.language), None),
        duration=round(sum(p.duration for p in parts), 3),
        segments=[s for p in parts for s in p.segments],
    )

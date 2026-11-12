"""Turn raw engine segments into clean output, and render SRT / VTT / TXT."""

from __future__ import annotations

import math

from app.engines import RawSegment
from app.schemas import OutputFormat, Segment, TranscriptionResult, format_timestamp

# Whisper hallucinates on silence/noise: high no-speech probability AND low
# token confidence together is a strong signal (same thresholds Whisper uses).
NO_SPEECH_THRESHOLD = 0.6
LOGPROB_THRESHOLD = -1.0


def clean_text(text: str) -> str:
    return " ".join(text.split())


def is_hallucination(seg: RawSegment) -> bool:
    return seg.no_speech_prob > NO_SPEECH_THRESHOLD and seg.avg_logprob < LOGPROB_THRESHOLD


def confidence(avg_logprob: float) -> float:
    """Mean per-token probability, clamped to [0, 1]."""
    return min(1.0, max(0.0, math.exp(avg_logprob)))


def finalize_segments(raw: list[RawSegment]) -> list[Segment]:
    """Normalise whitespace, drop empty segments and likely hallucinations, assign ids."""
    out: list[Segment] = []
    for seg in raw:
        text = clean_text(seg.text)
        if not text or is_hallucination(seg):
            continue
        out.append(
            Segment(
                id=len(out) + 1,
                start=round(seg.start, 3),
                end=round(seg.end, 3),
                text=text,
                confidence=round(confidence(seg.avg_logprob), 4),
            )
        )
    return out


def to_srt(segments: list[Segment]) -> str:
    blocks = [
        f"{s.id}\n{format_timestamp(s.start, ',')} --> {format_timestamp(s.end, ',')}\n{s.text}\n"
        for s in segments
    ]
    return "\n".join(blocks)


def to_vtt(segments: list[Segment]) -> str:
    return "WEBVTT\n\n" + "\n".join(f"{s.timeframe}\n{s.text}\n" for s in segments)


def to_txt(segments: list[Segment]) -> str:
    return "".join(f"[{s.timeframe}] {s.text}\n" for s in segments)


MEDIA_TYPES = {
    OutputFormat.json: "application/json",
    OutputFormat.srt: "application/x-subrip; charset=utf-8",
    OutputFormat.vtt: "text/vtt; charset=utf-8",
    OutputFormat.txt: "text/plain; charset=utf-8",
}


def render(result: TranscriptionResult, fmt: OutputFormat) -> str:
    if fmt is OutputFormat.json:
        return result.model_dump_json(indent=2)
    renderers = {OutputFormat.srt: to_srt, OutputFormat.vtt: to_vtt, OutputFormat.txt: to_txt}
    return renderers[fmt](result.segments)

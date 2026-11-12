"""Audio handling: probe, normalise, silence detection, chunk planning, extraction.

Everything after :func:`normalise` works on ONE canonical format
(16 kHz, mono, 16-bit PCM WAV), so no later stage cares what was uploaded.
"""

from __future__ import annotations

import json
import re
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

SAMPLE_RATE = 16_000


class AudioError(Exception):
    """Base class for input problems the caller can fix (bad file, too long...)."""


class UnsupportedAudioError(AudioError):
    """Not decodable audio: corrupt, not a media file, or media without an audio stream."""


class AudioTooLongError(AudioError):
    """Audio exceeds the configured maximum duration."""


@dataclass(frozen=True)
class AudioInfo:
    duration: float
    codec: str | None
    container: str | None
    sample_rate: int | None
    channels: int | None

    def as_metadata(self) -> dict[str, str | int | float | None]:
        return {
            "codec": self.codec,
            "container": self.container,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
        }


@dataclass(frozen=True)
class Chunk:
    """One slice of the normalised file.

    ``owned`` is the range this chunk is responsible for in the final transcript;
    ``cut`` is what is actually transcribed (owned + overlap at hard cuts).
    """

    index: int
    owned_start: float
    owned_end: float
    cut_start: float
    cut_end: float
    end_in_silence: bool  # True if the boundary at owned_end was snapped to a silence


# Uploaded files are untrusted: an HLS playlist could make ffmpeg fetch URLs (SSRF).
LOCAL_ONLY = ["-protocol_whitelist", "file"]


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def probe(path: Path) -> AudioInfo:
    """Inspect a file with ffprobe. Never trusts the extension: the content must
    contain a real audio stream with a positive duration."""
    proc = _run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", "-select_streams", "a", *LOCAL_ONLY, str(path),
        ]
    )
    if proc.returncode != 0:
        raise UnsupportedAudioError("File is not a readable audio or video file")
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams") or []
    if not streams:
        raise UnsupportedAudioError("File contains no audio stream")
    stream, fmt = streams[0], data.get("format") or {}

    duration = _to_float(fmt.get("duration")) or _to_float(stream.get("duration"))
    if not duration or duration <= 0:
        raise UnsupportedAudioError("Audio stream has no valid duration")
    return AudioInfo(
        duration=duration,
        codec=stream.get("codec_name"),
        container=fmt.get("format_name"),
        sample_rate=int(stream["sample_rate"]) if stream.get("sample_rate") else None,
        channels=stream.get("channels"),
    )


def _to_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def normalise(src: Path, dst: Path) -> Path:
    """Decode any input to 16 kHz mono 16-bit PCM WAV (drops video)."""
    proc = _run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-y", *LOCAL_ONLY, "-i", str(src),
            "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(dst),
        ]
    )
    if proc.returncode != 0 or not dst.exists():
        raise UnsupportedAudioError(f"Could not decode audio: {proc.stderr.strip()[-300:]}")
    return dst


def encode_flac(src_wav: Path, dst: Path) -> Path:
    """Lossless FLAC copy of a canonical WAV (for uploading to hosted APIs)."""
    proc = _run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src_wav), "-c:a", "flac", str(dst)])
    if proc.returncode != 0:
        raise AudioError(f"FLAC encoding failed: {proc.stderr.strip()[-300:]}")
    return dst


def wav_duration(path: Path) -> float:
    """Exact duration of a canonical WAV, from its frame count."""
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


_SILENCE_RE = re.compile(r"silence_(start|end): (-?[\d.]+)")


def detect_silences(
    path: Path, noise_db: float = -35.0, min_silence_s: float = 0.5
) -> list[tuple[float, float]]:
    """Return ``[(start, end), ...]`` silent ranges using ffmpeg ``silencedetect``."""
    proc = _run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-i", str(path),
            "-af", f"silencedetect=noise={noise_db}dB:d={min_silence_s}", "-f", "null", "-",
        ]
    )
    silences: list[tuple[float, float]] = []
    start: float | None = None
    for kind, value in _SILENCE_RE.findall(proc.stderr):
        if kind == "start":
            start = max(0.0, float(value))
        elif start is not None:
            silences.append((start, float(value)))
            start = None
    if start is not None:  # silence running to end of file
        silences.append((start, wav_duration(path)))
    return silences


def plan_chunks(
    duration: float,
    silences: list[tuple[float, float]],
    chunk_seconds: float,
    search_s: float = 30.0,
    overlap_s: float = 2.0,
) -> list[Chunk]:
    """Split ``[0, duration]`` into ~``chunk_seconds`` pieces.

    Each boundary moves to the midpoint of the nearest silence within
    ``+/- search_s`` of its target. With no silence nearby we hard-cut and widen
    both neighbouring chunks by ``overlap_s`` so no word is lost at the cut.
    """
    boundaries: list[float] = [0.0]
    snapped: list[bool] = [True]  # file start/end are natural boundaries: no overlap
    while duration - boundaries[-1] > chunk_seconds:
        last = boundaries[-1]
        target = last + chunk_seconds
        # Never move a boundary so far back that a chunk shrinks below half size.
        lo, hi = max(target - search_s, last + chunk_seconds / 2), min(target + search_s, duration)
        mids = [(s + e) / 2 for s, e in silences]
        candidates = [m for m in mids if lo <= m < hi]
        if candidates:
            boundaries.append(min(candidates, key=lambda m: abs(m - target)))
            snapped.append(True)
        else:
            boundaries.append(target)
            snapped.append(False)
    boundaries.append(duration)
    snapped.append(True)

    chunks = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        pad_left = 0.0 if snapped[i] else overlap_s
        pad_right = 0.0 if snapped[i + 1] else overlap_s
        chunks.append(
            Chunk(
                index=i,
                owned_start=start,
                owned_end=end,
                cut_start=max(0.0, start - pad_left),
                cut_end=min(duration, end + pad_right),
                end_in_silence=snapped[i + 1] and i + 1 < len(boundaries) - 1,
            )
        )
    return chunks


def extract(src_wav: Path, start: float, end: float, dst: Path) -> Path:
    """Copy ``[start, end)`` of a canonical WAV into ``dst``.

    Uses the stdlib ``wave`` module: the input is always PCM, so slicing by frame
    index is sample-exact and avoids spawning one ffmpeg per chunk.
    """
    with wave.open(str(src_wav), "rb") as r:
        rate = r.getframerate()
        r.setpos(int(start * rate))
        frames = r.readframes(int((end - start) * rate))
        with wave.open(str(dst), "wb") as w:
            w.setparams(r.getparams())
            w.writeframes(frames)
    return dst

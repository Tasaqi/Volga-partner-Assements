"""Audio input handling: validation, format probing, decoding and encoding."""

import io
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

import av
import numpy as np

# Cheap first gate, checked before the body is written to disk. The real check is
# `probe_audio`, which inspects the content: the extension is only a client claim.
SUPPORTED_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus",
    ".webm", ".mp4", ".aiff", ".aif", ".wma", ".amr",
}  # fmt: skip
WHISPER_SAMPLE_RATE = 16_000
COPY_CHUNK_BYTES = 1024 * 1024


class AudioError(Exception):
    """Base error for input the pipeline cannot accept. Carries an HTTP status."""

    status_code = 400


class UnsupportedFormatError(AudioError):
    status_code = 415


class FileTooLargeError(AudioError):
    status_code = 413


class EmptyFileError(AudioError):
    status_code = 400


class UndecodableAudioError(AudioError):
    """Right extension, but the bytes are corrupt, truncated or not audio at all."""

    status_code = 422


@dataclass(frozen=True)
class AudioInfo:
    container: str
    codec: str
    sample_rate: int
    channels: int
    duration: float | None  # None when the container does not record it (e.g. some streams)


def _check_extension(filename: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Unsupported file type '{suffix or filename}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return suffix


def save_upload(stream: BinaryIO, filename: str, dest_dir: str | Path, max_bytes: int) -> Path:
    """Stream an upload to disk, aborting as soon as it exceeds ``max_bytes``.

    The client-supplied filename is only used for its extension; the on-disk name is
    generated, so a crafted name like ``../../etc/passwd.wav`` cannot escape ``dest_dir``.
    """
    suffix = _check_extension(filename)
    fd, tmp = tempfile.mkstemp(suffix=suffix, dir=dest_dir)
    written = 0
    with open(fd, "wb") as out:
        while chunk := stream.read(COPY_CHUNK_BYTES):
            written += len(chunk)
            if written > max_bytes:
                raise FileTooLargeError(f"File exceeds the {max_bytes // (1024 * 1024)} MB limit")
            out.write(chunk)
    if written == 0:
        raise EmptyFileError("Uploaded file is empty")
    return Path(tmp)


def probe_audio(path: str | Path) -> AudioInfo:
    """Identify the file by its content (FFmpeg's demuxer probing), not its extension."""
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise UnsupportedFormatError("File contains no audio stream")
            stream = container.streams.audio[0]
            duration = round(container.duration / av.time_base, 3) if container.duration else None
            return AudioInfo(
                container=container.format.name,
                codec=stream.codec_context.name,
                sample_rate=stream.codec_context.sample_rate,
                channels=stream.codec_context.channels,
                duration=duration,
            )
    except av.error.FFmpegError as e:
        raise UndecodableAudioError(f"Could not read audio: {e}") from e


def _decode_blocks(path: str | Path) -> Iterator[np.ndarray]:
    """Stream-decode any container/codec as 16 kHz mono float32 PCM, one small block at a time.

    This is the single normalisation point: sample rate, channel count, bit depth and codec
    differences all disappear here, so nothing downstream handles formats. PyAV bundles the
    FFmpeg libraries, so there is no dependency on a system `ffmpeg` binary.
    """
    produced = False
    try:
        with av.open(str(path)) as container:
            resampler = av.AudioResampler(format="s16", layout="mono", rate=WHISPER_SAMPLE_RATE)
            for frame in [*container.decode(audio=0), None]:  # None flushes the resampler
                for out in resampler.resample(frame):
                    produced = True
                    yield out.to_ndarray().ravel().astype(np.float32) / 32768.0
    except av.error.FFmpegError as e:
        raise UndecodableAudioError(f"Could not decode audio: {e}") from e
    if not produced:
        raise UndecodableAudioError("Audio stream contains no samples")


def load_audio(path: str | Path) -> np.ndarray:
    """Decode the whole file into memory. Fine for short clips; use `iter_chunks` for long ones."""
    return np.concatenate(list(_decode_blocks(path)))


def _quietest_point(audio: np.ndarray, lo: int, hi: int, window: int) -> int:
    """Index of the centre of the lowest-energy window in audio[lo:hi]."""
    region = audio[lo:hi]
    n = len(region) // window
    if n == 0:
        return hi
    energy = np.square(region[: n * window].reshape(n, window)).mean(axis=1)
    quietest = n - 1 - int(energy[::-1].argmin())  # ties go to the latest window: longer chunks
    return lo + quietest * window + window // 2


def iter_chunks(
    path: str | Path, chunk_seconds: float, search_seconds: float = 5.0
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield ``(offset_seconds, samples)`` chunks of about ``chunk_seconds`` each.

    Memory is bounded by one chunk, whatever the file length. Each cut is placed at the
    quietest 100 ms within the last ``search_seconds`` before the target length, so chunk
    boundaries land in pauses between words instead of splitting one.
    """
    chunk_len = int(chunk_seconds * WHISPER_SAMPLE_RATE)
    search_len = min(int(search_seconds * WHISPER_SAMPLE_RATE), chunk_len // 2)
    window = WHISPER_SAMPLE_RATE // 10
    offset = 0
    pending: list[np.ndarray] = []
    pending_len = 0
    for block in _decode_blocks(path):
        pending.append(block)
        pending_len += len(block)
        if pending_len < chunk_len:
            continue
        # Concatenate once per chunk, not once per block (which would be quadratic).
        buf = np.concatenate(pending)
        while len(buf) >= chunk_len:
            cut = _quietest_point(buf, chunk_len - search_len, chunk_len, window)
            yield offset / WHISPER_SAMPLE_RATE, buf[:cut]
            offset += cut
            buf = buf[cut:]
        pending, pending_len = [buf], len(buf)
    if pending_len:
        yield offset / WHISPER_SAMPLE_RATE, np.concatenate(pending)


def encode_audio(
    samples: np.ndarray, container: str = "flac", codec: str = "flac", rate: int = WHISPER_SAMPLE_RATE
) -> bytes:
    """Encode mono float32 PCM in memory. FLAC by default: lossless and ~half the size of WAV."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).reshape(1, -1)
    frame = av.AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
    frame.sample_rate = rate
    buf = io.BytesIO()
    with av.open(buf, "w", format=container) as out:
        stream = out.add_stream(codec, rate=rate, layout="mono")
        for packet in [*stream.encode(frame), *stream.encode(None)]:
            out.mux(packet)
    return buf.getvalue()

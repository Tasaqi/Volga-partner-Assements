"""Audio input handling: validation, format probing, decoding and encoding."""

import io
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

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


def load_audio(path: str | Path) -> np.ndarray:
    """Decode any container/codec to the 16 kHz mono float32 PCM that Whisper expects.

    This is the single normalisation point: sample rate, channel count, bit depth and codec
    differences all disappear here, so nothing downstream handles formats. PyAV bundles the
    FFmpeg libraries, so there is no dependency on a system `ffmpeg` binary.
    """
    try:
        with av.open(str(path)) as container:
            resampler = av.AudioResampler(format="s16", layout="mono", rate=WHISPER_SAMPLE_RATE)
            frames = []
            for frame in container.decode(audio=0):
                frames.extend(f.to_ndarray() for f in resampler.resample(frame))
            frames.extend(f.to_ndarray() for f in resampler.resample(None))  # flush
    except av.error.FFmpegError as e:
        raise UndecodableAudioError(f"Could not decode audio: {e}") from e
    if not frames:
        raise UndecodableAudioError("Audio stream contains no samples")
    return np.concatenate(frames, axis=1).ravel().astype(np.float32) / 32768.0


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

"""Audio input handling: validation and safe persistence of uploaded files."""

import tempfile
from pathlib import Path
from typing import BinaryIO

SUPPORTED_EXTENSIONS = {".wav", ".mp3"}
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

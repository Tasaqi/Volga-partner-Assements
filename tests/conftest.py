import io
import math
import struct
import wave

import pytest
from fastapi.testclient import TestClient

from transcriber.api import app
from transcriber.audio import WHISPER_SAMPLE_RATE
from transcriber.engines import Segment, Transcription, get_transcriber


def make_wav(seconds: float = 1.0, rate: int = 16000, channels: int = 1) -> bytes:
    """Generate a 440 Hz tone as an in-memory WAV (mock audio, no fixtures on disk)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        frame = lambda i: struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate)))
        w.writeframes(b"".join(frame(i) * channels for i in range(int(seconds * rate))))
    return buf.getvalue()


class FakeTranscriber:
    """Mock engine: tests never download a model or call the network.

    Returns two segments splitting the chunk in half, like a real engine would.
    """

    name = "fake"

    def __init__(self, language="en"):
        self.calls = []
        self.language = language

    def transcribe(self, samples, *, prompt=None, language=None):
        self.calls.append({"samples": len(samples), "prompt": prompt, "language": language})
        duration = len(samples) / WHISPER_SAMPLE_RATE
        n = len(self.calls)
        return Transcription(
            text=f"hello{n} world{n}",
            language=language or self.language,
            duration=round(duration, 3),
            segments=[
                Segment(0.0, duration / 2, f"hello{n}"),
                Segment(duration / 2, duration, f"world{n}"),
            ],
        )


@pytest.fixture
def fake():
    return FakeTranscriber()


@pytest.fixture
def client(fake):
    app.dependency_overrides[get_transcriber] = lambda: fake
    yield TestClient(app)
    app.dependency_overrides.clear()

import io
import math
import struct
import wave

import pytest
from fastapi.testclient import TestClient

from transcriber.api import app


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


@pytest.fixture
def client():
    return TestClient(app)

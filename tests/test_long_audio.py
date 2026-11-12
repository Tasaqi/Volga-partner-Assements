import json

import numpy as np
import pytest

from transcriber import api
from transcriber.audio import WHISPER_SAMPLE_RATE as SR
from transcriber.audio import encode_audio, iter_chunks
from transcriber.config import Settings
from transcriber.engines import Segment, Transcription
from transcriber.pipeline import merge, transcribe_chunks


def speech_like(seconds: float, pauses=()) -> np.ndarray:
    """Continuous tone ("speech") with silent gaps at the given (start, end) seconds."""
    x = (0.3 * np.sin(2 * np.pi * 440 * np.arange(int(seconds * SR)) / SR)).astype(np.float32)
    for start, end in pauses:
        x[int(start * SR) : int(end * SR)] = 0
    return x


@pytest.fixture
def long_file(tmp_path):
    path = tmp_path / "long.flac"
    path.write_bytes(encode_audio(speech_like(25, pauses=[(8.5, 8.8)])))
    return path


def test_chunks_cover_audio_exactly_once(long_file):
    chunks = list(iter_chunks(long_file, chunk_seconds=10))
    assert len(chunks) == 3
    assert sum(len(c) for _, c in chunks) == 25 * SR  # no gaps, no overlap
    offsets = [o for o, _ in chunks]
    sizes = [len(c) / SR for _, c in chunks]
    assert offsets == pytest.approx(np.cumsum([0, *sizes[:-1]]))


def test_chunk_boundary_lands_in_a_pause(long_file):
    _, first = next(iter_chunks(long_file, chunk_seconds=10))
    assert 8.5 <= len(first) / SR <= 8.8


def test_chunks_never_exceed_target_length(tmp_path):
    path = tmp_path / "nopause.flac"
    path.write_bytes(encode_audio(speech_like(31)))
    assert all(len(c) <= 10 * SR for _, c in iter_chunks(path, chunk_seconds=10))


def test_timestamps_are_shifted_to_absolute_time(long_file, fake):
    result = merge(transcribe_chunks(long_file, fake, chunk_seconds=10))
    starts = [s.start for s in result.segments]
    assert starts == sorted(starts)
    assert result.segments[-1].end == pytest.approx(25.0, abs=0.01)
    assert result.duration == pytest.approx(25.0, abs=0.01)
    assert result.text.startswith("hello1 world1 hello2")


def test_language_and_context_carry_across_chunks(long_file, fake):
    list(transcribe_chunks(long_file, fake, chunk_seconds=10))
    first, second, third = fake.calls
    assert first["language"] is None and first["prompt"] is None  # detect on first chunk only
    assert second["language"] == "en" and second["prompt"] == "hello1 world1"
    assert third["prompt"] == "hello2 world2"


def test_segment_end_is_clamped_to_chunk(long_file):
    class Overshooting:
        name = "x"

        def transcribe(self, samples, **_):
            d = len(samples) / SR
            return Transcription("t", "en", d, [Segment(0.0, d + 5, "t")])  # Whisper overshoot

    segments = merge(transcribe_chunks(long_file, Overshooting(), 10)).segments
    assert all(a.end <= b.start + 1e-6 for a, b in zip(segments, segments[1:]))


def test_stream_returns_ndjson_events(client, long_file, monkeypatch):
    monkeypatch.setattr(api, "settings", Settings(chunk_seconds=10))
    r = client.post("/transcribe?stream=true", files={"file": ("long.flac", long_file.read_bytes())})
    assert r.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in r.text.splitlines()]
    assert events[0]["type"] == "start" and events[0]["source"]["container"] == "flac"
    assert [e["type"] for e in events[1:-1]] == ["segment"] * 6  # 3 chunks x 2 segments
    assert events[-1] == {"type": "done", "language": "en", "duration": pytest.approx(25.0, abs=0.01)}


def test_stream_still_rejects_bad_input_with_status_code(client):
    r = client.post("/transcribe?stream=true", files={"file": ("x.mp3", b"not audio" * 50)})
    assert r.status_code == 422


def test_stream_reports_mid_stream_failure_in_band(client, long_file, monkeypatch, fake):
    from transcriber.engines import TranscriptionError

    calls = {"n": 0}
    real = fake.transcribe

    def flaky(samples, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise TranscriptionError("upstream down")
        return real(samples, **kw)

    monkeypatch.setattr(fake, "transcribe", flaky)
    monkeypatch.setattr(api, "settings", Settings(chunk_seconds=10))
    r = client.post("/transcribe?stream=true", files={"file": ("long.flac", long_file.read_bytes())})
    events = [json.loads(line) for line in r.text.splitlines()]
    assert events[-1] == {"type": "error", "status": 502, "detail": "upstream down"}
    assert sum(e["type"] == "segment" for e in events) == 2  # first chunk was still delivered


def test_decoding_is_streamed_not_buffered(tmp_path, monkeypatch):
    """The first chunk must come back before the rest of the file is decoded."""
    import av

    path = tmp_path / "minute.flac"
    path.write_bytes(encode_audio(speech_like(60)))
    decoded = {"seconds": 0.0}
    real_decode = av.container.InputContainer.decode

    def counting_decode(self, *args, **kwargs):
        for frame in real_decode(self, *args, **kwargs):
            decoded["seconds"] += frame.samples / frame.sample_rate
            yield frame

    monkeypatch.setattr(av.container.InputContainer, "decode", counting_decode)
    next(iter_chunks(path, chunk_seconds=10))
    assert decoded["seconds"] < 12  # ~one chunk plus a frame, not all 60 s

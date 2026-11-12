from types import SimpleNamespace

import numpy as np
import openai
import pytest

from conftest import make_wav
from transcriber.audio import WHISPER_SAMPLE_RATE, load_audio
from transcriber.config import Settings
from transcriber.engines import OpenAIWhisperTranscriber, TranscriptionError, build_transcriber


def test_returns_transcribed_text(client, fake):
    r = client.post("/transcribe", files={"file": ("clip.wav", make_wav())})
    assert r.status_code == 200
    body = r.json()
    assert (body["filename"], body["engine"], body["language"]) == ("clip.wav", "fake", "en")
    assert body["text"] == "hello1 world1"
    assert len(fake.calls) == 1


def test_invalid_upload_never_reaches_engine(client, fake):
    client.post("/transcribe", files={"file": ("notes.txt", b"hi")})
    assert fake.calls == []


def test_load_audio_resamples_to_16k_mono(tmp_path):
    path = tmp_path / "stereo44k.wav"
    path.write_bytes(make_wav(seconds=2, rate=44_100, channels=2))
    audio = load_audio(path)
    assert audio.ndim == 1
    assert abs(len(audio) - 2 * WHISPER_SAMPLE_RATE) < 100
    assert audio.dtype.name == "float32" and abs(audio).max() <= 1.0


SAMPLES = np.zeros(WHISPER_SAMPLE_RATE, dtype=np.float32)


class StubClient:
    def __init__(self, error=None):
        self.kwargs = None
        self.error = error
        self.audio = SimpleNamespace(transcriptions=self)

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return SimpleNamespace(
            text=" hi there ",
            language="english",
            duration=1.23456,
            segments=[SimpleNamespace(start=0.0, end=1.23456, text=" hi there ")],
        )


def test_openai_engine_requests_verbose_json():
    stub = StubClient()
    result = OpenAIWhisperTranscriber("whisper-1", client=stub).transcribe(SAMPLES, prompt="ctx")
    assert result.text == "hi there"
    assert result.language == "en"
    assert stub.kwargs["model"] == "whisper-1"
    assert stub.kwargs["response_format"] == "verbose_json"
    assert stub.kwargs["prompt"] == "ctx" and "language" not in stub.kwargs  # unset = omitted
    assert stub.kwargs["file"][0] == "audio.flac"
    assert result.segments[0].end == 1.235 and result.segments[0].text == "hi there"


def test_openai_engine_maps_upstream_errors():
    engine = OpenAIWhisperTranscriber("whisper-1", client=StubClient(openai.OpenAIError("boom")))
    with pytest.raises(TranscriptionError):
        engine.transcribe(SAMPLES)


def test_unknown_engine_fails_fast():
    with pytest.raises(ValueError, match="Unknown TRANSCRIBER_ENGINE"):
        build_transcriber(Settings(engine="nope"))

from types import SimpleNamespace

import openai
import pytest

from conftest import make_wav
from transcriber.audio import WHISPER_SAMPLE_RATE, load_audio
from transcriber.config import Settings
from transcriber.engines import OpenAIWhisperTranscriber, TranscriptionError, build_transcriber


def test_returns_transcribed_text(client, fake):
    r = client.post("/transcribe", files={"file": ("clip.wav", make_wav())})
    assert r.status_code == 200
    assert r.json() == {"filename": "clip.wav", "engine": "fake", "language": "en", "text": "hello world"}
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


class StubClient:
    def __init__(self, error=None):
        self.kwargs = None
        self.error = error
        self.audio = SimpleNamespace(transcriptions=self)

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return SimpleNamespace(text=" hi there ", language="english")


def test_openai_engine_requests_verbose_json(tmp_path):
    path = tmp_path / "clip.wav"
    path.write_bytes(make_wav())
    stub = StubClient()
    result = OpenAIWhisperTranscriber("whisper-1", client=stub).transcribe(path)
    assert result.text == "hi there"
    assert stub.kwargs["model"] == "whisper-1"
    assert stub.kwargs["response_format"] == "verbose_json"


def test_openai_engine_maps_upstream_errors(tmp_path):
    path = tmp_path / "clip.wav"
    path.write_bytes(make_wav())
    engine = OpenAIWhisperTranscriber("whisper-1", client=StubClient(openai.OpenAIError("boom")))
    with pytest.raises(TranscriptionError):
        engine.transcribe(path)


def test_unknown_engine_fails_fast():
    with pytest.raises(ValueError, match="Unknown TRANSCRIBER_ENGINE"):
        build_transcriber(Settings(engine="nope"))

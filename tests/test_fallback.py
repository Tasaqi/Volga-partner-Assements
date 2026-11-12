import numpy as np
import pytest

from transcriber import engines
from transcriber.config import Settings
from transcriber.engines import (
    FallbackTranscriber,
    LocalWhisperTranscriber,
    Transcription,
    TranscriptionError,
    build_transcriber,
)

SAMPLES = np.zeros(16_000, dtype=np.float32)


class Engine:
    def __init__(self, name, fail=False):
        self.name, self.fail, self.calls = name, fail, []

    def transcribe(self, samples, *, prompt=None, language=None):
        self.calls.append({"prompt": prompt, "language": language})
        if self.fail:
            raise TranscriptionError(f"{self.name} down")
        return Transcription(f"from {self.name}", "en", 1.0, [])


def test_primary_success_never_touches_fallback():
    primary, fallback = Engine("openai"), Engine("local")
    result = FallbackTranscriber(primary, fallback).transcribe(SAMPLES)
    assert result.text == "from openai" and fallback.calls == []


def test_primary_failure_falls_back_with_same_context():
    primary, fallback = Engine("openai", fail=True), Engine("local")
    engine = FallbackTranscriber(primary, fallback)
    result = engine.transcribe(SAMPLES, prompt="ctx", language="en")
    assert result.text == "from local"
    assert fallback.calls == [{"prompt": "ctx", "language": "en"}]
    assert engine.name == "openai+local"


def test_both_failing_reports_both_errors():
    engine = FallbackTranscriber(Engine("openai", fail=True), Engine("local", fail=True))
    with pytest.raises(TranscriptionError, match="openai: openai down; local: local down"):
        engine.transcribe(SAMPLES)


def test_circuit_breaker_skips_primary_during_cooldown(monkeypatch):
    now = {"t": 1000.0}
    monkeypatch.setattr(engines.time, "monotonic", lambda: now["t"])
    primary, fallback = Engine("openai", fail=True), Engine("local")
    engine = FallbackTranscriber(primary, fallback, cooldown=60)

    engine.transcribe(SAMPLES)  # primary fails, breaker opens
    engine.transcribe(SAMPLES)  # within cooldown: primary not even tried
    assert len(primary.calls) == 1 and len(fallback.calls) == 2

    now["t"] += 61
    primary.fail = False
    assert engine.transcribe(SAMPLES).text == "from openai"  # recovered after cooldown


def test_local_runtime_errors_become_transcription_errors():
    class OOMModel:
        def transcribe(self, *a, **k):
            raise RuntimeError("CUDA out of memory")

    local = object.__new__(LocalWhisperTranscriber)  # skip loading a real model
    local._model, local._fp16, local._lock = OOMModel(), False, engines.threading.Lock()
    with pytest.raises(TranscriptionError) as exc:
        local.transcribe(SAMPLES)
    assert exc.value.status_code == 500


@pytest.fixture
def stub_engines(monkeypatch):
    monkeypatch.setattr(engines, "LocalWhisperTranscriber", lambda model: Engine("local"))
    monkeypatch.setattr(engines, "OpenAIWhisperTranscriber", lambda model: Engine("openai"))


@pytest.mark.parametrize(
    "engine,fallback,expected",
    [
        ("openai", "local", "openai+local"),  # default: hosted degrades to open source
        ("local", "openai", "local+openai"),
        ("openai", "none", "openai"),
        ("local", "local", "local"),  # fallback to itself is pointless: no wrapper
    ],
)
def test_build_wires_fallback_from_config(stub_engines, engine, fallback, expected):
    assert build_transcriber(Settings(engine=engine, fallback_engine=fallback)).name == expected


def test_unknown_fallback_fails_at_startup(stub_engines):
    with pytest.raises(ValueError, match="Unknown transcription engine 'nope'"):
        build_transcriber(Settings(engine="openai", fallback_engine="nope"))


def test_api_reports_engine_chain_and_still_answers(client, monkeypatch):
    from transcriber.api import app
    from transcriber.engines import get_transcriber
    from conftest import make_wav

    chain = FallbackTranscriber(Engine("openai", fail=True), Engine("local"))
    app.dependency_overrides[get_transcriber] = lambda: chain
    r = client.post("/transcribe", files={"file": ("clip.wav", make_wav())})
    assert r.status_code == 200
    assert r.json()["engine"] == "openai+local" and r.json()["text"] == "from local"

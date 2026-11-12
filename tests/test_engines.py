"""Engines and fallback. No network and no model download: OpenAI is a fake client."""

import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from app import engines
from app.api import create_app
from app.engines import (
    EngineResult,
    FallbackEngine,
    FasterWhisperEngine,
    MockEngine,
    OpenAIEngine,
    RawSegment,
    TranscriptionError,
    build_engine,
)
from app.pipeline import transcribe_file


class Stub:
    """Engine that returns one segment, or raises when ``fail`` is set."""

    def __init__(self, name: str, fail: bool = False):
        self.name, self.model, self.fail, self.calls = name, f"{name}-model", fail, []

    def is_loaded(self) -> bool:
        return True

    def transcribe(self, wav_path, language=None) -> EngineResult:
        self.calls.append(language)
        if self.fail:
            raise TranscriptionError(f"{self.name} down")
        return EngineResult([RawSegment(0, 1, f"from {self.name}")], "en")


def test_primary_success_never_touches_fallback():
    primary, fallback = Stub("openai"), Stub("faster-whisper")
    result = FallbackEngine(primary, fallback).transcribe(Path("x.wav"))
    assert result.segments[0].text == "from openai" and not result.fallback
    assert fallback.calls == []


def test_primary_failure_falls_back_with_same_language():
    primary, fallback = Stub("openai", fail=True), Stub("faster-whisper")
    engine = FallbackEngine(primary, fallback)
    result = engine.transcribe(Path("x.wav"), language="de")
    assert result.segments[0].text == "from faster-whisper" and result.fallback
    assert fallback.calls == ["de"]
    assert (engine.name, engine.model) == ("openai+faster-whisper", "openai-model+faster-whisper-model")


def test_both_failing_reports_both_errors():
    engine = FallbackEngine(Stub("openai", fail=True), Stub("faster-whisper", fail=True))
    with pytest.raises(TranscriptionError, match="openai: openai down; faster-whisper: faster-whisper down"):
        engine.transcribe(Path("x.wav"))


def test_circuit_breaker_skips_primary_during_cooldown(monkeypatch):
    now = {"t": 1000.0}
    monkeypatch.setattr(engines.time, "monotonic", lambda: now["t"])
    primary, fallback = Stub("openai", fail=True), Stub("faster-whisper")
    engine = FallbackEngine(primary, fallback, cooldown_s=60)

    engine.transcribe(Path("x.wav"))  # primary fails, breaker opens
    engine.transcribe(Path("x.wav"))  # inside cooldown: primary not even tried
    assert len(primary.calls) == 1 and len(fallback.calls) == 2

    now["t"] += 61
    primary.fail = False
    assert not engine.transcribe(Path("x.wav")).fallback  # primary used again after cooldown


def fake_openai(response=None, error: Exception | None = None):
    def create(**kwargs):
        assert kwargs["response_format"] == "verbose_json"
        if error:
            raise error
        return response

    return SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))


def test_openai_engine_maps_segments_and_language(make_gapped):
    wav = make_gapped(seconds=3)
    seg = SimpleNamespace(start=0.0, end=2.0, text=" hi ", avg_logprob=-0.2, no_speech_prob=0.1)
    engine = OpenAIEngine(client=fake_openai(SimpleNamespace(segments=[seg], language="english")))
    result = engine.transcribe(wav)
    assert result.language == "en"  # API says "english"; normalised to ISO code
    assert result.segments == [RawSegment(0.0, 2.0, " hi ", -0.2, 0.1)]
    assert not wav.with_suffix(".flac").exists()  # temp upload file cleaned up


def test_openai_errors_become_transcription_errors(make_gapped):
    error = openai.APIConnectionError(request=SimpleNamespace())  # type: ignore[arg-type]
    engine = OpenAIEngine(client=fake_openai(error=error))
    with pytest.raises(TranscriptionError) as exc:
        engine.transcribe(make_gapped(seconds=3))
    assert exc.value.status_code == 502


def test_local_runtime_errors_become_transcription_errors(make_gapped, monkeypatch):
    class OOM:
        def transcribe(self, *args, **kwargs):
            raise RuntimeError("out of memory")

    monkeypatch.setattr(FasterWhisperEngine, "_whisper", lambda self: OOM())
    with pytest.raises(TranscriptionError) as exc:
        FasterWhisperEngine().transcribe(make_gapped(seconds=3))
    assert exc.value.status_code == 500


@pytest.fixture
def offline_engines(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(FasterWhisperEngine, "_whisper", lambda self: None)  # no model download


@pytest.mark.parametrize(
    "engine,fallback,expected",
    [
        ("openai", "faster-whisper", "openai+faster-whisper"),  # default chain
        ("faster-whisper", "openai", "faster-whisper+openai"),
        ("openai", "none", "openai"),
        ("faster-whisper", "faster-whisper", "faster-whisper"),  # fallback to itself: no wrapper
        ("mock", "faster-whisper", "mock"),  # offline demo engine is never wrapped
    ],
)
def test_build_engine_wires_fallback(offline_engines, engine, fallback, expected):
    assert build_engine(_cfg(engine, fallback)).name == expected


def test_missing_openai_key_degrades_to_fallback(offline_engines, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    assert build_engine(_cfg("openai", "faster-whisper")).name == "faster-whisper"
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_engine(_cfg("openai", "none"))


def test_unknown_fallback_fails_at_startup(offline_engines):
    with pytest.raises(ValueError, match="Unknown engine 'nope'"):
        build_engine(_cfg("openai", "nope"))


def _cfg(engine: str, fallback: str):
    from app.config import Settings

    return Settings(engine=engine, fallback_engine=fallback)


class FailingMock(MockEngine):
    name = "openai"

    def transcribe(self, wav_path, language=None):
        raise TranscriptionError("openai down")


def test_pipeline_records_fallback_chunks(make_gapped, cfg):
    cfg = replace(cfg, chunk_seconds=10.0, silence_search_s=3.0)
    result = transcribe_file(make_gapped(seconds=30), cfg, FallbackEngine(FailingMock(), MockEngine()))
    assert result.metadata.engine == "openai+mock"
    assert result.metadata.fallback_chunks == result.metadata.chunks >= 3
    assert result.segments  # transcript still produced


def test_api_returns_502_when_every_engine_fails(cfg, make_tone):
    engine = FallbackEngine(FailingMock(), FailingMock())
    with TestClient(create_app(cfg, engine)) as client, make_tone("a.wav", 2.0).open("rb") as f:
        r = client.post("/transcriptions", files={"file": ("a.wav", f)})
    assert r.status_code == 502
    assert "openai down" in r.json()["detail"]


def test_request_error_falls_back_without_opening_breaker():
    class Rejects(Stub):
        def transcribe(self, wav_path, language=None):
            self.calls.append(language)
            raise TranscriptionError("400: bad input", outage=False)

    primary, fallback = Rejects("openai"), Stub("faster-whisper")
    engine = FallbackEngine(primary, fallback)
    assert engine.transcribe(Path("x.wav")).fallback  # this request still gets an answer
    engine.transcribe(Path("x.wav"))
    assert len(primary.calls) == 2  # breaker stayed closed: other users keep using OpenAI


@pytest.mark.parametrize(
    "error,outage",
    [
        (openai.BadRequestError("bad", response=httpx.Response(400, request=httpx.Request("POST", "http://x")), body=None), False),
        (openai.InternalServerError("boom", response=httpx.Response(500, request=httpx.Request("POST", "http://x")), body=None), True),
        (openai.RateLimitError("slow down", response=httpx.Response(429, request=httpx.Request("POST", "http://x")), body=None), True),
        (openai.APIConnectionError(request=httpx.Request("POST", "http://x")), True),
    ],
)
def test_openai_errors_are_classified_as_outage_or_request(make_gapped, error, outage):
    with pytest.raises(TranscriptionError) as exc:
        OpenAIEngine(client=fake_openai(error=error)).transcribe(make_gapped(seconds=3))
    assert exc.value.outage is outage


def test_missing_key_with_openai_as_fallback_degrades_too(offline_engines, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    assert build_engine(_cfg("faster-whisper", "openai")).name == "faster-whisper"


def test_model_loads_once_when_threads_race(monkeypatch):
    import faster_whisper

    created = []

    class SlowModel:
        def __init__(self, *args, **kwargs):
            time.sleep(0.05)  # widen the race window
            created.append(1)

    monkeypatch.setattr(faster_whisper, "WhisperModel", SlowModel)
    engines._load_whisper.cache_clear()
    try:
        engine = FasterWhisperEngine("race-test")
        assert not engine.is_loaded()
        threads = [threading.Thread(target=engine._whisper) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(created) == 1 and engine.is_loaded()
    finally:
        engines._load_whisper.cache_clear()

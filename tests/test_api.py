import asyncio
import tempfile
import threading
import time
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.api import RequestGuard, create_app
from app.engines import MockEngine


@pytest.fixture
def client(cfg, engine):
    app = create_app(replace(cfg, sync_max_duration_s=10.0, chunk_seconds=10.0,
                             silence_search_s=3.0, max_upload_mb=1), engine)
    with TestClient(app) as c:
        yield c


def post(client, path, **params):
    with path.open("rb") as f:
        return client.post("/transcriptions", files={"file": (path.name, f)}, params=params)


def test_health(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "engine": "mock", "model": "mock", "model_loaded": True}


def test_short_file_returns_json(client, make_tone):
    r = post(client, make_tone("hello.mp3", 5.0, "-c:a", "libmp3lame"), language="en")
    assert r.status_code == 200
    body = r.json()
    assert body["filename"] == "hello.mp3"
    assert body["language"] == "en"
    assert body["segments"][0] == {
        "id": 1, "start": 0.0, "end": 2.5, "text": "mock segment 1", "confidence": 0.9048,
        "timeframe": "00:00:00.000 --> 00:00:02.500",
    }
    assert set(body["metadata"]) == {
        "engine", "model", "chunks", "chunks_cut_in_silence", "fallback_chunks",
        "processing_seconds", "source_format",
    }
    assert body["metadata"]["source_format"]["container"] == "mp3"


def test_format_srt(client, make_tone):
    r = post(client, make_tone("hello.wav", 3.0), format="srt")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-subrip")
    assert r.text.startswith("1\n00:00:00,000 --> 00:00:02,500\nmock segment 1\n")


def test_long_file_becomes_job_and_completes(client, make_gapped):
    r = post(client, make_gapped(seconds=30), format="vtt")
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    assert r.headers["location"] == f"/transcriptions/{job_id}"

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status = client.get(f"/transcriptions/{job_id}")
        assert status.status_code == 200
        if status.headers["content-type"].startswith("text/vtt"):
            break
        body = status.json()
        assert body["status"] in {"queued", "running"}, body
        assert 0.0 <= body["progress"] < 1.0
        time.sleep(0.05)
    else:
        pytest.fail("job did not complete")
    assert status.text.startswith("WEBVTT")


def test_non_audio_returns_415(client, tmp_path):
    fake = tmp_path / "song.mp3"
    fake.write_text("not audio at all\n" * 20)
    r = post(client, fake)
    assert r.status_code == 415
    assert "detail" in r.json()


def test_empty_file_returns_400(client, tmp_path):
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    assert post(client, empty).status_code == 400


def test_oversized_upload_returns_413(client, tmp_path):
    big = tmp_path / "big.wav"
    big.write_bytes(b"\0" * (1024 * 1024 + 1))
    assert post(client, big).status_code == 413


def test_too_long_audio_returns_413(cfg, engine, make_tone):
    app = create_app(replace(cfg, max_duration_s=2.0), engine)
    with TestClient(app) as c:
        assert post(c, make_tone("long.wav", 3.0)).status_code == 413


def test_unknown_job_returns_404(client):
    assert client.get("/transcriptions/does-not-exist").status_code == 404


def test_api_key_required_when_configured(cfg, engine, make_tone):
    app = create_app(replace(cfg, api_key="s3cret"), engine)
    path = make_tone("hello.mp3", 2.0, "-c:a", "libmp3lame")
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200  # probes stay open
        assert post(c, path).status_code == 401
        assert c.get("/transcriptions/nope", headers={"X-API-Key": "wrong"}).status_code == 401
        with path.open("rb") as f:
            r = c.post("/transcriptions", files={"file": (path.name, f)},
                       headers={"X-API-Key": "s3cret"})
        assert r.status_code == 200


def test_language_names_are_accepted_and_unknown_ones_rejected(client, make_tone):
    path = make_tone("hello.wav", 2.0)
    r = post(client, path, language="English")
    assert r.status_code == 200 and r.json()["language"] == "en"
    r = post(client, path, language="klingon")
    assert r.status_code == 422  # used to reach the engines and crash with a raw 500
    assert "klingon" in r.json()["detail"]


def _call_guard(guard, headers, chunks=()):
    """Drive RequestGuard as a raw ASGI app; returns the response status."""
    sent, body = [], list(chunks)

    async def receive():
        return {"type": "http.request", "body": body.pop(0) if body else b"", "more_body": bool(body)}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "POST", "path": "/transcriptions", "headers": headers}
    asyncio.run(guard(scope, receive, send))
    return sent[0]["status"]


async def _must_not_run(scope, receive, send):
    raise AssertionError("request reached the app; the body should not have been read")


def test_guard_rejects_on_headers_before_reading_the_body():
    guard = RequestGuard(_must_not_run, api_key="k", max_body_bytes=10)
    assert _call_guard(guard, [(b"content-length", b"5")]) == 401  # no key
    assert _call_guard(guard, [(b"x-api-key", b"k"), (b"content-length", b"999999")]) == 413


def test_guard_counts_bytes_when_there_is_no_content_length():
    async def reads_everything(scope, receive, send):
        while (await receive())["more_body"]:
            pass
        raise AssertionError("an over-limit body was read to the end")

    guard = RequestGuard(reads_everything, api_key="", max_body_bytes=10)
    assert _call_guard(guard, [], chunks=[b"123456", b"123456", b"123456"]) == 413


def test_chunked_upload_over_the_limit_gets_413_through_fastapi(client):
    # No Content-Length (chunked transfer): the limit must hold on the bytes actually sent.
    body = (b'--b\r\nContent-Disposition: form-data; name="file"; filename="big.wav"\r\n\r\n'
            + b"\0" * (3 * 1024 * 1024) + b"\r\n--b--\r\n")
    pieces = (body[i:i + 65536] for i in range(0, len(body), 65536))
    r = client.post("/transcriptions", content=pieces,
                    headers={"content-type": "multipart/form-data; boundary=b"})
    assert r.status_code == 413


def test_jobs_cancelled_at_shutdown_delete_their_uploads(cfg, make_gapped, monkeypatch, tmp_path):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(uploads))
    release = threading.Event()

    class Blocking(MockEngine):
        def transcribe(self, wav_path, language=None):
            release.wait(10)
            return super().transcribe(wav_path, language)

    app = create_app(replace(cfg, sync_max_duration_s=10.0, job_workers=1), Blocking())
    audio = make_gapped(seconds=30)
    try:
        with TestClient(app) as c:
            assert post(c, audio).status_code == 202  # runs (blocked in the engine)
            assert post(c, audio).status_code == 202  # waits in the queue
            assert len(list(uploads.glob("tp-upload-*"))) == 2
        # shutdown cancelled the queued job: its upload is gone, the running one remains
        assert len(list(uploads.glob("tp-upload-*"))) == 1
    finally:
        release.set()

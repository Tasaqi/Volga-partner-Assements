import time
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.api import create_app


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
        "id": 1, "start": 0.0, "end": 2.5, "text": "mock segment 1", "confidence": 0.9048
    }
    assert set(body["metadata"]) == {
        "engine", "model", "chunks", "chunks_cut_in_silence", "processing_seconds", "source_format"
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

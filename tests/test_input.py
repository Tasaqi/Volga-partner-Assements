import io

import pytest

from conftest import make_wav
from transcriber import api
from transcriber.audio import save_upload
from transcriber.config import Settings


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


@pytest.mark.parametrize("name", ["clip.wav", "CLIP.MP3"])
def test_accepts_wav_and_mp3(client, name):
    r = client.post("/transcribe", files={"file": (name, make_wav())})
    assert r.status_code == 200
    assert r.json()["filename"] == name


@pytest.mark.parametrize(
    "name,data,status",
    [
        ("notes.txt", b"hello", 415),
        ("noextension", b"hello", 415),
        ("empty.wav", b"", 400),
    ],
)
def test_rejects_bad_upload(client, name, data, status):
    r = client.post("/transcribe", files={"file": (name, data)})
    assert r.status_code == status
    assert r.json()["detail"]


def test_rejects_oversized_upload(client, monkeypatch):
    monkeypatch.setattr(api, "settings", Settings(max_upload_mb=1))
    r = client.post("/transcribe", files={"file": ("clip.wav", make_wav(seconds=40))})  # ~1.3 MB
    assert r.status_code == 413


def test_client_filename_cannot_escape_dest_dir(tmp_path):
    path = save_upload(io.BytesIO(b"data"), "../../evil.wav", tmp_path, max_bytes=100)
    assert path.parent == tmp_path

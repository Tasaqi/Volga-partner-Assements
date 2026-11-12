import numpy as np
import pytest

from conftest import load_audio
from transcriber.audio import WHISPER_SAMPLE_RATE, encode_audio



def tone(rate: int, seconds: float = 2.0) -> np.ndarray:
    return (0.3 * np.sin(2 * np.pi * 440 * np.arange(int(seconds * rate)) / rate)).astype(np.float32)


TONE = tone(48_000)

# (extension, container, codec): lossy and lossless, PCM and compressed, audio and video containers
FORMATS = [
    ("mp3", "mp3", "libmp3lame"),
    ("flac", "flac", "flac"),
    ("m4a", "mp4", "aac"),
    ("opus", "ogg", "libopus"),
    ("webm", "webm", "libopus"),
    ("aiff", "aiff", "pcm_s16be"),
]


@pytest.mark.parametrize("ext,container,codec", FORMATS)
def test_every_format_normalises_to_16k_mono(tmp_path, ext, container, codec):
    path = tmp_path / f"tone.{ext}"
    rate = 48_000 if codec == "libopus" else 44_100  # Opus does not support 44.1 kHz
    path.write_bytes(encode_audio(tone(rate), container, codec, rate=rate))
    audio = load_audio(path)
    assert audio.ndim == 1 and audio.dtype == np.float32
    assert abs(len(audio) / WHISPER_SAMPLE_RATE - 2.0) < 0.1  # codec padding is a few ms


@pytest.mark.parametrize("ext,container,codec", FORMATS)
def test_api_accepts_format_and_reports_source(client, ext, container, codec):
    data = encode_audio(TONE, container, codec, rate=48_000)  # also exercises resampling
    r = client.post("/transcribe", files={"file": (f"tone.{ext}", data)})
    assert r.status_code == 200, r.text
    assert r.json()["source"]["channels"] == 1


@pytest.mark.parametrize(
    "name,data",
    [
        ("fake.mp3", b"this is not audio at all" * 100),  # right extension, wrong bytes
        ("truncated.flac", encode_audio(tone(16_000))[:40]),  # header only
    ],
)
def test_undecodable_content_is_422_and_skips_engine(client, fake, name, data):
    r = client.post("/transcribe", files={"file": (name, data)})
    assert r.status_code == 422, r.text
    assert fake.calls == []


def test_extension_is_not_trusted(client, fake):
    """An MP3 uploaded as .wav is detected by content and still transcribed."""
    r = client.post("/transcribe", files={"file": ("mislabelled.wav", encode_audio(TONE, "mp3", "libmp3lame", 48_000))})
    assert r.status_code == 200
    assert r.json()["source"]["container"] == "mp3"

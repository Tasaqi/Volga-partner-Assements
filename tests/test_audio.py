import subprocess
import wave
from pathlib import Path

import pytest

from app import audio
from app.audio import UnsupportedAudioError
from tests.conftest import ffmpeg

FORMATS = {
    "tone.wav": ("-c:a", "pcm_s16le"),
    "tone.mp3": ("-c:a", "libmp3lame"),
    "tone.ogg": ("-c:a", "libopus"),
    "tone.m4a": ("-c:a", "aac"),
    "tone.flac": ("-c:a", "flac"),
}


@pytest.mark.parametrize("name", FORMATS)
def test_every_format_normalises_to_canonical_wav(make_tone, tmp_path: Path, name: str):
    src = make_tone(name, 2.0, *FORMATS[name])
    info = audio.probe(src)
    assert info.duration == pytest.approx(2.0, abs=0.1)
    assert info.channels == 2

    out = audio.normalise(src, tmp_path / "out.wav")
    with wave.open(str(out), "rb") as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
    assert audio.wav_duration(out) == pytest.approx(2.0, abs=0.1)


def test_text_file_renamed_mp3_is_rejected(tmp_path: Path):
    fake = tmp_path / "notes.mp3"
    fake.write_text("this is definitely not audio\n" * 50)
    with pytest.raises(UnsupportedAudioError):
        audio.probe(fake)


def test_video_without_audio_is_rejected(tmp_path: Path):
    video = tmp_path / "silent.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc=duration=2:size=64x64:rate=10", "-c:v", "mpeg4", str(video))
    with pytest.raises(UnsupportedAudioError, match="no audio stream"):
        audio.probe(video)


def test_audio_track_inside_video_is_used(tmp_path: Path):
    video = tmp_path / "talk.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc=duration=2:size=64x64:rate=10",
           "-f", "lavfi", "-i", "sine=duration=2", "-c:v", "mpeg4", "-c:a", "aac", str(video))
    assert audio.probe(video).codec == "aac"
    out = audio.normalise(video, tmp_path / "out.wav")
    assert audio.wav_duration(out) == pytest.approx(2.0, abs=0.1)


def test_silence_detection_finds_gaps(make_gapped):
    wav = make_gapped(seconds=30, period=10, on=8)
    silences = audio.detect_silences(wav, noise_db=-35, min_silence_s=0.5)
    assert len(silences) == 3
    for (start, end), expected_start in zip(silences, (8, 18, 28)):
        assert start == pytest.approx(expected_start, abs=0.1)
        assert end - start == pytest.approx(2.0, abs=0.1)


def test_extract_is_sample_exact(make_gapped, tmp_path: Path):
    wav = make_gapped(seconds=10)
    part = audio.extract(wav, 2.0, 5.5, tmp_path / "part.wav")
    assert audio.wav_duration(part) == pytest.approx(3.5, abs=1e-4)


def test_normalise_stops_at_max_seconds(make_tone, tmp_path: Path):
    out = audio.normalise(make_tone("long.wav", 5.0), tmp_path / "out.wav", max_seconds=2.0)
    assert audio.wav_duration(out) == pytest.approx(2.0, abs=0.05)


def test_ffmpeg_timeout_becomes_unsupported_audio(monkeypatch, tmp_path: Path):
    def hang(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(audio.subprocess, "run", hang)
    with pytest.raises(UnsupportedAudioError, match="timed out"):
        audio.probe(tmp_path / "upload")


def test_ffmpeg_and_ffprobe_only_read_local_files(monkeypatch, tmp_path: Path):
    # An uploaded playlist must not make ffmpeg fetch URLs (SSRF): both tools get the whitelist.
    seen = []
    real_run = audio._run
    monkeypatch.setattr(audio, "_run", lambda cmd: seen.append(cmd) or real_run(cmd))
    playlist = tmp_path / "upload"
    playlist.write_text("ffconcat version 1.0\nfile http://127.0.0.1:9/secret.wav\n")
    with pytest.raises(UnsupportedAudioError):
        audio.probe(playlist)
    with pytest.raises(UnsupportedAudioError):
        audio.normalise(playlist, tmp_path / "out.wav")
    assert [cmd[0] for cmd in seen] == ["ffprobe", "ffmpeg"]
    assert all("-protocol_whitelist" in cmd and "file" in cmd for cmd in seen)

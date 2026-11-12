"""Fixtures. Audio is generated on the fly with ffmpeg lavfi: no binary files in the repo."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from app.config import Settings
from app.engines import MockEngine

MakeAudio = Callable[..., Path]


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *args], check=True)


@pytest.fixture
def make_tone(tmp_path: Path) -> MakeAudio:
    """``make_tone("a.mp3", seconds=3, "-c:a", "libmp3lame")`` -> path to a sine tone."""

    def _make(name: str, seconds: float = 3.0, *encode: str) -> Path:
        out = tmp_path / name
        ffmpeg("-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={seconds}",
               "-ac", "2", *encode, str(out))
        return out

    return _make


@pytest.fixture
def make_gapped(tmp_path: Path) -> MakeAudio:
    """Tone with periodic silences: ``on`` seconds of tone, then ``period - on`` of silence."""

    def _make(name: str = "gapped.wav", seconds: float = 30.0, period: float = 10.0,
              on: float = 8.0) -> Path:
        out = tmp_path / name
        expr = f"0.5*sin(2*PI*440*t)*lt(mod(t,{period}),{on})"
        ffmpeg("-f", "lavfi", "-i", f"aevalsrc='{expr}':s=16000:d={seconds}", str(out))
        return out

    return _make


@pytest.fixture
def cfg() -> Settings:
    return Settings(engine="mock")


@pytest.fixture
def engine() -> MockEngine:
    return MockEngine()

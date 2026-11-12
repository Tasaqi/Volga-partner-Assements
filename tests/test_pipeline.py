import threading
from dataclasses import replace

import pytest

from app.audio import AudioTooLongError
from app.engines import EngineResult, MockEngine, TranscriptionError
from app.pipeline import inspect, transcribe_file
from tests.conftest import ffmpeg

LONG = dict(chunk_seconds=10.0, silence_search_s=3.0, chunk_overlap_s=1.0)


def test_short_file_single_chunk(make_tone, cfg, engine):
    src = make_tone("short.mp3", 6.0, "-c:a", "libmp3lame")
    result = transcribe_file(src, cfg, engine, language="en")
    assert result.metadata.chunks == 1
    assert result.metadata.source_format["codec"] == "mp3"
    assert result.metadata.source_format["sample_rate"] == 44100
    assert result.duration == pytest.approx(6.0, abs=0.1)
    assert len(result.segments) == 3
    assert result.text.startswith("mock segment 1")


def test_progress_callback_reports_each_chunk(make_gapped, cfg, engine):
    wav = make_gapped(seconds=30)
    seen: list[float] = []
    transcribe_file(wav, replace(cfg, **LONG), engine, progress=seen.append)
    assert len(seen) >= 3
    assert seen == sorted(seen) and seen[-1] == 1.0


def test_max_duration_is_enforced(make_tone, cfg, engine):
    src = make_tone("long.wav", 5.0)
    with pytest.raises(AudioTooLongError):
        transcribe_file(src, replace(cfg, max_duration_s=4.0), engine)



def lying_mp3(tmp_path):
    """Probes as ~8 s but decodes to ~125 s: a 320 kb/s head on an 8 kb/s tail with no
    Xing header, so ffprobe estimates the whole length from the head's bitrate."""
    head, tail, out = (tmp_path / n for n in ("head.mp3", "tail.mp3", "lying.mp3"))
    ffmpeg("-f", "lavfi", "-i", "sine=f=440:d=5", "-ac", "1", "-ar", "44100",
           "-c:a", "libmp3lame", "-b:a", "320k", "-write_xing", "0", str(head))
    ffmpeg("-f", "lavfi", "-i", "anoisesrc=d=120:a=0.1", "-ac", "1", "-ar", "8000",
           "-c:a", "libmp3lame", "-b:a", "8k", "-write_xing", "0", "-id3v2_version", "0", str(tail))
    out.write_bytes(head.read_bytes() + tail.read_bytes())
    return out


def test_faked_duration_header_is_caught_after_decoding(tmp_path, cfg, engine):
    src = lying_mp3(tmp_path)
    limited = replace(cfg, max_duration_s=60.0)
    assert inspect(src, limited).duration < 60  # the header check alone is fooled
    with pytest.raises(AudioTooLongError, match="header"):
        transcribe_file(src, limited, engine)

@pytest.mark.parametrize("workers", [1, 3])
def test_long_file_has_ordered_absolute_timestamps(make_gapped, cfg, engine, workers):
    wav = make_gapped(seconds=30)
    result = transcribe_file(wav, replace(cfg, chunk_workers=workers, **LONG), engine)

    meta = result.metadata
    assert meta.chunks >= 3
    assert meta.chunks_cut_in_silence >= 2  # boundaries landed in the 8-10 s / 18-20 s gaps
    starts = [s.start for s in result.segments]
    assert starts == sorted(starts)
    assert starts[0] == 0.0
    assert result.segments[-1].end <= result.duration + 1e-6
    assert max(s.end for s in result.segments) > 25  # absolute, not chunk-relative
    assert [s.id for s in result.segments] == list(range(1, len(starts) + 1))


def test_chunk_files_are_removed_as_they_finish(make_gapped, cfg):
    seen_paths = []

    class Spy(MockEngine):
        def transcribe(self, wav_path, language=None) -> EngineResult:
            seen_paths.append(wav_path)
            return super().transcribe(wav_path, language)

    transcribe_file(make_gapped(seconds=30), replace(cfg, **LONG), Spy())
    assert len(seen_paths) >= 3
    assert not any(p.exists() for p in seen_paths)


def test_parallel_chunks_run_concurrently(make_gapped, cfg):
    barrier = threading.Barrier(2, timeout=5)

    class Concurrent(MockEngine):
        def transcribe(self, wav_path, language=None) -> EngineResult:
            barrier.wait()  # deadlocks (times out) unless two chunks run at once
            return super().transcribe(wav_path, language)

    wav = make_gapped(seconds=18, period=10, on=8)  # one gap -> exactly 2 chunks
    cfg = replace(cfg, chunk_workers=2, chunk_seconds=10.0, silence_search_s=3.0)
    result = transcribe_file(wav, cfg, Concurrent(), language="en")  # known language: all parallel
    assert result.metadata.chunks == 2


def test_language_from_first_speech_chunk_is_pinned_for_the_rest(make_gapped, cfg):
    class Detects(MockEngine):
        def __init__(self):
            self.seen = []

        def transcribe(self, wav_path, language=None):
            self.seen.append(language)
            return replace(super().transcribe(wav_path, language), language=language or "fr")

    engine = Detects()
    result = transcribe_file(make_gapped(seconds=30), replace(cfg, **LONG), engine)
    assert engine.seen[0] is None  # first chunk auto-detects...
    assert set(engine.seen[1:]) == {"fr"}  # ...and every later chunk reuses its answer
    assert result.language == "fr"


def test_failed_chunk_stops_the_remaining_chunks(make_gapped, cfg):
    class Down(MockEngine):
        calls = 0

        def transcribe(self, wav_path, language=None):
            Down.calls += 1
            raise TranscriptionError("both engines down")

    with pytest.raises(TranscriptionError):
        transcribe_file(make_gapped(seconds=30), replace(cfg, **LONG), Down(), language="en")
    # 4 chunks planned; a worker may already have picked up one more before the cancel
    assert Down.calls <= 2

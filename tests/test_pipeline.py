import threading
from dataclasses import replace

import pytest

from app.audio import AudioTooLongError
from app.engines import EngineResult, MockEngine
from app.pipeline import transcribe_file

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
    result = transcribe_file(wav, cfg, Concurrent())
    assert result.metadata.chunks == 2

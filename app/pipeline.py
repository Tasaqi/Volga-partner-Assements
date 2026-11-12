"""Orchestration: probe -> normalise -> plan chunks -> transcribe chunks -> merge -> clean.

The pipeline knows nothing about HTTP: the API calls :func:`transcribe_file`,
and it can be reused from a worker or script unchanged.
"""

from __future__ import annotations

import tempfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from pathlib import Path

from app import audio
from app.audio import AudioInfo, AudioTooLongError, Chunk
from app.config import Settings
from app.engines import EngineResult, RawSegment, TranscriptionEngine
from app.postprocess import finalize_segments
from app.schemas import Metadata, TranscriptionResult

ProgressCallback = Callable[[float], None]


def inspect(path: Path, cfg: Settings) -> AudioInfo:
    """Validate the input is real audio within the duration limit."""
    info = audio.probe(path)
    if info.duration > cfg.max_duration_s:
        raise AudioTooLongError(
            f"Audio is {info.duration:.0f}s; the maximum is {cfg.max_duration_s:.0f}s"
        )
    return info


def merge_chunks(results: list[tuple[Chunk, list[RawSegment]]]) -> list[RawSegment]:
    """Shift chunk-relative timestamps to absolute time and de-duplicate overlaps.

    A segment is kept only by the chunk whose OWNED range contains its midpoint,
    so speech inside an overlap region is emitted exactly once.
    """
    merged: list[RawSegment] = []
    last_index = max((c.index for c, _ in results), default=0)
    for chunk, segments in results:
        for seg in segments:
            start, end = seg.start + chunk.cut_start, seg.end + chunk.cut_start
            mid = (start + end) / 2
            in_range = chunk.owned_start <= mid < chunk.owned_end
            if in_range or (chunk.index == last_index and mid >= chunk.owned_end):
                merged.append(
                    RawSegment(start, min(end, chunk.cut_end), seg.text,
                               seg.avg_logprob, seg.no_speech_prob)
                )
    return sorted(merged, key=lambda s: s.start)


def _pick_language(results: list[EngineResult]) -> str | None:
    # Prefer a chunk that actually had speech; silent chunks give unreliable guesses.
    for r in results:
        if r.segments and r.language:
            return r.language
    return next((r.language for r in results if r.language), None)


def transcribe_file(
    path: Path,
    cfg: Settings,
    engine: TranscriptionEngine,
    *,
    language: str | None = None,
    filename: str | None = None,
    progress: ProgressCallback | None = None,
    info: AudioInfo | None = None,
) -> TranscriptionResult:
    """Transcribe any supported audio/video file. Raises :class:`audio.AudioError`."""
    started = time.monotonic()
    info = info or inspect(path, cfg)

    with tempfile.TemporaryDirectory(prefix="tp-") as tmp:
        # ffprobe's duration comes from headers, which can lie (a crafted MP3 probed as 95 s
        # decoded to an hour). Decode at most just past the limit, then check the real length.
        wav = audio.normalise(path, Path(tmp) / "audio.wav", max_seconds=cfg.max_duration_s + 1)
        duration = audio.wav_duration(wav)
        if duration > cfg.max_duration_s:
            raise AudioTooLongError(
                f"Audio is longer than its header says; the maximum is {cfg.max_duration_s:.0f}s"
            )

        silences = (
            audio.detect_silences(wav, cfg.silence_noise_db, cfg.silence_min_s)
            if duration > cfg.chunk_seconds
            else []
        )
        chunks = audio.plan_chunks(
            duration, silences, cfg.chunk_seconds, cfg.silence_search_s, cfg.chunk_overlap_s
        )

        lock = threading.Lock()
        done = 0

        def run_chunk(chunk: Chunk, lang: str | None) -> EngineResult:
            nonlocal done
            if len(chunks) == 1:
                result = engine.transcribe(wav, lang)
            else:
                part = audio.extract(wav, chunk.cut_start, chunk.cut_end,
                                     Path(tmp) / f"chunk-{chunk.index:04d}.wav")
                try:
                    result = engine.transcribe(part, lang)
                finally:
                    part.unlink(missing_ok=True)  # bounded disk use on long files
            with lock:
                done += 1
                if progress:
                    progress(done / len(chunks))
            return result

        # Without a requested language, transcribe chunks one by one until one contains
        # speech, then pin its language for the rest. Otherwise each chunk would detect
        # on its own, and a noisy chunk of an English file could come out as Japanese.
        results: dict[int, EngineResult] = {}
        for chunk in chunks:
            if language:
                break
            results[chunk.index] = first = run_chunk(chunk, None)
            if first.segments and first.language:
                language = first.language

        remaining = [c for c in chunks if c.index not in results]
        with ThreadPoolExecutor(max_workers=max(1, cfg.chunk_workers)) as pool:
            futures = {c.index: pool.submit(run_chunk, c, language) for c in remaining}
            finished, pending = wait(futures.values(), return_when=FIRST_EXCEPTION)
            failed = next((f for f in finished if f.exception()), None)
            if failed:
                for f in pending:
                    f.cancel()  # don't spend CPU or paid API minutes on a file that already failed
                raise failed.exception()  # type: ignore[misc]
        results.update({i: f.result() for i, f in futures.items()})
        engine_results = [results[c.index] for c in chunks]

    raw = merge_chunks([(c, r.segments) for c, r in zip(chunks, engine_results)])
    segments = finalize_segments(raw)
    return TranscriptionResult(
        filename=filename or path.name,
        language=language or _pick_language(engine_results),
        duration=round(duration, 3),
        text=" ".join(s.text for s in segments),
        segments=segments,
        metadata=Metadata(
            engine=engine.name,
            model=engine.model,
            chunks=len(chunks),
            chunks_cut_in_silence=sum(c.end_in_silence for c in chunks),
            fallback_chunks=sum(r.fallback for r in engine_results),
            processing_seconds=round(time.monotonic() - started, 3),
            source_format=info.as_metadata(),
        ),
    )

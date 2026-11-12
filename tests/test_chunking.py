import pytest

from app.audio import Chunk, plan_chunks
from app.engines import RawSegment
from app.pipeline import merge_chunks


def assert_covers(chunks: list[Chunk], duration: float) -> None:
    assert chunks[0].owned_start == 0
    assert chunks[-1].owned_end == duration
    for a, b in zip(chunks, chunks[1:]):
        assert a.owned_end == b.owned_start  # no gaps, no double ownership
    for c in chunks:
        assert c.cut_start <= c.owned_start < c.owned_end <= c.cut_end


def test_short_audio_is_one_chunk():
    chunks = plan_chunks(90, [], chunk_seconds=600)
    assert len(chunks) == 1
    assert (chunks[0].cut_start, chunks[0].cut_end) == (0, 90)


def test_boundaries_snap_to_nearest_silence():
    silences = [(570.0, 571.0), (610.0, 612.0), (1190.0, 1192.0)]
    chunks = plan_chunks(1500, silences, chunk_seconds=600, search_s=30, overlap_s=2)
    # 610-612 (mid 611) is nearer to 600 than 570-571 (mid 570.5)
    assert [c.owned_end for c in chunks] == [611.0, 1191.0, 1500]
    assert all(c.cut_start == c.owned_start and c.cut_end == c.owned_end for c in chunks)
    assert sum(c.end_in_silence for c in chunks) == 2
    assert_covers(chunks, 1500)


def test_hard_cut_adds_overlap_on_both_sides():
    chunks = plan_chunks(1000, [(100.0, 101.0)], chunk_seconds=600, search_s=30, overlap_s=2)
    assert len(chunks) == 2
    first, second = chunks
    assert first.owned_end == 600 and not first.end_in_silence
    assert first.cut_end == 602
    assert second.cut_start == 598
    assert_covers(chunks, 1000)


@pytest.mark.parametrize("duration", [600.0, 601.0, 1799.5, 3600.0, 14400.0])
def test_chunks_cover_whole_file_without_gaps(duration: float):
    silences = [(t, t + 1.0) for t in range(55, int(duration), 97)]
    chunks = plan_chunks(duration, silences, chunk_seconds=600, search_s=30, overlap_s=2)
    assert_covers(chunks, duration)
    assert all(c.owned_end - c.owned_start <= 630 for c in chunks)


def test_overlap_merge_keeps_spanning_segment_exactly_once():
    a, b = plan_chunks(20, [], chunk_seconds=10, search_s=0, overlap_s=2)
    assert (a.cut_end, b.cut_start) == (12, 8)
    # A segment at 9.0-11.5 absolute sits in both chunks' cut ranges.
    in_a = [RawSegment(0, 4, "early"), RawSegment(9.0, 11.5, "spans the cut")]
    in_b = [RawSegment(1.0, 3.5, "spans the cut"), RawSegment(5, 8, "late")]

    merged = merge_chunks([(a, in_a), (b, in_b)])

    assert [s.text for s in merged] == ["early", "spans the cut", "late"]
    spanning = merged[1]
    assert (spanning.start, spanning.end) == (9.0, 11.5)
    assert merged[2].start == 13  # shifted to absolute time

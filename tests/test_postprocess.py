import math

from app.engines import RawSegment
from app.postprocess import confidence, finalize_segments, to_srt, to_txt, to_vtt
from app.schemas import Segment


def test_hallucinations_and_empty_segments_are_dropped():
    raw = [
        RawSegment(0, 1, "  hello   world ", avg_logprob=-0.2, no_speech_prob=0.1),
        RawSegment(1, 2, "Thanks for watching!", avg_logprob=-1.5, no_speech_prob=0.9),
        RawSegment(2, 3, "   ", avg_logprob=-0.1, no_speech_prob=0.0),
        # high no-speech alone is not enough: confident text is kept
        RawSegment(3, 4, "quiet but sure", avg_logprob=-0.3, no_speech_prob=0.8),
    ]
    out = finalize_segments(raw)
    assert [s.text for s in out] == ["hello world", "quiet but sure"]
    assert [s.id for s in out] == [1, 2]


def test_confidence_is_clamped_exp_logprob():
    assert confidence(-0.5) == math.exp(-0.5)
    assert confidence(0.3) == 1.0
    assert confidence(-1e9) == 0.0


SEGS = [
    Segment(id=1, start=0.0, end=2.5, text="first", confidence=0.9),
    Segment(id=2, start=3661.25, end=3663.0, text="second", confidence=0.8),
]


def test_srt_format():
    assert to_srt(SEGS) == (
        "1\n00:00:00,000 --> 00:00:02,500\nfirst\n\n"
        "2\n01:01:01,250 --> 01:01:03,000\nsecond\n"
    )


def test_vtt_format():
    assert to_vtt(SEGS) == (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:02.500\nfirst\n\n"
        "01:01:01.250 --> 01:01:03.000\nsecond\n"
    )


def test_txt_format():
    assert to_txt(SEGS).splitlines() == [
        "[00:00:00.000 --> 00:00:02.500] first",
        "[01:01:01.250 --> 01:01:03.000] second",
    ]

from conftest import make_wav


def test_returns_timestamped_segments(client):
    body = client.post("/transcribe", files={"file": ("clip.wav", make_wav(seconds=2))}).json()
    assert body["duration"] == 2.0
    assert body["segments"] == [
        {"start": 0.0, "end": 1.0, "text": "hello1"},
        {"start": 1.0, "end": 2.0, "text": "world1"},
    ]


def test_segments_are_ordered_and_within_duration(client):
    body = client.post("/transcribe", files={"file": ("clip.wav", make_wav())}).json()
    ends = [s["end"] for s in body["segments"]]
    assert all(s["start"] <= s["end"] for s in body["segments"])
    assert ends == sorted(ends) and ends[-1] <= body["duration"]

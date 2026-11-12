import pytest

from app.config import Settings


@pytest.mark.parametrize("field", ["chunk_seconds", "chunk_workers", "job_workers", "max_upload_mb"])
def test_zero_or_negative_limits_fail_at_startup(field):
    # chunk_seconds=0 used to make chunk planning loop forever
    with pytest.raises(ValueError, match=f"TP_{field.upper()} must be > 0"):
        Settings(**{field: 0})


def test_blank_env_values_mean_default(monkeypatch):
    monkeypatch.setenv("TP_CHUNK_WORKERS", "")  # a blank line kept in .env
    monkeypatch.setenv("TP_CHUNK_SECONDS", "300")
    cfg = Settings.from_env()
    assert (cfg.chunk_workers, cfg.chunk_seconds) == (1, 300.0)


def test_float_env_value_for_max_duration(monkeypatch):
    monkeypatch.setenv("TP_MAX_DURATION_S", "14400.5")  # int default used to make this crash
    assert Settings.from_env().max_duration_s == 14400.5

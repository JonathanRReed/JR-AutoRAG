from app.core import rate_limiter as rate_limiter_module


def test_default_rate_limit_allows_operator_console_startup(monkeypatch):
    monkeypatch.delenv("JR_DEMO_MODE", raising=False)
    monkeypatch.delenv("AUTORAG_RATE_LIMIT_ENABLED", raising=False)
    monkeypatch.delenv("AUTORAG_RATE_LIMIT_RPM", raising=False)
    monkeypatch.delenv("AUTORAG_RATE_LIMIT_BURST", raising=False)
    monkeypatch.setattr(rate_limiter_module, "_rate_limiter", None)

    limiter = rate_limiter_module.get_rate_limiter()
    stats = limiter.get_stats()

    assert stats["enabled"] is True
    assert stats["requests_per_minute"] == 600
    assert stats["burst_capacity"] == 80
    assert all(limiter.allow("ip:127.0.0.1") for _ in range(40))

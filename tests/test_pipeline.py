"""End-to-end pipeline test using mock data + disabled LLM (forces fallback path)."""
from __future__ import annotations

from datetime import date

from finai.pipeline.etl import run_etl, run_full_pipeline
from finai.report.builder import build_daily_report


def test_etl_persists_snapshot(monkeypatch):
    from finai.config import settings
    monkeypatch.setattr(settings, "fetch_regions", "cn-a")  # mock source only covers A-share
    snap, regional, macro, effective = run_etl(trade_date=date(2026, 5, 6), source_name="mock")
    assert not snap.stocks.empty
    assert not snap.sectors.empty
    assert regional == {}
    assert macro is None
    assert effective == {"cn-a": date(2026, 5, 6)}


def test_full_pipeline_produces_html(monkeypatch):
    from finai.config import settings
    monkeypatch.setattr(settings, "fetch_regions", "cn-a")
    out = run_full_pipeline(trade_date=date(2026, 5, 6), source_name="mock")
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "FinAI" in html
    assert "市场总览" in html
    assert "板块强度" in html


def test_report_payload_has_fallback_when_llm_off(monkeypatch):
    from finai.config import settings
    monkeypatch.setattr(settings, "fetch_regions", "cn-a")
    snap, _, _, _ = run_etl(trade_date=date(2026, 5, 6), source_name="mock")
    payload = build_daily_report(snap)
    assert payload.fallback_used  # LLM disabled → attribution falls back
    assert payload.attributions
    assert payload.narrative["overview"]
    assert payload.macro == {}
    assert payload.cross_market == {}


def test_db_cache_backfill_when_today_empty(monkeypatch):
    """Today's fetch returns empty → ETL backfills from yesterday's cached snapshot."""
    from finai.config import settings
    from finai.data.base import MarketSnapshot
    from finai.data.mock_source import MockSource
    import pandas as pd
    monkeypatch.setattr(settings, "fetch_regions", "cn-a")

    # Seed DB with 2026-05-05 data via mock
    run_etl(trade_date=date(2026, 5, 5), source_name="mock")

    # Patch the source to return an empty snapshot for 2026-05-06
    class EmptySource(MockSource):
        def fetch_snapshot(self, trade_date):
            return MarketSnapshot(
                trade_date=trade_date,
                indices=pd.DataFrame(), stocks=pd.DataFrame(),
                sectors=pd.DataFrame(), capital=pd.DataFrame(), news=pd.DataFrame(),
            )

    monkeypatch.setattr("finai.pipeline.etl.get_source", lambda *a, **k: EmptySource())
    snap, _, _, effective = run_etl(trade_date=date(2026, 5, 6), source_name="mock")
    assert effective == {"cn-a": date(2026, 5, 5)}
    assert not snap.stocks.empty  # backfilled from 2026-05-05
    assert snap.trade_date == date(2026, 5, 5)


def test_stale_regions_marked_in_payload(monkeypatch):
    from finai.config import settings
    monkeypatch.setattr(settings, "fetch_regions", "cn-a")
    snap, _, _, _ = run_etl(trade_date=date(2026, 5, 6), source_name="mock")
    # Force a stale effective date
    payload = build_daily_report(snap, effective_dates={"cn-a": date(2026, 5, 4)})
    assert "cn-a" in payload.stale_regions
    assert payload.effective_dates["cn-a"] == "2026-05-04"

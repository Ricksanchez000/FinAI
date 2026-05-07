"""End-to-end pipeline test using mock data + disabled LLM (forces fallback path)."""
from __future__ import annotations

from datetime import date

from finai.pipeline.etl import run_etl, run_full_pipeline
from finai.report.builder import build_daily_report


def test_etl_persists_snapshot():
    snap = run_etl(trade_date=date(2026, 5, 6), source_name="mock")
    assert not snap.stocks.empty
    assert not snap.sectors.empty


def test_full_pipeline_produces_html():
    out = run_full_pipeline(trade_date=date(2026, 5, 6), source_name="mock")
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "FinAI" in html
    assert "市场总览" in html
    assert "板块强度" in html


def test_report_payload_has_fallback_when_llm_off():
    snap = run_etl(trade_date=date(2026, 5, 6), source_name="mock")
    payload = build_daily_report(snap)
    assert payload.fallback_used  # LLM disabled → attribution falls back
    assert payload.attributions
    assert payload.narrative["overview"]

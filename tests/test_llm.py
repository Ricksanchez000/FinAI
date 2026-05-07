"""LLM layer tests — verify fallback paths and Pydantic schema shape."""
from __future__ import annotations

import pandas as pd

from finai.llm.attribution import Attribution, attribute_anomalies
from finai.llm.client import LLMClient
from finai.llm.narrative import build_market_narrative
from finai.signals import compute_market_overview, compute_sector_rotation, detect_anomalies


def test_attribution_falls_back_when_llm_disabled(mock_source):
    from datetime import date
    snap = mock_source.fetch_snapshot(date(2026, 5, 6))
    rows = detect_anomalies(snap.stocks, top_n=5)
    result = attribute_anomalies(rows, snap.news, limit=5)
    assert result.fallback_used
    assert len(result.rows) == min(5, len(rows))
    for attr in result.rows:
        assert isinstance(attr, Attribution)
        assert attr.dimensions == ["technical"]


def test_narrative_falls_back_when_llm_disabled(mock_source):
    from datetime import date
    snap = mock_source.fetch_snapshot(date(2026, 5, 6))
    ov = compute_market_overview(snap.indices, snap.stocks, snap.capital, snap.trade_date)
    sv = compute_sector_rotation(snap.sectors, snap.trade_date)
    n = build_market_narrative(ov, sv)
    assert "LLM 不可用" in n.risk
    assert n.overview


def test_client_marks_disabled_without_api_key():
    c = LLMClient()
    assert c.enabled is False

"""Signal layer is pure-pandas — easiest place to assert correctness."""
from __future__ import annotations

from datetime import date

from finai.signals import (
    compute_market_overview,
    compute_sector_rotation,
    detect_anomalies,
)


def test_market_overview_counts_breadth(mock_source):
    snap = mock_source.fetch_snapshot(date(2026, 5, 6))
    ov = compute_market_overview(snap.indices, snap.stocks, snap.capital, snap.trade_date)
    assert ov.breadth["advance"] + ov.breadth["decline"] + ov.breadth["flat"] == len(snap.stocks)
    assert ov.total_amount_yi > 0
    assert len(ov.indices) == 5


def test_anomaly_detection_finds_injected_limits(mock_source):
    snap = mock_source.fetch_snapshot(date(2026, 5, 6))
    rows = detect_anomalies(snap.stocks, top_n=30)
    assert rows, "mock data injects ~12 limit-up/down candidates"
    reasons = {r.reason for r in rows}
    assert {"pct_up", "pct_down"} <= reasons


def test_sector_rotation_orders_by_pct(mock_source):
    snap = mock_source.fetch_snapshot(date(2026, 5, 6))
    sv = compute_sector_rotation(snap.sectors, snap.trade_date)
    assert sv.rising
    pcts = [r.pct_change for r in sv.rising]
    assert pcts == sorted(pcts, reverse=True)


def test_overview_handles_empty_frames():
    import pandas as pd
    ov = compute_market_overview(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), date(2026, 5, 6))
    assert ov.breadth == {}
    assert ov.indices == []

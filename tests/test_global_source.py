"""Global source — verify CoinGecko parsing + graceful degradation when offline."""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd

from finai.data.global_source import GlobalSource


def test_crypto_parses_coingecko_response():
    fake_payload = {
        "bitcoin":      {"usd": 70000.0, "usd_24h_change": 1.5, "last_updated_at": 1715600000},
        "ethereum":     {"usd": 3500.0,  "usd_24h_change": -0.8, "last_updated_at": 1715600000},
        "solana":       {"usd": 150.0,   "usd_24h_change": 2.3, "last_updated_at": 1715600000},
        "binancecoin":  {"usd": 580.0,   "usd_24h_change": 0.1, "last_updated_at": 1715600000},
    }
    mock_resp = MagicMock()
    mock_resp.json.return_value = fake_payload
    mock_resp.raise_for_status.return_value = None
    with patch("finai.data.global_source.httpx.get", return_value=mock_resp):
        src = GlobalSource.__new__(GlobalSource)
        src.ak = MagicMock()
        df = src._crypto()
    assert len(df) == 4
    assert set(df["code"]) == {"BTC", "ETH", "SOL", "BNB"}
    btc = df[df["code"] == "BTC"].iloc[0]
    assert btc["value"] == 70000.0
    assert btc["pct_change"] == 1.5


def test_crypto_returns_empty_on_network_failure():
    with patch("finai.data.global_source.httpx.get",
                side_effect=ConnectionError("no network")):
        src = GlobalSource.__new__(GlobalSource)
        src.ak = MagicMock()
        df = src._crypto()
    assert df.empty


def test_indices_falls_back_to_sina_when_em_throttled():
    src = GlobalSource.__new__(GlobalSource)
    src.ak = MagicMock()
    src.ak.index_global_spot_em.side_effect = ValueError("Expecting value: line 1 column 1 (char 0)")
    # Sina fallback returns a tiny 2-row history
    sina_df = pd.DataFrame({
        "date": ["2026-05-07", "2026-05-08"],
        "close": [5500.0, 5550.0],
    })
    src.ak.index_us_stock_sina.return_value = sina_df
    df = src._indices()
    assert not df.empty
    # 3 known fallback symbols
    assert set(df["code"]) <= {"INX", "IXIC", "DJI"}
    assert all(df["pct_change"] > 0)  # 5550 > 5500


def test_yields_extracts_latest_nonNan_per_series():
    """When latest row has NaN for some series (e.g. US data hasn't published),
    pick the most recent non-NaN per column independently.
    """
    src = GlobalSource.__new__(GlobalSource)
    src.ak = MagicMock()
    src.ak.bond_zh_us_rate.return_value = pd.DataFrame({
        "日期": ["2026-05-07", "2026-05-08", "2026-05-09"],
        "中国国债收益率10年":  [1.76, 1.77, 1.78],
        "美国国债收益率10年":  [4.38, 4.40, None],   # last day not yet published
    })
    df = src._yields()
    assert not df.empty
    us = df[df["code"] == "US10Y"].iloc[0]
    cn = df[df["code"] == "CN10Y"].iloc[0]
    assert us["value"] == 4.40  # backed off one day
    assert cn["value"] == 1.78  # latest

"""Build a StockHistory from AkShare for any A-share ticker.

Centralized so all callers get the same column schema. Adds best-effort
fundamentals (PE, PB, ROE) and recent institutional flow when reachable;
methods that need them check for None and degrade gracefully.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from finai.quant.base import StockHistory

log = logging.getLogger(__name__)


def load_history(symbol: str, lookback_days: int = 750) -> StockHistory:
    import akshare as ak

    code = symbol.lstrip("sh.").lstrip("sz.").lstrip("SH.").lstrip("SZ.")
    end = date.today()
    start = end - timedelta(days=lookback_days)
    bars = pd.DataFrame()
    try:
        bars = ak.stock_zh_a_hist(
            symbol=code, period="daily",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            adjust="qfq",
        )
    except Exception as exc:
        log.warning("hist fetch failed for %s: %s", code, exc)

    if not bars.empty:
        bars = bars.rename(columns={
            "日期": "trade_date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low",
            "成交量": "volume", "成交额": "amount",
            "涨跌幅": "pct_change",
        })
        bars["trade_date"] = pd.to_datetime(bars["trade_date"]).dt.date
        for col in ("open", "close", "high", "low", "volume", "amount", "pct_change"):
            if col in bars.columns:
                bars[col] = pd.to_numeric(bars[col], errors="coerce")

    name = _lookup_name(ak, code) or code
    fundamentals = _lookup_fundamentals(ak, code)
    fundamentals["institutional_flow"] = _lookup_inst_flow(ak, code)

    return StockHistory(
        symbol=code, name=name, bars=bars,
        as_of=end, fundamentals=fundamentals,
    )


def _lookup_name(ak, code: str) -> str | None:
    for fn in ("stock_individual_info_em",):
        try:
            df = getattr(ak, fn)(symbol=code)
            if df is not None and not df.empty and "value" in df.columns:
                row = df[df["item"] == "股票简称"]
                if not row.empty:
                    return str(row["value"].iloc[0])
        except Exception:
            continue
    return None


def _lookup_fundamentals(ak, code: str) -> dict:
    """Pull pe/pb from spot board if available. Best effort — many endpoints
    fluctuate, so we cap at one attempt and don't retry."""
    out = {}
    try:
        df = ak.stock_individual_info_em(symbol=code)
        if df is not None and not df.empty and "item" in df.columns:
            for _, r in df.iterrows():
                item, val = str(r["item"]), r.get("value")
                if "市盈率" in item and "动" not in item:
                    try: out["pe"] = float(val)
                    except (TypeError, ValueError): pass
                elif "市净率" in item:
                    try: out["pb"] = float(val)
                    except (TypeError, ValueError): pass
                elif "总市值" in item:
                    try: out["market_cap"] = float(val)
                    except (TypeError, ValueError): pass
    except Exception as exc:
        log.debug("fundamentals lookup failed: %s", exc)
    return out


def _lookup_inst_flow(ak, code: str) -> pd.DataFrame:
    """Last ~30 days of north + main capital flow for this ticker."""
    try:
        df = ak.stock_individual_fund_flow(stock=code)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns={
        "日期": "trade_date",
        "主力净流入-净额": "main_net_in",
        "主力净流入-净占比": "main_pct",
    })
    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    if "main_net_in" in df.columns:
        df["main_net_in"] = pd.to_numeric(df["main_net_in"], errors="coerce") / 1e8  # → 亿
    # Stub: 北向 per-stock is a separate AkShare endpoint and unreliable; leave 0
    df["north_net_in"] = 0.0
    return df.tail(30).reset_index(drop=True)

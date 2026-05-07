"""End-to-end pipeline: fetch → persist → signal → LLM → render."""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import math

from sqlalchemy import delete

from finai.data import MarketSnapshot, get_source
from finai.db import init_db, session_scope
from finai.models import (
    CapitalFlow,
    DailyReport,
    IndexQuote,
    NewsItem,
    SectorQuote,
    StockQuote,
)
from finai.report.builder import build_daily_report, write_report

log = logging.getLogger(__name__)


def run_etl(trade_date: date | None = None, source_name: str | None = None) -> MarketSnapshot:
    init_db()
    src = get_source(source_name)
    td = trade_date or src.latest_trade_date()
    log.info("etl: fetching snapshot for %s", td)
    snap = src.fetch_snapshot(td)
    _persist(snap)
    return snap


def run_full_pipeline(
    trade_date: date | None = None, source_name: str | None = None
) -> Path:
    snap = run_etl(trade_date, source_name)
    payload = build_daily_report(snap)
    out_path = write_report(payload)
    _persist_report(payload, out_path)
    log.info("pipeline complete: %s", out_path)
    return out_path


def _persist(snap: MarketSnapshot) -> None:
    with session_scope() as s:
        for table, df, mapper in (
            (IndexQuote, snap.indices, _to_index),
            (StockQuote, snap.stocks, _to_stock),
            (SectorQuote, snap.sectors, _to_sector),
            (CapitalFlow, snap.capital, _to_capital),
            (NewsItem, snap.news, _to_news),
        ):
            s.execute(delete(table).where(table.trade_date == snap.trade_date))
            if df is None or df.empty:
                continue
            objs = [mapper(snap.trade_date, r) for _, r in df.iterrows()]
            objs = [o for o in objs if o is not None]
            if objs:
                s.bulk_save_objects(objs)


def _f(v, default: float = 0.0) -> float:
    """Coerce pandas/numpy values (including NaN, None, '-') to a finite float."""
    if v is None:
        return default
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if math.isnan(x) or math.isinf(x):
        return default
    return x


def _to_index(td, r):
    return IndexQuote(
        trade_date=td, code=str(r["code"]), name=str(r["name"]),
        open=_f(r.get("open")), close=_f(r.get("close")),
        high=_f(r.get("high")), low=_f(r.get("low")),
        volume=_f(r.get("volume")), amount=_f(r.get("amount")),
        pct_change=_f(r.get("pct_change")),
    )


def _to_stock(td, r):
    return StockQuote(
        trade_date=td, code=str(r["code"]), name=str(r["name"]),
        open=_f(r.get("open")), close=_f(r.get("close")),
        high=_f(r.get("high")), low=_f(r.get("low")),
        pre_close=_f(r.get("pre_close")),
        volume=_f(r.get("volume")), amount=_f(r.get("amount")),
        turnover_rate=_f(r.get("turnover_rate")),
        pct_change=_f(r.get("pct_change")),
        market_cap=_f(r.get("market_cap")),
        sector=str(r.get("sector") or ""),
    )


def _to_sector(td, r):
    return SectorQuote(
        trade_date=td, sector=str(r["sector"]),
        pct_change=_f(r.get("pct_change")),
        amount=_f(r.get("amount")),
        leader_code=str(r.get("leader_code") or ""),
        leader_name=str(r.get("leader_name") or ""),
    )


def _to_capital(td, r):
    return CapitalFlow(
        trade_date=td, scope=str(r["scope"]), metric=str(r["metric"]),
        value=_f(r.get("value")),
    )


def _to_news(td, r):
    pub = r.get("published_at")
    if pub is None:
        return None
    return NewsItem(
        trade_date=td, code=str(r.get("code", "") or ""),
        source=str(r.get("source", "") or ""),
        title=str(r.get("title", ""))[:256],
        url=str(r.get("url", ""))[:512],
        published_at=pub.to_pydatetime() if hasattr(pub, "to_pydatetime") else pub,
        summary=str(r.get("summary", ""))[:4000],
    )


def _persist_report(payload, out_path: Path) -> None:
    with session_scope() as s:
        s.execute(delete(DailyReport).where(DailyReport.trade_date == payload.trade_date))
        s.add(DailyReport(
            trade_date=payload.trade_date,
            payload_json=payload.to_json(),
            html_path=str(out_path),
        ))

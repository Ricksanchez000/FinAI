"""End-to-end pipeline: A-share + optional US/HK/Global → persist → signal → LLM → render."""
from __future__ import annotations

import json
import logging
import math
from datetime import date
from pathlib import Path

from sqlalchemy import delete

from finai.config import settings
from finai.data import (
    MacroSnapshot,
    MarketSnapshot,
    RegionalSnapshot,
    get_macro_source,
    get_regional_source,
    get_source,
)
from finai.db import init_db, session_scope
from finai.models import (
    CapitalFlow,
    DailyReport,
    IndexQuote,
    MacroQuote,
    NewsItem,
    SectorQuote,
    StockQuote,
)
from finai.report.builder import build_daily_report, write_report

log = logging.getLogger(__name__)


def _enabled_regions() -> list[str]:
    return [r.strip() for r in settings.fetch_regions.split(",") if r.strip()]


def run_etl(trade_date: date | None = None, source_name: str | None = None
            ) -> tuple[MarketSnapshot, dict[str, RegionalSnapshot], MacroSnapshot | None]:
    init_db()
    src = get_source(source_name)
    td = trade_date or src.latest_trade_date()
    regions = _enabled_regions()

    log.info("etl: fetching A-share snapshot for %s", td)
    snap = src.fetch_snapshot(td)
    _persist_a(snap)

    regional: dict[str, RegionalSnapshot] = {}
    for market in ("us", "hk"):
        if market not in regions:
            continue
        log.info("etl: fetching %s snapshot (this can take 5–8 min)", market.upper())
        rsnap = get_regional_source(market).fetch_regional(td)
        regional[market] = rsnap
        _persist_regional(rsnap)

    macro: MacroSnapshot | None = None
    if "global" in regions:
        log.info("etl: fetching global macro snapshot")
        macro = get_macro_source().fetch_macro(td)
        _persist_macro(macro)

    return snap, regional, macro


def run_full_pipeline(
    trade_date: date | None = None, source_name: str | None = None
) -> Path:
    snap, regional, macro = run_etl(trade_date, source_name)
    payload = build_daily_report(snap, regional=regional, macro=macro)
    out_path = write_report(payload)
    _persist_report(payload, out_path)
    log.info("pipeline complete: %s", out_path)
    return out_path


def _f(v, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if math.isnan(x) or math.isinf(x):
        return default
    return x


def _persist_a(snap: MarketSnapshot) -> None:
    with session_scope() as s:
        s.execute(delete(IndexQuote).where(
            (IndexQuote.trade_date == snap.trade_date) & (IndexQuote.market == "cn-a")))
        s.execute(delete(StockQuote).where(
            (StockQuote.trade_date == snap.trade_date) & (StockQuote.market == "cn-a")))
        s.execute(delete(SectorQuote).where(SectorQuote.trade_date == snap.trade_date))
        s.execute(delete(CapitalFlow).where(CapitalFlow.trade_date == snap.trade_date))
        s.execute(delete(NewsItem).where(NewsItem.trade_date == snap.trade_date))
        for table, df, mapper in (
            (IndexQuote, snap.indices, _to_index),
            (StockQuote, snap.stocks, _to_stock),
            (SectorQuote, snap.sectors, _to_sector),
            (CapitalFlow, snap.capital, _to_capital),
            (NewsItem, snap.news, _to_news),
        ):
            if df is None or df.empty:
                continue
            objs = [mapper(snap.trade_date, r, "cn-a") for _, r in df.iterrows()]
            objs = [o for o in objs if o is not None]
            if objs:
                s.bulk_save_objects(objs)


def _persist_regional(rs: RegionalSnapshot) -> None:
    if rs.stocks is None or rs.stocks.empty:
        return
    with session_scope() as s:
        s.execute(delete(StockQuote).where(
            (StockQuote.trade_date == rs.trade_date) & (StockQuote.market == rs.market)))
        objs = [_to_stock(rs.trade_date, r, rs.market) for _, r in rs.stocks.iterrows()]
        s.bulk_save_objects([o for o in objs if o is not None])


def _persist_macro(snap: MacroSnapshot) -> None:
    with session_scope() as s:
        s.execute(delete(MacroQuote).where(MacroQuote.trade_date == snap.trade_date))
        for asset_class, df in (
            ("index_global", snap.indices_global),
            ("fx", snap.fx),
            ("yield", snap.yields),
            ("commodity", snap.commodities),
            ("crypto", snap.crypto),
        ):
            if df is None or df.empty:
                continue
            for _, r in df.iterrows():
                ts = r.get("as_of_ts")
                ts = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") and ts is not None else None
                s.add(MacroQuote(
                    trade_date=snap.trade_date,
                    asset_class=asset_class,
                    code=str(r["code"]),
                    name=str(r.get("name") or r["code"]),
                    value=_f(r.get("value")),
                    pct_change=_f(r.get("pct_change")),
                    as_of_ts=ts,
                ))


def _to_index(td, r, market: str = "cn-a"):
    return IndexQuote(
        trade_date=td, market=market,
        code=str(r["code"]), name=str(r["name"]),
        open=_f(r.get("open")), close=_f(r.get("close")),
        high=_f(r.get("high")), low=_f(r.get("low")),
        volume=_f(r.get("volume")), amount=_f(r.get("amount")),
        pct_change=_f(r.get("pct_change")),
    )


def _to_stock(td, r, market: str = "cn-a"):
    return StockQuote(
        trade_date=td, market=market,
        code=str(r["code"]), name=str(r["name"]),
        open=_f(r.get("open")), close=_f(r.get("close")),
        high=_f(r.get("high")), low=_f(r.get("low")),
        pre_close=_f(r.get("pre_close")),
        volume=_f(r.get("volume")), amount=_f(r.get("amount")),
        turnover_rate=_f(r.get("turnover_rate")),
        pct_change=_f(r.get("pct_change")),
        market_cap=_f(r.get("market_cap")),
        sector=str(r.get("sector") or ""),
    )


def _to_sector(td, r, market: str = "cn-a"):
    return SectorQuote(
        trade_date=td, sector=str(r["sector"]),
        pct_change=_f(r.get("pct_change")),
        amount=_f(r.get("amount")),
        leader_code=str(r.get("leader_code") or ""),
        leader_name=str(r.get("leader_name") or ""),
    )


def _to_capital(td, r, market: str = "cn-a"):
    return CapitalFlow(
        trade_date=td, scope=str(r["scope"]), metric=str(r["metric"]),
        value=_f(r.get("value")),
    )


def _to_news(td, r, market: str = "cn-a"):
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

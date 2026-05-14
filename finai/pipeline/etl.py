"""End-to-end pipeline: A-share + optional US/HK/Global → persist → signal → LLM → render."""
from __future__ import annotations

import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy import delete, select

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
            ) -> tuple[MarketSnapshot, dict[str, RegionalSnapshot], MacroSnapshot | None,
                       dict[str, date]]:
    """Returns (a_snap, regional_snaps, macro_snap, effective_dates).

    effective_dates maps region key → date the data is actually from. If a
    live endpoint returned empty and we backfilled from the DB, the date here
    will lag the requested trade_date. Empty regions are absent from the dict.
    """
    init_db()
    src = get_source(source_name)
    td = trade_date or src.latest_trade_date()
    regions = _enabled_regions()
    effective: dict[str, date] = {}

    log.info("etl: fetching A-share snapshot for %s", td)
    snap = src.fetch_snapshot(td)
    if _is_a_snap_usable(snap):
        _persist_a(snap)
        effective["cn-a"] = td
    else:
        log.warning("A-share fetch empty for %s — loading latest cached snapshot from DB", td)
        cached = _load_cached_a(td)
        if cached is not None:
            snap, eff_date = cached
            effective["cn-a"] = eff_date
            log.info("A-share backfilled from %s", eff_date)
        # else: leave snap as-is (empty); downstream handles it.

    # US, HK, global are independent — fire them in parallel. Each task is
    # mostly network-bound so GIL contention is a non-issue.
    regional: dict[str, RegionalSnapshot] = {}
    macro: MacroSnapshot | None = None
    parallel_tasks: dict = {}
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="finai-etl") as pool:
        for market in ("us", "hk"):
            if market in regions:
                log.info("etl: dispatch %s fetch (~5–8 min)", market.upper())
                parallel_tasks[pool.submit(get_regional_source(market).fetch_regional, td)] = ("regional", market)
        if "global" in regions:
            log.info("etl: dispatch global macro fetch")
            parallel_tasks[pool.submit(get_macro_source().fetch_macro, td)] = ("macro", None)

        for fut in as_completed(parallel_tasks):
            kind, market = parallel_tasks[fut]
            try:
                result = fut.result()
            except Exception as exc:
                log.warning("%s/%s fetch raised: %s", kind, market, exc)
                continue
            if kind == "regional":
                rsnap: RegionalSnapshot = result
                if rsnap.stocks is not None and not rsnap.stocks.empty:
                    regional[market] = rsnap
                    _persist_regional(rsnap)
                    effective[market] = td
                else:
                    log.warning("%s fetch empty — loading latest cached snapshot", market)
                    cached_r = _load_cached_regional(td, market)
                    if cached_r is not None:
                        regional[market], effective[market] = cached_r
                        log.info("%s backfilled from %s", market, effective[market])
            else:  # macro
                macro = result
                if _macro_has_data(macro):
                    _persist_macro(macro)
                    effective["global"] = td
                else:
                    cached_m = _load_cached_macro(td)
                    if cached_m is not None:
                        macro, effective["global"] = cached_m
                        log.info("global macro backfilled from %s", effective["global"])

    return snap, regional, macro, effective


def run_full_pipeline(
    trade_date: date | None = None, source_name: str | None = None
) -> Path:
    snap, regional, macro, effective = run_etl(trade_date, source_name)
    payload = build_daily_report(snap, regional=regional, macro=macro,
                                  effective_dates=effective)
    out_path = write_report(payload)
    _persist_report(payload, out_path)
    log.info("pipeline complete: %s", out_path)
    return out_path


def _is_a_snap_usable(snap: MarketSnapshot) -> bool:
    """A-share fetch is "good" if we got at least stocks. Indices alone are
    too thin — anomalies/sectors need the full universe."""
    return snap.stocks is not None and not snap.stocks.empty


def _macro_has_data(snap: MacroSnapshot) -> bool:
    for df in (snap.indices_global, snap.fx, snap.yields,
                snap.commodities, snap.crypto):
        if df is not None and not df.empty:
            return True
    return False


def _load_cached_a(target: date) -> tuple[MarketSnapshot, date] | None:
    """Return the most recent persisted A-share snapshot strictly before target."""
    with session_scope() as s:
        latest = s.execute(
            select(StockQuote.trade_date)
            .where(StockQuote.market == "cn-a")
            .where(StockQuote.trade_date <= target)
            .order_by(StockQuote.trade_date.desc()).limit(1)
        ).scalar_one_or_none()
        if latest is None:
            return None
        idx = pd.DataFrame([
            {k: getattr(r, k) for k in ("code", "name", "open", "close", "high",
                                          "low", "volume", "amount", "pct_change")}
            for r in s.execute(select(IndexQuote).where(
                (IndexQuote.trade_date == latest) & (IndexQuote.market == "cn-a"))
            ).scalars()
        ])
        stocks = pd.DataFrame([
            {k: getattr(r, k) for k in ("code", "name", "open", "close", "high",
                                          "low", "pre_close", "volume", "amount",
                                          "turnover_rate", "pct_change",
                                          "market_cap", "sector")}
            for r in s.execute(select(StockQuote).where(
                (StockQuote.trade_date == latest) & (StockQuote.market == "cn-a"))
            ).scalars()
        ])
        sectors = pd.DataFrame([
            {k: getattr(r, k) for k in ("sector", "pct_change", "amount",
                                          "leader_code", "leader_name")}
            for r in s.execute(select(SectorQuote).where(SectorQuote.trade_date == latest)
            ).scalars()
        ])
        capital = pd.DataFrame([
            {k: getattr(r, k) for k in ("scope", "metric", "value")}
            for r in s.execute(select(CapitalFlow).where(CapitalFlow.trade_date == latest)
            ).scalars()
        ])
        news = pd.DataFrame([
            {k: getattr(r, k) for k in ("code", "source", "title", "url",
                                          "published_at", "summary")}
            for r in s.execute(select(NewsItem).where(NewsItem.trade_date == latest)
            ).scalars()
        ])
    return MarketSnapshot(
        trade_date=latest, indices=idx, stocks=stocks,
        sectors=sectors, capital=capital, news=news,
    ), latest


def _load_cached_regional(target: date, market: str) -> tuple[RegionalSnapshot, date] | None:
    with session_scope() as s:
        latest = s.execute(
            select(StockQuote.trade_date)
            .where(StockQuote.market == market)
            .where(StockQuote.trade_date <= target)
            .order_by(StockQuote.trade_date.desc()).limit(1)
        ).scalar_one_or_none()
        if latest is None:
            return None
        stocks = pd.DataFrame([
            {k: getattr(r, k) for k in ("code", "name", "open", "close", "high",
                                          "low", "pre_close", "volume", "amount",
                                          "turnover_rate", "pct_change",
                                          "market_cap", "sector")}
            for r in s.execute(select(StockQuote).where(
                (StockQuote.trade_date == latest) & (StockQuote.market == market))
            ).scalars()
        ])
    if stocks.empty:
        return None
    return RegionalSnapshot(trade_date=latest, market=market, stocks=stocks), latest


def _load_cached_macro(target: date) -> tuple[MacroSnapshot, date] | None:
    with session_scope() as s:
        latest = s.execute(
            select(MacroQuote.trade_date)
            .where(MacroQuote.trade_date <= target)
            .order_by(MacroQuote.trade_date.desc()).limit(1)
        ).scalar_one_or_none()
        if latest is None:
            return None
        rows = list(s.execute(select(MacroQuote).where(MacroQuote.trade_date == latest)).scalars())
    by_class: dict[str, list[dict]] = {}
    for r in rows:
        by_class.setdefault(r.asset_class, []).append({
            "code": r.code, "name": r.name,
            "value": r.value, "pct_change": r.pct_change,
            "as_of_ts": r.as_of_ts,
        })
    def _df(key: str) -> pd.DataFrame:
        return pd.DataFrame(by_class.get(key, []))
    snap = MacroSnapshot(
        trade_date=latest,
        indices_global=_df("index_global"),
        fx=_df("fx"),
        yields=_df("yield"),
        commodities=_df("commodity"),
        crypto=_df("crypto"),
    )
    return snap, latest


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

"""Assemble signals + LLM output into the daily report payload, render HTML."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from finai.config import settings
from finai.data.base import MarketSnapshot
from finai.llm.attribution import AttributionResult, attribute_anomalies
from finai.llm.narrative import MarketNarrative, build_market_narrative
from finai.signals import (
    compute_market_overview,
    compute_sector_rotation,
    detect_anomalies,
    find_similar_days,
)

TEMPLATE_DIR = Path(__file__).parent / "templates"


@dataclass
class DailyReportPayload:
    trade_date: date
    generated_at: datetime
    market: dict          # MarketOverview
    sectors: dict         # SectorView
    anomalies: list[dict]
    attributions: list[dict]
    narrative: dict       # MarketNarrative
    similar_days: list[dict]
    fallback_used: bool

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, default=str, indent=2)


def build_daily_report(snapshot: MarketSnapshot) -> DailyReportPayload:
    market = compute_market_overview(
        snapshot.indices, snapshot.stocks, snapshot.capital, snapshot.trade_date
    )
    sectors = compute_sector_rotation(snapshot.sectors, snapshot.trade_date)
    anomalies = detect_anomalies(snapshot.stocks)
    attribution = attribute_anomalies(anomalies, snapshot.news)
    narrative = build_market_narrative(market, sectors)
    similar = find_similar_days(
        snapshot.trade_date,
        lookback=settings.similarity_lookback_days,
        top_k=settings.similarity_top_k,
    )

    return DailyReportPayload(
        trade_date=snapshot.trade_date,
        generated_at=datetime.now(timezone.utc),
        market=_market_dict(market),
        sectors=_sectors_dict(sectors),
        anomalies=[a.as_dict() for a in anomalies],
        attributions=[a.model_dump() for a in attribution.rows],
        narrative=narrative.model_dump(),
        similar_days=[d.as_dict() for d in similar],
        fallback_used=attribution.fallback_used,
    )


def _market_dict(m) -> dict:
    return {
        "trade_date": m.trade_date.isoformat(),
        "indices": m.indices,
        "breadth": m.breadth,
        "limit_up": m.limit_up,
        "limit_down": m.limit_down,
        "limit_up_broken": m.limit_up_broken,
        "consecutive_boards": m.consecutive_boards,
        "capital": m.capital,
        "total_amount_yi": m.total_amount_yi,
    }


def _sectors_dict(s) -> dict:
    return {
        "trade_date": s.trade_date.isoformat(),
        "rising": [r.as_dict() for r in s.rising],
        "falling": [r.as_dict() for r in s.falling],
        "heatmap": s.heatmap,
    }


def render_html(payload: DailyReportPayload) -> str:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("daily.html.j2")
    payload_dict = asdict(payload)
    payload_dict["payload_json"] = payload.to_json()
    return template.render(**payload_dict)


def write_report(payload: DailyReportPayload) -> Path:
    settings.report_dir.mkdir(parents=True, exist_ok=True)
    out = settings.report_dir / f"daily_{payload.trade_date.isoformat()}.html"
    out.write_text(render_html(payload), encoding="utf-8")
    json_out = settings.report_dir / f"daily_{payload.trade_date.isoformat()}.json"
    json_out.write_text(payload.to_json(), encoding="utf-8")
    return out

"""Market narrative: turn structured signals into a 4-paragraph daily brief."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict

from pydantic import BaseModel, Field

from finai.llm.client import LLMUnavailable, get_client
from finai.llm.prompts import NARRATIVE_SYSTEM
from finai.signals.market import MarketOverview
from finai.signals.sector import SectorView

log = logging.getLogger(__name__)


class MarketNarrative(BaseModel):
    overview: str = Field(max_length=180)
    capital: str = Field(max_length=180)
    rotation: str = Field(max_length=180)
    risk: str = Field(max_length=180)


def _fallback(ov: MarketOverview, sv: SectorView) -> MarketNarrative:
    idx_line = "、".join(
        f"{i['name']}{'+' if i['pct_change'] >= 0 else ''}{i['pct_change']:.2f}%"
        for i in ov.indices[:3]
    ) or "指数数据缺失"
    rising = "、".join(s.sector for s in sv.rising[:3]) or "无"
    falling = "、".join(s.sector for s in sv.falling[:3]) or "无"
    north = ov.capital.get("north_net_in", 0) / 1e8
    return MarketNarrative(
        overview=f"{idx_line}。涨{ov.breadth.get('advance', 0)}跌{ov.breadth.get('decline', 0)}，涨停 {ov.limit_up} 跌停 {ov.limit_down}。",
        capital=f"全市场成交 {ov.total_amount_yi:.0f} 亿，北向净流入 {north:.1f} 亿。",
        rotation=f"领涨：{rising}；领跌：{falling}。",
        risk="LLM 不可用，仅展示结构化数据。请关注高位股回撤与板块轮动节奏。",
    )


def build_market_narrative(ov: MarketOverview, sv: SectorView) -> MarketNarrative:
    client = get_client()
    if not client.enabled:
        return _fallback(ov, sv)

    payload = {
        "overview": {
            "trade_date": ov.trade_date.isoformat(),
            "indices": ov.indices,
            "breadth": ov.breadth,
            "limit_up": ov.limit_up,
            "limit_down": ov.limit_down,
            "limit_up_broken": ov.limit_up_broken,
            "total_amount_yi": round(ov.total_amount_yi, 2),
            "capital": {k: round(v / 1e8, 2) for k, v in ov.capital.items()},
        },
        "sectors": {
            "rising": [asdict(r) for r in sv.rising[:6]],
            "falling": [asdict(r) for r in sv.falling[:6]],
        },
    }
    user = (
        "请基于以下当日指标改写为四段日报。\n\n"
        f"<data>\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n</data>"
    )
    try:
        return client.parse(
            system=NARRATIVE_SYSTEM,
            user=user,
            schema=MarketNarrative,
            max_tokens=1024,
        )
    except LLMUnavailable:
        return _fallback(ov, sv)
    except Exception as exc:
        log.warning("narrative failed: %s", exc)
        return _fallback(ov, sv)

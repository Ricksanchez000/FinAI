"""Anomaly attribution: per-stock LLM analysis."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from finai.llm.client import LLMUnavailable, get_client
from finai.llm.prompts import ATTRIBUTION_SYSTEM
from finai.signals.anomaly import AnomalyRow

log = logging.getLogger(__name__)

Dimension = Literal["news", "sector", "capital", "fundamental", "technical"]


class Attribution(BaseModel):
    code: str
    dimensions: list[Dimension] = Field(min_length=1)
    summary: str = Field(max_length=120)
    risk: str = Field(default="", max_length=80)
    cited_news_urls: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


@dataclass
class AttributionResult:
    rows: list[Attribution]
    fallback_used: bool


def _payload(row: AnomalyRow, news_df: pd.DataFrame) -> str:
    related = news_df[news_df["title"].str.contains(row.name, na=False)]
    if related.empty:
        related = news_df[news_df["summary"].str.contains(row.sector, na=False)].head(5)
    news_items = related[["title", "url", "summary"]].head(8).to_dict("records") if not related.empty else []
    snapshot = {
        "code": row.code,
        "name": row.name,
        "sector": row.sector,
        "pct_change": round(row.pct_change, 2),
        "turnover_rate": round(row.turnover_rate, 2),
        "amount_yi": round(row.amount / 1e8, 2),
        "market_cap_yi": round(row.market_cap / 1e8, 2),
        "anomaly_reason": row.reason,
    }
    return (
        "请基于以下数据为该股票给出归因：\n\n"
        f"<snapshot>\n{json.dumps(snapshot, ensure_ascii=False, indent=2)}\n</snapshot>\n\n"
        f"<news>\n{json.dumps(news_items, ensure_ascii=False, indent=2)}\n</news>"
    )


def _fallback(row: AnomalyRow) -> Attribution:
    direction = "上涨" if row.pct_change >= 0 else "下跌"
    return Attribution(
        code=row.code,
        dimensions=["technical"],
        summary=f"{row.name} 当日{direction} {abs(row.pct_change):.2f}%（{row.reason}），暂无 LLM 解读。",
        risk="LLM 不可用，请人工复核。",
        cited_news_urls=[],
        confidence=0.0,
    )


def attribute_anomalies(
    anomalies: list[AnomalyRow],
    news_df: pd.DataFrame,
    *,
    limit: int = 20,
) -> AttributionResult:
    if not anomalies:
        return AttributionResult(rows=[], fallback_used=False)

    client = get_client()
    rows: list[Attribution] = []
    fallback = False

    for row in anomalies[:limit]:
        try:
            attr = client.parse(
                system=ATTRIBUTION_SYSTEM,
                user=_payload(row, news_df),
                schema=Attribution,
                max_tokens=512,
            )
            if attr.code != row.code:
                attr.code = row.code
            rows.append(attr)
        except LLMUnavailable:
            fallback = True
            rows.append(_fallback(row))
        except Exception as exc:
            log.warning("attribution failed for %s: %s", row.code, exc)
            fallback = True
            rows.append(_fallback(row))

    return AttributionResult(rows=rows, fallback_used=fallback)

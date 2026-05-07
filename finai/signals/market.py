"""Market overview indicators."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd


@dataclass
class MarketOverview:
    trade_date: date
    indices: list[dict[str, Any]] = field(default_factory=list)
    breadth: dict[str, int] = field(default_factory=dict)
    limit_up: int = 0
    limit_down: int = 0
    limit_up_broken: int = 0           # 炸板数
    consecutive_boards: list[dict[str, Any]] = field(default_factory=list)  # 连板梯队
    capital: dict[str, float] = field(default_factory=dict)
    total_amount_yi: float = 0.0       # 全市场成交额 (亿)


def _is_limit(pct: float, name: str) -> bool:
    threshold = 19.5 if any(t in name for t in ("ST",)) else 9.7
    if name.startswith(("3", "688")):
        threshold = 19.5
    return pct >= threshold


def compute_market_overview(
    indices_df: pd.DataFrame,
    stocks_df: pd.DataFrame,
    capital_df: pd.DataFrame,
    trade_date: date,
) -> MarketOverview:
    ov = MarketOverview(trade_date=trade_date)

    if not indices_df.empty:
        ov.indices = indices_df.to_dict("records")

    if not stocks_df.empty:
        s = stocks_df.copy()
        ov.breadth = {
            "advance": int((s["pct_change"] > 0).sum()),
            "decline": int((s["pct_change"] < 0).sum()),
            "flat": int((s["pct_change"] == 0).sum()),
        }
        # crude limit detection — scoreboard quality, not regulatory grade
        s["_is_up_limit"] = s.apply(lambda r: _is_limit(r["pct_change"], str(r["name"])), axis=1)
        s["_is_down_limit"] = s.apply(
            lambda r: _is_limit(-r["pct_change"], str(r["name"])), axis=1)
        ov.limit_up = int(s["_is_up_limit"].sum())
        ov.limit_down = int(s["_is_down_limit"].sum())
        # 炸板：日内最高触及涨停但收盘未封板
        if {"high", "pre_close"}.issubset(s.columns):
            high_pct = (s["high"] - s["pre_close"]) / s["pre_close"] * 100
            ov.limit_up_broken = int(((high_pct >= 9.7) & (~s["_is_up_limit"])).sum())
        # 连板梯队 v0：按涨停且涨幅近似 10% 简单分桶（无历史则只能给当日，未来由 history 表补全）
        ov.consecutive_boards = [
            {"boards": 1, "count": ov.limit_up}
        ]
        ov.total_amount_yi = float(s["amount"].sum() / 1e8)

    if not capital_df.empty:
        ov.capital = {
            f"{r['scope']}_{r['metric']}": float(r["value"])
            for _, r in capital_df.iterrows()
        }
    return ov

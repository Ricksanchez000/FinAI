"""Cross-market top-mover boards. Reuses the anomaly detector schema so the
existing report rendering stays uniform.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from finai.data.base import RegionalSnapshot
from finai.signals.anomaly import AnomalyRow


@dataclass
class CrossMarketBoard:
    market: str            # 'us' / 'hk'
    trade_date: date
    gainers: list[dict] = field(default_factory=list)
    losers: list[dict] = field(default_factory=list)


def compute_cross_market_board(snap: RegionalSnapshot, top_n: int = 15) -> CrossMarketBoard:
    board = CrossMarketBoard(market=snap.market, trade_date=snap.trade_date)
    df = snap.stocks
    if df is None or df.empty:
        return board

    df = df.copy()
    # 美股可能没有 turnover_rate / market_cap 时退化为 0
    for col in ("turnover_rate", "market_cap", "amount"):
        if col not in df.columns:
            df[col] = 0.0

    def _row(r: pd.Series, reason: str, score: float) -> dict:
        return AnomalyRow(
            code=str(r["code"]),
            name=str(r["name"]),
            sector=str(r.get("sector", "") or ""),
            pct_change=float(r["pct_change"]),
            turnover_rate=float(r.get("turnover_rate", 0) or 0),
            amount=float(r.get("amount", 0) or 0),
            market_cap=float(r.get("market_cap", 0) or 0),
            reason=reason,  # type: ignore[arg-type]
            score=score,
        ).as_dict()

    gainers = df.nlargest(top_n, "pct_change")
    losers = df.nsmallest(top_n, "pct_change")
    board.gainers = [_row(r, "pct_up", float(r["pct_change"])) for _, r in gainers.iterrows()]
    board.losers = [_row(r, "pct_down", abs(float(r["pct_change"]))) for _, r in losers.iterrows()]
    return board

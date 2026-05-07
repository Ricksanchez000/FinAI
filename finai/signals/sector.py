"""Sector rotation indicators."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date

import pandas as pd


@dataclass
class SectorRow:
    sector: str
    pct_change: float
    amount_yi: float
    leader_code: str
    leader_name: str
    rank: int

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SectorView:
    trade_date: date
    rising: list[SectorRow] = field(default_factory=list)
    falling: list[SectorRow] = field(default_factory=list)
    heatmap: list[dict] = field(default_factory=list)


def compute_sector_rotation(sectors_df: pd.DataFrame, trade_date: date,
                             top_n: int = 8) -> SectorView:
    view = SectorView(trade_date=trade_date)
    if sectors_df.empty:
        return view

    df = sectors_df.copy()
    df = df.sort_values("pct_change", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["amount_yi"] = df["amount"] / 1e8

    def _pack(sub: pd.DataFrame) -> list[SectorRow]:
        return [
            SectorRow(
                sector=str(r["sector"]),
                pct_change=float(r["pct_change"]),
                amount_yi=float(r["amount_yi"]),
                leader_code=str(r.get("leader_code", "")),
                leader_name=str(r.get("leader_name", "")),
                rank=int(r["rank"]),
            )
            for _, r in sub.iterrows()
        ]

    view.rising = _pack(df.head(top_n))
    view.falling = _pack(df.tail(top_n).iloc[::-1])
    view.heatmap = df[["sector", "pct_change", "amount_yi"]].to_dict("records")
    return view

"""Macro snapshot — turn MacroSnapshot frames into a structured view for the report."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from finai.data.base import MacroSnapshot


@dataclass
class MacroRow:
    code: str
    name: str
    value: float
    pct_change: float
    as_of: str  # ISO timestamp (or empty)

    @classmethod
    def from_series(cls, r: pd.Series) -> "MacroRow":
        ts = r.get("as_of_ts")
        as_of = ts.isoformat(timespec="minutes") if pd.notna(ts) else ""
        return cls(
            code=str(r["code"]),
            name=str(r.get("name") or r["code"]),
            value=float(r["value"]),
            pct_change=float(r.get("pct_change", 0.0) or 0.0),
            as_of=as_of,
        )

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class MacroView:
    trade_date: date
    indices_global: list[dict] = field(default_factory=list)
    fx: list[dict] = field(default_factory=list)
    yields: list[dict] = field(default_factory=list)
    commodities: list[dict] = field(default_factory=list)
    crypto: list[dict] = field(default_factory=list)


def _pack(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    return [MacroRow.from_series(r).as_dict() for _, r in df.iterrows()]


def compute_macro_view(snap: MacroSnapshot) -> MacroView:
    return MacroView(
        trade_date=snap.trade_date,
        indices_global=_pack(snap.indices_global),
        fx=_pack(snap.fx),
        yields=_pack(snap.yields),
        commodities=_pack(snap.commodities),
        crypto=_pack(snap.crypto),
    )

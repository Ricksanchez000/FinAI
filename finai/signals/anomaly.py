"""Anomaly detection on the day's stock snapshot."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import pandas as pd

from finai.config import settings


Reason = Literal["pct_up", "pct_down", "turnover_spike", "amount_spike"]


@dataclass
class AnomalyRow:
    code: str
    name: str
    sector: str
    pct_change: float
    turnover_rate: float
    amount: float
    market_cap: float
    reason: Reason
    score: float  # higher = more anomalous

    def as_dict(self) -> dict:
        return asdict(self)


def detect_anomalies(stocks_df: pd.DataFrame, top_n: int = 30) -> list[AnomalyRow]:
    if stocks_df.empty:
        return []

    df = stocks_df.copy()
    rows: list[AnomalyRow] = []

    pct_th = settings.anomaly_pct_threshold

    up = df[df["pct_change"] >= pct_th].nlargest(top_n, "pct_change")
    for _, r in up.iterrows():
        rows.append(_row(r, "pct_up", float(r["pct_change"])))

    down = df[df["pct_change"] <= -pct_th].nsmallest(top_n, "pct_change")
    for _, r in down.iterrows():
        rows.append(_row(r, "pct_down", abs(float(r["pct_change"]))))

    if "turnover_rate" in df.columns and df["turnover_rate"].std() > 0:
        z = (df["turnover_rate"] - df["turnover_rate"].mean()) / df["turnover_rate"].std()
        spike = df.assign(_z=z).nlargest(top_n, "_z")
        spike = spike[spike["_z"] >= settings.anomaly_turnover_z]
        for _, r in spike.iterrows():
            rows.append(_row(r, "turnover_spike", float(r["_z"])))

    if "amount" in df.columns:
        big = df.nlargest(top_n, "amount")
        for _, r in big.iterrows():
            rows.append(_row(r, "amount_spike", float(r["amount"]) / 1e8))

    seen, dedup = set(), []
    for row in sorted(rows, key=lambda x: x.score, reverse=True):
        key = (row.code, row.reason)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(row)
    return dedup[: top_n * 2]


def _row(r: pd.Series, reason: Reason, score: float) -> AnomalyRow:
    return AnomalyRow(
        code=str(r["code"]),
        name=str(r["name"]),
        sector=str(r.get("sector", "")),
        pct_change=float(r["pct_change"]),
        turnover_rate=float(r.get("turnover_rate", 0.0)),
        amount=float(r.get("amount", 0.0)),
        market_cap=float(r.get("market_cap", 0.0)),
        reason=reason,
        score=score,
    )

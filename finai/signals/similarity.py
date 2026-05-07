"""Historical similarity ★ differentiation hook.

Each trading day is encoded as a vector of (per-sector pct_change). Cosine
similarity against the rolling history surfaces "days that rhymed with
today" — the report layer renders these as clickable references so every
LLM observation can be checked against precedent.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import select

from finai.db import session_scope
from finai.models import SectorQuote


@dataclass
class SimilarDay:
    trade_date: date
    similarity: float

    def as_dict(self) -> dict:
        return {"trade_date": self.trade_date.isoformat(), "similarity": self.similarity}


def _vectorize(df: pd.DataFrame) -> tuple[list[date], np.ndarray, list[str]]:
    if df.empty:
        return [], np.empty((0, 0)), []
    pivot = df.pivot_table(index="trade_date", columns="sector",
                            values="pct_change", aggfunc="first").fillna(0.0)
    pivot = pivot.sort_index()
    return list(pivot.index), pivot.to_numpy(dtype=float), list(pivot.columns)


def find_similar_days(target: date, lookback: int = 750, top_k: int = 5) -> list[SimilarDay]:
    start = target - timedelta(days=lookback)
    with session_scope() as s:
        rows = s.execute(
            select(SectorQuote.trade_date, SectorQuote.sector, SectorQuote.pct_change)
            .where(SectorQuote.trade_date >= start)
            .where(SectorQuote.trade_date <= target)
        ).all()

    if not rows:
        return []

    df = pd.DataFrame(rows, columns=["trade_date", "sector", "pct_change"])
    dates, mat, _ = _vectorize(df)
    if not dates or target not in dates:
        return []

    idx = dates.index(target)
    target_vec = mat[idx]
    norms = np.linalg.norm(mat, axis=1)
    tn = norms[idx]
    if tn == 0:
        return []
    sims = (mat @ target_vec) / (norms * tn + 1e-12)

    candidates = [
        (dates[i], float(sims[i])) for i in range(len(dates))
        if dates[i] != target and not np.isnan(sims[i])
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [SimilarDay(trade_date=d, similarity=round(s, 4)) for d, s in candidates[:top_k]]

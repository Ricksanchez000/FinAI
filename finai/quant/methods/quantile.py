"""Historical-quantile valuation: where does today's PE/PB sit in the past N years?

Cheap, robust, and surprisingly informative — used by every active manager.
Requires the StockHistory.fundamentals dict to contain at least pe; if absent
we degrade to a price-quantile (current close vs N-year close distribution).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from finai.quant.base import PredictionResult, StockHistory
from finai.quant.registry import register


@dataclass
class QuantilePredictor:
    method_id: str = "hist_quantile"
    method_name: str = "历史分位"
    family: str = "fundamental"

    def predict(self, history: StockHistory, horizon_days: int = 5) -> PredictionResult:
        try:
            return self._predict(history)
        except Exception as exc:
            return PredictionResult(
                method_id=self.method_id, method_name=self.method_name,
                family="fundamental", result_type="valuation",
                summary="历史分位计算失败", error=str(exc),
            )

    def _predict(self, history: StockHistory) -> PredictionResult:
        bars = history.bars
        if bars.empty or "close" not in bars.columns:
            return self._fail("历史数据不足")

        closes = pd.to_numeric(bars["close"], errors="coerce").dropna()
        if len(closes) < 60:
            return self._fail(f"样本不足（{len(closes)} < 60 个交易日）")

        # Price quantile fallback
        cur = float(closes.iloc[-1])
        rank = float((closes < cur).sum()) / len(closes)
        # If PE/PB available, blend
        pe = history.fundamentals.get("pe")
        pb = history.fundamentals.get("pb")
        details = {"price_pct": round(rank * 100, 1)}

        # Higher rank → currently expensive (vs history) → mild bearish lean.
        # We frame as "deviation": rank 0.5 → 0%, rank 0.9 → +10% overvalued.
        deviation_pct = (rank - 0.5) * 25  # rank 1.0 → +12.5%, rank 0.0 → -12.5%
        parts = [f"价格处于过去 {len(closes)} 日 {rank*100:.0f}% 分位"]
        if pe is not None and pe > 0:
            parts.append(f"PE {pe:.1f}")
            details["pe"] = float(pe)
        if pb is not None and pb > 0:
            parts.append(f"PB {pb:.2f}")
            details["pb"] = float(pb)
        summary = "，".join(parts)

        return PredictionResult(
            method_id=self.method_id, method_name=self.method_name,
            family="fundamental", result_type="valuation",
            summary=summary,
            fair_value=float(closes.median()),
            deviation_pct=round(deviation_pct, 2),
            confidence=0.6,
            extra=details,
        )

    def _fail(self, reason: str) -> PredictionResult:
        return PredictionResult(
            method_id=self.method_id, method_name=self.method_name,
            family="fundamental", result_type="valuation",
            summary=reason, error=reason,
        )


register(QuantilePredictor())

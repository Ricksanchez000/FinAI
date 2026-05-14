"""Trailing returns + Sharpe-style summary.

Not strictly a "predictor" — it's a fact card. But individual investors find
it useful and it's part of the consensus picture: a stock that's down 30%
over 6m + has bearish technical → divergent vs ARIMA forecasting +2% / 5d.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from finai.quant.base import PredictionResult, StockHistory
from finai.quant.registry import register


@dataclass
class MomentumStatsPredictor:
    method_id: str = "momentum_stats"
    method_name: str = "收益率统计"
    family: str = "technical"

    def predict(self, history: StockHistory, horizon_days: int = 5) -> PredictionResult:
        bars = history.bars
        if bars.empty or "close" not in bars.columns:
            return PredictionResult(
                method_id=self.method_id, method_name=self.method_name,
                family="technical", result_type="signal",
                summary="历史数据缺失", error="empty bars",
            )
        c = pd.to_numeric(bars["close"], errors="coerce").dropna().reset_index(drop=True)
        if len(c) < 60:
            return PredictionResult(
                method_id=self.method_id, method_name=self.method_name,
                family="technical", result_type="signal",
                summary="样本不足", error="<60 bars",
            )
        cur = float(c.iloc[-1])

        windows = {"5日": 5, "20日": 20, "60日": 60, "120日": 120, "250日": 250}
        rets: dict[str, float] = {}
        for name, w in windows.items():
            if len(c) > w:
                rets[name] = (cur / c.iloc[-w-1] - 1) * 100

        # Annualized Sharpe using log returns over ~1y
        log_ret = np.log(c / c.shift(1)).dropna()
        recent = log_ret.iloc[-min(250, len(log_ret)):]
        ann_ret = recent.mean() * 252 * 100
        ann_vol = recent.std() * np.sqrt(252) * 100
        sharpe = (ann_ret - 2) / ann_vol if ann_vol > 1e-6 else 0.0  # rf ≈ 2%

        # Score: blended trailing return signal, mildly weighted toward longer
        # windows (markets mean-revert short-term, trend longer-term).
        weights = {"5日": 0.05, "20日": 0.15, "60日": 0.30, "120日": 0.25, "250日": 0.25}
        score_raw = sum(weights.get(k, 0) * (v / 30) for k, v in rets.items())
        score = max(-1.0, min(1.0, score_raw))

        parts = [f"近 1 年年化 {ann_ret:+.1f}%，波动 {ann_vol:.1f}%，Sharpe {sharpe:.2f}"]
        if "20日" in rets and "60日" in rets:
            parts.append(f"20/60 日累计 {rets['20日']:+.1f}% / {rets['60日']:+.1f}%")

        return PredictionResult(
            method_id=self.method_id, method_name=self.method_name,
            family="technical", result_type="signal",
            summary="；".join(parts),
            signal=("bullish" if score > 0.1 else "bearish" if score < -0.1 else "neutral"),
            signal_score=round(score, 3),
            confidence=0.5,
            extra={"trailing_returns_pct": {k: round(v, 2) for k, v in rets.items()},
                    "ann_return_pct": round(ann_ret, 2),
                    "ann_vol_pct": round(ann_vol, 2),
                    "sharpe": round(sharpe, 2)},
        )


register(MomentumStatsPredictor())

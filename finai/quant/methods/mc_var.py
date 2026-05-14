"""Historical Monte Carlo VaR / CVaR.

Resample daily log-returns with replacement, project horizon_days paths,
take the empirical 5% quantile. Not strictly "MC" (we draw from empirical
returns, not parametric), but accurate when returns are fat-tailed and is
the standard pedagogical example for risk.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from finai.quant.base import PredictionResult, StockHistory
from finai.quant.registry import register


@dataclass
class MonteCarloVaRPredictor:
    method_id: str = "mc_var"
    method_name: str = "Monte Carlo VaR"
    family: str = "risk"

    n_paths: int = 5000

    def predict(self, history: StockHistory, horizon_days: int = 5) -> PredictionResult:
        if history.bars.empty or "close" not in history.bars.columns:
            return PredictionResult(
                method_id=self.method_id, method_name=self.method_name,
                family="risk", result_type="risk",
                summary="历史数据缺失", error="empty bars",
            )
        c = pd.to_numeric(history.bars["close"], errors="coerce").dropna()
        if len(c) < 60:
            return PredictionResult(
                method_id=self.method_id, method_name=self.method_name,
                family="risk", result_type="risk",
                summary="样本不足", error="<60 bars",
            )
        log_r = np.log(c / c.shift(1)).dropna().to_numpy()
        rng = np.random.default_rng(seed=int(c.iloc[-1]) % 2**31)
        sampled = rng.choice(log_r, size=(self.n_paths, horizon_days), replace=True)
        cum_log_r = sampled.sum(axis=1)
        path_returns = (np.exp(cum_log_r) - 1) * 100  # pct returns over horizon

        var_95 = float(np.quantile(path_returns, 0.05))   # 5th percentile, negative
        cvar_95 = float(path_returns[path_returns <= var_95].mean())
        median = float(np.median(path_returns))
        p95 = float(np.quantile(path_returns, 0.95))

        cur = float(c.iloc[-1])
        return PredictionResult(
            method_id=self.method_id, method_name=self.method_name,
            family="risk", result_type="risk",
            summary=(f"{horizon_days} 日 95% VaR {var_95:.2f}% / "
                     f"CVaR {cvar_95:.2f}%；中位收益 {median:+.2f}%"),
            var_95_pct=round(var_95, 3),
            cvar_95_pct=round(cvar_95, 3),
            horizon_days=horizon_days,
            lower_band=cur * (1 + var_95 / 100),
            upper_band=cur * (1 + p95 / 100),
            point_forecast=cur * (1 + median / 100),
            forecast_return_pct=round(median, 3),
            confidence=0.65,
            extra={"n_paths": self.n_paths,
                    "horizon_days": horizon_days,
                    "p5_pct": round(var_95, 2),
                    "p50_pct": round(median, 2),
                    "p95_pct": round(p95, 2)},
        )


register(MonteCarloVaRPredictor())

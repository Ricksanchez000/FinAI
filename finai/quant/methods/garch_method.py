"""GARCH(1,1) volatility forecast.

Doesn't predict price direction — predicts how *volatile* the next N days
will be relative to recent history. Practical for sizing: rising forecast
vol → smaller positions, even if your directional view is bullish.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from finai.quant.base import PredictionResult, StockHistory
from finai.quant.registry import register


@dataclass
class GarchPredictor:
    method_id: str = "garch"
    method_name: str = "GARCH 波动率"
    family: str = "timeseries"

    def predict(self, history: StockHistory, horizon_days: int = 5) -> PredictionResult:
        try:
            from arch import arch_model
        except ImportError as exc:
            return PredictionResult(
                method_id=self.method_id, method_name=self.method_name,
                family="timeseries", result_type="risk",
                summary="arch 包未安装", error=str(exc),
            )
        if history.bars.empty or "close" not in history.bars.columns:
            return PredictionResult(
                method_id=self.method_id, method_name=self.method_name,
                family="timeseries", result_type="risk",
                summary="历史数据缺失", error="empty bars",
            )
        c = pd.to_numeric(history.bars["close"], errors="coerce").dropna()
        if len(c) < 120:
            return PredictionResult(
                method_id=self.method_id, method_name=self.method_name,
                family="timeseries", result_type="risk",
                summary="样本不足", error="<120 bars",
            )

        # arch_model expects returns in percent
        ret = c.pct_change().dropna() * 100
        ret = ret.iloc[-500:]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                m = arch_model(ret, vol="Garch", p=1, q=1, dist="normal")
                fit = m.fit(disp="off")
            except Exception as exc:
                return PredictionResult(
                    method_id=self.method_id, method_name=self.method_name,
                    family="timeseries", result_type="risk",
                    summary="GARCH 拟合失败", error=str(exc),
                )
            fc = fit.forecast(horizon=horizon_days, reindex=False)
            var_path = fc.variance.values[-1, :]  # variance, in pct^2 per day

        # Annualized vol for the *terminal* day of the horizon
        terminal_daily_vol = float(np.sqrt(var_path[-1]))    # daily, in pct
        terminal_ann_vol = terminal_daily_vol * np.sqrt(252)  # ann, in pct
        # Compare to recent realized vol over last 20 days
        recent_ann_vol = float(ret.iloc[-20:].std() * np.sqrt(252))
        delta_pct = (terminal_ann_vol - recent_ann_vol) / recent_ann_vol * 100 if recent_ann_vol else 0

        cur = float(c.iloc[-1])
        # 1-σ band over the horizon
        horizon_vol = float(np.sqrt(var_path.sum())) / 100  # convert pct → fraction
        lo = cur * (1 - horizon_vol)
        hi = cur * (1 + horizon_vol)

        regime = "上升" if delta_pct > 10 else ("下降" if delta_pct < -10 else "平稳")

        return PredictionResult(
            method_id=self.method_id, method_name=self.method_name,
            family="timeseries", result_type="risk",
            summary=(f"未来 {horizon_days} 日年化波动 {terminal_ann_vol:.1f}%"
                     f" (近 20 日 {recent_ann_vol:.1f}%，{regime})"),
            horizon_days=horizon_days,
            lower_band=round(lo, 4),
            upper_band=round(hi, 4),
            confidence=0.55,
            extra={"forecast_ann_vol_pct": round(terminal_ann_vol, 2),
                    "recent_ann_vol_pct": round(recent_ann_vol, 2),
                    "vol_regime": regime,
                    "vol_delta_pct": round(delta_pct, 2)},
        )


register(GarchPredictor())

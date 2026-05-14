"""ARIMA(p,d,q) on log-price.

We fit a small grid (p,q in [0,2], d=1) by AIC and forecast horizon_days.
Bands are the model's own 95% CI. ARIMA is famously bad at predicting stock
*direction* but useful as a baseline — its disagreement with technical or
ML methods is itself the signal.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from finai.quant.base import PredictionResult, StockHistory
from finai.quant.registry import register

log = logging.getLogger(__name__)


@dataclass
class ArimaPredictor:
    method_id: str = "arima"
    method_name: str = "ARIMA"
    family: str = "timeseries"

    def predict(self, history: StockHistory, horizon_days: int = 5) -> PredictionResult:
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError as exc:
            return PredictionResult(
                method_id=self.method_id, method_name=self.method_name,
                family="timeseries", result_type="forecast",
                summary="statsmodels 未安装", error=str(exc),
            )

        if history.bars.empty or "close" not in history.bars.columns:
            return PredictionResult(
                method_id=self.method_id, method_name=self.method_name,
                family="timeseries", result_type="forecast",
                summary="历史数据缺失", error="empty bars",
            )
        c = pd.to_numeric(history.bars["close"], errors="coerce").dropna()
        if len(c) < 120:
            return PredictionResult(
                method_id=self.method_id, method_name=self.method_name,
                family="timeseries", result_type="forecast",
                summary="样本不足", error="<120 bars",
            )
        # Fit on the last 250 bars — older data probably doesn't reflect
        # current regime, and full-history fits get slow.
        log_c = np.log(c.iloc[-250:].astype(float))

        best_aic = float("inf")
        best_fit = None
        best_order = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for p in range(3):
                for q in range(3):
                    try:
                        fit = ARIMA(log_c, order=(p, 1, q)).fit()
                        if fit.aic < best_aic:
                            best_aic = fit.aic
                            best_fit = fit
                            best_order = (p, 1, q)
                    except Exception:
                        continue

        if best_fit is None:
            return PredictionResult(
                method_id=self.method_id, method_name=self.method_name,
                family="timeseries", result_type="forecast",
                summary="ARIMA 拟合失败", error="no order converged",
            )

        fc = best_fit.get_forecast(steps=horizon_days)
        mean_log = fc.predicted_mean
        ci = fc.conf_int(alpha=0.05)
        mean_price = float(np.exp(mean_log.iloc[-1]))
        lo = float(np.exp(ci.iloc[-1, 0]))
        hi = float(np.exp(ci.iloc[-1, 1]))
        cur = float(c.iloc[-1])
        ret_pct = (mean_price - cur) / cur * 100

        return PredictionResult(
            method_id=self.method_id, method_name=self.method_name,
            family="timeseries", result_type="forecast",
            summary=(f"ARIMA{best_order} 预测 {horizon_days} 日"
                     f"价格 ¥{mean_price:.2f} ({ret_pct:+.2f}%)，"
                     f"95% CI ¥{lo:.2f}-¥{hi:.2f}"),
            horizon_days=horizon_days,
            point_forecast=round(mean_price, 4),
            forecast_return_pct=round(ret_pct, 3),
            lower_band=round(lo, 4), upper_band=round(hi, 4),
            confidence=0.4,
            fit_metrics={"aic": round(float(best_aic), 2)},
            extra={"order": list(best_order)},
        )


register(ArimaPredictor())

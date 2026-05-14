"""Predictor contract.

Every method (ARIMA, GARCH, technical, etc.) implements `Predictor` and
returns the same `PredictionResult` shape. The dashboard groups by family,
the LLM synthesis reasons over the union, and the comparison view spots
agreement/divergence across methods.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Literal, Protocol, runtime_checkable

import pandas as pd


MethodFamily = Literal[
    "timeseries",      # ARIMA, GARCH, VAR
    "ml",              # LSTM, XGBoost, Prophet
    "technical",       # 30-indicator composite
    "fundamental",     # PE/PB quantile, FF5
    "risk",            # MC VaR, stress test
    "institutional",   # 北向, 龙虎榜, 持仓变动
]

ResultType = Literal["forecast", "valuation", "signal", "risk"]


@dataclass
class StockHistory:
    """Standardized input to every predictor. AkShare-shaped K-line.

    Columns: trade_date, open, high, low, close, volume, amount, pct_change.
    Optionally a `fundamentals` dict with PE, PB, ROE, market_cap.
    """
    symbol: str
    name: str
    bars: pd.DataFrame          # OHLCV daily
    as_of: date
    fundamentals: dict[str, float] = field(default_factory=dict)


@dataclass
class PredictionResult:
    method_id: str           # 'arima' / 'mc_var' / 'tech_composite'
    method_name: str         # human-readable, shows on the card
    family: MethodFamily
    result_type: ResultType

    summary: str             # one-line human readable, shows under the card
    confidence: float | None = None  # 0-1, how much to trust this result

    # forecast type — projected price path
    horizon_days: int | None = None
    point_forecast: float | None = None   # predicted price at horizon
    forecast_return_pct: float | None = None
    lower_band: float | None = None
    upper_band: float | None = None

    # valuation type — current vs fair
    fair_value: float | None = None
    deviation_pct: float | None = None    # +ve = overvalued, -ve = undervalued

    # signal type — discrete direction
    signal: Literal["bullish", "bearish", "neutral"] | None = None
    signal_score: float | None = None     # -1 to +1

    # risk type — downside
    var_95_pct: float | None = None       # 1-day 95% VaR as negative pct
    cvar_95_pct: float | None = None

    # diagnostics & metadata
    fit_metrics: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    error: str | None = None              # set when the method failed; UI shows greyed out

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def direction_score(self) -> float | None:
        """Project to a uniform [-1, +1] axis so we can compute consensus.

        +1 = strongly bullish, -1 = strongly bearish, 0 = neutral.
        Returns None when the method's output doesn't have a direction
        (e.g. pure volatility forecast).
        """
        if self.error:
            return None
        if self.signal_score is not None:
            return max(-1.0, min(1.0, self.signal_score))
        if self.forecast_return_pct is not None:
            # >=10% over horizon → +1; >=5% → +0.5; etc.
            return max(-1.0, min(1.0, self.forecast_return_pct / 10.0))
        if self.deviation_pct is not None:
            # overvalued → bearish, undervalued → bullish
            return max(-1.0, min(1.0, -self.deviation_pct / 15.0))
        return None


@runtime_checkable
class Predictor(Protocol):
    method_id: str
    method_name: str
    family: MethodFamily

    def predict(self, history: StockHistory, horizon_days: int = 5) -> PredictionResult: ...

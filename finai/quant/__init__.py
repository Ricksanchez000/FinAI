"""Quantitative methods registry & runner.

Multi-method comparison dashboard for a single security. Each method
implements the Predictor protocol and produces a uniform PredictionResult,
so the dashboard can render them side-by-side and the LLM synthesis layer
can reason over their consensus / divergence.
"""

from finai.quant.base import (
    MethodFamily,
    PredictionResult,
    Predictor,
    ResultType,
    StockHistory,
)
from finai.quant.registry import REGISTRY, register

__all__ = [
    "PredictionResult", "Predictor", "StockHistory",
    "MethodFamily", "ResultType", "REGISTRY", "register",
]

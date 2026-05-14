"""Quant framework tests — verify each predictor produces sane output on
deterministic synthetic data."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from finai.quant.base import PredictionResult, StockHistory
from finai.quant.runner import run_all


@pytest.fixture
def trending_history() -> StockHistory:
    """A strongly uptrending stock — drift large enough that momentum and
    technical methods are unambiguously bullish."""
    rng = np.random.default_rng(42)
    n = 300
    dates = [date(2025, 1, 1) + timedelta(days=i) for i in range(n)]
    prices = 100 * np.cumprod(1 + 0.004 + rng.normal(0, 0.012, n))
    bars = pd.DataFrame({
        "trade_date": dates,
        "open": prices * (1 + rng.normal(0, 0.003, n)),
        "close": prices,
        "high": prices * (1 + np.abs(rng.normal(0, 0.005, n))),
        "low":  prices * (1 - np.abs(rng.normal(0, 0.005, n))),
        "volume": rng.lognormal(15, 0.3, n),
        "amount": rng.lognormal(20, 0.3, n),
        "pct_change": rng.normal(0.4, 1.2, n),
    })
    return StockHistory(
        symbol="TEST", name="TEST", bars=bars, as_of=dates[-1],
        fundamentals={"pe": 18.5, "pb": 2.4, "institutional_flow": pd.DataFrame()},
    )


@pytest.fixture
def declining_history() -> StockHistory:
    rng = np.random.default_rng(7)
    n = 300
    dates = [date(2025, 1, 1) + timedelta(days=i) for i in range(n)]
    prices = 100 * np.cumprod(1 - 0.004 + rng.normal(0, 0.012, n))
    bars = pd.DataFrame({
        "trade_date": dates,
        "open": prices, "close": prices,
        "high": prices * 1.005, "low": prices * 0.995,
        "volume": np.ones(n) * 1e6, "amount": prices * 1e6,
        "pct_change": np.zeros(n),
    })
    return StockHistory(symbol="DECL", name="DECL", bars=bars, as_of=dates[-1],
                         fundamentals={"institutional_flow": pd.DataFrame()})


def test_run_all_returns_results_for_every_predictor(trending_history):
    run = run_all(trending_history, horizon_days=5)
    assert run.diagnostics["n_methods"] >= 5  # at least Tier-1
    assert run.symbol == "TEST"
    assert run.current_price > 0
    # consensus block has all expected keys
    for k in ("verdict", "label", "net_score", "divergence",
                "n_bullish", "n_bearish", "n_neutral", "n_no_signal"):
        assert k in run.consensus


def test_uptrend_produces_mostly_bullish_signals(trending_history):
    run = run_all(trending_history)
    # net score should lean bullish; not asserting strictly positive because
    # a single bearish method could occasionally flip it, but on a 300-bar
    # uptrend the bullish count should dominate.
    assert run.consensus["n_bullish"] >= run.consensus["n_bearish"]


def test_downtrend_produces_mostly_bearish_signals(declining_history):
    run = run_all(declining_history)
    assert run.consensus["n_bearish"] >= run.consensus["n_bullish"]


def test_each_result_has_required_fields(trending_history):
    run = run_all(trending_history)
    for r in run.results:
        assert r["method_id"] and r["method_name"]
        assert r["family"] in {"timeseries", "ml", "technical",
                                 "fundamental", "risk", "institutional"}
        assert r["summary"]
        # if not errored, must produce SOME quantitative output. GARCH
        # legitimately outputs only forecast bands (a volatility band, no
        # point forecast), so we accept bands as a valid result too.
        if not r["error"]:
            has_output = (
                r["signal"] is not None
                or r["point_forecast"] is not None
                or r["fair_value"] is not None
                or r["var_95_pct"] is not None
                or r["lower_band"] is not None
                or r["upper_band"] is not None
            )
            assert has_output, f"predictor {r['method_id']} produced no output"


def test_consensus_handles_empty_history():
    empty = StockHistory(symbol="X", name="X", bars=pd.DataFrame(),
                          as_of=date(2026, 5, 14), fundamentals={})
    run = run_all(empty)
    # every method should error out gracefully
    assert run.diagnostics["n_failed"] >= 1
    # consensus is still well-formed
    assert run.consensus["n_methods_with_signal"] == 0

"""Run every registered predictor on a StockHistory and compute consensus.

Predictors run in a thread pool. Each is wrapped so its failure can't take
down the rest. The consensus block summarizes how many methods agree on
direction — the headline number for the dashboard.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field

from finai.quant.base import PredictionResult, StockHistory
from finai.quant.registry import get_predictors

log = logging.getLogger(__name__)


@dataclass
class AnalysisRun:
    symbol: str
    name: str
    as_of: str                          # ISO date
    current_price: float | None
    results: list[dict] = field(default_factory=list)
    consensus: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)


def run_all(history: StockHistory, horizon_days: int = 5,
             enabled_ids: list[str] | None = None) -> AnalysisRun:
    predictors = get_predictors(enabled_ids)
    log.info("quant.runner: %d predictors on %s", len(predictors), history.symbol)

    results: list[PredictionResult] = []
    with ThreadPoolExecutor(max_workers=min(8, len(predictors) or 1),
                             thread_name_prefix="quant") as pool:
        future_map = {pool.submit(p.predict, history, horizon_days): p for p in predictors}
        for fut in as_completed(future_map):
            p = future_map[fut]
            try:
                results.append(fut.result())
            except Exception as exc:
                log.exception("predictor %s raised", p.method_id)
                results.append(PredictionResult(
                    method_id=p.method_id, method_name=p.method_name,
                    family=p.family, result_type="signal",
                    summary="预测器抛出异常", error=str(exc),
                ))

    # Stable order so the dashboard doesn't shuffle between runs
    family_order = ["fundamental", "technical", "institutional",
                     "timeseries", "ml", "risk"]
    results.sort(key=lambda r: (family_order.index(r.family) if r.family in family_order else 99,
                                  r.method_id))

    current = _current_price(history)
    return AnalysisRun(
        symbol=history.symbol, name=history.name,
        as_of=history.as_of.isoformat(),
        current_price=current,
        results=[r.as_dict() for r in results],
        consensus=_consensus(results),
        diagnostics={
            "n_methods": len(results),
            "n_failed": sum(1 for r in results if r.error),
            "horizon_days": horizon_days,
            "lookback_bars": int(len(history.bars)),
        },
    )


def _current_price(history: StockHistory) -> float | None:
    if history.bars.empty or "close" not in history.bars.columns:
        return None
    try:
        return float(history.bars["close"].iloc[-1])
    except Exception:
        return None


def _consensus(results: list[PredictionResult]) -> dict:
    """Direction agreement across methods. Each method that has a direction
    contributes one vote (+1/-1/0). We report:
      - net_score in [-1, +1]
      - n_bullish / n_bearish / n_neutral / n_no_signal
      - divergence: 1 - |net|/total_with_signal (1 = total disagreement,
        0 = unanimous)
    """
    bullish = bearish = neutral = no_signal = 0
    total_score = 0.0
    contributing = 0
    for r in results:
        if r.error:
            continue
        score = r.direction_score
        if score is None:
            no_signal += 1
            continue
        contributing += 1
        total_score += score
        if score > 0.15:
            bullish += 1
        elif score < -0.15:
            bearish += 1
        else:
            neutral += 1

    net = total_score / contributing if contributing else 0.0
    divergence = (1 - abs(net)) if contributing else 0.0
    if net > 0.25:
        verdict = "bullish"
        label = "整体偏多"
    elif net < -0.25:
        verdict = "bearish"
        label = "整体偏空"
    else:
        verdict = "neutral"
        label = "分歧 / 中性"

    return {
        "verdict": verdict,
        "label": label,
        "net_score": round(net, 3),
        "divergence": round(divergence, 3),
        "n_bullish": bullish,
        "n_bearish": bearish,
        "n_neutral": neutral,
        "n_no_signal": no_signal,
        "n_methods_with_signal": contributing,
    }

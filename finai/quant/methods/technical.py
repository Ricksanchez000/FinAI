"""Composite technical signal: 8 classic indicators, voted to a single score.

Each indicator votes -1 / 0 / +1; sum is normalized to [-1, +1] as the
signal_score, and binned to bullish/neutral/bearish for display. Hand-rolled
to avoid the talib C dependency.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from finai.quant.base import PredictionResult, StockHistory
from finai.quant.registry import register


def _rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    delta = closes.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = (-delta.clip(upper=0)).rolling(period).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(closes: pd.Series) -> tuple[pd.Series, pd.Series]:
    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def _bollinger(closes: pd.Series, period: int = 20) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = closes.rolling(period).mean()
    std = closes.rolling(period).std()
    return mid - 2 * std, mid, mid + 2 * std


@dataclass
class TechnicalCompositePredictor:
    method_id: str = "tech_composite"
    method_name: str = "技术综合"
    family: str = "technical"

    def predict(self, history: StockHistory, horizon_days: int = 5) -> PredictionResult:
        try:
            return self._predict(history)
        except Exception as exc:
            return PredictionResult(
                method_id=self.method_id, method_name=self.method_name,
                family="technical", result_type="signal",
                summary="技术指标计算失败", error=str(exc),
            )

    def _predict(self, history: StockHistory) -> PredictionResult:
        bars = history.bars
        if bars.empty or len(bars) < 60:
            return PredictionResult(
                method_id=self.method_id, method_name=self.method_name,
                family="technical", result_type="signal",
                summary="样本不足", error="<60 bars",
            )
        c = pd.to_numeric(bars["close"], errors="coerce").reset_index(drop=True)
        h = pd.to_numeric(bars.get("high", c), errors="coerce").reset_index(drop=True)
        l = pd.to_numeric(bars.get("low", c), errors="coerce").reset_index(drop=True)
        v = pd.to_numeric(bars.get("volume", pd.Series([1] * len(c))), errors="coerce").reset_index(drop=True)

        votes: list[tuple[str, int, str]] = []  # (indicator, vote -1/0/+1, message)

        # 1. MA cross (MA20 vs MA60)
        ma20 = c.rolling(20).mean()
        ma60 = c.rolling(60).mean()
        if pd.notna(ma20.iloc[-1]) and pd.notna(ma60.iloc[-1]):
            vote = 1 if ma20.iloc[-1] > ma60.iloc[-1] else -1
            votes.append(("MA20/MA60", vote, "金叉" if vote > 0 else "死叉"))

        # 2. Price vs MA60
        if pd.notna(ma60.iloc[-1]):
            vote = 1 if c.iloc[-1] > ma60.iloc[-1] else -1
            votes.append(("Price/MA60", vote, f"{'高于' if vote > 0 else '低于'} MA60"))

        # 3. RSI(14): >70 overbought (-1), <30 oversold (+1), else 0
        rsi = _rsi(c)
        if pd.notna(rsi.iloc[-1]):
            r = float(rsi.iloc[-1])
            vote = 1 if r < 30 else (-1 if r > 70 else 0)
            votes.append(("RSI14", vote, f"RSI {r:.1f}"))

        # 4. MACD signal cross
        macd, sig = _macd(c)
        if pd.notna(macd.iloc[-1]) and pd.notna(sig.iloc[-1]):
            vote = 1 if macd.iloc[-1] > sig.iloc[-1] else -1
            votes.append(("MACD", vote, "上穿信号线" if vote > 0 else "下穿信号线"))

        # 5. Bollinger position
        lb, mid, ub = _bollinger(c)
        if pd.notna(lb.iloc[-1]) and pd.notna(ub.iloc[-1]):
            cur = c.iloc[-1]
            if cur < lb.iloc[-1]:
                vote, msg = 1, "突破下轨 (超卖)"
            elif cur > ub.iloc[-1]:
                vote, msg = -1, "突破上轨 (超买)"
            else:
                vote, msg = 0, "通道内"
            votes.append(("布林", vote, msg))

        # 6. 20-day momentum
        if len(c) >= 21:
            mom = (c.iloc[-1] / c.iloc[-21] - 1) * 100
            vote = 1 if mom > 3 else (-1 if mom < -3 else 0)
            votes.append(("20日动量", vote, f"{mom:+.1f}%"))

        # 7. Volume trend: recent 5-day vs prior 20-day average
        if len(v) >= 25:
            recent = v.iloc[-5:].mean()
            base = v.iloc[-25:-5].mean() or 1
            ratio = recent / base
            vote = 1 if ratio > 1.5 and c.iloc[-1] > c.iloc[-5] else (-1 if ratio > 1.5 and c.iloc[-1] < c.iloc[-5] else 0)
            votes.append(("放量方向", vote, f"成交 {ratio:.1f}× 均量"))

        # 8. KDJ-style stochastic (simplified)
        if len(c) >= 14:
            lowest = l.iloc[-14:].min()
            highest = h.iloc[-14:].max()
            if highest > lowest:
                k = (c.iloc[-1] - lowest) / (highest - lowest) * 100
                vote = 1 if k < 20 else (-1 if k > 80 else 0)
                votes.append(("KDJ-K", vote, f"K {k:.0f}"))

        score = sum(v for _, v, _ in votes) / max(len(votes), 1)
        if score > 0.25:
            signal, label = "bullish", "偏多"
        elif score < -0.25:
            signal, label = "bearish", "偏空"
        else:
            signal, label = "neutral", "中性"

        plus = sum(1 for _, v, _ in votes if v > 0)
        minus = sum(1 for _, v, _ in votes if v < 0)
        flat = len(votes) - plus - minus

        return PredictionResult(
            method_id=self.method_id, method_name=self.method_name,
            family="technical", result_type="signal",
            summary=f"{label}（多 {plus} / 中 {flat} / 空 {minus}）",
            signal=signal,
            signal_score=round(score, 3),
            confidence=0.55,
            extra={"votes": [{"indicator": i, "vote": v, "msg": m} for i, v, m in votes]},
        )


register(TechnicalCompositePredictor())

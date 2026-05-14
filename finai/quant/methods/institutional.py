"""Institutional flow read.

Two signals when available:
  - 近 N 日北向资金对该股的净流入方向
  - 近 N 日大单/主力净流入累计

Both are *flow* not *holding* — gives a short-term institutional sentiment
read. Holdings (公募基金, 股东户数) update quarterly so they're not in v1.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from finai.quant.base import PredictionResult, StockHistory
from finai.quant.registry import register


@dataclass
class InstitutionalFlowPredictor:
    method_id: str = "institutional_flow"
    method_name: str = "机构资金"
    family: str = "institutional"

    def predict(self, history: StockHistory, horizon_days: int = 5) -> PredictionResult:
        flows: pd.DataFrame = history.fundamentals.get("institutional_flow")  # type: ignore[assignment]
        if not isinstance(flows, pd.DataFrame) or flows.empty:
            return PredictionResult(
                method_id=self.method_id, method_name=self.method_name,
                family="institutional", result_type="signal",
                summary="该标的暂无可用机构资金数据",
                error="no institutional flow data available",
            )

        # Expected columns: trade_date, north_net_in (亿), main_net_in (亿)
        recent = flows.tail(20)
        north5 = recent.tail(5)["north_net_in"].sum() if "north_net_in" in recent else 0
        north20 = recent["north_net_in"].sum() if "north_net_in" in recent else 0
        main5 = recent.tail(5)["main_net_in"].sum() if "main_net_in" in recent else 0
        main20 = recent["main_net_in"].sum() if "main_net_in" in recent else 0

        # Composite score: both flows pointing same direction → strong
        north_score = 1 if north5 > 0.3 else (-1 if north5 < -0.3 else 0)
        main_score = 1 if main5 > 1 else (-1 if main5 < -1 else 0)
        score = (north_score + main_score) / 2

        msg_parts = []
        if "north_net_in" in recent:
            msg_parts.append(f"北向 5/20 日净 {north5:+.1f} / {north20:+.1f} 亿")
        if "main_net_in" in recent:
            msg_parts.append(f"主力 5/20 日净 {main5:+.1f} / {main20:+.1f} 亿")

        return PredictionResult(
            method_id=self.method_id, method_name=self.method_name,
            family="institutional", result_type="signal",
            summary="；".join(msg_parts) if msg_parts else "无资金流数据",
            signal=("bullish" if score > 0.25 else "bearish" if score < -0.25 else "neutral"),
            signal_score=round(score, 3),
            confidence=0.5,
            extra={"north_5d_yi": round(float(north5), 2),
                    "north_20d_yi": round(float(north20), 2),
                    "main_5d_yi": round(float(main5), 2),
                    "main_20d_yi": round(float(main20), 2)},
        )


register(InstitutionalFlowPredictor())

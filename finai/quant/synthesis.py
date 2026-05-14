"""LLM synthesis: read all method outputs, write 3-sentence consensus story.

Strict no-recommendation policy: never says "买" or "卖" — describes what
the models say and where they disagree. Falls back to template when LLM is
unavailable.
"""
from __future__ import annotations

import json
import logging

from pydantic import BaseModel, Field

from finai.llm.client import LLMUnavailable, get_client
from finai.quant.runner import AnalysisRun

log = logging.getLogger(__name__)

SYNTHESIS_SYSTEM = """\
你是一名严谨的量化分析助手。用户给你 N 个不同方法对同一只股票的结果，
你需要写一段克制、不带情绪、不给出买卖建议的综合判断。

强约束：
1. 你只描述模型说了什么，不预测涨跌，不写"建议买入/卖出/持有"。
2. 你不计算任何数字，全部从输入引用。
3. 必须明确指出方法之间的分歧（如有）。
4. 不超过 4 句话。
5. 不使用感叹号或夸张词汇。

输出结构化 JSON。
"""


class Synthesis(BaseModel):
    headline: str = Field(max_length=80)        # 一句话总结
    body: str = Field(max_length=320)            # 2-3 句详述
    divergence_note: str = Field(default="", max_length=140)  # 分歧点
    risk_note: str = Field(default="", max_length=140)


def _fallback(run: AnalysisRun) -> Synthesis:
    c = run.consensus
    head = f"{c['n_methods_with_signal']} 个有方向的方法中：{c['n_bullish']} 偏多 / {c['n_bearish']} 偏空 / {c['n_neutral']} 中性。"
    body_parts = []
    for r in run.results[:6]:
        if not r.get("error"):
            body_parts.append(f"{r['method_name']}：{r['summary']}")
    body = "；".join(body_parts)[:320]
    return Synthesis(
        headline=head,
        body=body or "无足够数据可解读",
        divergence_note=f"分歧度 {c['divergence']:.2f}（0 = 一致，1 = 完全分歧）",
        risk_note="LLM 未启用，仅模板拼接。",
    )


def synthesize(run: AnalysisRun) -> Synthesis:
    client = get_client()
    if not client.enabled:
        return _fallback(run)
    payload = {
        "symbol": run.symbol, "name": run.name, "as_of": run.as_of,
        "current_price": run.current_price,
        "consensus": run.consensus,
        "results": [
            {k: v for k, v in r.items()
             if k in ("method_id", "method_name", "family", "result_type",
                       "summary", "signal", "signal_score",
                       "forecast_return_pct", "fair_value", "deviation_pct",
                       "var_95_pct", "confidence")}
            for r in run.results if not r.get("error")
        ],
    }
    user = ("请基于以下多方法量化结果给出综合判断。\n\n"
            f"<data>\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n</data>")
    try:
        return client.parse(system=SYNTHESIS_SYSTEM, user=user,
                             schema=Synthesis, max_tokens=1024)
    except LLMUnavailable:
        return _fallback(run)
    except Exception as exc:
        log.warning("synthesis failed: %s", exc)
        return _fallback(run)

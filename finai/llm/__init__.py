"""LLM-powered narrative & attribution layer.

Design rules (enforced in code, not just docs):
- LLM never does arithmetic — all numbers come from the signal layer.
- Outputs are structured (Pydantic) so the report renderer can fail loudly.
- Every claim must reference a data id (anomaly code, news url) so the
  reader can drill back to the source.
- Disabled gracefully when ANTHROPIC_API_KEY is missing.
"""

from finai.llm.client import LLMClient, LLMUnavailable
from finai.llm.attribution import attribute_anomalies, AttributionResult
from finai.llm.narrative import build_market_narrative, MarketNarrative

__all__ = [
    "LLMClient", "LLMUnavailable",
    "attribute_anomalies", "AttributionResult",
    "build_market_narrative", "MarketNarrative",
]

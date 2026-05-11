"""Pure-pandas signal computations.

Each module returns a typed dataclass / dict so the report layer never has
to touch raw frames. Signals are deterministic given a snapshot — easy to
unit test, easy to cache.
"""

from finai.signals.market import compute_market_overview, MarketOverview
from finai.signals.anomaly import detect_anomalies, AnomalyRow
from finai.signals.sector import compute_sector_rotation, SectorView
from finai.signals.similarity import find_similar_days, SimilarDay
from finai.signals.global_macro import compute_macro_view, MacroView
from finai.signals.cross_market import compute_cross_market_board, CrossMarketBoard

__all__ = [
    "compute_market_overview", "MarketOverview",
    "detect_anomalies", "AnomalyRow",
    "compute_sector_rotation", "SectorView",
    "find_similar_days", "SimilarDay",
    "compute_macro_view", "MacroView",
    "compute_cross_market_board", "CrossMarketBoard",
]

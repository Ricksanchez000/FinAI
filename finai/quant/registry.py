"""Predictor registry. Methods register themselves on import.

Why a registry: lets users disable individual methods via config without
touching code, and the runner can simply iterate REGISTRY.values().
"""
from __future__ import annotations

import logging

from finai.quant.base import Predictor

log = logging.getLogger(__name__)

REGISTRY: dict[str, Predictor] = {}


def register(predictor: Predictor) -> Predictor:
    if predictor.method_id in REGISTRY:
        log.warning("predictor %s already registered; overriding", predictor.method_id)
    REGISTRY[predictor.method_id] = predictor
    return predictor


def get_predictors(enabled_ids: list[str] | None = None) -> list[Predictor]:
    _bootstrap()
    if enabled_ids is None:
        return list(REGISTRY.values())
    return [REGISTRY[i] for i in enabled_ids if i in REGISTRY]


_bootstrapped = False


def _bootstrap() -> None:
    """Import all method modules so they self-register. Heavy imports
    (statsmodels, prophet) are deferred until the registry is first used.
    """
    global _bootstrapped
    if _bootstrapped:
        return
    _bootstrapped = True
    # Importing the module triggers its `register(...)` call at bottom.
    from finai.quant.methods import (  # noqa: F401
        quantile, technical, momentum, mc_var, institutional,
        arima_method, garch_method,
    )

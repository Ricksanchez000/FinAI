"""Build & render the multi-method analysis page."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from finai.config import settings
from finai.quant.base import StockHistory
from finai.quant.runner import AnalysisRun, run_all
from finai.quant.synthesis import synthesize

TEMPLATE_DIR = Path(__file__).parent / "templates"


@dataclass
class AnalyzePayload:
    run: dict
    synthesis: dict
    kline: list[dict]              # last 180 bars: date, open, close, low, high
    fail_count: int
    generated_at: str


def build_analyze(history: StockHistory, horizon_days: int = 5) -> AnalyzePayload:
    run = run_all(history, horizon_days=horizon_days)
    syn = synthesize(run)

    bars = history.bars.tail(180) if not history.bars.empty else history.bars
    kline = []
    for _, r in bars.iterrows():
        kline.append({
            "date": r["trade_date"].isoformat() if hasattr(r["trade_date"], "isoformat") else str(r["trade_date"]),
            "open": float(r.get("open", 0) or 0),
            "close": float(r.get("close", 0) or 0),
            "low": float(r.get("low", 0) or 0),
            "high": float(r.get("high", 0) or 0),
        })
    return AnalyzePayload(
        run=asdict(run),
        synthesis=syn.model_dump(),
        kline=kline,
        fail_count=run.diagnostics.get("n_failed", 0),
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def render_analyze(payload: AnalyzePayload) -> str:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("analyze.html.j2")
    return template.render(**asdict(payload))


def write_analyze(payload: AnalyzePayload) -> Path:
    settings.report_dir.mkdir(parents=True, exist_ok=True)
    sym = payload.run["symbol"]
    out = settings.report_dir / f"analyze_{sym}_{payload.run['as_of']}.html"
    out.write_text(render_analyze(payload), encoding="utf-8")
    return out

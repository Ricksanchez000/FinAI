"""FastAPI server: serves rendered reports + structured JSON for the dashboard."""
from __future__ import annotations

import json
from datetime import date

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy import desc, select

from finai.db import init_db, session_scope
from finai.models import DailyReport
from finai.pipeline.etl import run_full_pipeline

app = FastAPI(title="FinAI", version="0.1.0")


@app.on_event("startup")
def _on_startup() -> None:
    init_db()


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/reports", response_model=list[dict])
def list_reports(limit: int = 30) -> list[dict]:
    with session_scope() as s:
        rows = s.execute(
            select(DailyReport).order_by(desc(DailyReport.trade_date)).limit(limit)
        ).scalars().all()
        return [
            {"trade_date": r.trade_date.isoformat(),
             "generated_at": r.generated_at.isoformat(),
             "html_path": r.html_path}
            for r in rows
        ]


@app.get("/reports/latest", response_class=HTMLResponse)
def latest_report() -> str:
    with session_scope() as s:
        row = s.execute(
            select(DailyReport).order_by(desc(DailyReport.trade_date)).limit(1)
        ).scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="no reports yet, run pipeline first")
    try:
        return open(row.html_path, encoding="utf-8").read()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/reports/{trade_date}")
def get_report(trade_date: date) -> dict:
    with session_scope() as s:
        row = s.execute(
            select(DailyReport).where(DailyReport.trade_date == trade_date)
        ).scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="not found")
    return json.loads(row.payload_json)


@app.post("/pipeline/run")
def trigger_run(trade_date: date | None = None, source: str | None = None) -> dict:
    path = run_full_pipeline(trade_date=trade_date, source_name=source)
    return {"status": "ok", "html_path": str(path)}


@app.get("/analyze/{symbol}", response_class=HTMLResponse)
def analyze(symbol: str, horizon: int = 5, lookback: int = 750) -> str:
    """Multi-method analysis page for a single A-share ticker.

    Example: /analyze/600519
    """
    from finai.quant.analyze_builder import build_analyze, render_analyze, write_analyze
    from finai.quant.loader import load_history
    history = load_history(symbol, lookback_days=lookback)
    if history.bars.empty:
        raise HTTPException(status_code=404,
                             detail=f"no history for {symbol} — bad ticker or network down")
    payload = build_analyze(history, horizon_days=horizon)
    write_analyze(payload)  # cache to disk
    return render_analyze(payload)


@app.get("/analyze/{symbol}.json")
def analyze_json(symbol: str, horizon: int = 5, lookback: int = 750) -> dict:
    """JSON view of the same analysis — for programmatic callers."""
    from finai.quant.analyze_builder import build_analyze
    from finai.quant.loader import load_history
    history = load_history(symbol, lookback_days=lookback)
    if history.bars.empty:
        raise HTTPException(status_code=404, detail=f"no history for {symbol}")
    payload = build_analyze(history, horizon_days=horizon)
    return {
        "run": payload.run,
        "synthesis": payload.synthesis,
        "fail_count": payload.fail_count,
        "generated_at": payload.generated_at,
    }

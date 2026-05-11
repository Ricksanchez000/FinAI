"""Click CLI: `finai run`, `finai serve`, `finai schedule`, `finai init-db`."""
from __future__ import annotations

import logging
from datetime import date

import click

from finai.config import settings
from finai.db import init_db


def _setup_logging() -> None:
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@click.group()
def cli() -> None:
    """FinAI command-line entrypoints."""
    _setup_logging()


@cli.command("init-db")
def cmd_init_db() -> None:
    """Create SQLite tables."""
    init_db()
    click.echo(f"db ready: {settings.db_url}")


@cli.command("run")
@click.option("--trade-date", "trade_date_s", default=None, help="YYYY-MM-DD")
@click.option("--source", default=None, help="akshare | mock")
@click.option("--regions", default=None,
              help="csv: cn-a,us,hk,global. Overrides FINAI_FETCH_REGIONS. "
                   "us+hk add ~5–8 min each.")
def cmd_run(trade_date_s: str | None, source: str | None, regions: str | None) -> None:
    """Run the full pipeline (fetch → signals → LLM → render)."""
    from finai.pipeline.etl import run_full_pipeline
    if regions is not None:
        settings.fetch_regions = regions
    td = date.fromisoformat(trade_date_s) if trade_date_s else None
    path = run_full_pipeline(trade_date=td, source_name=source)
    click.echo(f"report written: {path}")


@cli.command("etl")
@click.option("--trade-date", "trade_date_s", default=None)
@click.option("--source", default=None)
def cmd_etl(trade_date_s: str | None, source: str | None) -> None:
    """Run ETL only — fetch and persist; no signals or LLM."""
    from finai.pipeline.etl import run_etl
    td = date.fromisoformat(trade_date_s) if trade_date_s else None
    snap = run_etl(td, source)
    click.echo(f"persisted snapshot for {snap.trade_date}: "
               f"{len(snap.stocks)} stocks, {len(snap.sectors)} sectors, "
               f"{len(snap.news)} news")


@cli.command("serve")
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=8000, type=int)
def cmd_serve(host: str, port: int) -> None:
    """Start the FastAPI server."""
    import uvicorn
    uvicorn.run("finai.api.server:app", host=host, port=port, log_level=settings.log_level.lower())


@cli.command("schedule")
def cmd_schedule() -> None:
    """Run the blocking scheduler (weekdays 16:00 Asia/Shanghai)."""
    from finai.pipeline.scheduler import start
    start()


if __name__ == "__main__":
    cli()

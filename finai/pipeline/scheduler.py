"""APScheduler wrapper. Runs the full pipeline on Beijing-time weekdays at 16:00."""
from __future__ import annotations

import logging
from datetime import date

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from finai.pipeline.etl import run_full_pipeline

log = logging.getLogger(__name__)


def _job() -> None:
    try:
        path = run_full_pipeline()
        log.info("scheduled run done: %s", path)
    except Exception as exc:
        log.exception("scheduled run failed: %s", exc)


def start() -> None:
    sched = BlockingScheduler(timezone="Asia/Shanghai")
    sched.add_job(
        _job,
        CronTrigger(day_of_week="mon-fri", hour=16, minute=0),
        id="daily_report",
        max_instances=1,
        coalesce=True,
    )
    log.info("scheduler started: weekdays 16:00 Asia/Shanghai")
    sched.start()

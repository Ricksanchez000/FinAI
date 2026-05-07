"""Test fixtures: in-memory DB, mock data source, LLM disabled."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("FINAI_DB_URL", "sqlite:///./test_finai.db")
os.environ.setdefault("FINAI_LLM_ENABLED", "false")
os.environ.setdefault("FINAI_DATA_SOURCE", "mock")
os.environ.setdefault("FINAI_DATA_DIR", "./test_data")
os.environ.setdefault("FINAI_REPORT_DIR", "./test_reports")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from finai.config import settings
from finai.data.mock_source import MockSource
from finai.db import init_db


@pytest.fixture(autouse=True)
def _clean_db(tmp_path, monkeypatch):
    db_path = tmp_path / "finai.db"
    monkeypatch.setattr(settings, "db_url", f"sqlite:///{db_path}")
    monkeypatch.setattr(settings, "report_dir", tmp_path / "reports")
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    settings.report_dir.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    import finai.db as db_mod
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    db_mod._engine = create_engine(settings.db_url, future=True)
    db_mod.SessionLocal = sessionmaker(bind=db_mod._engine, autoflush=False, expire_on_commit=False)
    init_db()
    yield


@pytest.fixture
def mock_source() -> MockSource:
    return MockSource()

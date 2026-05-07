"""Centralized config loaded from env / .env file."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="FINAI_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    db_url: str = "sqlite:///./finai.db"
    data_dir: Path = Path("./data")
    report_dir: Path = Path("./reports")

    data_source: Literal["akshare", "mock"] = "akshare"

    llm_enabled: bool = True
    llm_model: str = "claude-opus-4-7"
    llm_fast_model: str = "claude-haiku-4-5"
    llm_max_tokens: int = 2048
    llm_anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    log_level: str = "INFO"

    anomaly_pct_threshold: float = 7.0
    anomaly_turnover_z: float = 2.5
    similarity_lookback_days: int = 750
    similarity_top_k: int = 5

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()

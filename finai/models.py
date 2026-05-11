"""ORM models for the daily snapshot store.

Schema is intentionally narrow: per-trading-day rows for indices, sectors,
stocks, plus rolled-up reports. The signal layer reads from these tables;
the report layer reads signals + DailyReport rows.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

from sqlalchemy import (
    Date,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class IndexQuote(Base):
    __tablename__ = "index_quote"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trade_date: Mapped[date] = mapped_column(Date, index=True)
    market: Mapped[str] = mapped_column(String(16), index=True, default="cn-a")  # cn-a / us / hk / global
    code: Mapped[str] = mapped_column(String(16), index=True)
    name: Mapped[str] = mapped_column(String(64))
    open: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)
    amount: Mapped[float] = mapped_column(Float)
    pct_change: Mapped[float] = mapped_column(Float)
    as_of_ts: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (UniqueConstraint("trade_date", "market", "code", name="uq_index_quote"),)


class StockQuote(Base):
    __tablename__ = "stock_quote"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trade_date: Mapped[date] = mapped_column(Date, index=True)
    market: Mapped[str] = mapped_column(String(16), index=True, default="cn-a")
    code: Mapped[str] = mapped_column(String(16), index=True)
    name: Mapped[str] = mapped_column(String(64))
    open: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    pre_close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)
    amount: Mapped[float] = mapped_column(Float)
    turnover_rate: Mapped[float] = mapped_column(Float, default=0.0)
    pct_change: Mapped[float] = mapped_column(Float)
    market_cap: Mapped[float] = mapped_column(Float, default=0.0)
    sector: Mapped[str] = mapped_column(String(64), default="")

    __table_args__ = (UniqueConstraint("trade_date", "market", "code", name="uq_stock_quote"),)


class MacroQuote(Base):
    """Non-equity instruments: FX, yields, commodities, crypto.

    Each row carries its own as_of_ts because global markets close at different
    times — the report renders the timestamp alongside the value rather than
    pretending they're synchronized.
    """
    __tablename__ = "macro_quote"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trade_date: Mapped[date] = mapped_column(Date, index=True)
    asset_class: Mapped[str] = mapped_column(String(16), index=True)  # fx / yield / commodity / crypto / index_global
    code: Mapped[str] = mapped_column(String(32))
    name: Mapped[str] = mapped_column(String(64))
    value: Mapped[float] = mapped_column(Float)
    pct_change: Mapped[float] = mapped_column(Float, default=0.0)
    as_of_ts: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    extra_json: Mapped[str] = mapped_column(Text, default="")

    __table_args__ = (UniqueConstraint("trade_date", "asset_class", "code", name="uq_macro_quote"),)


class SectorQuote(Base):
    __tablename__ = "sector_quote"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trade_date: Mapped[date] = mapped_column(Date, index=True)
    sector: Mapped[str] = mapped_column(String(64), index=True)
    pct_change: Mapped[float] = mapped_column(Float)
    amount: Mapped[float] = mapped_column(Float)
    leader_code: Mapped[str] = mapped_column(String(16), default="")
    leader_name: Mapped[str] = mapped_column(String(64), default="")

    __table_args__ = (UniqueConstraint("trade_date", "sector", name="uq_sector_quote"),)


class CapitalFlow(Base):
    __tablename__ = "capital_flow"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trade_date: Mapped[date] = mapped_column(Date, index=True)
    scope: Mapped[str] = mapped_column(String(16))  # north / margin / main
    metric: Mapped[str] = mapped_column(String(32))
    value: Mapped[float] = mapped_column(Float)


class NewsItem(Base):
    __tablename__ = "news_item"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trade_date: Mapped[date] = mapped_column(Date, index=True)
    code: Mapped[str] = mapped_column(String(16), default="", index=True)
    source: Mapped[str] = mapped_column(String(32))
    title: Mapped[str] = mapped_column(String(256))
    url: Mapped[str] = mapped_column(String(512), default="")
    published_at: Mapped[datetime] = mapped_column(DateTime)
    summary: Mapped[str] = mapped_column(Text, default="")


class DailyReport(Base):
    __tablename__ = "daily_report"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trade_date: Mapped[date] = mapped_column(Date, unique=True, index=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    payload_json: Mapped[str] = mapped_column(Text)  # full structured report
    html_path: Mapped[str] = mapped_column(String(256), default="")

"""SQLAlchemy models for scalpbot persistence.

Every table carries a ``market`` column ("crypto" | "stock" | "option") so a
single schema serves all three avenues and queries can slice by avenue.

Semantics worth noting:

* **Position.leverage** -- for OPTIONS this holds the *number of contracts*
  (each contract = 100 shares of exposure). For crypto/stock it is the notional
  leverage multiplier (``1.0`` for spot/cash).
* **Position.entry_price** -- for OPTIONS this is the *premium per contract*
  (per share); realized/unrealized P&L for options is premium-based, i.e.
  ``(exit_premium - entry_premium) * contracts * 100``.

The connection URL comes from ``DATABASE_URL`` (a Doppler secret). Nothing here
hard-codes credentials.
"""
from __future__ import annotations

import datetime as dt
import os
from typing import Optional

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


# ---------------------------------------------------------------------------
# market / side / status vocabulary (kept as plain strings for portability)
# ---------------------------------------------------------------------------
MARKET_CRYPTO = "crypto"
MARKET_STOCK = "stock"
MARKET_OPTION = "option"
MARKETS = (MARKET_CRYPTO, MARKET_STOCK, MARKET_OPTION)

SIDE_LONG = "long"
SIDE_SHORT = "short"

POSITION_OPEN = "open"
POSITION_CLOSED = "closed"

TRADE_FILLED = "filled"
TRADE_SUBMITTED = "submitted"
TRADE_REJECTED = "rejected"
TRADE_CANCELED = "canceled"


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Base(DeclarativeBase):
    pass


class Trade(Base):
    """An executed (or attempted) order. Immutable audit record."""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market: Mapped[str] = mapped_column(String(16), index=True)
    symbol: Mapped[str] = mapped_column(String(64), index=True)
    side: Mapped[str] = mapped_column(String(8))
    action: Mapped[str] = mapped_column(String(16))  # open | close | add | reduce
    qty: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)         # fill price (premium for options)
    notional: Mapped[float] = mapped_column(Float, default=0.0)
    fees: Mapped[float] = mapped_column(Float, default=0.0)
    pnl: Mapped[float] = mapped_column(Float, default=0.0)  # realized on close trades
    order_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default=TRADE_SUBMITTED)
    mode: Mapped[str] = mapped_column(String(8), default="paper")  # paper | live
    reason: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<Trade {self.market}:{self.symbol} {self.action} {self.side} "
            f"qty={self.qty} px={self.price} status={self.status}>"
        )


class Position(Base):
    """A currently-open (or historically-closed) position."""

    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market: Mapped[str] = mapped_column(String(16), index=True)
    symbol: Mapped[str] = mapped_column(String(64), index=True)
    side: Mapped[str] = mapped_column(String(8), default=SIDE_LONG)
    qty: Mapped[float] = mapped_column(Float)
    # For options: number of contracts. For crypto/stock: leverage multiplier.
    leverage: Mapped[float] = mapped_column(Float, default=1.0)
    # For options: entry premium per contract (per share). Else: entry price.
    entry_price: Mapped[float] = mapped_column(Float)
    current_price: Mapped[float] = mapped_column(Float, default=0.0)
    entry_notional: Mapped[float] = mapped_column(Float, default=0.0)
    entry_fees: Mapped[float] = mapped_column(Float, default=0.0)
    # Options underlying + contract metadata (null for crypto/stock).
    underlying: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    strike: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    option_type: Mapped[Optional[str]] = mapped_column(String(4), nullable=True)  # call | put
    expiration: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)  # YYYY-MM-DD
    sentiment_score: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(16), default=POSITION_OPEN, index=True)
    opened_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    closed_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    extra: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<Position {self.market}:{self.symbol} {self.side} qty={self.qty} "
            f"entry={self.entry_price} status={self.status}>"
        )


class BotConfig(Base):
    """Runtime override store -- a tiny key/value table for mutable settings that
    can be flipped through the API without a redeploy (e.g. a kill switch or a
    paper/live toggle within the constraints of the live-trading guard).

    This does NOT replace env/Doppler config; it only holds operator overrides.
    """

    __tablename__ = "bot_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    value: Mapped[str] = mapped_column(String(256))
    market: Mapped[str] = mapped_column(String(16), default="all")  # all | crypto | stock | option
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class SignalLog(Base):
    """A snapshot of an aggregated sentiment signal for a symbol in one cycle."""

    __tablename__ = "signal_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market: Mapped[str] = mapped_column(String(16), index=True)
    symbol: Mapped[str] = mapped_column(String(64), index=True)
    score: Mapped[float] = mapped_column(Float)
    label: Mapped[str] = mapped_column(String(24))
    acted: Mapped[bool] = mapped_column(default=False)  # did it trigger an order?
    detail: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)


# ---------------------------------------------------------------------------
# engine / session factory
# ---------------------------------------------------------------------------
def _url_from_postgres_parts() -> Optional[str]:
    """Assemble a SQLAlchemy URL from the POSTGRES_* Doppler secrets, if present."""
    user = os.environ.get("POSTGRES_USER")
    password = os.environ.get("POSTGRES_PASSWORD")
    db = os.environ.get("POSTGRES_DB")
    host = os.environ.get("POSTGRES_HOST")
    port = os.environ.get("POSTGRES_PORT", "5432")
    if user and password and db and host:
        return f"postgresql://{user}:{password}@{host}:{port}/{db}"
    return None


def make_engine(url: Optional[str] = None, echo: bool = False):
    """Create a SQLAlchemy engine.

    Resolution order for the connection URL:
    1. explicit ``url`` argument,
    2. ``DATABASE_URL`` (Doppler secret, if the operator prefers a single URL),
    3. assembled from the ``POSTGRES_*`` Doppler secrets,
    4. a local SQLite file (so unit tests and local smoke runs work without
       Postgres).
    """
    url = (
        url
        or os.environ.get("DATABASE_URL")
        or _url_from_postgres_parts()
        or "sqlite:///scalpbot_local.db"
    )
    # SQLAlchemy 2.x wants postgresql:// (not the legacy postgres://).
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return create_engine(url, echo=echo, future=True, pool_pre_ping=True)


def make_session_factory(url: Optional[str] = None, echo: bool = False):
    engine = make_engine(url, echo=echo)
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_db(url: Optional[str] = None, echo: bool = False):
    """Create all tables (idempotent). Returns the engine."""
    engine = make_engine(url, echo=echo)
    Base.metadata.create_all(engine)
    return engine

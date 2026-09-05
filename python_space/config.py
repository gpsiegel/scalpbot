"""Central configuration for scalpbot.

This module is the single source of truth for runtime configuration. It reads
from the process environment, which is populated two ways (see the project
STANDING RULE):

* **Doppler secrets** -> API keys, DB credentials, and the ticker/coin LISTS
  (``STOCK_TICKERS``, ``CRYPTO_CORE_COINS``, ``CRYPTO_SATELLITE_COINS``).
* **Plain .env operational config** -> avenue toggles, position caps, market
  hours gate, short toggle, options quality filters, and the
  ``SENTIMENT_WEIGHT_*`` values.

Nothing here hard-codes a secret. Coin symbols listed in the crypto lists must
have a matching entry in ``sentiment.base.COIN_METADATA``; :func:`validate`
enforces this so a typo fails fast instead of silently dropping a coin.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List

from sentiment.base import COIN_METADATA


# ---------------------------------------------------------------------------
# small env helpers
# ---------------------------------------------------------------------------
def _get_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y", "t"}


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def _get_list(name: str, default: List[str]) -> List[str]:
    """Parse a comma-separated env var into an upper-cased, de-duped list."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return list(default)
    seen: Dict[str, None] = {}
    for item in raw.split(","):
        token = item.strip().upper()
        if token:
            seen.setdefault(token, None)
    return list(seen.keys())


# ---------------------------------------------------------------------------
# config dataclasses
# ---------------------------------------------------------------------------
@dataclass
class AvenueToggles:
    """Per-avenue master switches. When off, the avenue still MANAGES and EXITS
    existing positions but opens NO new entries."""

    crypto: bool = True
    stocks: bool = True
    options: bool = True

    @classmethod
    def from_env(cls) -> "AvenueToggles":
        return cls(
            crypto=_get_bool("ENABLE_CRYPTO", True),
            stocks=_get_bool("ENABLE_STOCKS", True),
            options=_get_bool("ENABLE_OPTIONS", True),
        )


@dataclass
class PositionCaps:
    """Hard limits on concurrent open positions."""

    total: int = 3
    crypto: int = 2
    stock: int = 2
    option: int = 2

    @classmethod
    def from_env(cls) -> "PositionCaps":
        return cls(
            total=_get_int("MAX_TOTAL_POSITIONS", 3),
            crypto=_get_int("MAX_CRYPTO_POSITIONS", 2),
            stock=_get_int("MAX_STOCK_POSITIONS", 2),
            option=_get_int("MAX_OPTION_POSITIONS", 2),
        )


@dataclass
class OptionsFilters:
    """Quality gates for options entries. Options are buy-to-open only
    (long calls / puts)."""

    min_dte: int = 7          # skip near-expiry / 0-DTE contracts
    max_otm_pct: float = 0.10  # only near-the-money, no deep-OTM lottery tickets
    min_premium: float = 0.10  # skip near-worthless contracts
    max_spread_pct: float = 0.25  # skip illiquid, wide bid/ask contracts

    @classmethod
    def from_env(cls) -> "OptionsFilters":
        return cls(
            min_dte=_get_int("OPTION_MIN_DTE", 7),
            max_otm_pct=_get_float("OPTION_MAX_OTM_PCT", 0.10),
            min_premium=_get_float("OPTION_MIN_PREMIUM", 0.10),
            max_spread_pct=_get_float("OPTION_MAX_SPREAD_PCT", 0.25),
        )


@dataclass
class SentimentWeights:
    reddit: float = 0.30
    cryptocv: float = 0.25
    coingecko: float = 0.20
    lunarcrush: float = 0.10
    trends: float = 0.08
    fear_greed: float = 0.07

    @classmethod
    def from_env(cls) -> "SentimentWeights":
        return cls(
            reddit=_get_float("SENTIMENT_WEIGHT_REDDIT", 0.30),
            cryptocv=_get_float("SENTIMENT_WEIGHT_CRYPTOCV", 0.25),
            coingecko=_get_float("SENTIMENT_WEIGHT_COINGECKO", 0.20),
            lunarcrush=_get_float("SENTIMENT_WEIGHT_LUNARCRUSH", 0.10),
            trends=_get_float("SENTIMENT_WEIGHT_TRENDS", 0.08),
            fear_greed=_get_float("SENTIMENT_WEIGHT_FEAR_GREED", 0.07),
        )


@dataclass
class Config:
    """Top-level resolved configuration for a bot run."""

    # environment / trading mode
    app_env: str = "nonprod"          # nonprod | staging | prod
    paper_trading: bool = True
    live_trading: bool = False

    # avenue behaviour
    avenues: AvenueToggles = field(default_factory=AvenueToggles)
    respect_market_hours: bool = True  # stocks/options gate on US market open; crypto 24/7
    allow_short: bool = True           # stock long-only mode when False

    # limits + filters
    caps: PositionCaps = field(default_factory=PositionCaps)
    options: OptionsFilters = field(default_factory=OptionsFilters)
    weights: SentimentWeights = field(default_factory=SentimentWeights)

    # instrument universe (from Doppler)
    stock_tickers: List[str] = field(default_factory=list)
    crypto_core_coins: List[str] = field(default_factory=list)
    crypto_satellite_coins: List[str] = field(default_factory=list)

    # fees
    taker_fee_pct: float = 0.0025  # 0.25% per side

    # ---- derived helpers -------------------------------------------------
    @property
    def all_crypto_coins(self) -> List[str]:
        """Core + satellite, de-duplicated, order-preserving."""
        seen: Dict[str, None] = {}
        for c in list(self.crypto_core_coins) + list(self.crypto_satellite_coins):
            seen.setdefault(c, None)
        return list(seen.keys())

    def crypto_symbol(self, coin: str) -> str:
        """Alpaca crypto market symbol, e.g. 'SOL' -> 'SOL/USD'."""
        return f"{coin.upper()}/USD"

    def round_trip_fees(self, size: float) -> float:
        """Round-trip taker fees for a position of ``size`` notional."""
        return size * 2 * self.taker_fee_pct

    def is_live(self) -> bool:
        """Live trading requires ALL THREE guards. Any other APP_ENV forces
        paper trading regardless of the LIVE_TRADING setting."""
        return (
            self.app_env == "prod"
            and self.paper_trading is False
            and self.live_trading is True
        )

    # ---- construction ----------------------------------------------------
    @classmethod
    def from_env(cls) -> "Config":
        app_env = os.environ.get("APP_ENV", "nonprod").strip().lower() or "nonprod"
        return cls(
            app_env=app_env,
            paper_trading=_get_bool("PAPER_TRADING", True),
            live_trading=_get_bool("LIVE_TRADING", False),
            avenues=AvenueToggles.from_env(),
            respect_market_hours=_get_bool("RESPECT_MARKET_HOURS", True),
            allow_short=_get_bool("ALLOW_SHORT", True),
            caps=PositionCaps.from_env(),
            options=OptionsFilters.from_env(),
            weights=SentimentWeights.from_env(),
            stock_tickers=_get_list("STOCK_TICKERS", []),
            crypto_core_coins=_get_list("CRYPTO_CORE_COINS", []),
            crypto_satellite_coins=_get_list("CRYPTO_SATELLITE_COINS", []),
        )

    # ---- validation ------------------------------------------------------
    def validate(self) -> List[str]:
        """Return a list of human-readable problems (empty == valid)."""
        problems: List[str] = []

        for coin in self.all_crypto_coins:
            if coin not in COIN_METADATA:
                problems.append(
                    f"crypto coin '{coin}' has no COIN_METADATA entry "
                    f"(add it in sentiment/base.py or fix the CRYPTO_*_COINS list)"
                )

        if self.avenues.crypto and not self.all_crypto_coins:
            problems.append("ENABLE_CRYPTO is on but no CRYPTO_CORE/SATELLITE coins configured")
        if self.avenues.stocks and not self.stock_tickers:
            problems.append("ENABLE_STOCKS is on but STOCK_TICKERS is empty")

        if self.app_env == "prod" and self.paper_trading is False and self.live_trading:
            # live trading is intentional here; nothing to warn about
            pass

        return problems


# Convenience module-level singleton, resolved lazily.
_config: "Config | None" = None


def get_config(reload: bool = False) -> Config:
    global _config
    if _config is None or reload:
        _config = Config.from_env()
    return _config

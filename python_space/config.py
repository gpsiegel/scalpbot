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

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from sentiment.base import COIN_METADATA, EQUITY_CAPABLE_SOURCES

logger = logging.getLogger("scalpbot.config")

# Recognised environments. There is NO "staging" -- nonprod and prod only.
KNOWN_ENVIRONMENTS = ("nonprod", "prod")
_ENV_DIR = Path(__file__).resolve().parent / "environments"


# ---------------------------------------------------------------------------
# committed, PR-controlled operational config loader
# ---------------------------------------------------------------------------
def load_env_file(app_env: str | None = None) -> None:
    """Load ``environments/<APP_ENV>.env`` as operational DEFAULTS.

    These committed, per-environment files hold NON-SECRET operational config
    (avenue toggles, risk tier, caps, behaviour flags) so that changing them is
    a reviewable GitHub PR. Secrets are NEVER read from here -- they come from
    Doppler.

    Precedence: values are applied with :func:`os.environ.setdefault`, so any
    real environment variable already present (injected by Doppler / CI /
    shell) ALWAYS wins over the committed default.

    Fully defensive: a missing/unreadable file is a no-op. An unknown APP_ENV
    falls back to the nonprod file (the safe, paper-only default).
    """
    env = (app_env or os.environ.get("APP_ENV") or "nonprod").strip().lower() or "nonprod"
    file_env = env if env in KNOWN_ENVIRONMENTS else "nonprod"
    path = _ENV_DIR / f"{file_env}.env"
    if not path.is_file():
        logger.debug("no operational config file at %s (skipping)", path)
        return
    try:
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if not key:
                continue
            value = value.strip()
            # strip an inline "  # comment" (only when preceded by whitespace)
            for i in range(1, len(value)):
                if value[i] == "#" and value[i - 1].isspace():
                    value = value[:i].rstrip()
                    break
            # strip surrounding quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            os.environ.setdefault(key, value)
    except OSError as exc:  # pragma: no cover - defensive
        logger.warning("could not read operational config %s: %s", path, exc)


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
    max_dte: int = 45         # bound the contract query on the far side too
    max_otm_pct: float = 0.10  # only near-the-money, no deep-OTM lottery tickets
    min_premium: float = 0.10  # skip near-worthless contracts
    max_spread_pct: float = 0.08  # skip illiquid, wide bid/ask contracts

    @classmethod
    def from_env(cls) -> "OptionsFilters":
        return cls(
            min_dte=_get_int("OPTION_MIN_DTE", 7),
            max_dte=_get_int("OPTION_MAX_DTE", 45),
            max_otm_pct=_get_float("OPTION_MAX_OTM_PCT", 0.10),
            min_premium=_get_float("OPTION_MIN_PREMIUM", 0.10),
            max_spread_pct=_get_float("OPTION_MAX_SPREAD_PCT", 0.08),
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
class MarketCosts:
    """Per-market round-trip execution cost model, fed into the EV gate and
    realized P&L.

    Alpaca charges a taker fee only on crypto; stocks are commission-free but
    still cross a bid/ask spread on both legs of a round trip. Options are
    handled differently: a long option is already bought near the ask and
    sold near the bid (engine/options.py), so the spread cost is captured
    structurally in the premium itself -- adding a separate spread charge on
    top would double-count it. Only optional per-contract regulatory/exchange
    fees are modeled for options, defaulted to 0.
    """

    crypto_taker_fee_pct: float = 0.0025    # 0.25% per side (Alpaca crypto taker fee)
    crypto_half_spread_pct: float = 0.0005  # estimated per-side crypto spread
    stock_half_spread_pct: float = 0.0005   # estimated per-side equity spread
    option_regulatory_fee_pct: float = 0.0  # e.g. OCC/exchange fees, if modeled

    @classmethod
    def from_env(cls) -> "MarketCosts":
        return cls(
            crypto_taker_fee_pct=_get_float("CRYPTO_TAKER_FEE_PCT", 0.0025),
            crypto_half_spread_pct=_get_float("CRYPTO_HALF_SPREAD_PCT", 0.0005),
            stock_half_spread_pct=_get_float("STOCK_HALF_SPREAD_PCT", 0.0005),
            option_regulatory_fee_pct=_get_float("OPTION_REGULATORY_FEE_PCT", 0.0),
        )

    def round_trip_pct(self, market: str) -> float:
        """Total round-trip cost as a fraction of notional, for ``market``."""
        if market == "crypto":
            return 2 * (self.crypto_taker_fee_pct + self.crypto_half_spread_pct)
        if market == "stock":
            return 2 * self.stock_half_spread_pct
        if market == "option":
            return 2 * self.option_regulatory_fee_pct
        return 0.0


@dataclass
class Config:
    """Top-level resolved configuration for a bot run."""

    # environment / trading mode
    app_env: str = "nonprod"          # nonprod | prod
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
    costs: MarketCosts = field(default_factory=MarketCosts)

    # options management (not an entry filter, so it lives outside OptionsFilters):
    # close a held option once it is this close to expiry, regardless of P&L --
    # Alpaca auto-exercises/force-sells ITM contracts near expiry.
    option_exit_dte: int = 2  # OPTION_EXIT_DTE

    # crypto/stock management: close a position after this many minutes
    # regardless of P&L, independent of the ATR-scaled take-profit/stop-loss.
    # 0 disables the check.
    max_hold_minutes: int = 240  # MAX_HOLD_MINUTES

    # instrument universe (from Doppler)
    stock_tickers: List[str] = field(default_factory=list)
    crypto_core_coins: List[str] = field(default_factory=list)
    crypto_satellite_coins: List[str] = field(default_factory=list)

    # risk profile (resolved to a TierParams bundle via engine.risk.get_tier).
    # Non-secret operational tunable -> lives in plain .env as RISK_TIER.
    risk_tier: str = "moderate"    # conservative | moderate | high

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

    def round_trip_fees(self, market: str, size: float) -> float:
        """Round-trip execution cost in dollars for a ``size``-notional
        position in ``market`` (see :class:`MarketCosts`)."""
        return size * self.costs.round_trip_pct(market)

    @property
    def is_live_capable_env(self) -> bool:
        """prod is the ONLY environment permitted to touch the live account.
        Being live-capable does not mean it IS live -- see :meth:`is_live`."""
        return self.app_env == "prod"

    def is_live(self) -> bool:
        """Live trading requires ALL THREE guards:
        ``APP_ENV=prod`` AND ``PAPER_TRADING=false`` AND ``LIVE_TRADING=true``.

        nonprod (or any non-prod APP_ENV) can NEVER be live: the first guard
        fails, so PAPER_TRADING/LIVE_TRADING are irrelevant and paper is forced.
        prod defaults to paper too -- going live is an explicit, reviewable flip
        of the two mode flags in ``environments/prod.env``."""
        return (
            self.app_env == "prod"
            and self.paper_trading is False
            and self.live_trading is True
        )

    # ---- construction ----------------------------------------------------
    @classmethod
    def from_env(cls) -> "Config":
        # Load committed, PR-controlled operational defaults for this APP_ENV
        # FIRST (setdefault -> real env / Doppler still wins), then resolve.
        load_env_file()
        app_env = os.environ.get("APP_ENV", "nonprod").strip().lower() or "nonprod"
        risk_tier = os.environ.get("RISK_TIER", "moderate").strip().lower() or "moderate"
        return cls(
            app_env=app_env,
            risk_tier=risk_tier,
            paper_trading=_get_bool("PAPER_TRADING", True),
            live_trading=_get_bool("LIVE_TRADING", False),
            avenues=AvenueToggles.from_env(),
            respect_market_hours=_get_bool("RESPECT_MARKET_HOURS", True),
            allow_short=_get_bool("ALLOW_SHORT", True),
            caps=PositionCaps.from_env(),
            options=OptionsFilters.from_env(),
            weights=SentimentWeights.from_env(),
            costs=MarketCosts.from_env(),
            option_exit_dte=_get_int("OPTION_EXIT_DTE", 2),
            max_hold_minutes=_get_int("MAX_HOLD_MINUTES", 240),
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
        if (self.avenues.stocks or self.avenues.options) and not EQUITY_CAPABLE_SOURCES:
            problems.append(
                "ENABLE_STOCKS/ENABLE_OPTIONS is on but no sentiment source is "
                "equity-capable (every source but fear_greed only resolves crypto "
                "tickers, and fear_greed is a market-wide crypto index) -- "
                "stocks/options would trade on an insufficient-coverage signal"
            )

        if self.app_env == "prod" and self.paper_trading is False and self.live_trading:
            # live trading is intentional here; nothing to warn about
            pass
        elif self.app_env != "prod" and (self.paper_trading is False or self.live_trading):
            # Non-prod can never go live; the flags are ignored and paper forced.
            problems.append(
                f"LIVE_TRADING/PAPER_TRADING ignored in APP_ENV='{self.app_env}': "
                "only APP_ENV=prod can trade live; forcing paper."
            )

        return problems


# Convenience module-level singleton, resolved lazily.
_config: "Config | None" = None


def get_config(reload: bool = False) -> Config:
    global _config
    if _config is None or reload:
        _config = Config.from_env()
    return _config

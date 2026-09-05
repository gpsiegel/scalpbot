"""Base primitives for the scalpbot sentiment stack.

Every sentiment source returns a :class:`SentimentSignal` with a normalized
score in the range ``[-1.0, +1.0]``:

* ``+1.0`` -> maximally bullish
* ``0.0``  -> neutral / no signal
* ``-1.0`` -> maximally bearish

Sources are intentionally defensive: a network error, missing API key, or
empty payload results in an *unavailable* signal (``available=False``) rather
than an exception, so the aggregator can gracefully re-weight the surviving
sources.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger("scalpbot.sentiment")

# Canonical coin metadata used to translate an internal coin key (e.g. "SOL")
# into the identifiers each upstream provider expects. SOL/USD and DOGE/USD are
# the tradeable crypto markets for this bot.
COIN_METADATA: Dict[str, Dict[str, Any]] = {
    "SOL": {
        "name": "Solana",
        "symbol": "SOL",
        "coingecko_id": "solana",
        "cryptocv_ticker": "SOL",
        "lunarcrush_symbol": "SOL",
        "subreddits": ["solana"],
        "trends_keywords": ["Solana", "SOL crypto"],
    },
    "DOGE": {
        "name": "Dogecoin",
        "symbol": "DOGE",
        "coingecko_id": "dogecoin",
        "cryptocv_ticker": "DOGE",
        "lunarcrush_symbol": "DOGE",
        "subreddits": ["dogecoin"],
        "trends_keywords": ["Dogecoin", "DOGE crypto"],
    },
}

# Broad crypto subreddits scanned in addition to the coin-specific ones.
GENERAL_SUBREDDITS = ["CryptoCurrency", "CryptoMarkets"]


def clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    """Clamp ``value`` into the ``[low, high]`` interval."""
    return max(low, min(high, value))


def scale_to_unit(value: float, min_value: float, max_value: float) -> float:
    """Linearly map ``value`` from ``[min_value, max_value]`` to ``[-1, 1]``."""
    if max_value == min_value:
        return 0.0
    pct = (value - min_value) / (max_value - min_value)  # 0..1
    return clamp(pct * 2.0 - 1.0)


@dataclass
class SentimentSignal:
    """Normalized output of a single sentiment source for a single coin."""

    source: str
    coin: str
    score: float = 0.0  # normalized [-1, 1]
    available: bool = True
    confidence: float = 1.0  # 0..1, how much data backed this score
    raw: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def __post_init__(self) -> None:
        self.score = clamp(float(self.score))
        self.confidence = clamp(float(self.confidence), 0.0, 1.0)

    @classmethod
    def unavailable(cls, source: str, coin: str, error: str) -> "SentimentSignal":
        return cls(
            source=source,
            coin=coin,
            score=0.0,
            available=False,
            confidence=0.0,
            error=error,
        )


class BaseSentimentSource:
    """Common interface for all sentiment sources.

    Subclasses implement :meth:`_fetch` and return a :class:`SentimentSignal`.
    The public :meth:`get_signal` wraps it with error handling so a single
    failing provider never takes the aggregator down.
    """

    name: str = "base"

    def get_signal(self, coin: str) -> SentimentSignal:
        try:
            signal = self._fetch(coin)
            if signal is None:
                return SentimentSignal.unavailable(self.name, coin, "no data returned")
            return signal
        except Exception as exc:  # noqa: BLE001 - defensive by design
            logger.warning("[%s] failed for %s: %s", self.name, coin, exc)
            return SentimentSignal.unavailable(self.name, coin, str(exc))

    def _fetch(self, coin: str) -> Optional[SentimentSignal]:  # pragma: no cover
        raise NotImplementedError

    # -- shared helpers ---------------------------------------------------
    @staticmethod
    def coin_meta(coin: str) -> Dict[str, Any]:
        meta = COIN_METADATA.get(coin.upper())
        if not meta:
            raise ValueError(f"unknown coin '{coin}'; expected one of {list(COIN_METADATA)}")
        return meta

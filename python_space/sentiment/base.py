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
# into the identifiers each upstream provider expects.
#
# This is a REGISTRY of supported coins, not the active trading set. The active
# core/satellite lists are configured via CRYPTO_CORE_COINS / CRYPTO_SATELLITE_COINS
# (Doppler) and read through python_space/config.py. Every symbol referenced by
# those lists MUST have an entry here so the sentiment sources can resolve the
# per-provider identifiers. All entries below are Alpaca-tradeable USD markets.
COIN_METADATA: Dict[str, Dict[str, Any]] = {
    "BTC": {
        "name": "Bitcoin", "symbol": "BTC", "coingecko_id": "bitcoin",
        "cryptocv_ticker": "BTC", "lunarcrush_symbol": "BTC",
        "subreddits": ["Bitcoin"], "trends_keywords": ["Bitcoin", "BTC price"],
    },
    "ETH": {
        "name": "Ethereum", "symbol": "ETH", "coingecko_id": "ethereum",
        "cryptocv_ticker": "ETH", "lunarcrush_symbol": "ETH",
        "subreddits": ["ethereum", "ethtrader"], "trends_keywords": ["Ethereum", "ETH crypto"],
    },
    "SOL": {
        "name": "Solana", "symbol": "SOL", "coingecko_id": "solana",
        "cryptocv_ticker": "SOL", "lunarcrush_symbol": "SOL",
        "subreddits": ["solana"], "trends_keywords": ["Solana", "SOL crypto"],
    },
    "DOGE": {
        "name": "Dogecoin", "symbol": "DOGE", "coingecko_id": "dogecoin",
        "cryptocv_ticker": "DOGE", "lunarcrush_symbol": "DOGE",
        "subreddits": ["dogecoin"], "trends_keywords": ["Dogecoin", "DOGE crypto"],
    },
    "AVAX": {
        "name": "Avalanche", "symbol": "AVAX", "coingecko_id": "avalanche-2",
        "cryptocv_ticker": "AVAX", "lunarcrush_symbol": "AVAX",
        "subreddits": ["Avax"], "trends_keywords": ["Avalanche", "AVAX crypto"],
    },
    "LINK": {
        "name": "Chainlink", "symbol": "LINK", "coingecko_id": "chainlink",
        "cryptocv_ticker": "LINK", "lunarcrush_symbol": "LINK",
        "subreddits": ["Chainlink"], "trends_keywords": ["Chainlink", "LINK crypto"],
    },
    "LTC": {
        "name": "Litecoin", "symbol": "LTC", "coingecko_id": "litecoin",
        "cryptocv_ticker": "LTC", "lunarcrush_symbol": "LTC",
        "subreddits": ["litecoin"], "trends_keywords": ["Litecoin", "LTC crypto"],
    },
    "DOT": {
        "name": "Polkadot", "symbol": "DOT", "coingecko_id": "polkadot",
        "cryptocv_ticker": "DOT", "lunarcrush_symbol": "DOT",
        "subreddits": ["Polkadot"], "trends_keywords": ["Polkadot", "DOT crypto"],
    },
    "UNI": {
        "name": "Uniswap", "symbol": "UNI", "coingecko_id": "uniswap",
        "cryptocv_ticker": "UNI", "lunarcrush_symbol": "UNI",
        "subreddits": ["UniSwap"], "trends_keywords": ["Uniswap", "UNI crypto"],
    },
    "AAVE": {
        "name": "Aave", "symbol": "AAVE", "coingecko_id": "aave",
        "cryptocv_ticker": "AAVE", "lunarcrush_symbol": "AAVE",
        "subreddits": ["Aave_Official"], "trends_keywords": ["Aave", "AAVE crypto"],
    },
    "BCH": {
        "name": "Bitcoin Cash", "symbol": "BCH", "coingecko_id": "bitcoin-cash",
        "cryptocv_ticker": "BCH", "lunarcrush_symbol": "BCH",
        "subreddits": ["Bitcoincash"], "trends_keywords": ["Bitcoin Cash", "BCH crypto"],
    },
    "SHIB": {
        "name": "Shiba Inu", "symbol": "SHIB", "coingecko_id": "shiba-inu",
        "cryptocv_ticker": "SHIB", "lunarcrush_symbol": "SHIB",
        "subreddits": ["SHIBArmy"], "trends_keywords": ["Shiba Inu", "SHIB crypto"],
    },
    "XRP": {
        "name": "XRP", "symbol": "XRP", "coingecko_id": "ripple",
        "cryptocv_ticker": "XRP", "lunarcrush_symbol": "XRP",
        "subreddits": ["Ripple", "XRP"], "trends_keywords": ["XRP", "Ripple crypto"],
    },
    "MKR": {
        "name": "Maker", "symbol": "MKR", "coingecko_id": "maker",
        "cryptocv_ticker": "MKR", "lunarcrush_symbol": "MKR",
        "subreddits": ["MakerDAO"], "trends_keywords": ["Maker", "MKR crypto"],
    },
    "CRV": {
        "name": "Curve DAO", "symbol": "CRV", "coingecko_id": "curve-dao-token",
        "cryptocv_ticker": "CRV", "lunarcrush_symbol": "CRV",
        "subreddits": ["CurveFinance"], "trends_keywords": ["Curve Finance", "CRV crypto"],
    },
    "GRT": {
        "name": "The Graph", "symbol": "GRT", "coingecko_id": "the-graph",
        "cryptocv_ticker": "GRT", "lunarcrush_symbol": "GRT",
        "subreddits": ["TheGraph"], "trends_keywords": ["The Graph", "GRT crypto"],
    },
    "XTZ": {
        "name": "Tezos", "symbol": "XTZ", "coingecko_id": "tezos",
        "cryptocv_ticker": "XTZ", "lunarcrush_symbol": "XTZ",
        "subreddits": ["tezos"], "trends_keywords": ["Tezos", "XTZ crypto"],
    },
    "SUSHI": {
        "name": "SushiSwap", "symbol": "SUSHI", "coingecko_id": "sushi",
        "cryptocv_ticker": "SUSHI", "lunarcrush_symbol": "SUSHI",
        "subreddits": ["SushiSwap"], "trends_keywords": ["SushiSwap", "SUSHI crypto"],
    },
    "YFI": {
        "name": "yearn.finance", "symbol": "YFI", "coingecko_id": "yearn-finance",
        "cryptocv_ticker": "YFI", "lunarcrush_symbol": "YFI",
        "subreddits": ["yearn_finance"], "trends_keywords": ["Yearn Finance", "YFI crypto"],
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

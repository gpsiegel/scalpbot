"""Sentiment aggregator for scalpbot.

Combines the six sentiment sources into a single normalized score per coin,
using weights pulled from environment variables (plain ``.env`` operational
config -- NOT Doppler secrets):

  SENTIMENT_WEIGHT_REDDIT       (default 0.30)  -> Reddit OAuth
  SENTIMENT_WEIGHT_CRYPTOCV     (default 0.25)  -> cryptocurrency.cv
  SENTIMENT_WEIGHT_COINGECKO    (default 0.20)  -> CoinGecko (Basic tier)
  SENTIMENT_WEIGHT_LUNARCRUSH   (default 0.10)  -> LunarCrush (hobby tier)
  SENTIMENT_WEIGHT_TRENDS       (default 0.08)  -> Google Trends (trendspyg)
  SENTIMENT_WEIGHT_FEAR_GREED   (default 0.07)  -> Fear & Greed Index

When a source is unavailable in a given cycle, its weight is dropped and the
remaining weights are renormalized so the final score always spans [-1, 1].
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .base import SentimentSignal
from .coingecko import CoinGeckoSource
from .cryptocurrency_cv import CryptoCurrencyCVSource
from .fear_greed import FearGreedSource
from .google_trends import GoogleTrendsSource
from .lunarcrush import LunarCrushSource
from .reddit_source import RedditSource

logger = logging.getLogger("scalpbot.sentiment")

# env var name -> (source key, default weight)
WEIGHT_ENV = {
    "reddit": ("SENTIMENT_WEIGHT_REDDIT", 0.30),
    "cryptocurrency_cv": ("SENTIMENT_WEIGHT_CRYPTOCV", 0.25),
    "coingecko": ("SENTIMENT_WEIGHT_COINGECKO", 0.20),
    "lunarcrush": ("SENTIMENT_WEIGHT_LUNARCRUSH", 0.10),
    "google_trends": ("SENTIMENT_WEIGHT_TRENDS", 0.08),
    "fear_greed": ("SENTIMENT_WEIGHT_FEAR_GREED", 0.07),
}

# env var name -> (source key, default TTL in seconds). Hit twice per symbol
# per crypto cycle, some of these (Google Trends especially) will rate-limit
# without a cache.
TTL_ENV = {
    "fear_greed": ("SENTIMENT_TTL_FEAR_GREED", 3600),
    "google_trends": ("SENTIMENT_TTL_TRENDS", 3600),
    "reddit": ("SENTIMENT_TTL_REDDIT", 600),
    "cryptocurrency_cv": ("SENTIMENT_TTL_CRYPTOCV", 600),
    "lunarcrush": ("SENTIMENT_TTL_LUNARCRUSH", 900),
    "coingecko": ("SENTIMENT_TTL_COINGECKO", 300),
}

# An unavailable result (dead API, missing key) is cached only briefly so a
# down source is retried in about a minute instead of waiting out its full
# (possibly hour-long) normal TTL.
UNAVAILABLE_TTL_SECONDS = 60.0

MIN_COVERAGE_ENV = "SENTIMENT_MIN_COVERAGE"
DEFAULT_MIN_COVERAGE = 0.5


def _weight(env_name: str, default: float) -> float:
    raw = os.getenv(env_name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning("invalid %s=%r; using default %.2f", env_name, raw, default)
        return default


def load_weights() -> Dict[str, float]:
    """Load sentiment source weights from the environment."""
    return {key: _weight(env, default) for key, (env, default) in WEIGHT_ENV.items()}


def load_ttls() -> Dict[str, float]:
    """Load per-source cache TTLs (seconds) from the environment."""
    return {key: _weight(env, default) for key, (env, default) in TTL_ENV.items()}


def load_min_coverage() -> float:
    """Minimum fraction of positive source weight that must be available for
    a score to be considered actionable (see SentimentAggregator.get_sentiment)."""
    return _weight(MIN_COVERAGE_ENV, DEFAULT_MIN_COVERAGE)


@dataclass
class AggregatedSentiment:
    coin: str
    score: float  # weighted, normalized [-1, 1]
    label: str
    signals: Dict[str, SentimentSignal] = field(default_factory=dict)
    weights_used: Dict[str, float] = field(default_factory=dict)
    sources_available: List[str] = field(default_factory=list)
    # Fraction of total positive source weight actually available this call.
    # See SentimentAggregator.get_sentiment for how this and `actionable` are
    # computed for a real aggregator; fakes/stubs that build this directly
    # default to a fully-covered, actionable signal.
    coverage: float = 1.0
    actionable: bool = True

    def as_dict(self) -> Dict[str, object]:
        return {
            "coin": self.coin,
            "score": round(self.score, 4),
            "label": self.label,
            "coverage": round(self.coverage, 4),
            "actionable": self.actionable,
            "sources_available": self.sources_available,
            "weights_used": {k: round(v, 4) for k, v in self.weights_used.items()},
            "signals": {
                name: {
                    "score": round(sig.score, 4),
                    "available": sig.available,
                    "confidence": round(sig.confidence, 4),
                    "error": sig.error,
                }
                for name, sig in self.signals.items()
            },
        }


def _label(score: float) -> str:
    if score >= 0.5:
        return "very_bullish"
    if score >= 0.15:
        return "bullish"
    if score > -0.15:
        return "neutral"
    if score > -0.5:
        return "bearish"
    return "very_bearish"


class SentimentAggregator:
    """Fetches all sources and produces a weighted sentiment per coin."""

    def __init__(
        self,
        sources=None,
        weights: Dict[str, float] | None = None,
        ttls: Dict[str, float] | None = None,
        min_coverage: float | None = None,
        clock: Optional[Callable[[], float]] = None,
    ):
        self.sources = sources if sources is not None else self._default_sources()
        self.weights = weights if weights is not None else load_weights()
        self.ttls = ttls if ttls is not None else load_ttls()
        self.min_coverage = min_coverage if min_coverage is not None else load_min_coverage()
        self._clock = clock if clock is not None else time.monotonic
        # (source_key, COIN) -> (expires_at, signal)
        self._cache: Dict[Tuple[str, str], Tuple[float, SentimentSignal]] = {}

    @staticmethod
    def _default_sources():
        return {
            "reddit": RedditSource(),
            "cryptocurrency_cv": CryptoCurrencyCVSource(),
            "coingecko": CoinGeckoSource(),
            "lunarcrush": LunarCrushSource(),
            "google_trends": GoogleTrendsSource(),
            "fear_greed": FearGreedSource(),
        }

    def _get_signal(self, key: str, source, coin: str) -> SentimentSignal:
        cache_key = (key, coin.upper())
        now = self._clock()
        cached = self._cache.get(cache_key)
        if cached is not None and now < cached[0]:
            return cached[1]

        signal = source.get_signal(coin)
        ttl = self.ttls.get(key, 0.0)
        if not signal.available:
            ttl = min(ttl, UNAVAILABLE_TTL_SECONDS) if ttl > 0 else UNAVAILABLE_TTL_SECONDS
        if ttl > 0:
            self._cache[cache_key] = (now + ttl, signal)
        return signal

    def get_sentiment(self, coin: str) -> AggregatedSentiment:
        signals: Dict[str, SentimentSignal] = {}
        for key, source in self.sources.items():
            signals[key] = self._get_signal(key, source, coin)

        weighted_sum = 0.0
        total_weight = 0.0
        weights_used: Dict[str, float] = {}
        available: List[str] = []

        for key, sig in signals.items():
            base_w = self.weights.get(key, 0.0)
            if not sig.available or base_w <= 0.0:
                continue
            # Fold in the source's own confidence so thin signals count less.
            effective_w = base_w * max(sig.confidence, 1e-6)
            weighted_sum += sig.score * effective_w
            total_weight += effective_w
            weights_used[key] = base_w
            available.append(key)

        total_positive_weight = sum(w for w in self.weights.values() if w > 0.0)
        coverage = (
            sum(weights_used.values()) / total_positive_weight
            if total_positive_weight > 0
            else 0.0
        )
        actionable = coverage >= self.min_coverage

        if actionable:
            score = weighted_sum / total_weight if total_weight > 0 else 0.0
            label = _label(score)
        else:
            score = 0.0
            label = "insufficient_data"

        return AggregatedSentiment(
            coin=coin.upper(),
            score=score,
            label=label,
            signals=signals,
            weights_used=weights_used,
            sources_available=available,
            coverage=coverage,
            actionable=actionable,
        )

    def get_all(self, coins: List[str]) -> Dict[str, AggregatedSentiment]:
        return {coin.upper(): self.get_sentiment(coin) for coin in coins}

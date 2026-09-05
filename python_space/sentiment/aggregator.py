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
from dataclasses import dataclass, field
from typing import Dict, List

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


@dataclass
class AggregatedSentiment:
    coin: str
    score: float  # weighted, normalized [-1, 1]
    label: str
    signals: Dict[str, SentimentSignal] = field(default_factory=dict)
    weights_used: Dict[str, float] = field(default_factory=dict)
    sources_available: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "coin": self.coin,
            "score": round(self.score, 4),
            "label": self.label,
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

    def __init__(self, sources=None, weights: Dict[str, float] | None = None):
        self.sources = sources if sources is not None else self._default_sources()
        self.weights = weights if weights is not None else load_weights()

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

    def get_sentiment(self, coin: str) -> AggregatedSentiment:
        signals: Dict[str, SentimentSignal] = {}
        for key, source in self.sources.items():
            signals[key] = source.get_signal(coin)

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

        score = weighted_sum / total_weight if total_weight > 0 else 0.0
        return AggregatedSentiment(
            coin=coin.upper(),
            score=score,
            label=_label(score),
            signals=signals,
            weights_used=weights_used,
            sources_available=available,
        )

    def get_all(self, coins: List[str]) -> Dict[str, AggregatedSentiment]:
        return {coin.upper(): self.get_sentiment(coin) for coin in coins}

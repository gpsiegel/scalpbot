"""scalpbot sentiment stack.

Public API::

    from sentiment import SentimentAggregator

    agg = SentimentAggregator()
    result = agg.get_sentiment("SOL")
    print(result.score, result.label)

Sources & weights (weights come from plain .env operational config):

  * Reddit (OAuth)               30%  SENTIMENT_WEIGHT_REDDIT
  * cryptocurrency.cv (no key)   25%  SENTIMENT_WEIGHT_CRYPTOCV
  * CoinGecko (Basic tier)       20%  SENTIMENT_WEIGHT_COINGECKO
  * LunarCrush (hobby tier)      10%  SENTIMENT_WEIGHT_LUNARCRUSH
  * Google Trends (trendspyg)     8%  SENTIMENT_WEIGHT_TRENDS
  * Fear & Greed Index            7%  SENTIMENT_WEIGHT_FEAR_GREED

Removed (no longer part of the stack): CryptoPanic, Twitter/X, pytrends, NewsAPI.
"""
from .aggregator import (
    AggregatedSentiment,
    SentimentAggregator,
    load_weights,
)
from .base import COIN_METADATA, SentimentSignal
from .coingecko import CoinGeckoSource
from .cryptocurrency_cv import CryptoCurrencyCVSource
from .fear_greed import FearGreedSource
from .google_trends import GoogleTrendsSource
from .lunarcrush import LunarCrushSource
from .reddit_source import RedditSource

__all__ = [
    "SentimentAggregator",
    "AggregatedSentiment",
    "SentimentSignal",
    "COIN_METADATA",
    "load_weights",
    "RedditSource",
    "CryptoCurrencyCVSource",
    "CoinGeckoSource",
    "LunarCrushSource",
    "GoogleTrendsSource",
    "FearGreedSource",
]

"""cryptocurrency.cv sentiment source.

Free Crypto News API (https://github.com/nirholas/cryptocurrency.cv) that
aggregates 300+ crypto news sources with built-in AI sentiment. No API key or
signup required -- pure HTTP REST.

We use two endpoints, filtered per coin:
  * ``/api/sentiment``     -> aggregated sentiment, filtered by ticker
  * ``/api/archive?ticker=`` -> recent ticker-tagged articles (each with a
    sentiment label) as a fallback / confidence signal.

Weight in the aggregator: SENTIMENT_WEIGHT_CRYPTOCV (default 0.25).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

from .base import BaseSentimentSource, SentimentSignal, clamp

BASE_URL = "https://cryptocurrency.cv"

# Textual sentiment labels -> numeric score in [-1, 1].
_LABEL_MAP = {
    "very_bullish": 1.0,
    "very bullish": 1.0,
    "bullish": 0.6,
    "positive": 0.6,
    "somewhat_bullish": 0.3,
    "neutral": 0.0,
    "mixed": 0.0,
    "somewhat_bearish": -0.3,
    "bearish": -0.6,
    "negative": -0.6,
    "very_bearish": -1.0,
    "very bearish": -1.0,
}


class CryptoCurrencyCVSource(BaseSentimentSource):
    name = "cryptocurrency_cv"

    def __init__(self, base_url: str = BASE_URL, timeout: float = 10.0, article_limit: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.article_limit = article_limit
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    # ------------------------------------------------------------------
    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}{path}"
        resp = self.session.get(url, params=params or {}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _label_to_score(label: Any) -> Optional[float]:
        if label is None:
            return None
        if isinstance(label, (int, float)):
            # Some endpoints return a numeric score in [-1,1] or [0,100].
            val = float(label)
            if val > 1.0 or val < -1.0:
                val = clamp(val / 100.0 * 2.0 - 1.0) if val >= 0 else clamp(val / 100.0)
            return clamp(val)
        return _LABEL_MAP.get(str(label).strip().lower())

    def _fetch(self, coin: str) -> Optional[SentimentSignal]:
        meta = self.coin_meta(coin)
        ticker = meta["cryptocv_ticker"]

        scores: List[float] = []
        raw: Dict[str, Any] = {"ticker": ticker}

        # 1) Aggregated sentiment endpoint, filtered by ticker.
        try:
            data = self._get("/api/sentiment", {"ticker": ticker})
            agg = self._extract_aggregate_sentiment(data)
            if agg is not None:
                scores.append(agg)
                raw["sentiment_endpoint"] = agg
        except requests.RequestException:
            pass

        # 2) Ticker-tagged archive articles, each carrying a sentiment label.
        try:
            articles = self._get(
                "/api/archive",
                {"ticker": ticker, "limit": self.article_limit},
            )
            article_scores = self._extract_article_scores(articles)
            if article_scores:
                avg = sum(article_scores) / len(article_scores)
                scores.append(avg)
                raw["article_count"] = len(article_scores)
                raw["article_avg"] = avg
        except requests.RequestException:
            pass

        if not scores:
            return SentimentSignal.unavailable(self.name, coin, "no sentiment/articles returned")

        final = sum(scores) / len(scores)
        # Confidence grows with the number of articles observed.
        n = int(raw.get("article_count", 0))
        confidence = clamp(0.3 + min(n, 20) / 20.0 * 0.7, 0.0, 1.0)
        return SentimentSignal(
            source=self.name,
            coin=coin,
            score=final,
            confidence=confidence,
            raw=raw,
        )

    # ------------------------------------------------------------------
    def _extract_aggregate_sentiment(self, data: Any) -> Optional[float]:
        if data is None:
            return None
        if isinstance(data, dict):
            # Try common keys returned by the sentiment endpoint.
            for key in ("sentiment", "sentiment_score", "score", "overall", "average"):
                if key in data:
                    val = self._label_to_score(data[key])
                    if val is not None:
                        return val
            # Bullish/bearish counts.
            bull = data.get("bullish") or data.get("positive")
            bear = data.get("bearish") or data.get("negative")
            if isinstance(bull, (int, float)) and isinstance(bear, (int, float)):
                total = bull + bear
                if total:
                    return clamp((bull - bear) / total)
        return None

    def _extract_article_scores(self, data: Any) -> List[float]:
        items: List[Dict[str, Any]] = []
        if isinstance(data, dict):
            for key in ("articles", "news", "data", "results", "items"):
                if isinstance(data.get(key), list):
                    items = data[key]
                    break
        elif isinstance(data, list):
            items = data

        scores: List[float] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            label = (
                item.get("sentiment")
                or item.get("sentiment_label")
                or item.get("sentiment_score")
            )
            val = self._label_to_score(label)
            if val is not None:
                scores.append(val)
        return scores

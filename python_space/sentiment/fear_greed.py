"""Crypto Fear & Greed Index sentiment source (alternative.me).

Kept as-is: no API key required. The index is market-wide (not per-coin), so
the same value is applied to every coin in a cycle.

  * 0   = Extreme Fear   -> bearish (-1.0)
  * 50  = Neutral        -> 0.0
  * 100 = Extreme Greed  -> bullish (+1.0)

Endpoint: https://api.alternative.me/fng/

Weight in the aggregator: SENTIMENT_WEIGHT_FEAR_GREED (default 0.07).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import requests

from .base import BaseSentimentSource, SentimentSignal, scale_to_unit

URL = "https://api.alternative.me/fng/"


class FearGreedSource(BaseSentimentSource):
    name = "fear_greed"

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def _fetch(self, coin: str) -> Optional[SentimentSignal]:
        resp = self.session.get(URL, params={"limit": 1}, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        rows = data.get("data") if isinstance(data, dict) else None
        if not rows:
            return SentimentSignal.unavailable(self.name, coin, "empty fng payload")
        entry = rows[0]
        value = entry.get("value")
        try:
            value_num = float(value)
        except (TypeError, ValueError):
            return SentimentSignal.unavailable(self.name, coin, f"bad value: {value!r}")
        raw: Dict[str, Any] = {
            "value": value_num,
            "classification": entry.get("value_classification"),
        }
        score = scale_to_unit(value_num, 0.0, 100.0)
        return SentimentSignal(
            source=self.name, coin=coin, score=score, confidence=1.0, raw=raw
        )

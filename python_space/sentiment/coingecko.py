"""CoinGecko sentiment source (paid Basic tier).

Uses the CoinGecko **Basic** tier API. The API key is read from the
``COINGECKO_API_KEY`` environment variable (stored in Doppler as a secret) and
sent via the ``x-cg-demo-api-key`` / ``x-cg-pro-api-key`` header depending on
the configured host.

Signals combined per coin (all filtered to SOL / DOGE):
  * ``/coins/markets``            -> 24h/7d price momentum + up/down vote %
  * ``/search/trending``          -> is the coin currently trending (momentum)
  * ``/news``                     -> recent coin-relevant news headlines
    (scored with the local lexicon since CoinGecko news carries no sentiment)

Weight in the aggregator: SENTIMENT_WEIGHT_COINGECKO (default 0.20).

Basic tier notes: rate limit is far higher than the free/demo tier (~500
calls/min per CoinGecko Basic docs), so the three calls per coin per cycle are
comfortably within budget.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests

from .base import BaseSentimentSource, SentimentSignal, clamp
from .text_scoring import score_many

# Pro/Basic paid host. The paid tiers (Basic/Pro) share this host + header.
PRO_HOST = "https://pro-api.coingecko.com/api/v3"
PUBLIC_HOST = "https://api.coingecko.com/api/v3"


class CoinGeckoSource(BaseSentimentSource):
    name = "coingecko"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
        vs_currency: str = "usd",
    ):
        self.api_key = api_key if api_key is not None else os.getenv("COINGECKO_API_KEY", "")
        self.timeout = timeout
        self.vs_currency = vs_currency
        # Paid key -> pro host + pro header. Without a key we fall back to the
        # public host (degraded, but keeps dev environments functional).
        if self.api_key:
            self.host = PRO_HOST
            self.auth_header = {"x-cg-pro-api-key": self.api_key}
        else:
            self.host = PUBLIC_HOST
            self.auth_header = {}
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json", **self.auth_header})

    # ------------------------------------------------------------------
    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        resp = self.session.get(f"{self.host}{path}", params=params or {}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _fetch(self, coin: str) -> Optional[SentimentSignal]:
        if not self.api_key:
            return SentimentSignal.unavailable(
                self.name, coin, "COINGECKO_API_KEY not set (Basic tier required)"
            )
        meta = self.coin_meta(coin)
        coin_id = meta["coingecko_id"]
        symbol = meta["symbol"]

        components: List[float] = []
        weights: List[float] = []
        raw: Dict[str, Any] = {"coin_id": coin_id}

        # 1) Market momentum (price change + community up/down votes).
        market_score = self._market_score(coin_id, raw)
        if market_score is not None:
            components.append(market_score)
            weights.append(0.5)

        # 2) Trending momentum.
        trending_score = self._trending_score(coin_id, raw)
        if trending_score is not None:
            components.append(trending_score)
            weights.append(0.2)

        # 3) News headline sentiment.
        news_score = self._news_score(symbol, meta["name"], raw)
        if news_score is not None:
            components.append(news_score)
            weights.append(0.3)

        if not components:
            return SentimentSignal.unavailable(self.name, coin, "no CoinGecko data returned")

        total_w = sum(weights)
        final = sum(c * w for c, w in zip(components, weights)) / total_w
        confidence = clamp(0.4 + 0.2 * len(components), 0.0, 1.0)
        return SentimentSignal(
            source=self.name, coin=coin, score=final, confidence=confidence, raw=raw
        )

    # ------------------------------------------------------------------
    def _market_score(self, coin_id: str, raw: Dict[str, Any]) -> Optional[float]:
        try:
            data = self._get(
                "/coins/markets",
                {
                    "vs_currency": self.vs_currency,
                    "ids": coin_id,
                    "price_change_percentage": "24h,7d",
                    "sparkline": "false",
                },
            )
        except requests.RequestException:
            return None
        if not isinstance(data, list) or not data:
            return None
        row = data[0]
        parts: List[float] = []
        ch24 = row.get("price_change_percentage_24h_in_currency")
        if ch24 is None:
            ch24 = row.get("price_change_percentage_24h")
        ch7 = row.get("price_change_percentage_7d_in_currency")
        if ch24 is not None:
            # +-10% maps to +-1.0
            parts.append(clamp(float(ch24) / 10.0))
            raw["price_change_24h"] = ch24
        if ch7 is not None:
            parts.append(clamp(float(ch7) / 20.0))
            raw["price_change_7d"] = ch7
        if not parts:
            return None
        return sum(parts) / len(parts)

    def _trending_score(self, coin_id: str, raw: Dict[str, Any]) -> Optional[float]:
        try:
            data = self._get("/search/trending")
        except requests.RequestException:
            return None
        coins = (data or {}).get("coins", []) if isinstance(data, dict) else []
        ids = []
        for entry in coins:
            item = entry.get("item", {}) if isinstance(entry, dict) else {}
            if item.get("id"):
                ids.append(item["id"])
        raw["trending_ids"] = ids
        if coin_id in ids:
            # Rank-weighted: top of the trending list is a stronger signal.
            rank = ids.index(coin_id)
            raw["trending_rank"] = rank
            return clamp(1.0 - rank / max(len(ids), 1) * 0.5)
        # Not trending is mildly neutral-negative (no momentum), not bearish.
        return 0.0

    def _news_score(self, symbol: str, name: str, raw: Dict[str, Any]) -> Optional[float]:
        try:
            data = self._get("/news")
        except requests.RequestException:
            return None
        items: List[Dict[str, Any]] = []
        if isinstance(data, dict):
            items = data.get("data") or data.get("articles") or []
        elif isinstance(data, list):
            items = data
        if not items:
            return None
        # Filter to headlines mentioning this coin by name or symbol.
        needle_name = name.lower()
        needle_sym = symbol.lower()
        relevant: List[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            text = " ".join(
                str(item.get(k, "")) for k in ("title", "description", "summary")
            ).lower()
            if needle_name in text or f" {needle_sym} " in f" {text} ":
                relevant.append(text)
        raw["news_relevant_count"] = len(relevant)
        if not relevant:
            return None
        score, _hits = score_many(relevant)
        return score

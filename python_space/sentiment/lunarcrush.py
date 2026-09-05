"""LunarCrush sentiment source (free Hobby tier).

The free Hobby tier exposes point-in-time market/coin metrics but NOT the
paywalled social time-series endpoints. We therefore use **Galaxy Score** and
**AltRank** as a social-momentum proxy:

  * Galaxy Score (0-100): higher = healthier price + social activity -> bullish.
  * AltRank (1 = best): lower rank number = outperforming the market -> bullish.

API key is read from ``LUNARCRUSH_API_KEY`` (Doppler secret) and sent as a
Bearer token. Endpoint: ``/public/coins/{symbol}/v1`` on the v4 API.

Weight in the aggregator: SENTIMENT_WEIGHT_LUNARCRUSH (default 0.10).
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests

from .base import BaseSentimentSource, SentimentSignal, clamp, scale_to_unit

BASE_URL = "https://lunarcrush.com/api4"

# AltRank is 1..~few-thousand. Treat the top slice as meaningfully bullish.
ALT_RANK_BEST = 1
ALT_RANK_WORST = 500


class LunarCrushSource(BaseSentimentSource):
    name = "lunarcrush"

    def __init__(self, api_key: Optional[str] = None, timeout: float = 10.0):
        self.api_key = api_key if api_key is not None else os.getenv("LUNARCRUSH_API_KEY", "")
        self.timeout = timeout
        self.session = requests.Session()
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self.session.headers.update(headers)

    # ------------------------------------------------------------------
    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        resp = self.session.get(f"{BASE_URL}{path}", params=params or {}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _fetch(self, coin: str) -> Optional[SentimentSignal]:
        if not self.api_key:
            return SentimentSignal.unavailable(
                self.name, coin, "LUNARCRUSH_API_KEY not set"
            )
        meta = self.coin_meta(coin)
        symbol = meta["lunarcrush_symbol"]

        try:
            data = self._get(f"/public/coins/{symbol}/v1")
        except requests.RequestException as exc:
            return SentimentSignal.unavailable(self.name, coin, f"request failed: {exc}")

        payload = data.get("data", data) if isinstance(data, dict) else {}
        if isinstance(payload, list) and payload:
            payload = payload[0]
        if not isinstance(payload, dict):
            return SentimentSignal.unavailable(self.name, coin, "unexpected payload shape")

        galaxy = payload.get("galaxy_score")
        alt_rank = payload.get("alt_rank")
        raw: Dict[str, Any] = {
            "galaxy_score": galaxy,
            "alt_rank": alt_rank,
            "market_cap": payload.get("market_cap"),
            "volume_24h": payload.get("volume_24h"),
            "percent_change_24h": payload.get("percent_change_24h"),
        }

        components = []
        if isinstance(galaxy, (int, float)):
            # Galaxy score 0..100 -> [-1, 1]; 50 is neutral.
            components.append(scale_to_unit(float(galaxy), 0.0, 100.0))
        if isinstance(alt_rank, (int, float)):
            # Lower rank number is better, so invert: best rank -> +1, worst -> -1.
            rank = clamp(float(alt_rank), ALT_RANK_BEST, ALT_RANK_WORST)
            components.append(-scale_to_unit(rank, ALT_RANK_BEST, ALT_RANK_WORST))

        if not components:
            return SentimentSignal.unavailable(
                self.name, coin, "no galaxy_score/alt_rank in payload"
            )

        score = sum(components) / len(components)
        confidence = clamp(0.4 + 0.3 * len(components), 0.0, 1.0)
        return SentimentSignal(
            source=self.name, coin=coin, score=score, confidence=confidence, raw=raw
        )

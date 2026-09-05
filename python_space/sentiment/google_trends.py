"""Google Trends sentiment source (trendspyg).

Replaces the archived ``pytrends`` library with ``trendspyg`` (actively
maintained). We keep the *same public interface* within scalpbot -- the
aggregator still just calls :meth:`get_signal` -- so this is a drop-in swap
from the bot's point of view.

Signal: rising search interest for a coin over the recent window is treated as
bullish momentum; falling interest as bearish. We compare the most recent
interest points to the trailing average.

Query keywords per coin (e.g. "Solana", "SOL crypto").

Weight in the aggregator: SENTIMENT_WEIGHT_TRENDS (default 0.08).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BaseSentimentSource, SentimentSignal, clamp


class GoogleTrendsSource(BaseSentimentSource):
    name = "google_trends"

    def __init__(self, timeframe: str = "now 7-d", geo: str = "US", timeout: float = 30.0):
        self.timeframe = timeframe
        self.geo = geo
        self.timeout = timeout

    # ------------------------------------------------------------------
    def _interest_series(self, keyword: str) -> List[float]:
        """Return the interest-over-time values for a keyword via trendspyg.

        trendspyg's ``download_google_trends_interest_over_time`` is the direct
        successor to pytrends' ``interest_over_time``.
        """
        import trendspyg

        data = trendspyg.download_google_trends_interest_over_time(
            keyword,
            geo=self.geo,
            timeframe=self.timeframe,
            output_format="dict",
        )
        values: List[float] = []
        if isinstance(data, list):
            for point in data:
                if not isinstance(point, dict):
                    continue
                val = point.get("value")
                if val is None:
                    # Some rows key the value by the keyword itself.
                    val = point.get(keyword)
                if isinstance(val, (list, tuple)) and val:
                    val = val[0]
                if isinstance(val, (int, float)):
                    values.append(float(val))
        return values

    @staticmethod
    def _momentum(values: List[float]) -> Optional[float]:
        """Compare recent interest to the trailing baseline -> [-1, 1]."""
        if len(values) < 4:
            return None
        recent = values[-3:]
        baseline = values[:-3]
        recent_avg = sum(recent) / len(recent)
        base_avg = sum(baseline) / len(baseline) if baseline else recent_avg
        if base_avg == 0:
            return 0.0 if recent_avg == 0 else 1.0
        pct_change = (recent_avg - base_avg) / base_avg
        # +-50% swing in search interest maps to +-1.0.
        return clamp(pct_change / 0.5)

    def _fetch(self, coin: str) -> Optional[SentimentSignal]:
        meta = self.coin_meta(coin)
        keywords = meta["trends_keywords"]

        scores: List[float] = []
        raw: Dict[str, Any] = {"keywords": keywords}
        for kw in keywords:
            try:
                series = self._interest_series(kw)
            except Exception:  # noqa: BLE001 - a single keyword failing is fine
                continue
            m = self._momentum(series)
            if m is not None:
                scores.append(m)
                raw[f"momentum::{kw}"] = m

        if not scores:
            return SentimentSignal.unavailable(self.name, coin, "no trends data returned")

        score = sum(scores) / len(scores)
        confidence = clamp(0.3 + 0.35 * len(scores), 0.0, 1.0)
        return SentimentSignal(
            source=self.name, coin=coin, score=score, confidence=confidence, raw=raw
        )

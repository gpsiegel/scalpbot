"""Offline unit tests for the sentiment stack.

These tests avoid live network calls: sources are stubbed so the aggregator's
weighting / renormalization / labeling logic is verified deterministically.

Run:  python -m pytest python_space/tests/test_sentiment.py
Or:   python python_space/tests/test_sentiment.py   (falls back to a runner)
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sentiment.aggregator import SentimentAggregator, load_weights  # noqa: E402
from sentiment.base import (  # noqa: E402
    BaseSentimentSource,
    SentimentSignal,
    clamp,
    scale_to_unit,
)
from sentiment.text_scoring import score_many, score_text  # noqa: E402


class _StubSource(BaseSentimentSource):
    def __init__(self, name, score, available=True, confidence=1.0):
        self.name = name
        self._score = score
        self._available = available
        self._confidence = confidence

    def _fetch(self, coin):
        if not self._available:
            return SentimentSignal.unavailable(self.name, coin, "stubbed unavailable")
        return SentimentSignal(
            source=self.name, coin=coin, score=self._score, confidence=self._confidence
        )


def test_clamp_and_scale():
    assert clamp(5) == 1.0
    assert clamp(-5) == -1.0
    assert scale_to_unit(50, 0, 100) == 0.0
    assert scale_to_unit(100, 0, 100) == 1.0
    assert scale_to_unit(0, 0, 100) == -1.0


def test_text_scoring():
    pos, hits = score_text("SOL to the moon, massive bullish breakout rally")
    assert pos > 0 and hits >= 3
    neg, _ = score_text("total crash, dump incoming, bearish scam")
    assert neg < 0
    neutral, h = score_text("the price of the coin today")
    assert neutral == 0.0 and h == 0
    agg, _ = score_many(["bullish pump", "bearish dump"])
    assert -0.5 <= agg <= 0.5


def test_weighted_aggregation_all_available():
    sources = {
        "reddit": _StubSource("reddit", 1.0),
        "cryptocurrency_cv": _StubSource("cryptocurrency_cv", 1.0),
        "coingecko": _StubSource("coingecko", 1.0),
        "lunarcrush": _StubSource("lunarcrush", 1.0),
        "google_trends": _StubSource("google_trends", 1.0),
        "fear_greed": _StubSource("fear_greed", 1.0),
    }
    agg = SentimentAggregator(sources=sources, weights=load_weights())
    res = agg.get_sentiment("SOL")
    assert abs(res.score - 1.0) < 1e-9
    assert res.label == "very_bullish"
    assert len(res.sources_available) == 6


def test_renormalization_when_sources_drop():
    # Only two sources available; result must still span [-1, 1].
    sources = {
        "reddit": _StubSource("reddit", 1.0),
        "coingecko": _StubSource("coingecko", -1.0),
        "fear_greed": _StubSource("fear_greed", 0.0, available=False),
    }
    weights = {"reddit": 0.30, "coingecko": 0.20, "fear_greed": 0.07}
    agg = SentimentAggregator(sources=sources, weights=weights)
    res = agg.get_sentiment("DOGE")
    # weighted: (1*0.30 + -1*0.20) / 0.50 = 0.2
    assert abs(res.score - 0.2) < 1e-9
    assert "fear_greed" not in res.sources_available


def test_all_unavailable_is_insufficient_data():
    # A fully-blind aggregator (0% coverage) must not be mistaken for a
    # genuinely neutral market -- that was exactly the bug PR2 fixes.
    sources = {"reddit": _StubSource("reddit", 1.0, available=False)}
    agg = SentimentAggregator(sources=sources, weights={"reddit": 0.30})
    res = agg.get_sentiment("SOL")
    assert res.score == 0.0
    assert res.actionable is False
    assert res.label == "insufficient_data"


def test_low_coverage_stock_ticker_is_not_actionable():
    # Reproduces the PR2 bug: only fear_greed resolves a stock ticker, and its
    # weight alone (0.07 of 1.00) is far short of the 0.5 coverage floor.
    sources = {
        "reddit": _StubSource("reddit", 0.0, available=False),
        "cryptocurrency_cv": _StubSource("cryptocurrency_cv", 0.0, available=False),
        "coingecko": _StubSource("coingecko", 0.0, available=False),
        "lunarcrush": _StubSource("lunarcrush", 0.0, available=False),
        "google_trends": _StubSource("google_trends", 0.0, available=False),
        "fear_greed": _StubSource("fear_greed", 0.65),
    }
    agg = SentimentAggregator(sources=sources, weights=load_weights())
    res = agg.get_sentiment("PLTR")
    assert res.actionable is False
    assert res.score == 0.0
    assert res.label == "insufficient_data"


def test_sufficient_crypto_coverage_is_actionable():
    sources = {
        "reddit": _StubSource("reddit", 0.5),
        "cryptocurrency_cv": _StubSource("cryptocurrency_cv", 0.5),
        "coingecko": _StubSource("coingecko", 0.5),
    }
    weights = {"reddit": 0.30, "cryptocurrency_cv": 0.25, "coingecko": 0.20,
               "lunarcrush": 0.10, "google_trends": 0.08, "fear_greed": 0.07}
    agg = SentimentAggregator(sources=sources, weights=weights)
    res = agg.get_sentiment("SOL")
    assert res.coverage == pytest.approx(0.75 / 1.00)
    assert res.actionable is True


def test_ttl_cache_avoids_refetch_within_ttl():
    calls = {"n": 0}

    class CountingSource(BaseSentimentSource):
        name = "reddit"

        def _fetch(self, coin):
            calls["n"] += 1
            return SentimentSignal(source=self.name, coin=coin, score=0.5, confidence=1.0)

    clock = {"t": 0.0}
    agg = SentimentAggregator(
        sources={"reddit": CountingSource()},
        weights={"reddit": 0.30},
        ttls={"reddit": 600.0},
        clock=lambda: clock["t"],
    )
    agg.get_sentiment("SOL")
    clock["t"] = 100.0  # still within the 600s TTL
    agg.get_sentiment("SOL")
    assert calls["n"] == 1

    clock["t"] = 601.0  # TTL expired -> refetch
    agg.get_sentiment("SOL")
    assert calls["n"] == 2


def test_unavailable_result_cached_only_briefly():
    calls = {"n": 0}

    class FlakySource(BaseSentimentSource):
        name = "fear_greed"

        def _fetch(self, coin):
            calls["n"] += 1
            return SentimentSignal.unavailable(self.name, coin, "down")

    clock = {"t": 0.0}
    agg = SentimentAggregator(
        sources={"fear_greed": FlakySource()},
        weights={"fear_greed": 0.07},
        ttls={"fear_greed": 3600.0},  # long normal TTL
        clock=lambda: clock["t"],
    )
    agg.get_sentiment("SOL")
    clock["t"] = 61.0  # well past the ~60s unavailable-result cap, far short of 3600s
    agg.get_sentiment("SOL")
    assert calls["n"] == 2


def test_env_weight_override(monkeypatch=None):
    os.environ["SENTIMENT_WEIGHT_REDDIT"] = "0.99"
    try:
        w = load_weights()
        assert w["reddit"] == 0.99
    finally:
        del os.environ["SENTIMENT_WEIGHT_REDDIT"]


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  ok  {fn.__name__}")
    print(f"\n{passed}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run()

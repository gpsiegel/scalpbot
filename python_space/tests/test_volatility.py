"""Unit tests for the pure ATR helpers in engine/volatility.py."""
from __future__ import annotations

import pytest

from engine.volatility import compute_atr, true_range


def test_true_range_picks_the_largest_component():
    # gap up from prior close dominates.
    assert true_range(high=105, low=103, prev_close=100) == pytest.approx(5)
    # gap down from prior close dominates.
    assert true_range(high=101, low=99, prev_close=105) == pytest.approx(6)
    # ordinary bar: high-low is largest.
    assert true_range(high=102, low=98, prev_close=100) == pytest.approx(4)


def test_compute_atr_is_average_of_true_ranges():
    # closes flat at 100, but each bar has a 2-point high-low range.
    bars = [(101.0, 99.0, 100.0)] * 15  # 15 bars -> period=14 possible
    atr = compute_atr(bars, period=14)
    assert atr == pytest.approx(2.0)


def test_compute_atr_none_with_insufficient_bars():
    bars = [(101.0, 99.0, 100.0)] * 10  # need period+1 = 15
    assert compute_atr(bars, period=14) is None


def test_compute_atr_uses_only_the_most_recent_period():
    # an old, huge-range bar falls outside the most recent 14 true ranges
    # once there are enough newer bars to push it out of the window.
    bars = [(50.0, 50.0, 50.0), (200.0, 0.0, 100.0)] + [(101.0, 99.0, 100.0)] * 14
    atr = compute_atr(bars, period=14)
    assert atr == pytest.approx(2.0)

"""Tests for the scheduler (python_space/runner.py).

Uses an injected clock/sleep so the loop runs instantly and deterministically
-- no real time passes and no real TradingEngine/DB/Alpaca is touched.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("APP_ENV", "nonprod")

from engine.engine import CycleReport, MARKET_CRYPTO, MARKET_OPTION, MARKET_STOCK  # noqa: E402
from runner import Runner  # noqa: E402


class _FakeClock:
    """A clock advanced only by the injected `sleep` calls it also serves."""

    def __init__(self):
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


class FakeEngine:
    def __init__(self):
        self.exit_calls = []
        self.entry_calls = []

    def manage_exits(self, market):
        self.exit_calls.append(market)
        return CycleReport(market=market, mode="paper")

    def open_entries(self, market):
        self.entry_calls.append(market)
        return CycleReport(market=market, mode="paper")


def test_runner_schedules_exits_more_often_than_entries():
    clock = _FakeClock()
    engine = FakeEngine()
    r = Runner(
        engine=engine, exit_interval=10, entry_interval=100,
        sleep=clock.sleep, clock=clock.now,
    )
    r.run_forever(max_iterations=50)
    assert len(engine.exit_calls) > len(engine.entry_calls)
    assert len(engine.entry_calls) > 0
    # every tick covers all three markets
    assert set(engine.exit_calls) == {MARKET_CRYPTO, MARKET_STOCK, MARKET_OPTION}


def test_runner_survives_exception_in_one_market():
    clock = _FakeClock()

    class FlakyEngine(FakeEngine):
        def manage_exits(self, market):
            self.exit_calls.append(market)
            if market == MARKET_STOCK and len(self.exit_calls) == 2:
                raise RuntimeError("boom")
            return CycleReport(market=market, mode="paper")

    engine = FlakyEngine()
    r = Runner(
        engine=engine, exit_interval=1, entry_interval=1000,
        sleep=clock.sleep, clock=clock.now,
    )
    r.run_forever(max_iterations=10)  # must not raise
    # the exception on one market in one iteration didn't stop later ticks
    assert len(engine.exit_calls) > 3
    assert MARKET_OPTION in engine.exit_calls  # sibling market in the same tick still ran


def test_runner_stops_via_request_stop():
    clock = _FakeClock()
    engine = FakeEngine()
    r = Runner(engine=engine, exit_interval=1, entry_interval=1, sleep=clock.sleep, clock=clock.now)

    calls = {"n": 0}
    real_sleep = clock.sleep

    def sleep_and_maybe_stop(seconds):
        real_sleep(seconds)
        calls["n"] += 1
        if calls["n"] >= 3:
            r.request_stop()

    r._sleep = sleep_and_maybe_stop
    r.run_forever()  # no max_iterations -- relies on request_stop()
    assert calls["n"] == 3

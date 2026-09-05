"""Tests for the trading engine, avenues, and options logic.

Everything runs against an in-memory/temp SQLite DB with fake Alpaca and fake
sentiment -- no network, no Postgres, no real orders.
"""
from __future__ import annotations

import os

import pytest
from sqlalchemy.orm import sessionmaker

# Configure a deterministic paper environment BEFORE importing config.
os.environ.setdefault("APP_ENV", "nonprod")
os.environ.setdefault("STOCK_TICKERS", "PLTR,SCHD")
os.environ.setdefault("CRYPTO_CORE_COINS", "SOL,BTC")
os.environ.setdefault("CRYPTO_SATELLITE_COINS", "DOGE")

from config import Config, OptionsFilters  # noqa: E402
from engine import avenues as av  # noqa: E402
from engine import options as opt  # noqa: E402
from engine.alpaca_client import OptionContract  # noqa: E402
from engine.engine import TradingEngine  # noqa: E402
from engine.models import Base, Position, POSITION_OPEN, Trade, make_engine  # noqa: E402
from sentiment.aggregator import AggregatedSentiment  # noqa: E402


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------
class FakeAgg:
    def __init__(self, score: float):
        self.score = score

    def get_sentiment(self, coin: str):
        return AggregatedSentiment(coin=coin.upper(), score=self.score, label="x")


class FakeAlpaca:
    def __init__(self, price=100.0, spot=20.0):
        self.price = price
        self.spot = spot
        self.orders = []

    def get_crypto_price(self, s):
        return self.price

    def get_stock_price(self, s):
        return self.spot

    def is_market_open(self):
        return True

    def list_option_contracts(self, underlying, option_type, **kw):
        return [
            OptionContract(symbol=f"{underlying}_ATM", underlying=underlying,
                           option_type=option_type, strike=self.spot + 1,
                           expiration="2099-01-17", bid=0.50, ask=0.55, open_interest=100),
            OptionContract(symbol=f"{underlying}_DEEP", underlying=underlying,
                           option_type=option_type, strike=self.spot + 10,
                           expiration="2099-01-17", bid=0.03, ask=0.05, open_interest=5),
        ]

    def submit_crypto_order(self, *a, **k):
        self.orders.append(("crypto", a))

    def submit_stock_order(self, *a, **k):
        self.orders.append(("stock", a))

    def submit_option_order(self, *a, **k):
        self.orders.append(("option", a))


@pytest.fixture()
def session_factory(tmp_path):
    db = tmp_path / "t.db"
    eng = make_engine(f"sqlite:///{db}")
    Base.metadata.create_all(eng)
    return sessionmaker(bind=eng, expire_on_commit=False, future=True)


def _cfg(**over) -> Config:
    base = dict(
        app_env="nonprod",
        stock_tickers=["PLTR", "SCHD"],
        crypto_core_coins=["SOL", "BTC"],
        crypto_satellite_coins=["DOGE"],
    )
    base.update(over)
    return Config(**base)


# --------------------------------------------------------------------------
# avenues (pure logic)
# --------------------------------------------------------------------------
def test_crypto_entry_bullish_only():
    assert av.crypto_entry_decision("SOL/USD", 0.5).is_open
    assert not av.crypto_entry_decision("SOL/USD", 0.0).is_open


def test_stock_short_gated_by_allow_short():
    d_no = av.stock_entry_decision("F", -0.5, allow_short=False)
    assert d_no.action == "skip"
    d_yes = av.stock_entry_decision("F", -0.5, allow_short=True)
    assert d_yes.is_open and d_yes.side == "short"


def test_linear_exit_take_profit_and_stop():
    tp = av.linear_exit_decision("crypto", "SOL/USD", "long", 100.0, 104.0, 0.5)
    assert tp.is_close and "take profit" in tp.reason
    sl = av.linear_exit_decision("crypto", "SOL/USD", "long", 100.0, 97.0, 0.5)
    assert sl.is_close and "stop loss" in sl.reason
    hold = av.linear_exit_decision("crypto", "SOL/USD", "long", 100.0, 100.5, 0.5)
    assert hold.action == "hold"


def test_position_pnl_direction():
    assert av.position_pnl_pct("long", 100, 110) == pytest.approx(0.10)
    assert av.position_pnl_pct("short", 100, 90) == pytest.approx(0.10)


# --------------------------------------------------------------------------
# options logic
# --------------------------------------------------------------------------
def test_options_filters_reject_deep_otm_and_low_premium():
    f = OptionsFilters()
    good = OptionContract(symbol="G", underlying="P", option_type="call",
                          strike=21, expiration="2099-01-17", bid=0.5, ask=0.55)
    deep = OptionContract(symbol="D", underlying="P", option_type="call",
                          strike=40, expiration="2099-01-17", bid=0.5, ask=0.55)
    cheap = OptionContract(symbol="C", underlying="P", option_type="call",
                           strike=21, expiration="2099-01-17", bid=0.01, ask=0.02)
    assert opt.passes_filters(good, 20, f)
    assert not opt.passes_filters(deep, 20, f)
    assert not opt.passes_filters(cheap, 20, f)


def test_options_reject_near_expiry():
    import datetime as dt
    f = OptionsFilters()
    soon = (dt.date.today() + dt.timedelta(days=2)).isoformat()
    c = OptionContract(symbol="S", underlying="P", option_type="call",
                       strike=21, expiration=soon, bid=0.5, ask=0.55)
    assert not opt.passes_filters(c, 20, f)


def test_options_select_prefers_atm():
    f = OptionsFilters()
    contracts = [
        OptionContract(symbol="ATM", underlying="P", option_type="call",
                       strike=21, expiration="2099-01-17", bid=0.5, ask=0.55),
        OptionContract(symbol="OTM", underlying="P", option_type="call",
                       strike=22, expiration="2099-01-17", bid=0.3, ask=0.35),
    ]
    choice = opt.select_contract(contracts, 20, "call", f)
    assert choice is not None and choice.contract.symbol == "ATM"


def test_options_exit_ladder_and_pnl():
    assert opt.exit_decision(1.0, 1.25) == "take_profit_20"
    assert opt.exit_decision(1.0, 1.45) == "take_profit_40"
    assert opt.exit_decision(1.0, 1.65) == "take_profit_60"
    assert opt.exit_decision(1.0, 0.60) == "stop_loss"
    assert opt.exit_decision(1.0, 1.05) is None
    assert opt.option_pnl(1.0, 1.5, 2) == pytest.approx(100.0)


def test_contracts_for_budget():
    assert opt.contracts_for_budget(0.50, 100) == 2   # $50 each
    assert opt.contracts_for_budget(2.00, 100) == 0    # $200 each, unaffordable


# --------------------------------------------------------------------------
# engine cycles
# --------------------------------------------------------------------------
def test_crypto_cycle_opens_on_bullish(session_factory):
    e = TradingEngine(config=_cfg(), session_factory=session_factory,
                      alpaca=FakeAlpaca(price=150.0), aggregator=FakeAgg(0.6))
    r = e.run_crypto_cycle()
    assert r.opened >= 1
    s = session_factory()
    assert s.query(Position).filter(Position.market == "crypto",
                                    Position.status == POSITION_OPEN).count() == r.opened
    s.close()


def test_total_cap_enforced(session_factory):
    cfg = _cfg()
    cfg.caps.total = 2
    cfg.caps.crypto = 5
    e = TradingEngine(config=cfg, session_factory=session_factory,
                      alpaca=FakeAlpaca(price=150.0), aggregator=FakeAgg(0.9))
    r = e.run_crypto_cycle()
    assert r.opened == 2  # capped at total even though 3 coins are bullish


def test_no_double_open_same_symbol(session_factory):
    e = TradingEngine(config=_cfg(), session_factory=session_factory,
                      alpaca=FakeAlpaca(price=150.0), aggregator=FakeAgg(0.6))
    e.run_crypto_cycle()
    before = _count_open(session_factory)
    e.run_crypto_cycle()  # second cycle should not re-open the same coins
    after = _count_open(session_factory)
    assert before == after


def test_stock_cycle_respects_market_hours(session_factory):
    fake = FakeAlpaca()
    fake.is_market_open = lambda: False
    e = TradingEngine(config=_cfg(), session_factory=session_factory,
                      alpaca=fake, aggregator=FakeAgg(0.6))
    r = e.run_stock_cycle()
    assert r.opened == 0 and "market closed" in r.errors


def test_options_cycle_opens_call_on_strong_bull(session_factory):
    e = TradingEngine(config=_cfg(), session_factory=session_factory,
                      alpaca=FakeAlpaca(spot=20.0), aggregator=FakeAgg(0.6))
    r = e.run_options_cycle()
    assert r.opened >= 1
    s = session_factory()
    pos = s.query(Position).filter(Position.market == "option").first()
    assert pos.option_type == "call" and pos.leverage >= 1
    s.close()


def test_mode_paper_by_default(session_factory):
    e = TradingEngine(config=_cfg(), session_factory=session_factory,
                      alpaca=FakeAlpaca(), aggregator=FakeAgg(0.6))
    assert e.mode == "paper"


def test_live_requires_all_guards():
    assert _cfg(app_env="prod", paper_trading=False, live_trading=True).is_live()
    assert not _cfg(app_env="staging", paper_trading=False, live_trading=True).is_live()
    assert not _cfg(app_env="prod", paper_trading=True, live_trading=True).is_live()


def _count_open(session_factory) -> int:
    s = session_factory()
    try:
        return s.query(Position).filter(Position.status == POSITION_OPEN).count()
    finally:
        s.close()

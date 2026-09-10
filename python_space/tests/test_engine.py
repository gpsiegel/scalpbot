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
from engine.alpaca_client import OptionContract, OrderResult  # noqa: E402
from engine.engine import MARKET_CRYPTO, MARKET_OPTION, MARKET_STOCK, TradingEngine  # noqa: E402
from engine.models import (  # noqa: E402
    Base,
    Position,
    POSITION_CLOSED,
    POSITION_OPEN,
    SignalLog,
    Trade,
    TRADE_REJECTED,
    make_engine,
)
from sentiment.aggregator import AggregatedSentiment  # noqa: E402


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------
class FakeAgg:
    def __init__(self, score: float, actionable: bool = True, coverage: float = 1.0):
        self.score = score
        self.actionable = actionable
        self.coverage = coverage

    def get_sentiment(self, coin: str):
        return AggregatedSentiment(
            coin=coin.upper(), score=self.score, label="x",
            actionable=self.actionable, coverage=self.coverage,
        )


class FakeAlpaca:
    """Fake Alpaca client for engine tests.

    Supports rejecting the next order (``reject_next_order``), tracking
    per-symbol broker-held quantities (``held_qty``) so ``close_position``
    can return a qty that differs from what the engine thinks it bought, and
    a crypto buy-side fee haircut (``crypto_fee_pct``) modeling fees paid in
    the asset received.
    """

    def __init__(self, price=100.0, spot=20.0):
        self.price = price
        self.spot = spot
        self.orders = []
        self.close_calls = 0
        self.held_qty: dict = {}
        self.option_quotes: dict = {}  # symbol -> (bid, ask), absent == no quote
        self.crypto_fee_pct = 0.0
        self._reject_next = False
        self._reject_error = "simulated rejection"
        self._order_seq = 0

    def reject_next_order(self, error="simulated rejection"):
        self._reject_next = True
        self._reject_error = error

    def _next_order_id(self):
        self._order_seq += 1
        return f"fake-{self._order_seq}"

    def get_crypto_price(self, s):
        return self.price

    def get_stock_price(self, s):
        return self.spot

    def is_market_open(self):
        return True

    def list_option_contracts(self, underlying, option_type, spot=None, filters=None, **kw):
        return [
            OptionContract(symbol=f"{underlying}_ATM", underlying=underlying,
                           option_type=option_type, strike=self.spot + 1,
                           expiration="2099-01-17", bid=0.52, ask=0.55, open_interest=100),
            OptionContract(symbol=f"{underlying}_DEEP", underlying=underlying,
                           option_type=option_type, strike=self.spot + 10,
                           expiration="2099-01-17", bid=0.03, ask=0.05, open_interest=5),
        ]

    def get_option_quote(self, symbol):
        return self.option_quotes.get(symbol)

    def _fill(self, symbol, side, qty, price, market) -> OrderResult:
        if self._reject_next:
            self._reject_next = False
            return OrderResult(ok=False, error=self._reject_error)
        filled_qty = qty
        if market == "crypto" and side == "buy":
            filled_qty = qty * (1 - self.crypto_fee_pct)
        held = self.held_qty.get(symbol, 0.0)
        held = held + filled_qty if side == "buy" else held - qty
        self.held_qty[symbol] = held
        return OrderResult(
            ok=True, order_id=self._next_order_id(), filled_qty=filled_qty,
            filled_price=price, status="filled",
        )

    def submit_crypto_order(self, symbol, side, qty):
        self.orders.append(("crypto", symbol, side, qty))
        return self._fill(symbol, side, qty, self.price, "crypto")

    def submit_stock_order(self, symbol, side, qty):
        self.orders.append(("stock", symbol, side, qty))
        return self._fill(symbol, side, qty, self.spot, "stock")

    def submit_option_order(self, symbol, side, contracts):
        self.orders.append(("option", symbol, side, contracts))
        return self._fill(symbol, side, contracts, 0.55, "option")

    def close_position(self, symbol) -> OrderResult:
        self.close_calls += 1
        if self._reject_next:
            self._reject_next = False
            return OrderResult(ok=False, error=self._reject_error)
        qty = self.held_qty.get(symbol, 0.0)
        self.held_qty[symbol] = 0.0
        if qty <= 0:
            return OrderResult(ok=False, error="no position")
        return OrderResult(
            ok=True, order_id=self._next_order_id(), filled_qty=qty,
            filled_price=self.price, status="filled",
        )

    def get_position_qty(self, symbol):
        return self.held_qty.get(symbol)

    def list_positions(self):
        return []


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
    # bid/ask kept under the 0.08 default spread cap -- this test is about
    # OTM/premium rejection, not spread.
    good = OptionContract(symbol="G", underlying="P", option_type="call",
                          strike=21, expiration="2099-01-17", bid=0.52, ask=0.55)
    deep = OptionContract(symbol="D", underlying="P", option_type="call",
                          strike=40, expiration="2099-01-17", bid=0.52, ask=0.55)
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
                       strike=21, expiration=soon, bid=0.52, ask=0.55)
    assert not opt.passes_filters(c, 20, f)


def test_options_select_prefers_atm():
    f = OptionsFilters()
    contracts = [
        OptionContract(symbol="ATM", underlying="P", option_type="call",
                       strike=21, expiration="2099-01-17", bid=0.52, ask=0.55),
        OptionContract(symbol="OTM", underlying="P", option_type="call",
                       strike=22, expiration="2099-01-17", bid=0.335, ask=0.35),
    ]
    choice = opt.select_contract(contracts, 20, "call", f)
    assert choice is not None and choice.contract.symbol == "ATM"


def test_options_exit_single_target_and_pnl():
    assert opt.exit_decision(1.0, 1.45, take_profit_pct=0.40) == "take_profit"
    assert opt.exit_decision(1.0, 0.60, take_profit_pct=0.40) == "stop_loss"
    assert opt.exit_decision(1.0, 1.05, take_profit_pct=0.40) is None
    assert opt.option_pnl(1.0, 1.5, 2) == pytest.approx(100.0)


def test_options_exit_float_boundary():
    # (0.70 - 0.50) / 0.50 computes as 0.3999999999999999 in float -- must
    # still clear a 0.40 target rather than falling through to "hold".
    assert opt.exit_decision(0.50, 0.70, take_profit_pct=0.40) == "take_profit"


def test_contracts_for_budget():
    assert opt.contracts_for_budget(0.50, 100) == 2   # $50 each
    assert opt.contracts_for_budget(2.00, 100) == 0    # $200 each, unaffordable


def test_default_spread_filter_rejects_wide_spread():
    # OPTION_MAX_SPREAD_PCT default is 0.08; a 24% spread must be rejected.
    f = OptionsFilters()
    wide = OptionContract(symbol="W", underlying="P", option_type="call",
                          strike=21, expiration="2099-01-17", bid=0.44, ask=0.56)
    assert not opt.passes_filters(wide, 20, f)


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


def test_options_cycle_no_duplicate_underlying(session_factory):
    e = TradingEngine(config=_cfg(), session_factory=session_factory,
                      alpaca=FakeAlpaca(spot=20.0), aggregator=FakeAgg(0.6))
    e.run_options_cycle()
    first_count = _count_open(session_factory)
    assert first_count >= 1
    e.run_options_cycle()  # still strongly bullish -- must not open a second contract
    assert _count_open(session_factory) == first_count


def _open_option(symbol="PLTR_ATM", underlying="PLTR", entry_price=0.55,
                  expiration="2099-01-17"):
    # Marked simulated so TradingEngine.reconcile() (which runs at the top of
    # every cycle) doesn't treat this directly-inserted row as a real
    # position missing at the broker and close it out from under the test.
    return Position(
        market=MARKET_OPTION, symbol=symbol, side="long", qty=1, leverage=1.0,
        entry_price=entry_price, current_price=entry_price,
        entry_notional=entry_price * 100, entry_fees=0.0,
        underlying=underlying, strike=21.0, option_type="call",
        expiration=expiration, sentiment_score=0.5, status=POSITION_OPEN,
        extra={"simulated": True},
    )


def test_option_management_closes_on_bid_target(session_factory):
    fake = FakeAlpaca()
    e = TradingEngine(config=_cfg(), session_factory=session_factory,
                      alpaca=fake, aggregator=FakeAgg(0.0))
    s = session_factory()
    pos = _open_option(entry_price=0.50)
    s.add(pos)
    s.commit()
    s.close()
    fake.option_quotes[pos.symbol] = (0.70, 0.72)  # bid up 40% -> take profit
    r = e.run_options_cycle()
    assert r.closed == 1
    s2 = session_factory()
    reloaded = s2.query(Position).filter(Position.symbol == pos.symbol).first()
    assert reloaded.status == POSITION_CLOSED
    s2.close()


def test_option_management_holds_with_no_quote(session_factory):
    fake = FakeAlpaca()
    e = TradingEngine(config=_cfg(), session_factory=session_factory,
                      alpaca=fake, aggregator=FakeAgg(0.0))
    s = session_factory()
    pos = _open_option()
    s.add(pos)
    s.commit()
    s.close()
    r = e.run_options_cycle()  # no quote set -> hold, no DTE due either
    assert r.held >= 1
    s2 = session_factory()
    reloaded = s2.query(Position).filter(Position.symbol == pos.symbol).first()
    assert reloaded.status == POSITION_OPEN
    s2.close()


def test_option_dte_exit_fires_without_quote(session_factory):
    import datetime as _dt
    fake = FakeAlpaca()
    e = TradingEngine(config=_cfg(), session_factory=session_factory,
                      alpaca=fake, aggregator=FakeAgg(0.0))
    s = session_factory()
    soon = (_dt.date.today() + _dt.timedelta(days=1)).isoformat()  # <= OPTION_EXIT_DTE (2)
    pos = _open_option(expiration=soon)
    s.add(pos)
    s.commit()
    s.close()
    r = e.run_options_cycle()  # no quote available, but DTE exit must still fire
    assert r.closed == 1
    s2 = session_factory()
    reloaded = s2.query(Position).filter(Position.symbol == pos.symbol).first()
    assert reloaded.status == POSITION_CLOSED
    s2.close()


def test_mode_paper_by_default(session_factory):
    e = TradingEngine(config=_cfg(), session_factory=session_factory,
                      alpaca=FakeAlpaca(), aggregator=FakeAgg(0.6))
    assert e.mode == "paper"


def test_live_requires_all_guards():
    assert _cfg(app_env="prod", paper_trading=False, live_trading=True).is_live()
    assert not _cfg(app_env="dev", paper_trading=False, live_trading=True).is_live()
    assert not _cfg(app_env="prod", paper_trading=True, live_trading=True).is_live()


# --------------------------------------------------------------------------
# PR 1: order truthfulness and simulated positions
# --------------------------------------------------------------------------
def _open_position(status_market, symbol, side="long", score=0.5):
    return av.AvenueDecision(
        action="open", market=status_market, symbol=symbol, side=side,
        reason="test entry", score=score,
    )


def test_rejected_open_creates_no_position_and_rejected_trade(session_factory):
    fake = FakeAlpaca(price=100.0)
    fake.reject_next_order("insufficient buying power")
    e = TradingEngine(config=_cfg(), session_factory=session_factory,
                      alpaca=fake, aggregator=FakeAgg(0.5))
    s = session_factory()
    decision = _open_position(MARKET_CRYPTO, "SOL/USD")
    status, err = e._open_position(s, MARKET_CRYPTO, "SOL/USD", decision, 100.0, place_order=True)
    assert status == "rejected"
    assert err == "insufficient buying power"
    assert s.query(Position).count() == 0
    trades = s.query(Trade).filter(Trade.status == TRADE_REJECTED).all()
    assert len(trades) == 1
    assert trades[0].reason == "insufficient buying power"
    s.close()


def test_rejected_close_leaves_position_open(session_factory):
    fake = FakeAlpaca(price=100.0)
    e = TradingEngine(config=_cfg(), session_factory=session_factory,
                      alpaca=fake, aggregator=FakeAgg(0.5))
    s = session_factory()
    pos = Position(market=MARKET_CRYPTO, symbol="SOL/USD", side="long", qty=1.0,
                   leverage=1.0, entry_price=100.0, current_price=100.0,
                   entry_notional=100.0, entry_fees=0.0, status=POSITION_OPEN)
    s.add(pos)
    s.commit()
    fake.reject_next_order("close rejected")
    closed = e._close_position(s, pos, 105.0, "take profit")
    assert closed is False
    assert pos.status == POSITION_OPEN
    assert e.loss_tracker._counts.get(MARKET_CRYPTO, 0) == 0
    s.close()


def test_simulated_close_makes_zero_broker_calls(session_factory):
    fake = FakeAlpaca(price=100.0)
    e = TradingEngine(config=_cfg(), session_factory=session_factory,
                      alpaca=fake, aggregator=FakeAgg(0.5))
    s = session_factory()
    pos = Position(market=MARKET_CRYPTO, symbol="SOL/USD", side="long", qty=1.0,
                   leverage=1.0, entry_price=100.0, current_price=100.0,
                   entry_notional=100.0, entry_fees=0.0, status=POSITION_OPEN,
                   extra={"simulated": True})
    s.add(pos)
    s.commit()
    closed = e._close_position(s, pos, 105.0, "take profit")
    assert closed is True
    assert pos.status == POSITION_CLOSED
    assert fake.close_calls == 0
    assert fake.orders == []
    s.close()


def test_crypto_close_uses_broker_held_qty_not_pos_qty(session_factory):
    fake = FakeAlpaca(price=100.0)
    e = TradingEngine(config=_cfg(), session_factory=session_factory,
                      alpaca=fake, aggregator=FakeAgg(0.5))
    s = session_factory()
    pos = Position(market=MARKET_CRYPTO, symbol="SOL/USD", side="long", qty=1.0,
                   leverage=1.0, entry_price=100.0, current_price=100.0,
                   entry_notional=100.0, entry_fees=0.0, status=POSITION_OPEN)
    s.add(pos)
    s.commit()
    fake.held_qty["SOL/USD"] = 0.98  # broker holds less than pos.qty (crypto fee)
    closed = e._close_position(s, pos, 105.0, "take profit")
    assert closed is True
    close_trade = s.query(Trade).filter(Trade.action == "close").first()
    assert close_trade.qty == pytest.approx(0.98)
    s.close()


def test_short_sizing_whole_shares(session_factory):
    fake = FakeAlpaca(spot=20.0)
    e = TradingEngine(config=_cfg(), session_factory=session_factory,
                      alpaca=fake, aggregator=FakeAgg(-0.5))
    s = session_factory()
    decision = _open_position(MARKET_STOCK, "F", side="short", score=-0.5)

    status, err = e._open_position(s, MARKET_STOCK, "F", decision, 20.0, place_order=True)
    assert status == "opened" and err is None
    pos = s.query(Position).filter(Position.symbol == "F").first()
    assert pos.qty == 5

    status2, err2 = e._open_position(s, MARKET_STOCK, "G", decision, 150.0, place_order=True)
    assert status2 == "skip"
    assert err2 == "short requires whole shares"
    assert s.query(Position).filter(Position.symbol == "G").count() == 0
    s.close()


def test_reconcile_closes_missing_broker_position(session_factory):
    fake = FakeAlpaca(price=100.0)
    e = TradingEngine(config=_cfg(), session_factory=session_factory,
                      alpaca=fake, aggregator=FakeAgg(0.5))
    s = session_factory()
    pos = Position(market=MARKET_CRYPTO, symbol="SOL/USD", side="long", qty=1.0,
                   leverage=1.0, entry_price=100.0, current_price=100.0,
                   entry_notional=100.0, entry_fees=0.0, status=POSITION_OPEN)
    s.add(pos)
    s.commit()
    # fake.held_qty has no entry for SOL/USD -> get_position_qty() returns None
    warnings = e.reconcile(s, MARKET_CRYPTO)
    assert len(warnings) == 1
    assert pos.status == POSITION_CLOSED
    assert pos.extra["reconciled"] == "missing_at_broker"
    s.close()


def test_no_entry_when_sentiment_not_actionable(session_factory):
    e = TradingEngine(config=_cfg(), session_factory=session_factory,
                      alpaca=FakeAlpaca(price=150.0),
                      aggregator=FakeAgg(0.9, actionable=False, coverage=0.07))
    r = e.run_crypto_cycle()
    assert r.opened == 0
    assert r.skipped >= 1
    s = session_factory()
    logs = s.query(SignalLog).all()
    assert any("insufficient sentiment coverage" in (log.detail or "") for log in logs)
    s.close()


def test_reconcile_skips_simulated_position(session_factory):
    fake = FakeAlpaca(price=100.0)
    e = TradingEngine(config=_cfg(), session_factory=session_factory,
                      alpaca=fake, aggregator=FakeAgg(0.5))
    s = session_factory()
    pos = Position(market=MARKET_CRYPTO, symbol="SOL/USD", side="long", qty=1.0,
                   leverage=1.0, entry_price=100.0, current_price=100.0,
                   entry_notional=100.0, entry_fees=0.0, status=POSITION_OPEN,
                   extra={"simulated": True})
    s.add(pos)
    s.commit()
    warnings = e.reconcile(s, MARKET_CRYPTO)
    assert warnings == []
    assert pos.status == POSITION_OPEN
    s.close()


def _count_open(session_factory) -> int:
    s = session_factory()
    try:
        return s.query(Position).filter(Position.status == POSITION_OPEN).count()
    finally:
        s.close()

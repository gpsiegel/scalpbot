"""Unit tests for AlpacaClient's order-fill polling and position helpers.

Exercises _submit_market / close_position / get_position_qty directly
against a minimal fake ``_trading`` client (no network, no alpaca-py SDK
calls -- the client is constructed ``lazy=True`` and ``_trading`` is
monkeypatched in directly, so ``_ensure()`` never imports alpaca-py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pytest

from engine.alpaca_client import AlpacaClient


@dataclass
class FakeOrder:
    id: str
    status: str = "new"
    filled_qty: float = 0.0
    filled_avg_price: float = 0.0


@dataclass
class FakePosition:
    symbol: str
    qty: float


class FakeTradingClient:
    """Fake alpaca-py TradingClient: a submitted order starts ``new`` and
    becomes ``filled`` after N polls (default 1), simulating the brief delay
    between a market order's submission and its fill."""

    def __init__(self, polls_to_fill: int = 1, positions: Optional[dict] = None):
        self.polls_to_fill = polls_to_fill
        self.positions = positions or {}
        self._orders: dict = {}
        self._poll_counts: dict = {}
        self._seq = 0
        self.close_requests = []

    def _next_id(self) -> str:
        self._seq += 1
        return f"order-{self._seq}"

    def submit_order(self, req):
        order = FakeOrder(id=self._next_id())
        self._orders[order.id] = order
        self._poll_counts[order.id] = 0
        return order

    def get_order_by_id(self, order_id):
        order = self._orders[order_id]
        self._poll_counts[order_id] += 1
        if self._poll_counts[order_id] >= self.polls_to_fill:
            order.status = "filled"
            order.filled_qty = 1.0
            order.filled_avg_price = 100.0
        return order

    def get_open_position(self, symbol):
        if symbol not in self.positions:
            raise Exception(f"position not found: {symbol}")
        return FakePosition(symbol=symbol, qty=self.positions[symbol])

    def close_position(self, symbol):
        self.close_requests.append(symbol)
        order = FakeOrder(id=self._next_id(), status="filled",
                          filled_qty=self.positions.get(symbol, 0.0),
                          filled_avg_price=100.0)
        return order


def _client(trading, **kw) -> AlpacaClient:
    c = AlpacaClient(api_key="x", secret_key="y", lazy=True,
                      order_poll_timeout=0.2, order_poll_interval=0.01, **kw)
    c._trading = trading
    return c


def test_submit_market_polls_until_filled():
    trading = FakeTradingClient(polls_to_fill=2)
    client = _client(trading)
    result = client.submit_crypto_order("SOL/USD", "buy", 1.0)
    assert result.ok is True
    assert result.filled_qty == 1.0
    assert result.filled_price == 100.0
    assert result.status == "filled"
    assert result.order_id is not None


def test_submit_market_times_out_with_zero_fill_is_not_ok():
    # polls_to_fill absurdly high -> never reaches "filled" within the timeout
    trading = FakeTradingClient(polls_to_fill=10_000)
    client = _client(trading)
    result = client.submit_crypto_order("SOL/USD", "buy", 1.0)
    assert result.ok is False
    assert result.filled_qty == 0.0


def test_submit_market_never_raises_on_broker_error():
    class RaisingTrading:
        def submit_order(self, req):
            raise RuntimeError("network down")

    client = _client(RaisingTrading())
    result = client.submit_crypto_order("SOL/USD", "buy", 1.0)
    assert result.ok is False
    assert "network down" in result.error


def test_get_position_qty_normalizes_crypto_symbol():
    trading = FakeTradingClient(positions={"SOLUSD": 2.5})
    client = _client(trading)
    assert client.get_position_qty("SOL/USD") == pytest.approx(2.5)
    assert client.get_position_qty("MISSING/USD") is None


def test_close_position_returns_broker_held_qty():
    trading = FakeTradingClient(positions={"SOLUSD": 0.97})
    client = _client(trading)
    result = client.close_position("SOL/USD")
    assert result.ok is True
    assert result.filled_qty == pytest.approx(0.97)
    assert trading.close_requests == ["SOLUSD"]


def test_close_position_never_raises_on_broker_error():
    class RaisingTrading:
        def close_position(self, symbol):
            raise RuntimeError("no position")

    client = _client(RaisingTrading())
    result = client.close_position("SOL/USD")
    assert result.ok is False
    assert "no position" in result.error


# --------------------------------------------------------------------------
# PR 3: bounded options contract query + single-symbol quote
# --------------------------------------------------------------------------
class FakeContractsTradingClient:
    def __init__(self):
        self.last_request = None

    def get_option_contracts(self, req):
        self.last_request = req
        return []


def test_list_option_contracts_bounds_expiration_and_strike():
    from config import OptionsFilters

    trading = FakeContractsTradingClient()
    client = _client(trading)
    filters = OptionsFilters(min_dte=7, max_dte=45, max_otm_pct=0.10)

    client.list_option_contracts("PLTR", "call", 20.0, filters)

    req = trading.last_request
    assert req is not None
    # alpaca-py types these Optional[str] -- must be strings, not floats.
    assert isinstance(req.strike_price_gte, str)
    assert isinstance(req.strike_price_lte, str)
    assert float(req.strike_price_gte) < 20.0 < float(req.strike_price_lte)
    assert req.expiration_date_gte is not None
    assert req.expiration_date_lte is not None
    assert req.expiration_date_gte < req.expiration_date_lte


@dataclass
class FakeQuote:
    bid_price: float
    ask_price: float


class FakeOptionDataClient:
    def __init__(self, quotes=None):
        self.quotes = quotes or {}

    def get_option_latest_quote(self, req):
        symbol = req.symbol_or_symbols
        return {symbol: self.quotes[symbol]} if symbol in self.quotes else {}


def test_get_option_quote_returns_bid_ask():
    client = _client(FakeTradingClient())
    client._option_data = FakeOptionDataClient(
        quotes={"PLTR250117C00020000": FakeQuote(bid_price=0.50, ask_price=0.55)}
    )
    assert client.get_option_quote("PLTR250117C00020000") == (0.50, 0.55)


def test_get_option_quote_none_when_missing():
    client = _client(FakeTradingClient())
    client._option_data = FakeOptionDataClient()
    assert client.get_option_quote("NOPE") is None


# --------------------------------------------------------------------------
# backlog: volatility-scaled exits -- ATR fetch
# --------------------------------------------------------------------------
@dataclass
class FakeBar:
    high: float
    low: float
    close: float


class FakeBarSet:
    def __init__(self, data):
        self.data = data  # {symbol: [FakeBar, ...]}


class FakeStockDataClient:
    def __init__(self, bars_by_symbol=None):
        self.bars_by_symbol = bars_by_symbol or {}

    def get_stock_bars(self, req):
        symbol = req.symbol_or_symbols
        return FakeBarSet({symbol: self.bars_by_symbol.get(symbol, [])})


class FakeCryptoDataClient:
    def __init__(self, bars_by_symbol=None):
        self.bars_by_symbol = bars_by_symbol or {}

    def get_crypto_bars(self, req):
        symbol = req.symbol_or_symbols
        return FakeBarSet({symbol: self.bars_by_symbol.get(symbol, [])})


def test_get_atr_stock_computes_from_bars():
    bars = [FakeBar(high=101.0, low=99.0, close=100.0) for _ in range(15)]
    client = _client(FakeTradingClient())
    client._stock_data = FakeStockDataClient({"PLTR": bars})
    assert client.get_atr("PLTR", "stock", period=14) == pytest.approx(2.0)


def test_get_atr_crypto_computes_from_bars():
    bars = [FakeBar(high=101.0, low=99.0, close=100.0) for _ in range(15)]
    client = _client(FakeTradingClient())
    client._crypto_data = FakeCryptoDataClient({"SOL/USD": bars})
    assert client.get_atr("SOL/USD", "crypto", period=14) == pytest.approx(2.0)


def test_get_atr_none_on_insufficient_history():
    bars = [FakeBar(high=101.0, low=99.0, close=100.0) for _ in range(5)]
    client = _client(FakeTradingClient())
    client._stock_data = FakeStockDataClient({"PLTR": bars})
    assert client.get_atr("PLTR", "stock", period=14) is None


def test_get_atr_never_raises_on_broker_error():
    class RaisingStockData:
        def get_stock_bars(self, req):
            raise RuntimeError("data feed down")

    client = _client(FakeTradingClient())
    client._stock_data = RaisingStockData()
    assert client.get_atr("PLTR", "stock", period=14) is None

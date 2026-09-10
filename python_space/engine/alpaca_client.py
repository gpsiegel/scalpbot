"""Alpaca brokerage + market-data wrapper for scalpbot.

Covers all three avenues through one object:

* **crypto** -- 24/7 spot, symbols like ``SOL/USD``, ``time_in_force=gtc``.
* **stocks** -- US equities, gated on market hours.
* **options** -- US equity options (buy-to-open long calls/puts only).

Design notes
------------
* ``alpaca-py`` is imported lazily inside :meth:`_ensure` so this module (and
  everything that imports it) loads even when the package is absent -- unit
  tests construct the client with ``lazy=True`` and never touch the network.
* Paper vs. live is a single ``paper`` flag wired straight into
  ``TradingClient(paper=...)``. The live-trading guard lives in
  :class:`config.Config`; this wrapper just does what it is told.
* API credentials come from the environment (Doppler secrets):
  ``ALPACA_API_KEY`` / ``ALPACA_SECRET_KEY``. Nothing is hard-coded.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from .volatility import compute_atr

logger = logging.getLogger("scalpbot.engine.alpaca")

# Order statuses that mean "the broker is done deciding" -- stop polling.
_TERMINAL_STATUSES = {"filled", "canceled", "expired", "rejected", "done_for_day"}


@dataclass
class OrderResult:
    """Normalized result of an order submission."""

    ok: bool
    order_id: Optional[str] = None
    filled_qty: float = 0.0
    filled_price: float = 0.0
    status: str = ""
    raw: Optional[object] = None
    error: Optional[str] = None


@dataclass
class OptionContract:
    """A single option contract candidate from the chain."""

    symbol: str            # OCC symbol, e.g. SOFI250117C00010000
    underlying: str
    option_type: str       # "call" | "put"
    strike: float
    expiration: str        # YYYY-MM-DD
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    open_interest: int = 0

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.last or self.ask or self.bid

    @property
    def spread_pct(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.ask - self.bid) / ((self.ask + self.bid) / 2.0)
        return 1.0  # unknown / illiquid -> treat as maximally wide


class AlpacaClient:
    """Thin, defensive wrapper over the Alpaca trading + data SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: bool = True,
        lazy: bool = False,
        order_poll_timeout: Optional[float] = None,
        order_poll_interval: Optional[float] = None,
    ):
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")
        self.paper = paper
        self._trading = None
        self._crypto_data = None
        self._stock_data = None
        self._option_data = None
        self._sdk = None
        self.order_poll_timeout = (
            order_poll_timeout
            if order_poll_timeout is not None
            else float(os.environ.get("ORDER_POLL_TIMEOUT_SECONDS", "10") or "10")
        )
        self.order_poll_interval = (
            order_poll_interval
            if order_poll_interval is not None
            else float(os.environ.get("ORDER_POLL_INTERVAL_SECONDS", "0.5") or "0.5")
        )
        if not lazy:
            self._ensure()

    # -- lazy SDK bootstrap -------------------------------------------------
    def _ensure(self):
        """Import alpaca-py and construct the SDK clients on first use."""
        if self._trading is not None:
            return
        if not self.api_key or not self.secret_key:
            raise RuntimeError(
                "ALPACA_API_KEY / ALPACA_SECRET_KEY not set (expected from Doppler)"
            )
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical.crypto import CryptoHistoricalDataClient
            from alpaca.data.historical.stock import StockHistoricalDataClient
            try:
                from alpaca.data.historical.option import OptionHistoricalDataClient
            except Exception:  # older alpaca-py without option data client
                OptionHistoricalDataClient = None
            import alpaca as _sdk
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "alpaca-py is not installed; add it from requirements.txt"
            ) from exc

        self._sdk = _sdk
        self._trading = TradingClient(self.api_key, self.secret_key, paper=self.paper)
        self._crypto_data = CryptoHistoricalDataClient(self.api_key, self.secret_key)
        self._stock_data = StockHistoricalDataClient(self.api_key, self.secret_key)
        self._option_data = (
            OptionHistoricalDataClient(self.api_key, self.secret_key)
            if OptionHistoricalDataClient
            else None
        )

    # -- market clock -------------------------------------------------------
    def is_market_open(self) -> bool:
        """True if the US equities market is open right now (crypto is 24/7)."""
        self._ensure()
        try:
            return bool(self._trading.get_clock().is_open)
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("get_clock failed: %s", exc)
            return False

    # -- prices -------------------------------------------------------------
    def get_crypto_price(self, symbol: str) -> Optional[float]:
        """Latest trade price for a crypto pair like ``SOL/USD``."""
        self._ensure()
        try:
            from alpaca.data.requests import CryptoLatestTradeRequest

            req = CryptoLatestTradeRequest(symbol_or_symbols=symbol)
            resp = self._crypto_data.get_crypto_latest_trade(req)
            return float(resp[symbol].price)
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("get_crypto_price(%s) failed: %s", symbol, exc)
            return None

    def get_stock_price(self, symbol: str) -> Optional[float]:
        """Latest trade price for a US equity."""
        self._ensure()
        try:
            from alpaca.data.requests import StockLatestTradeRequest

            req = StockLatestTradeRequest(symbol_or_symbols=symbol)
            resp = self._stock_data.get_stock_latest_trade(req)
            return float(resp[symbol].price)
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("get_stock_price(%s) failed: %s", symbol, exc)
            return None

    def _get_daily_bars(self, symbol: str, market: str, lookback_days: int) -> List[Tuple[float, float, float]]:
        """Recent daily ``(high, low, close)`` bars, oldest first. ``[]`` on
        any error -- callers must handle that as "no data available"."""
        self._ensure()
        try:
            import datetime as _dt

            from alpaca.data.timeframe import TimeFrame

            # A little slack for weekends/holidays.
            start = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=lookback_days * 2)

            if market == "crypto":
                from alpaca.data.requests import CryptoBarsRequest

                req = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=start)
                resp = self._crypto_data.get_crypto_bars(req)
            else:
                from alpaca.data.requests import StockBarsRequest

                req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=start)
                resp = self._stock_data.get_stock_bars(req)

            raw_bars = resp.data.get(symbol, []) if hasattr(resp, "data") else []
            return [(float(b.high), float(b.low), float(b.close)) for b in raw_bars]
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("_get_daily_bars(%s) failed: %s", symbol, exc)
            return []

    def get_atr(self, symbol: str, market: str, period: int = 14) -> Optional[float]:
        """Average True Range for ``symbol`` (an absolute price, not a
        percentage) from recent daily bars. ``market`` is ``"crypto"`` or
        ``"stock"``. ``None`` on any error or insufficient history -- callers
        must fall back to a static percentage. Never raises."""
        bars = self._get_daily_bars(symbol, market, lookback_days=period + 5)
        return compute_atr(bars, period=period)

    def get_recent_return(self, symbol: str, market: str, days: int = 5) -> Optional[float]:
        """Fractional price change (e.g. ``0.03`` == +3%) from the close
        ``days`` bars ago to the most recent close. ``None`` on any error or
        insufficient history. Never raises."""
        bars = self._get_daily_bars(symbol, market, lookback_days=days + 3)
        if len(bars) < days + 1:
            return None
        start_close = bars[-(days + 1)][2]
        end_close = bars[-1][2]
        if not start_close:
            return None
        return (end_close - start_close) / start_close

    # -- options chain ------------------------------------------------------
    def list_option_contracts(
        self,
        underlying: str,
        option_type: str,
        spot: float,
        filters: Any,
        limit: int = 100,
    ) -> List[OptionContract]:
        """Fetch tradable option contracts for an underlying, then enrich with a
        latest quote (bid/ask) so the caller can apply spread/premium filters.

        Bounds the query itself rather than relying on Alpaca's default window
        (which only returns contracts expiring before the upcoming weekend):
        expiration between ``filters.min_dte``/``max_dte`` days out, and
        strike within ``filters.max_otm_pct`` of ``spot`` (widened by a fixed
        5 points so contracts right at the boundary aren't excluded before
        ``engine.options.passes_filters`` runs its own precise check).
        """
        self._ensure()
        contracts: List[OptionContract] = []
        try:
            import datetime as _dt

            from alpaca.trading.requests import GetOptionContractsRequest
            from alpaca.trading.enums import ContractType, AssetStatus

            today = _dt.date.today()
            exp_gte = today + _dt.timedelta(days=filters.min_dte)
            exp_lte = today + _dt.timedelta(days=filters.max_dte)

            pad = filters.max_otm_pct + 0.05
            strike_lo = max(0.01, spot * (1 - pad))
            strike_hi = spot * (1 + pad)

            ctype = ContractType.CALL if option_type == "call" else ContractType.PUT
            req = GetOptionContractsRequest(
                underlying_symbols=[underlying],
                status=AssetStatus.ACTIVE,
                type=ctype,
                expiration_date_gte=exp_gte,
                expiration_date_lte=exp_lte,
                # alpaca-py types these as str, not float.
                strike_price_gte=f"{strike_lo:.2f}",
                strike_price_lte=f"{strike_hi:.2f}",
                limit=limit,
            )
            resp = self._trading.get_option_contracts(req)
            raw_list = getattr(resp, "option_contracts", resp) or []
            for c in raw_list:
                contracts.append(
                    OptionContract(
                        symbol=c.symbol,
                        underlying=underlying,
                        option_type=option_type,
                        strike=float(c.strike_price),
                        expiration=str(c.expiration_date),
                        open_interest=int(getattr(c, "open_interest", 0) or 0),
                    )
                )
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("list_option_contracts(%s) failed: %s", underlying, exc)
            return []

        self._enrich_option_quotes(contracts)
        return contracts

    def _enrich_option_quotes(self, contracts: List[OptionContract]) -> None:
        if not contracts or self._option_data is None:
            return
        try:
            from alpaca.data.requests import OptionLatestQuoteRequest

            symbols = [c.symbol for c in contracts]
            req = OptionLatestQuoteRequest(symbol_or_symbols=symbols)
            quotes = self._option_data.get_option_latest_quote(req)
            for c in contracts:
                q = quotes.get(c.symbol)
                if q is not None:
                    c.bid = float(getattr(q, "bid_price", 0.0) or 0.0)
                    c.ask = float(getattr(q, "ask_price", 0.0) or 0.0)
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("option quote enrich failed: %s", exc)

    def get_option_quote(self, symbol: str) -> Optional[Tuple[float, float]]:
        """Latest ``(bid, ask)`` for a single OCC option symbol.

        ``None`` if there is no option data client, no quote, or the call
        fails -- the management loop holds the position rather than acting on
        a stale/missing price. Never raises.
        """
        self._ensure()
        if self._option_data is None:
            return None
        try:
            from alpaca.data.requests import OptionLatestQuoteRequest

            req = OptionLatestQuoteRequest(symbol_or_symbols=symbol)
            resp = self._option_data.get_option_latest_quote(req)
            q = resp.get(symbol) if hasattr(resp, "get") else resp[symbol]
            if q is None:
                return None
            bid = float(getattr(q, "bid_price", 0.0) or 0.0)
            ask = float(getattr(q, "ask_price", 0.0) or 0.0)
            if bid <= 0 and ask <= 0:
                return None
            return bid, ask
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("get_option_quote(%s) failed: %s", symbol, exc)
            return None

    # -- orders -------------------------------------------------------------
    def submit_crypto_order(self, symbol: str, side: str, qty: float) -> OrderResult:
        """Market order for a crypto pair. TIF=GTC (crypto requirement)."""
        return self._submit_market(symbol, side, qty, market="crypto")

    def submit_stock_order(self, symbol: str, side: str, qty: float) -> OrderResult:
        """Market order for a US equity. TIF=DAY."""
        return self._submit_market(symbol, side, qty, market="stock")

    def submit_option_order(self, symbol: str, side: str, contracts: int) -> OrderResult:
        """Market order for an option contract (OCC symbol). TIF=DAY."""
        return self._submit_market(symbol, side, contracts, market="option")

    def _submit_market(self, symbol: str, side: str, qty: float, market: str) -> OrderResult:
        self._ensure()
        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            order_side = OrderSide.BUY if side.lower() in ("buy", "long") else OrderSide.SELL
            # Crypto must use GTC; equities/options use DAY.
            tif = TimeInForce.GTC if market == "crypto" else TimeInForce.DAY
            req = MarketOrderRequest(
                symbol=symbol,
                side=order_side,
                time_in_force=tif,
                qty=qty,
                client_order_id=f"scalpbot-{uuid.uuid4()}",
            )
            order = self._trading.submit_order(req)
            order = self._poll_order(order)
            return self._order_result_from(order)
        except Exception as exc:  # pragma: no cover - network dependent
            logger.error("submit order %s %s %s failed: %s", market, symbol, side, exc)
            return OrderResult(ok=False, error=str(exc))

    # -- order fill polling --------------------------------------------------
    @staticmethod
    def _status_of(order) -> str:
        status = getattr(order, "status", "")
        value = getattr(status, "value", status)
        return str(value).lower()

    def _poll_order(self, order):
        """Poll ``get_order_by_id`` until the order is terminal or times out.

        A ``partially_filled`` order still sitting open at the timeout is
        returned as-is; the caller treats any positive ``filled_qty`` as ok.
        Never raises -- returns the last order state seen on any failure.
        """
        order_id = getattr(order, "id", None)
        if order_id is None:
            return order
        deadline = time.monotonic() + self.order_poll_timeout
        current = order
        while self._status_of(current) not in _TERMINAL_STATUSES:
            if time.monotonic() >= deadline:
                break
            time.sleep(self.order_poll_interval)
            try:
                current = self._trading.get_order_by_id(order_id)
            except Exception as exc:  # pragma: no cover - network dependent
                logger.warning("get_order_by_id(%s) failed: %s", order_id, exc)
                break
        return current

    def _order_result_from(self, order) -> OrderResult:
        filled_qty = float(getattr(order, "filled_qty", 0) or 0)
        filled_avg = float(getattr(order, "filled_avg_price", 0) or 0)
        order_id = getattr(order, "id", None)
        return OrderResult(
            ok=filled_qty > 0,
            order_id=str(order_id) if order_id is not None else None,
            filled_qty=filled_qty,
            filled_price=filled_avg,
            status=self._status_of(order),
            raw=order,
        )

    # -- account/positions --------------------------------------------------
    def get_account(self):  # pragma: no cover - network dependent
        self._ensure()
        return self._trading.get_account()

    def list_positions(self):  # pragma: no cover - network dependent
        self._ensure()
        try:
            return self._trading.get_all_positions()
        except Exception as exc:
            logger.warning("list_positions failed: %s", exc)
            return []

    def get_position_qty(self, symbol: str) -> Optional[float]:
        """Broker-held qty for ``symbol``, or ``None`` if there is no position.

        Crypto positions are listed without the pair slash (orders use
        ``SOL/USD``, positions show ``SOLUSD``), so the symbol is normalized
        before the lookup.
        """
        self._ensure()
        try:
            norm = symbol.replace("/", "")
            pos = self._trading.get_open_position(norm)
            return float(getattr(pos, "qty", 0) or 0)
        except Exception as exc:
            logger.warning("get_position_qty(%s) failed: %s", symbol, exc)
            return None

    def close_position(self, symbol: str) -> OrderResult:
        """Close 100% of whatever the broker actually holds for ``symbol``.

        Delegates direction and quantity entirely to Alpaca's close-position
        endpoint, so it is correct for both long and short positions and for
        crypto positions whose held quantity has drifted from the recorded
        entry size (fees are paid in the asset received). Never raises.
        """
        self._ensure()
        try:
            norm = symbol.replace("/", "")
            order = self._trading.close_position(norm)
            order = self._poll_order(order)
            return self._order_result_from(order)
        except Exception as exc:  # pragma: no cover - network dependent
            logger.error("close_position(%s) failed: %s", symbol, exc)
            return OrderResult(ok=False, error=str(exc))

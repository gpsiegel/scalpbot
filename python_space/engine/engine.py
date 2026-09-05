"""The scalpbot cycle orchestrator.

A *cycle* for an avenue does three things, in order:

1. **Manage open positions** -- fetch the latest price/premium and apply the
   avenue's exit rules (take-profit / stop-loss / sentiment-reversal for
   crypto+stock; the +20/40/60% ladder and -35% stop for options).
2. **Open new entries** -- only if the avenue toggle is on, market hours allow
   (equities/options), and the per-avenue and total position caps have room.
3. **Persist** -- write a :class:`SignalLog` row per evaluated symbol, a
   :class:`Trade` row per order, and create/close :class:`Position` rows.

The engine is deliberately dependency-injected: the Alpaca client, the DB
session factory, and the sentiment aggregator are all passed in, so the whole
cycle can be exercised with fakes in tests (no network, no Postgres).

Live vs. paper: ``Config.is_live()`` is the single source of truth. When it is
False every order is tagged ``mode="paper"``; the Alpaca client itself is also
constructed against the paper endpoint. There is no way to place a live order
unless APP_ENV=prod AND PAPER_TRADING=false AND LIVE_TRADING=true.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import Config, get_config
from sentiment.aggregator import SentimentAggregator

from . import avenues as av
from . import options as opt
from .alpaca_client import AlpacaClient
from .models import (
    MARKET_CRYPTO,
    MARKET_OPTION,
    MARKET_STOCK,
    POSITION_CLOSED,
    POSITION_OPEN,
    TRADE_FILLED,
    BotConfig,
    Position,
    SignalLog,
    Trade,
    init_db,
    make_session_factory,
)

logger = logging.getLogger("scalpbot.engine")

# Per-position sizing (operational tunable -> plain .env).
POSITION_BUDGET_USD = float(os.environ.get("POSITION_BUDGET_USD", "100") or "100")


@dataclass
class CycleReport:
    """Summary of one avenue cycle -- returned to the API and logged."""

    market: str
    mode: str
    evaluated: int = 0
    opened: int = 0
    closed: int = 0
    skipped: int = 0
    held: int = 0
    errors: List[str] = field(default_factory=list)
    details: List[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "market": self.market,
            "mode": self.mode,
            "evaluated": self.evaluated,
            "opened": self.opened,
            "closed": self.closed,
            "skipped": self.skipped,
            "held": self.held,
            "errors": self.errors,
            "details": self.details,
        }


class TradingEngine:
    def __init__(
        self,
        config: Optional[Config] = None,
        session_factory=None,
        alpaca: Optional[AlpacaClient] = None,
        aggregator: Optional[SentimentAggregator] = None,
    ):
        self.config = config or get_config()
        self.session_factory = session_factory or make_session_factory()
        self.aggregator = aggregator or SentimentAggregator()
        # Construct the Alpaca client against the correct endpoint if not given.
        if alpaca is not None:
            self.alpaca = alpaca
        else:
            self.alpaca = AlpacaClient(paper=not self.config.is_live(), lazy=True)

    @property
    def mode(self) -> str:
        return "live" if self.config.is_live() else "paper"

    # -- helpers ------------------------------------------------------------
    def _open_positions(self, session, market: str) -> List[Position]:
        return (
            session.query(Position)
            .filter(Position.market == market, Position.status == POSITION_OPEN)
            .all()
        )

    def _count_open(self, session, market: Optional[str] = None) -> int:
        q = session.query(Position).filter(Position.status == POSITION_OPEN)
        if market:
            q = q.filter(Position.market == market)
        return q.count()

    def _has_open(self, session, market: str, symbol: str) -> bool:
        return (
            session.query(Position)
            .filter(
                Position.market == market,
                Position.symbol == symbol,
                Position.status == POSITION_OPEN,
            )
            .count()
            > 0
        )

    def _log_signal(self, session, market, symbol, score, label, acted, detail=None):
        session.add(
            SignalLog(
                market=market, symbol=symbol, score=score,
                label=label, acted=acted, detail=detail,
            )
        )

    # -- crypto -------------------------------------------------------------
    def run_crypto_cycle(self) -> CycleReport:
        report = CycleReport(market=MARKET_CRYPTO, mode=self.mode)
        if not self.config.crypto_core_coins and not self.config.crypto_satellite_coins:
            report.errors.append("no crypto coins configured")
            return report

        session = self.session_factory()
        try:
            # 1) manage exits
            for pos in self._open_positions(session, MARKET_CRYPTO):
                coin = pos.symbol.split("/")[0]
                price = self.alpaca.get_crypto_price(pos.symbol)
                if price is None:
                    continue
                sent = self.aggregator.get_sentiment(coin)
                decision = av.linear_exit_decision(
                    MARKET_CRYPTO, pos.symbol, pos.side,
                    pos.entry_price, price, sent.score,
                )
                if decision.is_close:
                    self._close_position(session, pos, price, decision.reason)
                    report.closed += 1
                else:
                    report.held += 1

            # 2) entries
            for coin in self.config.all_crypto_coins:
                symbol = self.config.crypto_symbol(coin)
                sent = self.aggregator.get_sentiment(coin)
                report.evaluated += 1
                if self._has_open(session, MARKET_CRYPTO, symbol):
                    self._log_signal(session, MARKET_CRYPTO, symbol, sent.score, sent.label, False)
                    continue
                if not self.config.avenues.crypto:
                    self._log_signal(session, MARKET_CRYPTO, symbol, sent.score, sent.label, False)
                    report.skipped += 1
                    continue
                if not self._has_cap_room(session, MARKET_CRYPTO):
                    report.skipped += 1
                    continue
                decision = av.crypto_entry_decision(symbol, sent.score)
                if decision.is_open:
                    price = self.alpaca.get_crypto_price(symbol)
                    if price:
                        self._open_position(session, MARKET_CRYPTO, symbol, decision, price)
                        report.opened += 1
                    self._log_signal(session, MARKET_CRYPTO, symbol, sent.score, sent.label, True)
                else:
                    report.skipped += 1
                    self._log_signal(session, MARKET_CRYPTO, symbol, sent.score, sent.label, False)

            session.commit()
        except Exception as exc:  # pragma: no cover - defensive
            session.rollback()
            logger.exception("crypto cycle failed")
            report.errors.append(str(exc))
        finally:
            session.close()
        return report

    # -- stocks -------------------------------------------------------------
    def run_stock_cycle(self) -> CycleReport:
        report = CycleReport(market=MARKET_STOCK, mode=self.mode)
        if not self.config.stock_tickers:
            report.errors.append("no stock tickers configured")
            return report
        if self.config.respect_market_hours and not self.alpaca.is_market_open():
            report.errors.append("market closed")
            return report

        session = self.session_factory()
        try:
            for pos in self._open_positions(session, MARKET_STOCK):
                price = self.alpaca.get_stock_price(pos.symbol)
                if price is None:
                    continue
                sent = self.aggregator.get_sentiment(pos.symbol)
                decision = av.linear_exit_decision(
                    MARKET_STOCK, pos.symbol, pos.side,
                    pos.entry_price, price, sent.score,
                )
                if decision.is_close:
                    self._close_position(session, pos, price, decision.reason)
                    report.closed += 1
                else:
                    report.held += 1

            for ticker in self.config.stock_tickers:
                sent = self.aggregator.get_sentiment(ticker)
                report.evaluated += 1
                if self._has_open(session, MARKET_STOCK, ticker):
                    self._log_signal(session, MARKET_STOCK, ticker, sent.score, sent.label, False)
                    continue
                if not self.config.avenues.stocks or not self._has_cap_room(session, MARKET_STOCK):
                    report.skipped += 1
                    self._log_signal(session, MARKET_STOCK, ticker, sent.score, sent.label, False)
                    continue
                decision = av.stock_entry_decision(ticker, sent.score, self.config.allow_short)
                if decision.is_open:
                    price = self.alpaca.get_stock_price(ticker)
                    if price:
                        self._open_position(session, MARKET_STOCK, ticker, decision, price)
                        report.opened += 1
                    self._log_signal(session, MARKET_STOCK, ticker, sent.score, sent.label, True)
                else:
                    report.skipped += 1
                    self._log_signal(session, MARKET_STOCK, ticker, sent.score, sent.label, False)

            session.commit()
        except Exception as exc:  # pragma: no cover - defensive
            session.rollback()
            logger.exception("stock cycle failed")
            report.errors.append(str(exc))
        finally:
            session.close()
        return report

    # -- options ------------------------------------------------------------
    def run_options_cycle(self) -> CycleReport:
        report = CycleReport(market=MARKET_OPTION, mode=self.mode)
        if not self.config.stock_tickers:
            report.errors.append("no underlyings configured (uses STOCK_TICKERS)")
            return report
        if self.config.respect_market_hours and not self.alpaca.is_market_open():
            report.errors.append("market closed")
            return report

        session = self.session_factory()
        try:
            # 1) manage open option positions via the premium ladder
            for pos in self._open_positions(session, MARKET_OPTION):
                # For a real fill we'd re-quote pos.symbol; premium tracked in current_price.
                current_premium = pos.current_price or pos.entry_price
                action = opt.exit_decision(pos.entry_price, current_premium)
                if action:
                    self._close_option_position(session, pos, current_premium, action)
                    report.closed += 1
                else:
                    report.held += 1

            # 2) entries
            for underlying in self.config.stock_tickers:
                sent = self.aggregator.get_sentiment(underlying)
                report.evaluated += 1
                side = av.option_side_for_score(sent.score)
                if side is None:
                    report.skipped += 1
                    self._log_signal(session, MARKET_OPTION, underlying, sent.score, sent.label, False)
                    continue
                if not self.config.avenues.options or not self._has_cap_room(session, MARKET_OPTION):
                    report.skipped += 1
                    self._log_signal(session, MARKET_OPTION, underlying, sent.score, sent.label, False)
                    continue
                spot = self.alpaca.get_stock_price(underlying)
                if not spot:
                    report.skipped += 1
                    continue
                contracts = self.alpaca.list_option_contracts(
                    underlying, side,
                    expiration_gte=None, expiration_lte=None,
                )
                choice = opt.select_contract(contracts, spot, side, self.config.options)
                if choice is None:
                    report.skipped += 1
                    self._log_signal(session, MARKET_OPTION, underlying, sent.score, sent.label, False)
                    continue
                n = opt.contracts_for_budget(choice.premium, POSITION_BUDGET_USD)
                if n < 1:
                    report.skipped += 1
                    continue
                self._open_option_position(session, underlying, side, choice, n, sent.score)
                report.opened += 1
                self._log_signal(session, MARKET_OPTION, choice.contract.symbol, sent.score, sent.label, True)

            session.commit()
        except Exception as exc:  # pragma: no cover - defensive
            session.rollback()
            logger.exception("options cycle failed")
            report.errors.append(str(exc))
        finally:
            session.close()
        return report

    # -- cap enforcement ----------------------------------------------------
    def _has_cap_room(self, session, market: str) -> bool:
        caps = self.config.caps
        if self._count_open(session) >= caps.total:
            return False
        per_market = {
            MARKET_CRYPTO: caps.crypto,
            MARKET_STOCK: caps.stock,
            MARKET_OPTION: caps.option,
        }[market]
        return self._count_open(session, market) < per_market

    # -- position lifecycle -------------------------------------------------
    def _open_position(self, session, market, symbol, decision, price):
        qty = POSITION_BUDGET_USD / price if price else 0.0
        notional = qty * price
        fees = self.config.round_trip_fees(notional) / 2  # entry side only
        pos = Position(
            market=market, symbol=symbol, side=decision.side, qty=qty,
            leverage=1.0, entry_price=price, current_price=price,
            entry_notional=notional, entry_fees=fees,
            sentiment_score=decision.score, status=POSITION_OPEN,
        )
        session.add(pos)
        session.add(Trade(
            market=market, symbol=symbol, side=decision.side, action="open",
            qty=qty, price=price, notional=notional, fees=fees,
            status=TRADE_FILLED, mode=self.mode, reason=decision.reason,
        ))
        # place the real order (paper or live per endpoint)
        if market == MARKET_CRYPTO:
            self.alpaca.submit_crypto_order(symbol, "buy", qty)
        else:
            self.alpaca.submit_stock_order(symbol, "buy" if decision.side == "long" else "sell", qty)

    def _close_position(self, session, pos: Position, price: float, reason: str):
        pnl = av.position_pnl_pct(pos.side, pos.entry_price, price) * pos.entry_notional
        exit_fees = self.config.round_trip_fees(pos.entry_notional) / 2
        pos.status = POSITION_CLOSED
        pos.current_price = price
        pos.realized_pnl = pnl - exit_fees - pos.entry_fees
        import datetime as _dt
        pos.closed_at = _dt.datetime.now(_dt.timezone.utc)
        session.add(Trade(
            market=pos.market, symbol=pos.symbol, side=pos.side, action="close",
            qty=pos.qty, price=price, notional=pos.qty * price, fees=exit_fees,
            pnl=pos.realized_pnl, status=TRADE_FILLED, mode=self.mode, reason=reason,
        ))
        if pos.market == MARKET_CRYPTO:
            self.alpaca.submit_crypto_order(pos.symbol, "sell", pos.qty)
        else:
            self.alpaca.submit_stock_order(pos.symbol, "sell" if pos.side == "long" else "buy", pos.qty)

    def _open_option_position(self, session, underlying, side, choice, contracts, score):
        premium = choice.premium
        cost = premium * opt.CONTRACT_MULTIPLIER * contracts
        pos = Position(
            market=MARKET_OPTION, symbol=choice.contract.symbol, side="long",
            qty=contracts, leverage=float(contracts), entry_price=premium,
            current_price=premium, entry_notional=cost, entry_fees=0.0,
            underlying=underlying, strike=choice.contract.strike,
            option_type=side, expiration=choice.contract.expiration,
            sentiment_score=score, status=POSITION_OPEN,
        )
        session.add(pos)
        session.add(Trade(
            market=MARKET_OPTION, symbol=choice.contract.symbol, side="long",
            action="open", qty=contracts, price=premium, notional=cost, fees=0.0,
            status=TRADE_FILLED, mode=self.mode,
            reason=f"{side} entry, score {score:.3f}",
        ))
        self.alpaca.submit_option_order(choice.contract.symbol, "buy", contracts)

    def _close_option_position(self, session, pos: Position, current_premium: float, action: str):
        contracts = pos.leverage or pos.qty
        pnl = opt.option_pnl(pos.entry_price, current_premium, contracts)
        pos.status = POSITION_CLOSED
        pos.current_price = current_premium
        pos.realized_pnl = pnl
        import datetime as _dt
        pos.closed_at = _dt.datetime.now(_dt.timezone.utc)
        session.add(Trade(
            market=MARKET_OPTION, symbol=pos.symbol, side="long", action="close",
            qty=contracts, price=current_premium,
            notional=current_premium * opt.CONTRACT_MULTIPLIER * contracts,
            fees=0.0, pnl=pnl, status=TRADE_FILLED, mode=self.mode, reason=action,
        ))
        self.alpaca.submit_option_order(pos.symbol, "sell", int(contracts))

    # -- convenience --------------------------------------------------------
    def ensure_schema(self):
        """Create tables if they do not yet exist."""
        init_db()

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

import datetime as _dt
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import Config, get_config
from sentiment.aggregator import SentimentAggregator

from . import avenues as av
from . import options as opt
from .alpaca_client import AlpacaClient
from .groq_advisor import GroqAdvisor
from .risk import TierParams, ev_positive, get_tier
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

# Per-position sizing (operational tunable -> plain .env). Superseded per-trade
# by the active risk tier's ``position_budget_usd``; kept as a global fallback.
POSITION_BUDGET_USD = float(os.environ.get("POSITION_BUDGET_USD", "100") or "100")


def _utcnow() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


class ConsecutiveLossTracker:
    """Circuit breaker: pause a market after too many consecutive losses.

    Scalping's fastest way to ruin is revenge-trading a losing streak. This
    tracker counts *consecutive* losing closes per market (in memory) and, once
    the streak hits :data:`LOSS_THRESHOLD`, writes a cooldown timestamp
    :data:`COOLDOWN_MINUTES` into the future to the persistent ``bot_config``
    table. While ``is_cooling_down`` is True the engine opens no new positions in
    that market. A single winning close resets the streak; the cooldown itself
    simply expires with the clock.
    """

    COOLDOWN_MINUTES = 60
    LOSS_THRESHOLD = 3
    KEY_PREFIX = "cooldown_until_"
    LOSS_PREFIX = "consec_losses_"

    def __init__(self) -> None:
        # in-memory consecutive-loss counts per market
        self._counts: Dict[str, int] = {}

    def record_win(self, market: str) -> None:
        """A winning close breaks the streak."""
        self._counts[market] = 0

    def record_loss(self, market: str, session) -> None:
        """A losing close extends the streak; trip the breaker at the threshold."""
        self._counts[market] = self._counts.get(market, 0) + 1
        if self._counts[market] >= self.LOSS_THRESHOLD:
            self._set_cooldown(market, session)
            self._counts[market] = 0  # reset streak once the breaker trips

    def is_cooling_down(self, market: str, session) -> bool:
        """True while a cooldown timestamp for ``market`` is still in the future."""
        row = (
            session.query(BotConfig)
            .filter_by(key=f"{self.KEY_PREFIX}{market}")
            .first()
        )
        if row is None or not row.value:
            return False
        try:
            until = _dt.datetime.fromisoformat(row.value)
        except ValueError:
            return False
        if until.tzinfo is None:
            until = until.replace(tzinfo=_dt.timezone.utc)
        return _utcnow() < until

    def _set_cooldown(self, market: str, session) -> None:
        until = _utcnow() + _dt.timedelta(minutes=self.COOLDOWN_MINUTES)
        key = f"{self.KEY_PREFIX}{market}"
        row = session.query(BotConfig).filter_by(key=key).first()
        if row is None:
            session.add(BotConfig(key=key, value=until.isoformat(), market=market))
        else:
            row.value = until.isoformat()
        logger.warning(
            "consecutive-loss circuit breaker tripped for %s: cooling down until %s",
            market,
            until.isoformat(),
        )


class ConfidenceGate:
    """Paper-validate a tier before it is allowed to place real orders.

    When a (market, tier) pair first runs -- or whenever the active tier changes
    -- the gate opens and the next :data:`PAPER_TRADES` closes are taken as paper
    trades and tallied. If at least :data:`WINS_REQUIRED` of them win, the gate
    passes (closes) and the market may trade for real under that tier. If the
    validation window fills without enough wins, the tally resets and another
    paper window begins. State lives in the persistent ``bot_config`` table so it
    survives restarts.
    """

    PAPER_TRADES = 5
    WINS_REQUIRED = 3
    ACTIVE_PREFIX = "gate_active_"
    WINS_PREFIX = "gate_wins_"
    TOTAL_PREFIX = "gate_total_"
    TIER_PREFIX = "gate_tier_"

    def _get(self, session, key: str) -> Optional[str]:
        row = session.query(BotConfig).filter_by(key=key).first()
        return row.value if row else None

    def _set(self, session, key: str, value: str, market: str) -> None:
        row = session.query(BotConfig).filter_by(key=key).first()
        if row is None:
            session.add(BotConfig(key=key, value=value, market=market))
        else:
            row.value = value

    def initialize_if_needed(self, market: str, tier: TierParams, session) -> None:
        """Open a fresh validation window when unset or when the tier changed."""
        active = self._get(session, f"{self.ACTIVE_PREFIX}{market}")
        current_tier = self._get(session, f"{self.TIER_PREFIX}{market}")
        if active is None or current_tier != tier.name:
            self.reset(market, session)
            self._set(session, f"{self.TIER_PREFIX}{market}", tier.name, market)

    def reset(self, market: str, session) -> None:
        """(Re)open the validation window: active, zero wins, zero total."""
        self._set(session, f"{self.ACTIVE_PREFIX}{market}", "1", market)
        self._set(session, f"{self.WINS_PREFIX}{market}", "0", market)
        self._set(session, f"{self.TOTAL_PREFIX}{market}", "0", market)

    def is_active(self, market: str, tier: TierParams, session) -> bool:
        """True while the tier is still being paper-validated for ``market``."""
        self.initialize_if_needed(market, tier, session)
        return self._get(session, f"{self.ACTIVE_PREFIX}{market}") == "1"

    def record_paper_trade(
        self, market: str, tier: TierParams, won: bool, session
    ) -> bool:
        """Record one paper-trade outcome; return True if the gate just passed.

        Fills the validation window; at :data:`PAPER_TRADES` trades it either
        passes (>= :data:`WINS_REQUIRED` wins -> gate closes) or resets for
        another window.
        """
        self.initialize_if_needed(market, tier, session)
        wins = int(self._get(session, f"{self.WINS_PREFIX}{market}") or "0")
        total = int(self._get(session, f"{self.TOTAL_PREFIX}{market}") or "0")
        wins += 1 if won else 0
        total += 1

        if total >= self.PAPER_TRADES:
            if wins >= self.WINS_REQUIRED:
                self._set(session, f"{self.ACTIVE_PREFIX}{market}", "0", market)
                self._set(session, f"{self.WINS_PREFIX}{market}", str(wins), market)
                self._set(session, f"{self.TOTAL_PREFIX}{market}", str(total), market)
                logger.info(
                    "confidence gate PASSED for %s tier=%s (%d/%d wins)",
                    market, tier.name, wins, total,
                )
                return True
            # window filled without enough wins -> reset and try again
            self.reset(market, session)
            logger.info(
                "confidence gate reset for %s tier=%s (%d/%d wins, retrying)",
                market, tier.name, wins, total,
            )
            return False

        self._set(session, f"{self.WINS_PREFIX}{market}", str(wins), market)
        self._set(session, f"{self.TOTAL_PREFIX}{market}", str(total), market)
        return False


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
        tier: Optional[TierParams] = None,
        groq: Optional[GroqAdvisor] = None,
    ):
        self.config = config or get_config()
        self.session_factory = session_factory or make_session_factory()
        self.aggregator = aggregator or SentimentAggregator()
        # Active risk tier (resolved from RISK_TIER via config) and the Groq
        # advisor. Both are injectable so tests can substitute fakes.
        self.tier = tier or get_tier(self.config.risk_tier)
        self.groq = groq or GroqAdvisor()
        # Risk-management state machines.
        self.loss_tracker = ConsecutiveLossTracker()
        self.conf_gate = ConfidenceGate()
        # Construct the Alpaca client against the correct endpoint if not given.
        if alpaca is not None:
            self.alpaca = alpaca
        else:
            self.alpaca = AlpacaClient(paper=not self.config.is_live(), lazy=True)

    # -- risk gates ---------------------------------------------------------
    def _passes_entry_gates(self, session, market, symbol, score, tp, sl):
        """Run the tier + cooldown + Groq + confidence-floor + EV gates.

        Returns ``(ok, reason, paper_only)``:

        * ``ok`` -- whether an entry may be taken at all;
        * ``reason`` -- human-readable skip/accept reason (for logs);
        * ``paper_only`` -- when True the position must be opened as a paper
          trade (no real order) because the confidence gate is still validating
          this tier for ``market``.
        """
        # 1) consecutive-loss circuit breaker
        if self.loss_tracker.is_cooling_down(market, session):
            return False, "cooldown active after consecutive losses", False
        # 2) tier conviction gate (stricter than the avenue's base threshold)
        if abs(score) < self.tier.entry_score_threshold:
            return (
                False,
                f"score {score:.3f} below tier gate {self.tier.entry_score_threshold:.2f}",
                False,
            )
        # 3) + 4) Groq advisor confidence floor and EV projection gate. Both
        # depend on the advisor's verdict, so they are only enforced when Groq is
        # configured. With no GROQ_API_KEY the advisor is disabled and the engine
        # falls back to sentiment-only entries (the pre-advisor behaviour).
        if not self.groq.is_disabled:
            advice = self.groq.validate_trade(market, symbol, score, self.tier)
            # 3) Groq advisor confidence floor
            if advice.confidence < self.tier.groq_confidence_min:
                return (
                    False,
                    f"groq confidence {advice.confidence:.2f} < {self.tier.groq_confidence_min:.2f}",
                    False,
                )
            # 4) expected-value projection gate (must clear fee drag)
            round_trip_fee = self.config.taker_fee_pct * 2
            if not ev_positive(tp, sl, advice.projected_win_probability, round_trip_fee):
                return (
                    False,
                    f"negative EV (win_prob {advice.projected_win_probability:.2f}) after fees",
                    False,
                )
        # 5) confidence gate -- paper-validate the tier before real orders
        paper_only = self.conf_gate.is_active(market, self.tier, session)
        return True, "risk gates passed", paper_only

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
                    take_profit_pct=self.tier.crypto_take_profit_pct,
                    stop_loss_pct=self.tier.crypto_stop_loss_pct,
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
                    ok, reason, paper_only = self._passes_entry_gates(
                        session, MARKET_CRYPTO, symbol, sent.score,
                        self.tier.crypto_take_profit_pct,
                        self.tier.crypto_stop_loss_pct,
                    )
                    if not ok:
                        report.skipped += 1
                        self._log_signal(session, MARKET_CRYPTO, symbol, sent.score, sent.label, False, reason)
                        continue
                    price = self.alpaca.get_crypto_price(symbol)
                    if price:
                        self._open_position(
                            session, MARKET_CRYPTO, symbol, decision, price,
                            place_order=not paper_only,
                        )
                        report.opened += 1
                    self._log_signal(session, MARKET_CRYPTO, symbol, sent.score, sent.label, True, reason)
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
                    take_profit_pct=self.tier.stock_take_profit_pct,
                    stop_loss_pct=self.tier.stock_stop_loss_pct,
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
                    ok, reason, paper_only = self._passes_entry_gates(
                        session, MARKET_STOCK, ticker, sent.score,
                        self.tier.stock_take_profit_pct,
                        self.tier.stock_stop_loss_pct,
                    )
                    if not ok:
                        report.skipped += 1
                        self._log_signal(session, MARKET_STOCK, ticker, sent.score, sent.label, False, reason)
                        continue
                    price = self.alpaca.get_stock_price(ticker)
                    if price:
                        self._open_position(
                            session, MARKET_STOCK, ticker, decision, price,
                            place_order=not paper_only,
                        )
                        report.opened += 1
                    self._log_signal(session, MARKET_STOCK, ticker, sent.score, sent.label, True, reason)
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
                # risk gates: cooldown, tier conviction, Groq, EV. For options the
                # EV projection uses the tier's premium profit target vs. the -35%
                # premium stop encoded in the options ladder.
                ok, reason, paper_only = self._passes_entry_gates(
                    session, MARKET_OPTION, underlying, sent.score,
                    self.tier.option_profit_target_pct, abs(opt.STOP_LOSS),
                )
                if not ok:
                    report.skipped += 1
                    self._log_signal(session, MARKET_OPTION, underlying, sent.score, sent.label, False, reason)
                    continue
                n = opt.contracts_for_budget(choice.premium, self.tier.position_budget_usd)
                if n < 1:
                    report.skipped += 1
                    continue
                self._open_option_position(
                    session, underlying, side, choice, n, sent.score,
                    place_order=not paper_only,
                )
                report.opened += 1
                self._log_signal(session, MARKET_OPTION, choice.contract.symbol, sent.score, sent.label, True, reason)

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
    def _open_position(self, session, market, symbol, decision, price, place_order: bool = True):
        # size with the active risk tier's per-position budget
        budget = self.tier.position_budget_usd
        qty = budget / price if price else 0.0
        notional = qty * price
        fees = self.config.round_trip_fees(notional) / 2  # entry side only
        gate_mode = self.mode if place_order else "paper"
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
            status=TRADE_FILLED, mode=gate_mode, reason=decision.reason,
        ))
        # place the real order (paper or live per endpoint). When the confidence
        # gate is still validating this tier, skip the order entirely (paper-only).
        if not place_order:
            return
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
        self._record_outcome(session, pos.market, pos.realized_pnl)

    def _record_outcome(self, session, market: str, realized_pnl: float) -> None:
        """Feed a closed trade's result into the risk state machines.

        Updates the consecutive-loss circuit breaker and, while the confidence
        gate is still validating this tier, tallies the paper-trade outcome.
        """
        won = (realized_pnl or 0.0) > 0
        if won:
            self.loss_tracker.record_win(market)
        else:
            self.loss_tracker.record_loss(market, session)
        if self.conf_gate.is_active(market, self.tier, session):
            self.conf_gate.record_paper_trade(market, self.tier, won, session)

    def _open_option_position(self, session, underlying, side, choice, contracts, score, place_order: bool = True):
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
            status=TRADE_FILLED, mode=self.mode if place_order else "paper",
            reason=f"{side} entry, score {score:.3f}",
        ))
        # skip the real order while the confidence gate is paper-validating.
        if place_order:
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
        self._record_outcome(session, pos.market, pos.realized_pnl)

    # -- convenience --------------------------------------------------------
    def ensure_schema(self):
        """Create tables if they do not yet exist."""
        init_db()

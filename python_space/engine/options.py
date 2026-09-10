"""Options avenue helpers: contract selection, quality filters, and P&L.

Scope (intentionally narrow):

* **Buy-to-open long only** -- bullish sentiment -> long CALL, bearish -> long
  PUT. No spreads, no selling premium, no assignment risk.
* **Quality gates** (from :class:`config.OptionsFilters`):
    - ``min_dte`` / ``max_dte`` -- bound the contract query to a tradeable
      expiry window.
    - ``max_otm_pct``    -- only near-the-money; no deep-OTM lottery tickets.
    - ``min_premium``    -- skip near-worthless contracts.
    - ``max_spread_pct`` -- skip illiquid, wide bid/ask contracts.
* **P&L is premium-based**: a position stores contracts in ``leverage`` and the
  entry premium (per share, at the ask -- the real cost basis) in
  ``entry_price``. One contract = 100 shares.

Management (long options): a single take-profit at the active risk tier's
``option_profit_target_pct`` (matching what the EV gate assumed at entry),
hard stop at **-35%**, plus a DTE-based expiry exit (``Config.option_exit_dte``)
independent of P&L -- Alpaca auto-exercises/force-sells ITM contracts near
expiry.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import List, Optional

from config import OptionsFilters
from .alpaca_client import OptionContract

logger = logging.getLogger("scalpbot.engine.options")

# Long-option management: a single take-profit target (the active risk
# tier's `option_profit_target_pct`) plus a hard stop.
STOP_LOSS = -0.35
CONTRACT_MULTIPLIER = 100  # shares per contract
# Float-rounding tolerance for the take-profit/stop-loss boundary comparisons
# (e.g. entry 0.50 -> current 0.70 computes as 0.3999999999999999, not 0.40).
_EPSILON = 1e-9


def days_to_expiration(expiration: str, today: Optional[dt.date] = None) -> int:
    """Whole calendar days from ``today`` to ``expiration`` (YYYY-MM-DD)."""
    today = today or dt.date.today()
    exp = dt.date.fromisoformat(str(expiration)[:10])
    return (exp - today).days


def otm_pct(spot: float, strike: float, option_type: str) -> float:
    """How far out-of-the-money the strike is, as a fraction of spot.

    Negative == in-the-money, 0 == at-the-money. For a call, higher strikes are
    more OTM; for a put, lower strikes are more OTM.
    """
    if spot <= 0:
        return 1.0
    if option_type == "call":
        return (strike - spot) / spot
    return (spot - strike) / spot


@dataclass
class ContractScore:
    contract: OptionContract
    premium: float
    dte: int
    otm: float
    spread: float


def passes_filters(
    contract: OptionContract,
    spot: float,
    filters: OptionsFilters,
    today: Optional[dt.date] = None,
) -> bool:
    """True if the contract clears every quality gate."""
    dte = days_to_expiration(contract.expiration, today)
    if dte < filters.min_dte:
        return False

    otm = otm_pct(spot, contract.strike, contract.option_type)
    # Reject deep OTM. Allow ITM/ATM (otm <= 0) and near-OTM up to the cap.
    if otm > filters.max_otm_pct:
        return False

    # A long option is bought near the ask, not the mid -- that's the real
    # cost basis, so it's what the quality floor and sizing should use.
    premium = contract.ask
    if premium < filters.min_premium:
        return False

    if contract.spread_pct > filters.max_spread_pct:
        return False

    return True


def select_contract(
    contracts: List[OptionContract],
    spot: float,
    option_type: str,
    filters: OptionsFilters,
    today: Optional[dt.date] = None,
) -> Optional[ContractScore]:
    """Pick the best contract of ``option_type`` for the given spot.

    Strategy: keep only contracts passing every filter, then prefer the one
    closest to at-the-money (smallest absolute OTM), breaking ties toward the
    nearer expiration and the tighter spread.
    """
    candidates: List[ContractScore] = []
    for c in contracts:
        if c.option_type != option_type:
            continue
        if not passes_filters(c, spot, filters, today):
            continue
        candidates.append(
            ContractScore(
                contract=c,
                premium=c.ask,
                dte=days_to_expiration(c.expiration, today),
                otm=otm_pct(spot, c.strike, c.option_type),
                spread=c.spread_pct,
            )
        )

    if not candidates:
        return None

    candidates.sort(key=lambda s: (abs(s.otm), s.dte, s.spread))
    return candidates[0]


def contracts_for_budget(premium: float, budget: float) -> int:
    """How many whole contracts fit in ``budget`` at ``premium`` per share.

    One contract costs ``premium * 100``. Returns 0 when even one is unaffordable.
    """
    if premium <= 0:
        return 0
    per_contract_cost = premium * CONTRACT_MULTIPLIER
    return int(budget // per_contract_cost)


def option_pnl(entry_premium: float, exit_premium: float, contracts: float) -> float:
    """Premium-based P&L for a long option position."""
    return (exit_premium - entry_premium) * contracts * CONTRACT_MULTIPLIER


def pnl_pct(entry_premium: float, current_premium: float) -> float:
    """Fractional gain/loss on premium (e.g. 0.20 == +20%)."""
    if entry_premium <= 0:
        return 0.0
    return (current_premium - entry_premium) / entry_premium


def exit_decision(
    entry_premium: float,
    current_premium: float,
    take_profit_pct: float,
    stop_loss_pct: float = abs(STOP_LOSS),
) -> Optional[str]:
    """Return a management action string, or None to hold.

    A single take-profit at ``take_profit_pct`` (the active risk tier's
    ``option_profit_target_pct``, matching what the EV gate assumed at entry)
    plus the ``stop_loss_pct`` hard stop. Compared with a small epsilon so
    float rounding never misses an exact-boundary target.
    """
    pct = pnl_pct(entry_premium, current_premium)
    if pct <= -stop_loss_pct + _EPSILON:
        return "stop_loss"
    if pct >= take_profit_pct - _EPSILON:
        return "take_profit"
    return None

"""Risk tiers and expected-value (EV) math for scalpbot.

Why tiers?
----------
A scalper's edge is not any single take-profit number -- it is keeping the
*expected value* per trade positive after fees, then sizing to survive variance.
This module packages three coherent risk profiles (CONSERVATIVE / MODERATE /
HIGH), each a self-consistent bundle of:

* take-profit % / stop-loss %  (the reward:risk ratio),
* the sentiment entry-score gate (how much conviction is required to act),
* the Groq confidence floor (how sure the AI advisor must be),
* the per-position dollar budget (variance / exposure control),
* the option premium profit target (which rung of the +20/40/60% ladder).

Fee economics (the part everyone underestimates)
------------------------------------------------
Alpaca charges a **taker fee of 0.25% per side**, so a round trip costs
**0.50%**. That fee is symmetric drag: it shrinks every win and deepens every
loss. With take-profit ``tp`` and stop-loss ``sl`` (both positive fractions):

    net_win  = tp - fee      # what you actually keep on a winner
    net_loss = sl + fee      # what you actually give back on a loser

The break-even win-rate (the win-rate at which EV == 0) is::

    breakeven_WR = net_loss / (net_win + net_loss)

Worked per tier (fee = 0.50%):

* CONSERVATIVE crypto  tp=4%  sl=2%  -> net_win=3.5% net_loss=2.5% -> BE 41.7%
* CONSERVATIVE stock   tp=3%  sl=1.5%-> net_win=2.5% net_loss=2.0% -> BE 44.4%
* MODERATE     crypto  tp=6%  sl=3%  -> net_win=5.5% net_loss=3.5% -> BE 38.9%
* MODERATE     stock   tp=4%  sl=2%  -> net_win=3.5% net_loss=2.5% -> BE 41.7%
* HIGH         crypto  tp=9%  sl=4%  -> net_win=8.5% net_loss=4.5% -> BE 34.6%
* HIGH         stock   tp=6%  sl=2.5%-> net_win=5.5% net_loss=3.0% -> BE 35.3%

Notice the pattern: wider targets *lower* the break-even win-rate (each winner
pays for more losers), but they also fire less often and demand the trade
actually run further before reversing. That is the real trade-off the tiers
encode -- CONSERVATIVE needs to be right ~42% of the time on smaller, quicker
moves; HIGH can be right only ~35% of the time but needs bigger, rarer runs.

Trader rationale per tier
-------------------------
* **CONSERVATIVE** -- tight 2:1 reward:risk on modest moves, the strictest
  conviction gate (entry score 0.25, Groq >= 60%), and the smallest size ($75).
  Grinds many small, high-probability scalps; minimizes drawdown. Good default
  for an unproven signal or a live account you are protecting.
* **MODERATE** -- the balanced middle. 2:1 R:R on larger moves, a slightly
  looser gate (0.20 / 55%), standard $100 size. This is the shipped default.
* **HIGH** -- momentum/runner profile. ~2.25-2.4:1 R:R on big moves, the
  *highest* conviction gate (0.30 / 60%) precisely because the bets are larger
  ($150) and fire less often. Fewer, bigger, more selective trades.

Nothing here is a secret -- the active tier is chosen with the ``RISK_TIER``
plain-``.env`` variable and resolved through :func:`get_tier`.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RiskTier(str, Enum):
    """The three supported risk profiles. Inherits ``str`` so the value compares
    cleanly against env strings and serializes as plain text."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    HIGH = "high"


@dataclass(frozen=True)
class TierParams:
    """Immutable bundle of every risk parameter for one tier."""

    name: str

    # crypto exit thresholds (fractions of entry price). Used as the static
    # fallback when ATR data is unavailable, and for crypto_breakeven_wr's
    # nominal estimate; the live exit distance is ATR-scaled -- see the
    # *_atr_mult fields below.
    crypto_take_profit_pct: float
    crypto_stop_loss_pct: float

    # stock exit thresholds (fractions of entry price) -- same role as above.
    stock_take_profit_pct: float
    stock_stop_loss_pct: float

    # ATR multiples for volatility-scaled exits (engine._manage_crypto_exits /
    # _manage_stock_exits): tp_pct = (ATR * tp_mult) / price, so the same
    # tier stakes a wider/tighter distance depending on how volatile the
    # asset actually is right now, rather than a fixed percentage. Chosen to
    # preserve each tier's reward:risk ratio above (e.g. moderate crypto's
    # 2:1 is 3.0/1.5 here too); only the absolute scale is now ATR-driven.
    crypto_tp_atr_mult: float
    crypto_sl_atr_mult: float
    stock_tp_atr_mult: float
    stock_sl_atr_mult: float

    # option management: which rung of the +20/40/60% premium ladder to target
    option_profit_target_pct: float

    # entry gates
    entry_score_threshold: float  # min |sentiment score| to consider an entry
    groq_confidence_min: float    # min Groq advisor confidence to act

    # sizing
    position_budget_usd: float

    # ---- derived helpers -------------------------------------------------
    def crypto_breakeven_wr(self, fee_pct: float = 0.005) -> float:
        """Break-even win-rate for the crypto TP/SL after round-trip fees."""
        net_win = self.crypto_take_profit_pct - fee_pct
        net_loss = self.crypto_stop_loss_pct + fee_pct
        return net_loss / (net_win + net_loss)

    def stock_breakeven_wr(self, fee_pct: float = 0.005) -> float:
        """Break-even win-rate for the stock TP/SL after round-trip fees."""
        net_win = self.stock_take_profit_pct - fee_pct
        net_loss = self.stock_stop_loss_pct + fee_pct
        return net_loss / (net_win + net_loss)


# ---------------------------------------------------------------------------
# the three tiers (exact values per spec)
# ---------------------------------------------------------------------------
_TIERS: dict[str, TierParams] = {
    RiskTier.CONSERVATIVE.value: TierParams(
        name=RiskTier.CONSERVATIVE.value,
        crypto_take_profit_pct=0.04,   # 2:1 R:R; break-even ~41.7%
        crypto_stop_loss_pct=0.02,
        stock_take_profit_pct=0.03,    # 2:1 R:R; break-even ~44.4%
        stock_stop_loss_pct=0.015,
        crypto_tp_atr_mult=2.0,        # 2:1 R:R, tightest ATR distance
        crypto_sl_atr_mult=1.0,
        stock_tp_atr_mult=2.0,
        stock_sl_atr_mult=1.0,
        option_profit_target_pct=0.20,  # +20% premium (ladder rung 1)
        entry_score_threshold=0.25,     # tighter conviction gate
        groq_confidence_min=0.60,       # Groq must be >= 60% confident
        position_budget_usd=75.0,       # smaller position size
    ),
    RiskTier.MODERATE.value: TierParams(
        name=RiskTier.MODERATE.value,
        crypto_take_profit_pct=0.06,   # 2:1 R:R; break-even ~38.9%
        crypto_stop_loss_pct=0.03,
        stock_take_profit_pct=0.04,    # 2:1 R:R; break-even ~41.7%
        stock_stop_loss_pct=0.02,
        crypto_tp_atr_mult=3.0,        # 2:1 R:R, standard ATR distance
        crypto_sl_atr_mult=1.5,
        stock_tp_atr_mult=3.0,
        stock_sl_atr_mult=1.5,
        option_profit_target_pct=0.40,  # +40% premium (ladder rung 2)
        entry_score_threshold=0.20,
        groq_confidence_min=0.55,
        position_budget_usd=100.0,
    ),
    RiskTier.HIGH.value: TierParams(
        name=RiskTier.HIGH.value,
        crypto_take_profit_pct=0.09,   # 2.25:1 R:R; break-even ~34.6%
        crypto_stop_loss_pct=0.04,
        stock_take_profit_pct=0.06,    # 2.4:1 R:R; break-even ~35.3%
        stock_stop_loss_pct=0.025,
        crypto_tp_atr_mult=4.5,        # 2.25:1 R:R, widest ATR distance
        crypto_sl_atr_mult=2.0,
        stock_tp_atr_mult=4.8,         # 2.4:1 R:R
        stock_sl_atr_mult=2.0,
        option_profit_target_pct=0.60,  # +60% premium (ladder rung 3)
        entry_score_threshold=0.30,     # higher conviction required
        groq_confidence_min=0.60,       # high confidence needed for larger bets
        position_budget_usd=150.0,
    ),
}


def get_tier(name: str) -> TierParams:
    """Return the :class:`TierParams` for ``name`` (case-insensitive).

    Unknown or empty names fall back to MODERATE, the balanced default, so a
    typo in ``RISK_TIER`` degrades safely instead of crashing the engine.
    """
    key = (name or "").strip().lower()
    return _TIERS.get(key, _TIERS[RiskTier.MODERATE.value])


def ev_positive(
    take_profit_pct: float,
    stop_loss_pct: float,
    win_prob: float,
    fee_pct: float = 0.005,
) -> bool:
    """Is the expected value of this trade positive after round-trip fees?

    ::

        net_win  = take_profit_pct - fee_pct
        net_loss = stop_loss_pct   + fee_pct
        ev = win_prob * net_win - (1 - win_prob) * net_loss
        return ev > 0

    ``fee_pct`` defaults to 0.005 (0.25% taker fee x 2 sides = 0.50% round trip).
    ``win_prob`` is the projected probability of hitting the take-profit before
    the stop-loss -- in the engine this comes from the Groq advisor's
    ``projected_win_probability``. The gate blocks any trade whose expected
    value does not clear the fee drag, no matter how strong the sentiment.
    """
    net_win = take_profit_pct - fee_pct
    net_loss = stop_loss_pct + fee_pct
    ev = win_prob * net_win - (1.0 - win_prob) * net_loss
    return ev > 0

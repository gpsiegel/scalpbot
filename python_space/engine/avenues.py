"""Signal -> decision logic for the crypto and stock avenues.

These functions are pure and network-free: the caller (``engine.py``) supplies
the aggregated sentiment score and the latest price, and gets back a plain
:class:`AvenueDecision`. That keeps the trading rules unit-testable without
Alpaca or Postgres.

Entry rules
-----------
* **crypto** (spot, long-only): open LONG when score >= ``CRYPTO_ENTRY_SCORE``.
* **stock**: open LONG when score >= ``STOCK_ENTRY_SCORE``; open SHORT when
  score <= -``STOCK_ENTRY_SCORE`` *and* shorting is allowed.

Exit rules (crypto/stock)
-------------------------
Price-based: take profit at ``+TAKE_PROFIT_PCT``, stop out at ``-STOP_LOSS_PCT``
(measured in the direction of the position). Also exit when sentiment flips
hard against an open position.

All thresholds are operational tunables -> plain ``.env`` (NOT Doppler), with
conservative scalping defaults.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from .models import MARKET_CRYPTO, MARKET_STOCK, SIDE_LONG, SIDE_SHORT


def _f(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# Entry sentiment thresholds (aggregated score is in [-1, 1]).
CRYPTO_ENTRY_SCORE = _f("CRYPTO_ENTRY_SCORE", 0.15)
STOCK_ENTRY_SCORE = _f("STOCK_ENTRY_SCORE", 0.15)
OPTION_ENTRY_SCORE = _f("OPTION_ENTRY_SCORE", 0.25)  # options need stronger conviction

# Price-based exit thresholds for crypto/stock (fraction of entry).
TAKE_PROFIT_PCT = _f("TAKE_PROFIT_PCT", 0.03)   # +3%
STOP_LOSS_PCT = _f("STOP_LOSS_PCT", 0.02)       # -2%
# Sentiment reversal exit: close a long if score falls below this (and vice versa).
SENTIMENT_FLIP = _f("SENTIMENT_FLIP", 0.0)


@dataclass
class AvenueDecision:
    action: str            # "open" | "close" | "hold" | "skip"
    market: str
    symbol: str
    side: str = ""
    reason: str = ""
    score: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.action == "open"

    @property
    def is_close(self) -> bool:
        return self.action == "close"


# ---------------------------------------------------------------------------
# entries
# ---------------------------------------------------------------------------
def crypto_entry_decision(symbol: str, score: float) -> AvenueDecision:
    """Crypto is spot/long-only."""
    if score >= CRYPTO_ENTRY_SCORE:
        return AvenueDecision(
            action="open", market=MARKET_CRYPTO, symbol=symbol,
            side=SIDE_LONG, score=score,
            reason=f"bullish score {score:.3f} >= {CRYPTO_ENTRY_SCORE:.3f}",
        )
    return AvenueDecision(
        action="skip", market=MARKET_CRYPTO, symbol=symbol, score=score,
        reason=f"score {score:.3f} below entry {CRYPTO_ENTRY_SCORE:.3f}",
    )


def stock_entry_decision(symbol: str, score: float, allow_short: bool) -> AvenueDecision:
    """Stocks can go long or (optionally) short."""
    if score >= STOCK_ENTRY_SCORE:
        return AvenueDecision(
            action="open", market=MARKET_STOCK, symbol=symbol,
            side=SIDE_LONG, score=score,
            reason=f"bullish score {score:.3f} >= {STOCK_ENTRY_SCORE:.3f}",
        )
    if allow_short and score <= -STOCK_ENTRY_SCORE:
        return AvenueDecision(
            action="open", market=MARKET_STOCK, symbol=symbol,
            side=SIDE_SHORT, score=score,
            reason=f"bearish score {score:.3f} <= {-STOCK_ENTRY_SCORE:.3f}",
        )
    return AvenueDecision(
        action="skip", market=MARKET_STOCK, symbol=symbol, score=score,
        reason=f"score {score:.3f} within neutral band",
    )


def option_side_for_score(score: float) -> Optional[str]:
    """Return "call" for strong bullish, "put" for strong bearish, else None."""
    if score >= OPTION_ENTRY_SCORE:
        return "call"
    if score <= -OPTION_ENTRY_SCORE:
        return "put"
    return None


# ---------------------------------------------------------------------------
# exits (crypto / stock, price + sentiment based)
# ---------------------------------------------------------------------------
def position_pnl_pct(side: str, entry_price: float, current_price: float) -> float:
    """Signed P&L fraction for a directional position."""
    if entry_price <= 0:
        return 0.0
    raw = (current_price - entry_price) / entry_price
    return raw if side == SIDE_LONG else -raw


def linear_exit_decision(
    market: str,
    symbol: str,
    side: str,
    entry_price: float,
    current_price: float,
    score: float,
) -> AvenueDecision:
    """Exit logic shared by crypto and stock (price + sentiment reversal)."""
    pnl = position_pnl_pct(side, entry_price, current_price)

    if pnl >= TAKE_PROFIT_PCT:
        return AvenueDecision(
            action="close", market=market, symbol=symbol, side=side, score=score,
            reason=f"take profit {pnl*100:.2f}% >= {TAKE_PROFIT_PCT*100:.2f}%",
        )
    if pnl <= -STOP_LOSS_PCT:
        return AvenueDecision(
            action="close", market=market, symbol=symbol, side=side, score=score,
            reason=f"stop loss {pnl*100:.2f}% <= {-STOP_LOSS_PCT*100:.2f}%",
        )
    # sentiment reversal
    if side == SIDE_LONG and score < SENTIMENT_FLIP - CRYPTO_ENTRY_SCORE:
        return AvenueDecision(
            action="close", market=market, symbol=symbol, side=side, score=score,
            reason=f"sentiment reversed to {score:.3f}",
        )
    if side == SIDE_SHORT and score > SENTIMENT_FLIP + STOCK_ENTRY_SCORE:
        return AvenueDecision(
            action="close", market=market, symbol=symbol, side=side, score=score,
            reason=f"sentiment reversed to {score:.3f}",
        )
    return AvenueDecision(
        action="hold", market=market, symbol=symbol, side=side, score=score,
        reason=f"pnl {pnl*100:.2f}% within band",
    )

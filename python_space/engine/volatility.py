"""Average True Range (ATR) and related volatility helpers.

Pure, network-free functions only -- :func:`compute_atr` takes plain
``(high, low, close)`` bar tuples, so it's trivially unit-testable.
``AlpacaClient.get_atr`` (engine/alpaca_client.py) is what actually fetches
bars from Alpaca and calls into this module; nothing here touches the
network or the SDK.

Why this exists: the engine's crypto/stock exits used fixed take-profit /
stop-loss percentages per risk tier, so a 6% target meant the same thing on
a quiet day and a violent one. Scaling the exit distance by the asset's own
recent volatility (ATR) keeps the tier's reward:risk *ratio* intact while
letting the absolute distance breathe with the market.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

Bar = Tuple[float, float, float]  # (high, low, close)


def true_range(high: float, low: float, prev_close: float) -> float:
    """True range for one bar given the *previous* bar's close."""
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def compute_atr(bars: Sequence[Bar], period: int = 14) -> Optional[float]:
    """Average True Range over the most recent ``period`` bars.

    ``bars`` is a sequence of ``(high, low, close)`` tuples in chronological
    order (oldest first). Each true-range value needs the *previous* bar's
    close, so ``period + 1`` bars are required to produce ``period`` true
    ranges. Returns ``None`` (never raises) when there isn't enough data.
    """
    if period <= 0 or len(bars) < period + 1:
        return None
    true_ranges = [
        true_range(bars[i][0], bars[i][1], bars[i - 1][2])
        for i in range(1, len(bars))
    ]
    recent = true_ranges[-period:]
    return sum(recent) / len(recent)

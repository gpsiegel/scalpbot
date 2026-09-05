"""Lightweight lexicon-based text sentiment scoring.

Used by sources that return raw text (Reddit titles/bodies) but no built-in
sentiment score. Deliberately dependency-free (no NLTK/vader download at
runtime) so it works in the locked-down production container. Returns a score
in ``[-1, 1]``.
"""
from __future__ import annotations

import re
from typing import Iterable, Tuple

_POSITIVE = {
    "moon", "mooning", "bullish", "bull", "pump", "pumping", "rally", "surge",
    "surging", "breakout", "gains", "gain", "up", "green", "buy", "buying",
    "long", "hodl", "hold", "accumulate", "accumulating", "support", "strong",
    "strength", "rocket", "ath", "record", "soar", "soaring", "outperform",
    "adoption", "partnership", "upgrade", "win", "winning", "profit", "profits",
    "undervalued", "opportunity", "optimistic", "confident", "explode", "explosive",
}

_NEGATIVE = {
    "dump", "dumping", "bearish", "bear", "crash", "crashing", "drop", "dropping",
    "fall", "falling", "red", "sell", "selling", "short", "fear", "fud", "scam",
    "rug", "rugpull", "hack", "hacked", "exploit", "weak", "weakness", "resistance",
    "correction", "capitulation", "loss", "losses", "overvalued", "risk", "risky",
    "warning", "collapse", "plummet", "plunge", "tank", "tanking", "liquidated",
    "liquidation", "down", "bleed", "bleeding", "panic", "worried", "avoid",
}

_NEGATORS = {"not", "no", "never", "isn't", "isnt", "aren't", "arent", "won't", "wont", "dont", "don't"}

_WORD_RE = re.compile(r"[a-zA-Z']+")


def _tokens(text: str) -> Iterable[str]:
    return (m.group(0).lower() for m in _WORD_RE.finditer(text or ""))


def score_text(text: str) -> Tuple[float, int]:
    """Return ``(score, hits)`` for a single piece of text.

    ``score`` is in ``[-1, 1]``; ``hits`` is the number of sentiment-bearing
    tokens found (used as a confidence proxy).
    """
    toks = list(_tokens(text))
    if not toks:
        return 0.0, 0
    pos = neg = 0
    for i, tok in enumerate(toks):
        negated = i > 0 and toks[i - 1] in _NEGATORS
        if tok in _POSITIVE:
            neg += 1 if negated else 0
            pos += 0 if negated else 1
        elif tok in _NEGATIVE:
            pos += 1 if negated else 0
            neg += 0 if negated else 1
    hits = pos + neg
    if hits == 0:
        return 0.0, 0
    return (pos - neg) / hits, hits


def score_many(texts: Iterable[str]) -> Tuple[float, int]:
    """Aggregate sentiment across many texts, weighted by hit count.

    Returns ``(aggregate_score, total_hits)``.
    """
    weighted_sum = 0.0
    total_hits = 0
    for text in texts:
        s, hits = score_text(text)
        if hits:
            weighted_sum += s * hits
            total_hits += hits
    if total_hits == 0:
        return 0.0, 0
    return weighted_sum / total_hits, total_hits

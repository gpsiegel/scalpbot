"""Groq-backed trade advisor for scalpbot.

What this is
------------
A thin, *defensive* wrapper around Groq's OpenAI-compatible chat-completions
endpoint. Given a candidate entry (market, symbol, sentiment score, and the
active risk tier) it asks a fast reasoning model to act as a professional scalp
trader and return a structured verdict: an action, a confidence, and -- most
importantly for the engine -- a ``projected_win_probability`` that feeds the
expected-value gate in :func:`engine.risk.ev_positive`.

Design principle: **fail safe, never fail loud.**
------------------------------------------------
The advisor is an *optional edge*, not a hard dependency. The trading loop must
keep running even if Groq is unconfigured, rate-limited, slow, returns garbage
JSON, or is down entirely. Therefore *every* failure path -- missing API key,
network error, timeout, non-200 status, unparseable body, missing fields --
collapses to the same conservative, neutral result: ``action="skip"`` with
``confidence=0.0`` and a ``projected_win_probability`` of 0.5 (a coin flip,
which after fees fails the EV gate and blocks the trade). A missing advisor thus
never *opens* a position; it only ever declines to endorse one.

The ``GROQ_API_KEY`` lives in Doppler (it is a real secret). Everything else
about the advisor -- model, thresholds -- is non-secret configuration.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

from .risk import TierParams

logger = logging.getLogger(__name__)

# Groq OpenAI-compatible endpoint + model. Non-secret; safe to keep in source.
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-120b"
REQUEST_TIMEOUT_SECONDS = 10

_SYSTEM_PROMPT = (
    "You are a professional cryptocurrency and equities scalp trader AI. You "
    "evaluate short-hold, high-turnover trade candidates and give a disciplined, "
    "risk-aware verdict. Your edge comes from three things: (1) momentum "
    "alignment -- only endorse trades whose short-term momentum agrees with the "
    "sentiment signal; (2) fee-drag awareness -- every round trip costs about "
    "0.50% in taker fees, so a trade must have room to clear that drag before it "
    "is worth taking; (3) market microstructure -- consider spread, likely "
    "slippage, and how far price must travel to the take-profit before the "
    "stop-loss. Be skeptical: most marginal scalps are negative expected value "
    "after fees. Respond with a single JSON object and nothing else, using the "
    "keys: action (one of \"enter\", \"skip\", \"hold\"), confidence (0.0-1.0), "
    "projected_win_probability (0.0-1.0, your estimate of hitting take-profit "
    "before stop-loss), reasoning (one short sentence), risk_tier_fit (one of "
    "\"good\", \"neutral\", \"poor\"), and signal_strength (one of \"strong\", "
    "\"moderate\", \"weak\")."
)


@dataclass
class GroqAdvice:
    """Structured advisor verdict consumed by the trading engine.

    Defaults are the *neutral hold*: they are exactly what every failure path
    returns, so the engine can treat "advisor unavailable" and "advisor says
    don't" identically -- both simply fail to endorse the trade.
    """

    action: str = "skip"
    confidence: float = 0.0
    projected_win_probability: float = 0.5
    reasoning: str = "groq unavailable - defaulting to hold"
    risk_tier_fit: str = "neutral"
    signal_strength: str = "weak"
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def endorses_entry(self) -> bool:
        """True only when the advisor actively recommends entering."""
        return self.action == "enter"


def _neutral_hold(reasoning: str = "groq unavailable - defaulting to hold") -> GroqAdvice:
    """Canonical safe result for every failure / disabled path."""
    return GroqAdvice(
        action="skip",
        confidence=0.0,
        projected_win_probability=0.5,
        reasoning=reasoning,
        risk_tier_fit="neutral",
        signal_strength="weak",
        raw={},
    )


class GroqAdvisor:
    """Defensive client for Groq trade validation.

    Parameters
    ----------
    api_key:
        Optional explicit key. When omitted, falls back to the ``GROQ_API_KEY``
        environment variable (populated from Doppler). When neither is present
        the advisor is *disabled* and every call returns a neutral hold.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = (api_key or os.environ.get("GROQ_API_KEY") or "").strip()

    @property
    def is_disabled(self) -> bool:
        """True when no API key is configured -- the advisor is inert."""
        return not self._api_key

    def validate_trade(
        self,
        market: str,
        symbol: str,
        sentiment_score: float,
        tier: TierParams,
        extra_context: Optional[dict[str, Any]] = None,
    ) -> GroqAdvice:
        """Ask Groq to validate a candidate entry.

        Returns a :class:`GroqAdvice`. This method never raises: any problem --
        disabled advisor, network error, timeout, bad status, malformed JSON --
        yields the neutral-hold result, which fails downstream gates safely.
        """
        if self.is_disabled:
            return _neutral_hold("groq disabled - no api key configured")

        try:
            payload = self._build_payload(
                market, symbol, sentiment_score, tier, extra_context
            )
            resp = requests.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if resp.status_code != 200:
                logger.warning(
                    "groq advisor: non-200 status %s for %s/%s; defaulting to hold",
                    resp.status_code,
                    market,
                    symbol,
                )
                return _neutral_hold(
                    f"groq http {resp.status_code} - defaulting to hold"
                )

            body = resp.json()
            content = body["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            return self._parse_advice(parsed)
        except Exception as exc:  # noqa: BLE001 - intentionally catch-all: fail safe
            logger.warning(
                "groq advisor: failed for %s/%s (%s); defaulting to hold",
                market,
                symbol,
                exc,
            )
            return _neutral_hold("groq error - defaulting to hold")

    # ---- internals -------------------------------------------------------
    def _build_payload(
        self,
        market: str,
        symbol: str,
        sentiment_score: float,
        tier: TierParams,
        extra_context: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        """Assemble the chat-completions request body."""
        user_context = {
            "market": market,
            "symbol": symbol,
            "sentiment_score": round(float(sentiment_score), 4),
            "risk_tier": {
                "name": tier.name,
                "crypto_take_profit_pct": tier.crypto_take_profit_pct,
                "crypto_stop_loss_pct": tier.crypto_stop_loss_pct,
                "stock_take_profit_pct": tier.stock_take_profit_pct,
                "stock_stop_loss_pct": tier.stock_stop_loss_pct,
                "entry_score_threshold": tier.entry_score_threshold,
                "groq_confidence_min": tier.groq_confidence_min,
            },
            "round_trip_fee_pct": 0.005,
        }
        if extra_context:
            user_context["extra_context"] = extra_context

        user_prompt = (
            "Evaluate this scalp trade candidate and respond with the JSON "
            "verdict object described in your instructions.\n\n"
            + json.dumps(user_context)
        )
        return {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "reasoning_effort": "low",
            "max_tokens": 2500,
            "response_format": {"type": "json_object"},
        }

    def _parse_advice(self, parsed: dict[str, Any]) -> GroqAdvice:
        """Coerce a parsed JSON dict into a :class:`GroqAdvice`, clamping ranges.

        Missing or malformed fields fall back to neutral defaults so a partial
        response still yields a usable, conservative verdict.
        """
        def _clamp01(value: Any, default: float) -> float:
            try:
                f = float(value)
            except (TypeError, ValueError):
                return default
            return max(0.0, min(1.0, f))

        action = str(parsed.get("action", "skip")).strip().lower()
        if action not in {"enter", "skip", "hold"}:
            action = "skip"

        return GroqAdvice(
            action=action,
            confidence=_clamp01(parsed.get("confidence"), 0.0),
            projected_win_probability=_clamp01(
                parsed.get("projected_win_probability"), 0.5
            ),
            reasoning=str(parsed.get("reasoning", "")).strip() or "no reasoning given",
            risk_tier_fit=str(parsed.get("risk_tier_fit", "neutral")).strip().lower()
            or "neutral",
            signal_strength=str(parsed.get("signal_strength", "weak")).strip().lower()
            or "weak",
            raw=parsed,
        )

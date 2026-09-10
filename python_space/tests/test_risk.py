"""Tests for the risk-management layer:

* expected-value math and tier lookup (engine.risk),
* the defensive Groq advisor when disabled (engine.groq_advisor),
* the consecutive-loss circuit breaker and the confidence gate (engine.engine).

Everything runs against an in-memory SQLite DB with no network and no real
Groq calls (the advisor is exercised only in its disabled/offline path).
"""
from __future__ import annotations

import datetime as dt
import os

import pytest
from sqlalchemy.orm import sessionmaker

# Deterministic paper environment BEFORE importing config-dependent modules.
os.environ.setdefault("APP_ENV", "nonprod")
os.environ.setdefault("STOCK_TICKERS", "PLTR,SCHD")
os.environ.setdefault("CRYPTO_CORE_COINS", "SOL,BTC")

from engine.risk import RiskTier, TierParams, ev_positive, get_tier  # noqa: E402
from engine.groq_advisor import GroqAdvice, GroqAdvisor  # noqa: E402
from engine.engine import ConfidenceGate, ConsecutiveLossTracker  # noqa: E402
from engine.models import Base, BotConfig, make_engine  # noqa: E402


@pytest.fixture()
def session_factory(tmp_path):
    db = tmp_path / "risk.db"
    eng = make_engine(f"sqlite:///{db}")
    Base.metadata.create_all(eng)
    return sessionmaker(bind=eng, expire_on_commit=False, future=True)


# --------------------------------------------------------------------------
# expected-value math
# --------------------------------------------------------------------------
def test_ev_positive_basic_fee_drag():
    # tp=6% sl=3% round-trip fee=0.5%. break-even ~38.9%.
    assert ev_positive(0.06, 0.03, win_prob=0.50) is True     # well above BE
    assert ev_positive(0.06, 0.03, win_prob=0.30) is False    # below BE
    # right at a coin flip, a tight symmetric trade cannot clear the fee.
    assert ev_positive(0.02, 0.02, win_prob=0.50) is False


def test_ev_positive_respects_custom_fee():
    # with zero fees a 50/50 on positive-skew tp>sl is positive EV.
    assert ev_positive(0.06, 0.03, win_prob=0.50, fee_pct=0.0) is True
    # crank fees high enough and the same trade turns negative.
    assert ev_positive(0.06, 0.03, win_prob=0.50, fee_pct=0.05) is False


def test_ev_matches_tier_breakeven_boundaries():
    tier = get_tier("moderate")
    be = tier.crypto_breakeven_wr()  # ~0.389
    # just above break-even -> positive; just below -> negative.
    assert ev_positive(tier.crypto_take_profit_pct, tier.crypto_stop_loss_pct, be + 0.02)
    assert not ev_positive(tier.crypto_take_profit_pct, tier.crypto_stop_loss_pct, be - 0.02)


# --------------------------------------------------------------------------
# tier lookup
# --------------------------------------------------------------------------
def test_get_tier_case_insensitive():
    assert get_tier("CONSERVATIVE").name == RiskTier.CONSERVATIVE.value
    assert get_tier("Moderate").name == RiskTier.MODERATE.value
    assert get_tier("  high  ").name == RiskTier.HIGH.value


def test_get_tier_unknown_falls_back_to_moderate():
    assert get_tier("nonsense").name == RiskTier.MODERATE.value
    assert get_tier("").name == RiskTier.MODERATE.value
    assert get_tier(None).name == RiskTier.MODERATE.value  # type: ignore[arg-type]


def test_tier_values_are_self_consistent():
    cons, mod, high = get_tier("conservative"), get_tier("moderate"), get_tier("high")
    # wider targets should lower the break-even win-rate.
    assert cons.crypto_breakeven_wr() > mod.crypto_breakeven_wr() > high.crypto_breakeven_wr()
    # sizing grows with risk appetite.
    assert cons.position_budget_usd < mod.position_budget_usd < high.position_budget_usd
    # every tier is a dataclass bundle of the expected type.
    assert isinstance(mod, TierParams)


# --------------------------------------------------------------------------
# Groq advisor -- disabled / offline path (no network)
# --------------------------------------------------------------------------
def test_groq_advisor_disabled_returns_neutral_hold():
    advisor = GroqAdvisor(api_key="")
    assert advisor.is_disabled is True
    advice = advisor.validate_trade("crypto", "SOL/USD", 0.6, get_tier("moderate"))
    assert isinstance(advice, GroqAdvice)
    assert advice.action == "skip"
    assert advice.confidence == 0.0
    assert advice.projected_win_probability == 0.5   # coin-flip; EV-positive on its own,
    # but action="skip" (not "enter") is what keeps this from endorsing a trade
    assert advice.endorses_entry is False


def test_groq_advisor_enabled_flag_from_key():
    assert GroqAdvisor(api_key="sk-test-123").is_disabled is False


# --------------------------------------------------------------------------
# consecutive-loss circuit breaker
# --------------------------------------------------------------------------
def test_loss_tracker_trips_after_threshold(session_factory):
    s = session_factory()
    t = ConsecutiveLossTracker()
    t.record_loss("crypto", s)
    t.record_loss("crypto", s)
    assert t.is_cooling_down("crypto", s) is False   # 2 losses, not yet
    t.record_loss("crypto", s)                        # 3rd loss trips breaker
    assert t.is_cooling_down("crypto", s) is True
    # a cooldown row was persisted with a future timestamp.
    row = s.query(BotConfig).filter_by(key="cooldown_until_crypto").first()
    assert row is not None and row.value
    s.close()


def test_loss_tracker_win_resets_streak(session_factory):
    s = session_factory()
    t = ConsecutiveLossTracker()
    t.record_loss("crypto", s)
    t.record_loss("crypto", s)
    t.record_win("crypto", s)         # streak reset
    t.record_loss("crypto", s)
    t.record_loss("crypto", s)        # only 2 consecutive again
    assert t.is_cooling_down("crypto", s) is False
    s.close()


def test_loss_tracker_isolated_per_market(session_factory):
    s = session_factory()
    t = ConsecutiveLossTracker()
    for _ in range(3):
        t.record_loss("crypto", s)
    assert t.is_cooling_down("crypto", s) is True
    assert t.is_cooling_down("stock", s) is False     # unaffected
    s.close()


def test_loss_tracker_persists_across_new_instance(session_factory):
    # A fresh ConsecutiveLossTracker (as created on every redeploy) must pick
    # up an in-progress streak from bot_config rather than starting at zero.
    s = session_factory()
    t1 = ConsecutiveLossTracker()
    t1.record_loss("crypto", s)
    t1.record_loss("crypto", s)

    t2 = ConsecutiveLossTracker()  # simulates a restart: no in-memory state
    assert t2.is_cooling_down("crypto", s) is False
    t2.record_loss("crypto", s)    # the 3rd consecutive loss, from t2's view
    assert t2.is_cooling_down("crypto", s) is True
    s.close()


def test_loss_tracker_expired_cooldown_is_not_active(session_factory):
    s = session_factory()
    t = ConsecutiveLossTracker()
    past = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=5)).isoformat()
    s.add(BotConfig(key="cooldown_until_crypto", value=past, market="crypto"))
    s.flush()
    assert t.is_cooling_down("crypto", s) is False
    s.close()


# --------------------------------------------------------------------------
# confidence gate
# --------------------------------------------------------------------------
def test_confidence_gate_active_by_default(session_factory):
    s = session_factory()
    g = ConfidenceGate()
    assert g.is_active("crypto", get_tier("moderate"), s) is True
    s.close()


def test_confidence_gate_passes_with_positive_expectancy(session_factory):
    s = session_factory()
    g = ConfidenceGate()
    tier = get_tier("moderate")
    # 29 small after-fee losses, then one large win -> net positive over the
    # 30-trade window, so it passes on actual expectancy, not a win count
    # (only 1 of 30 trades here actually won).
    results = [g.record_paper_trade("crypto", tier, -1.0, s) for _ in range(29)]
    results.append(g.record_paper_trade("crypto", tier, 50.0, s))
    assert results[-1] is True          # the 30th trade closes the gate
    assert g.is_active("crypto", tier, s) is False
    s.close()


def test_confidence_gate_fails_closed_and_does_not_auto_retry(session_factory):
    s = session_factory()
    g = ConfidenceGate()
    tier = get_tier("moderate")
    # 30 losing trades -> negative expectancy, gate fails.
    results = [g.record_paper_trade("stock", tier, -1.0, s) for _ in range(30)]
    assert results[-1] is False
    assert g.is_active("stock", tier, s) is True   # still blocks real orders
    assert g._get(s, "gate_failed_stock") == "1"

    # Unlike the old win-tally design, a failed window does NOT quietly start
    # counting a fresh one -- further paper trades are simply ignored.
    total_before = g._get(s, "gate_total_stock")
    passed = g.record_paper_trade("stock", tier, 1000.0, s)
    assert passed is False
    assert g._get(s, "gate_total_stock") == total_before
    assert g.is_active("stock", tier, s) is True


def test_confidence_gate_manual_reset_via_config(session_factory):
    s = session_factory()
    g = ConfidenceGate()
    tier = get_tier("moderate")
    for _ in range(30):
        g.record_paper_trade("option", tier, -1.0, s)
    assert g.is_active("option", tier, s) is True
    assert g._get(s, "gate_failed_option") == "1"

    # An operator resets it the same way POST /config would: writing a new
    # value under key=f"gate_reset_{market}".
    g._set(s, "gate_reset_option", "reset-1", "option")
    assert g.is_active("option", tier, s) is True   # fresh window, still validating
    assert g._get(s, "gate_failed_option") == "0"
    assert g._get(s, "gate_total_option") == "0"

    # posting the same reset value again is a no-op (only a *change* resets).
    for _ in range(30):
        g.record_paper_trade("option", tier, -1.0, s)
    assert g._get(s, "gate_failed_option") == "1"
    g._set(s, "gate_reset_option", "reset-1", "option")
    assert g._get(s, "gate_failed_option") == "1"


def test_confidence_gate_resets_on_tier_change(session_factory):
    s = session_factory()
    g = ConfidenceGate()
    mod, high = get_tier("moderate"), get_tier("high")
    # bank two trades under moderate...
    g.record_paper_trade("crypto", mod, -1.0, s)
    g.record_paper_trade("crypto", mod, -1.0, s)
    assert g._get(s, "gate_total_crypto") == "2"
    # switching the tier re-opens a fresh window.
    assert g.is_active("crypto", high, s) is True
    assert g._get(s, "gate_total_crypto") == "0"
    assert g._get(s, "gate_tier_crypto") == high.name
    s.close()

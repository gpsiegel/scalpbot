"""Offline unit tests for python_space/config.py.

Run from python_space/:  python3 -m pytest tests/test_config.py -q
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest  # noqa: E402

from config import Config  # noqa: E402
from sentiment.base import COIN_METADATA  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Strip any scalpbot env vars so each test starts from known defaults."""
    for key in list(os.environ):
        if (
            key.startswith(("SENTIMENT_WEIGHT_", "MAX_", "OPTION_", "ENABLE_", "CRYPTO_"))
            or key in {
                "APP_ENV", "PAPER_TRADING", "LIVE_TRADING", "RESPECT_MARKET_HOURS",
                "ALLOW_SHORT", "STOCK_TICKERS",
            }
        ):
            monkeypatch.delenv(key, raising=False)
    yield


def test_defaults():
    c = Config.from_env()
    assert c.app_env == "nonprod"
    assert c.paper_trading is True
    assert c.live_trading is False
    # Stocks/options are off by default (no equity-capable sentiment source
    # yet -- see sentiment.base.EQUITY_CAPABLE_SOURCES); crypto stays on.
    assert c.avenues.crypto is True
    assert c.avenues.stocks is False
    assert c.avenues.options is False
    assert c.caps.total == 3
    assert c.options.min_dte == 7
    assert c.costs.crypto_taker_fee_pct == 0.0025
    assert c.max_hold_minutes == 240


def test_max_hold_minutes_env_override(monkeypatch):
    monkeypatch.setenv("MAX_HOLD_MINUTES", "30")
    assert Config.from_env().max_hold_minutes == 30


def test_list_parsing_upper_and_dedup(monkeypatch):
    monkeypatch.setenv("CRYPTO_CORE_COINS", "sol, doge ,SOL")
    monkeypatch.setenv("STOCK_TICKERS", "schd,xlf")
    c = Config.from_env()
    assert c.crypto_core_coins == ["SOL", "DOGE"]  # de-duped, upper-cased
    assert c.stock_tickers == ["SCHD", "XLF"]


def test_all_crypto_coins_merges_core_and_satellite(monkeypatch):
    monkeypatch.setenv("CRYPTO_CORE_COINS", "BTC,ETH,SOL")
    monkeypatch.setenv("CRYPTO_SATELLITE_COINS", "DOGE,ETH")  # ETH dupes core
    c = Config.from_env()
    assert c.all_crypto_coins == ["BTC", "ETH", "SOL", "DOGE"]


def test_crypto_symbol_and_fees():
    c = Config.from_env()
    assert c.crypto_symbol("sol") == "SOL/USD"
    # crypto: taker fee + spread, both round-tripped -> 1000 * 2 * (0.0025 + 0.0005)
    assert c.round_trip_fees("crypto", 1000) == pytest.approx(6.0)
    # stocks: commission-free at Alpaca, spread only -> 1000 * 2 * 0.0005
    assert c.round_trip_fees("stock", 1000) == pytest.approx(1.0)
    # options: spread is already captured via ask/bid execution, not a
    # separate fee -- only the (default-zero) regulatory fee applies here.
    assert c.round_trip_fees("option", 1000) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "app_env,paper,live,expected",
    [
        ("prod", "false", "true", True),    # all three guards satisfied
        ("prod", "true", "true", False),    # paper still on
        ("prod", "false", "false", False),  # live flag off
        ("dev", "false", "true", False),  # non-prod env forces paper
        ("nonprod", "false", "true", False),  # wrong env forces paper
    ],
)
def test_live_guard(monkeypatch, app_env, paper, live, expected):
    monkeypatch.setenv("APP_ENV", app_env)
    monkeypatch.setenv("PAPER_TRADING", paper)
    monkeypatch.setenv("LIVE_TRADING", live)
    assert Config.from_env().is_live() is expected


def test_validate_flags_unknown_coin(monkeypatch):
    monkeypatch.setenv("CRYPTO_CORE_COINS", "SOL,NOTACOIN")
    monkeypatch.setenv("STOCK_TICKERS", "SCHD")
    problems = Config.from_env().validate()
    assert any("NOTACOIN" in p for p in problems)


def test_validate_flags_empty_universes(monkeypatch):
    # crypto on but no coins, stocks on but no tickers
    monkeypatch.setenv("ENABLE_STOCKS", "true")
    problems = Config.from_env().validate()
    assert any("CRYPTO" in p for p in problems)
    assert any("STOCK_TICKERS" in p for p in problems)


def test_validate_clean_with_known_coins(monkeypatch):
    monkeypatch.setenv("CRYPTO_CORE_COINS", "BTC,ETH,SOL,DOGE")
    monkeypatch.setenv("STOCK_TICKERS", "SCHD,XLF")
    monkeypatch.setenv("ENABLE_STOCKS", "false")
    monkeypatch.setenv("ENABLE_OPTIONS", "false")
    assert Config.from_env().validate() == []


def test_validate_flags_stocks_enabled_without_equity_sentiment(monkeypatch):
    # Stocks/options currently have no equity-capable sentiment source (every
    # source but fear_greed is crypto-only) -- enabling either must warn.
    monkeypatch.setenv("ENABLE_STOCKS", "true")
    monkeypatch.setenv("STOCK_TICKERS", "SCHD")
    monkeypatch.setenv("ENABLE_OPTIONS", "false")
    problems = Config.from_env().validate()
    assert any("equity-capable" in p for p in problems)


def test_toggle_off(monkeypatch):
    monkeypatch.setenv("ENABLE_STOCKS", "false")
    monkeypatch.setenv("ALLOW_SHORT", "no")
    c = Config.from_env()
    assert c.avenues.stocks is False
    assert c.allow_short is False


def test_registry_entries_are_well_formed():
    required = {"name", "symbol", "coingecko_id", "cryptocv_ticker",
               "lunarcrush_symbol", "subreddits", "trends_keywords"}
    assert len(COIN_METADATA) >= 12  # enough to cover 8 core + 4 satellite
    for sym, meta in COIN_METADATA.items():
        assert required.issubset(meta), f"{sym} missing keys"
        assert meta["symbol"] == sym

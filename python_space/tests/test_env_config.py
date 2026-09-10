"""Tests for PR-controlled per-environment operational config and the
nonprod-paper / prod-only-live safety model.

Run from python_space/:  python3 -m pytest tests/test_env_config.py -q
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest  # noqa: E402

import config as cfg  # noqa: E402
from config import Config, load_env_file  # noqa: E402


@pytest.fixture(autouse=True)
def _snapshot_env():
    """Fully snapshot and restore os.environ.

    ``load_env_file`` mutates os.environ via ``setdefault`` (not monkeypatch),
    so we restore the exact environment after every test to avoid leakage.
    """
    saved = dict(os.environ)
    # Start from a clean slate for the vars these tests touch.
    for key in list(os.environ):
        if (
            key.startswith(("ENABLE_", "MAX_", "OPTION_", "CRYPTO_", "SENTIMENT_WEIGHT_"))
            or key in {
                "APP_ENV", "PAPER_TRADING", "LIVE_TRADING", "RESPECT_MARKET_HOURS",
                "ALLOW_SHORT", "STOCK_TICKERS", "RISK_TIER",
            }
        ):
            del os.environ[key]
    yield
    os.environ.clear()
    os.environ.update(saved)


# ---------------------------------------------------------------------------
# load_env_file precedence (setdefault: real env always wins)
# ---------------------------------------------------------------------------
def test_load_env_file_sets_unset_key(tmp_path, monkeypatch):
    env_dir = tmp_path / "environments"
    env_dir.mkdir()
    (env_dir / "nonprod.env").write_text(
        "APP_ENV=nonprod\nENABLE_STOCKS=false\nRISK_TIER=high  # inline comment\n"
        "# a full comment line\n\nQUOTED=\"quoted value\"\n"
    )
    monkeypatch.setattr(cfg, "_ENV_DIR", env_dir)
    assert "ENABLE_STOCKS" not in os.environ
    load_env_file("nonprod")
    assert os.environ["ENABLE_STOCKS"] == "false"
    assert os.environ["RISK_TIER"] == "high"          # inline comment stripped
    assert os.environ["QUOTED"] == "quoted value"      # quotes stripped
    os.environ.pop("QUOTED", None)


def test_load_env_file_does_not_override_existing(tmp_path, monkeypatch):
    env_dir = tmp_path / "environments"
    env_dir.mkdir()
    (env_dir / "nonprod.env").write_text("ENABLE_STOCKS=false\n")
    monkeypatch.setattr(cfg, "_ENV_DIR", env_dir)
    os.environ["ENABLE_STOCKS"] = "true"   # a real env var already present
    load_env_file("nonprod")
    assert os.environ["ENABLE_STOCKS"] == "true"   # file did NOT clobber it


def test_load_env_file_missing_file_is_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "_ENV_DIR", tmp_path / "does_not_exist")
    load_env_file("nonprod")   # must not raise


def test_unknown_app_env_falls_back_to_nonprod_file(tmp_path, monkeypatch):
    env_dir = tmp_path / "environments"
    env_dir.mkdir()
    (env_dir / "nonprod.env").write_text("RISK_TIER=conservative\n")
    monkeypatch.setattr(cfg, "_ENV_DIR", env_dir)
    load_env_file("weirdenv")   # unknown -> nonprod file
    assert os.environ["RISK_TIER"] == "conservative"


# ---------------------------------------------------------------------------
# committed real files load through from_env()
# ---------------------------------------------------------------------------
def test_real_nonprod_file_loads_defaults(monkeypatch):
    monkeypatch.setenv("APP_ENV", "nonprod")
    c = Config.from_env()
    assert c.app_env == "nonprod"
    # Stocks/options are off by default (no equity-capable sentiment source
    # yet -- see sentiment.base.EQUITY_CAPABLE_SOURCES); crypto stays on.
    assert c.avenues.crypto is True
    assert c.avenues.stocks is False
    assert c.avenues.options is False
    assert c.risk_tier == "moderate"


def test_avenue_toggle_from_env_file(tmp_path, monkeypatch):
    env_dir = tmp_path / "environments"
    env_dir.mkdir()
    (env_dir / "nonprod.env").write_text("APP_ENV=nonprod\nENABLE_STOCKS=false\n")
    monkeypatch.setattr(cfg, "_ENV_DIR", env_dir)
    c = Config.from_env()
    assert c.avenues.stocks is False       # avenue disabled purely via the file
    assert c.avenues.crypto is True


# ---------------------------------------------------------------------------
# environment safety model
# ---------------------------------------------------------------------------
def test_nonprod_can_never_be_live(monkeypatch):
    monkeypatch.setenv("APP_ENV", "nonprod")
    monkeypatch.setenv("PAPER_TRADING", "false")
    monkeypatch.setenv("LIVE_TRADING", "true")
    c = Config.from_env()
    assert c.is_live() is False
    assert c.is_live_capable_env is False
    # validate() should warn that the flags are ignored
    assert any("only APP_ENV=prod can trade live" in p for p in c.validate())


def test_prod_all_guards_is_live(monkeypatch):
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("PAPER_TRADING", "false")
    monkeypatch.setenv("LIVE_TRADING", "true")
    c = Config.from_env()
    assert c.is_live() is True
    assert c.is_live_capable_env is True


def test_prod_defaults_to_paper(monkeypatch):
    monkeypatch.setenv("APP_ENV", "prod")
    # PAPER_TRADING/LIVE_TRADING come from prod.env defaults (paper/false)
    c = Config.from_env()
    assert c.is_live() is False           # prod but not all guards set
    assert c.is_live_capable_env is True  # ...still the only live-capable env


def test_is_live_capable_env_only_prod(monkeypatch):
    for env, expected in (("nonprod", False), ("prod", True), ("dev", False)):
        monkeypatch.setenv("APP_ENV", env)
        assert Config.from_env().is_live_capable_env is expected

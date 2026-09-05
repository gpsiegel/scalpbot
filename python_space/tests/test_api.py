"""Tests for the control-plane API security model.

Verifies that state-changing endpoints require X-API-Key and that switching to
live mode requires a valid confirmation token.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

os.environ["SCALPBOT_API_KEY"] = "test-secret-key"
os.environ.setdefault("STOCK_TICKERS", "PLTR")
os.environ.setdefault("CRYPTO_CORE_COINS", "SOL")

from config import Config  # noqa: E402
from engine.api import create_app  # noqa: E402
from engine.engine import TradingEngine  # noqa: E402
from engine.models import Base, make_engine  # noqa: E402
from sentiment.aggregator import AggregatedSentiment  # noqa: E402


class FakeAgg:
    def get_sentiment(self, coin):
        return AggregatedSentiment(coin=coin.upper(), score=0.0, label="neutral")


class FakeAlpaca:
    def get_crypto_price(self, s):
        return 100.0

    def get_stock_price(self, s):
        return 20.0

    def is_market_open(self):
        return True

    def list_option_contracts(self, *a, **k):
        return []

    def submit_crypto_order(self, *a, **k):
        return None

    def submit_stock_order(self, *a, **k):
        return None

    def submit_option_order(self, *a, **k):
        return None


@pytest.fixture()
def client(tmp_path):
    eng = make_engine(f"sqlite:///{tmp_path/'api.db'}")
    Base.metadata.create_all(eng)
    sf = sessionmaker(bind=eng, expire_on_commit=False, future=True)
    cfg = Config(app_env="nonprod", stock_tickers=["PLTR"], crypto_core_coins=["SOL"])
    engine = TradingEngine(config=cfg, session_factory=sf,
                           alpaca=FakeAlpaca(), aggregator=FakeAgg())
    return TestClient(create_app(engine))


def test_health_open(client):
    assert client.get("/health").status_code == 200


def test_status_open_and_reports_paper(client):
    r = client.get("/status")
    assert r.status_code == 200
    assert r.json()["mode"] == "paper"


def test_run_cycle_requires_key(client):
    assert client.post("/run-cycle").status_code == 401
    assert client.post("/run-cycle", headers={"X-API-Key": "wrong"}).status_code == 401
    ok = client.post("/run-cycle", headers={"X-API-Key": "test-secret-key"})
    assert ok.status_code == 200
    assert ok.json()["market"] == "crypto"


def test_stocks_and_options_cycle_require_key(client):
    assert client.post("/stocks/run-cycle").status_code == 401
    assert client.post("/options/run-cycle").status_code == 401
    h = {"X-API-Key": "test-secret-key"}
    assert client.post("/stocks/run-cycle", headers=h).status_code == 200
    assert client.post("/options/run-cycle", headers=h).status_code == 200


def test_live_switch_requires_confirmation_token(client):
    h = {"X-API-Key": "test-secret-key"}
    # Without a token -> 428 Precondition Required.
    r = client.post("/mode", headers=h, json={
        "app_env": "prod", "paper_trading": False, "live_trading": True,
    })
    assert r.status_code == 428

    # Mint a token, then the switch succeeds.
    tok = client.get("/confirmation-token", headers=h).json()["confirmation_token"]
    r2 = client.post("/mode", headers=h, json={
        "app_env": "prod", "paper_trading": False, "live_trading": True,
        "confirmation_token": tok,
    })
    assert r2.status_code == 200
    assert r2.json()["is_live"] is True


def test_confirmation_token_requires_key(client):
    assert client.get("/confirmation-token").status_code == 401

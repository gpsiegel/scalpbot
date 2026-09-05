"""Local control-plane API for scalpbot.

Security model
--------------
* **Bind address**: the server is meant to run on ``127.0.0.1`` only (see
  :func:`run`). It is a loopback control plane, not a public API.
* **API key**: every *state-changing* endpoint requires the ``X-API-Key``
  header to match ``SCALPBOT_API_KEY`` (a Doppler secret). Read-only endpoints
  (``/health``, ``/status``, ``/positions``) are open on loopback.
* **Live-trading confirmation**: flipping to live mode is a two-step handshake.
  ``GET /confirmation-token`` (authenticated) mints a short-lived token; that
  exact token must be echoed back to ``POST /mode`` to authorize a live switch.
  This makes an accidental single request incapable of enabling live trading.

Endpoints requiring ``X-API-Key``:
  POST /run-cycle, POST /stocks/run-cycle, POST /options/run-cycle,
  POST /config, POST /mode, GET /confirmation-token
"""
from __future__ import annotations

import logging
import os
import secrets
import time
from typing import Optional

try:
    from fastapi import Depends, FastAPI, Header, HTTPException
    from pydantic import BaseModel
except ImportError:  # pragma: no cover - allows import without fastapi installed
    FastAPI = None  # type: ignore
    BaseModel = object  # type: ignore

from config import get_config
from .engine import TradingEngine
from .models import Position, POSITION_OPEN

logger = logging.getLogger("scalpbot.engine.api")

# Confirmation tokens for live-mode switches: token -> expiry epoch.
_CONFIRM_TOKENS: dict[str, float] = {}
_CONFIRM_TTL_SECONDS = 120


class ModeRequest(BaseModel):
    app_env: Optional[str] = None
    paper_trading: Optional[bool] = None
    live_trading: Optional[bool] = None
    confirmation_token: Optional[str] = None


class ConfigRequest(BaseModel):
    key: str
    value: str
    market: str = "all"


def _api_key() -> Optional[str]:
    # Prefer the scalpbot-specific name; fall back to the repo's generic API_KEY
    # Doppler secret so either convention works.
    return os.environ.get("SCALPBOT_API_KEY") or os.environ.get("API_KEY")


def _require_key(x_api_key: Optional[str]) -> None:
    expected = _api_key()
    if not expected:
        raise HTTPException(status_code=503, detail="SCALPBOT_API_KEY not configured")
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


def create_app(engine: Optional[TradingEngine] = None):
    """Build the FastAPI app. ``engine`` is injectable for tests."""
    if FastAPI is None:  # pragma: no cover
        raise RuntimeError("fastapi is not installed; add it from requirements.txt")

    app = FastAPI(title="scalpbot control plane", version="1.0.0")
    engine = engine or TradingEngine()

    def key_dep(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
        _require_key(x_api_key)
        return True

    # -- read-only (open on loopback) --------------------------------------
    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/status")
    def status():
        cfg = engine.config
        return {
            "mode": engine.mode,
            "app_env": cfg.app_env,
            "is_live": cfg.is_live(),
            "avenues": {
                "crypto": cfg.avenues.crypto,
                "stocks": cfg.avenues.stocks,
                "options": cfg.avenues.options,
            },
            "caps": {
                "total": cfg.caps.total, "crypto": cfg.caps.crypto,
                "stock": cfg.caps.stock, "option": cfg.caps.option,
            },
            "crypto_coins": cfg.all_crypto_coins,
            "stock_tickers": cfg.stock_tickers,
            "config_problems": cfg.validate(),
        }

    @app.get("/positions")
    def positions():
        session = engine.session_factory()
        try:
            rows = session.query(Position).filter(Position.status == POSITION_OPEN).all()
            return {
                "open": [
                    {
                        "market": p.market, "symbol": p.symbol, "side": p.side,
                        "qty": p.qty, "entry_price": p.entry_price,
                        "current_price": p.current_price, "score": p.sentiment_score,
                    }
                    for p in rows
                ]
            }
        finally:
            session.close()

    # -- state-changing (X-API-Key required) -------------------------------
    @app.post("/run-cycle")
    def run_cycle(_: bool = Depends(key_dep)):
        return engine.run_crypto_cycle().as_dict()

    @app.post("/stocks/run-cycle")
    def stocks_run_cycle(_: bool = Depends(key_dep)):
        return engine.run_stock_cycle().as_dict()

    @app.post("/options/run-cycle")
    def options_run_cycle(_: bool = Depends(key_dep)):
        return engine.run_options_cycle().as_dict()

    @app.get("/confirmation-token")
    def confirmation_token(_: bool = Depends(key_dep)):
        token = secrets.token_urlsafe(24)
        _CONFIRM_TOKENS[token] = time.time() + _CONFIRM_TTL_SECONDS
        return {"confirmation_token": token, "ttl_seconds": _CONFIRM_TTL_SECONDS}

    @app.post("/mode")
    def set_mode(req: ModeRequest, _: bool = Depends(key_dep)):
        # Determine whether this request is trying to ENABLE live trading.
        wants_live = (
            (req.app_env or engine.config.app_env) == "prod"
            and req.paper_trading is False
            and req.live_trading is True
        )
        if wants_live:
            tok = req.confirmation_token
            expiry = _CONFIRM_TOKENS.pop(tok, None) if tok else None
            if not expiry or expiry < time.time():
                raise HTTPException(
                    status_code=428,
                    detail="live switch requires a valid confirmation_token "
                           "(GET /confirmation-token first)",
                )
        # Apply overrides to the environment, then reload config.
        if req.app_env is not None:
            os.environ["APP_ENV"] = req.app_env
        if req.paper_trading is not None:
            os.environ["PAPER_TRADING"] = "true" if req.paper_trading else "false"
        if req.live_trading is not None:
            os.environ["LIVE_TRADING"] = "true" if req.live_trading else "false"
        engine.config = get_config(reload=True)
        return {"mode": engine.mode, "is_live": engine.config.is_live()}

    @app.post("/config")
    def set_config(req: ConfigRequest, _: bool = Depends(key_dep)):
        from .models import BotConfig

        session = engine.session_factory()
        try:
            row = session.query(BotConfig).filter(BotConfig.key == req.key).one_or_none()
            if row is None:
                row = BotConfig(key=req.key, value=req.value, market=req.market)
                session.add(row)
            else:
                row.value = req.value
                row.market = req.market
            session.commit()
            return {"key": row.key, "value": row.value, "market": row.market}
        finally:
            session.close()

    return app


def run(host: str = "127.0.0.1", port: int = 8000):  # pragma: no cover
    """Run the control plane. Bound to loopback by default -- do NOT expose."""
    import uvicorn

    engine = TradingEngine()
    engine.ensure_schema()
    uvicorn.run(create_app(engine), host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    run()

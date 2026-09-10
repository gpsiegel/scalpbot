"""Local control-plane API for scalpbot.

Security model
--------------
* **Bind address**: the server is meant to run on ``127.0.0.1`` only (see
  :func:`run`). It is a loopback control plane, not a public API.
* **API key**: every *state-changing* endpoint requires the ``X-API-Key``
  header to match ``SCALPBOT_API_KEY`` (a Doppler secret). Read-only endpoints
  (``/health``, ``/status``, ``/positions``) are open on loopback.
* **Mode changes require a restart, not an API call**: ``TradingEngine.alpaca``
  (the ``AlpacaClient``) is constructed once at startup against the paper/live
  endpoint matching ``Config.is_live()`` *at that time*. Reloading ``Config``
  alone does not rebuild it, so accepting a runtime app_env/paper_trading/
  live_trading change here would silently desync the two: trades could be
  labeled "live" while still hitting the paper endpoint, or -- far worse --
  labeled "paper" while actually hitting the live endpoint. ``POST /mode``
  therefore only ever *reports* the current mode; any request that would
  actually change ``app_env``, ``paper_trading``, or ``live_trading`` is
  rejected with 409. Going live is still exactly what ``config.Config.is_live``
  documents: an explicit, reviewable flip of ``environments/prod.env``,
  applied by restarting the process.

Endpoints requiring ``X-API-Key``:
  POST /run-cycle, POST /stocks/run-cycle, POST /options/run-cycle,
  POST /config, POST /mode
"""
from __future__ import annotations

import logging
import os
import secrets
from typing import Optional

try:
    from fastapi import Depends, FastAPI, Header, HTTPException
    from pydantic import BaseModel
except ImportError:  # pragma: no cover - allows import without fastapi installed
    FastAPI = None  # type: ignore
    BaseModel = object  # type: ignore

from .engine import TradingEngine
from .models import Position, POSITION_OPEN

logger = logging.getLogger("scalpbot.engine.api")


class ModeRequest(BaseModel):
    app_env: Optional[str] = None
    paper_trading: Optional[bool] = None
    live_trading: Optional[bool] = None


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

    @app.post("/mode")
    def set_mode(req: ModeRequest, _: bool = Depends(key_dep)):
        """Report the current mode. Never mutates it -- see the module
        docstring: rebuilding Config without rebuilding engine.alpaca to
        match could silently desync the two, so any request that would
        actually change app_env/paper_trading/live_trading is rejected."""
        if req.app_env is not None and req.app_env != engine.config.app_env:
            raise HTTPException(
                status_code=409,
                detail="app_env cannot be changed at runtime; restart the "
                       "process with the new APP_ENV instead",
            )
        if (
            (req.paper_trading is not None and req.paper_trading != engine.config.paper_trading)
            or (req.live_trading is not None and req.live_trading != engine.config.live_trading)
        ):
            raise HTTPException(
                status_code=409,
                detail="paper_trading/live_trading cannot be changed at runtime "
                       "(the Alpaca client is bound to the endpoint chosen at "
                       "startup); restart the process with the new values instead",
            )
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

"""scalpbot trading engine.

Modules:
* ``models``        -- SQLAlchemy ORM (Trade, Position, BotConfig, SignalLog).
* ``alpaca_client`` -- thin wrapper over alpaca-py for crypto/stock/option
                       trading + market data, honoring paper/live mode.
* ``options``       -- options contract selection + quality filters + P&L.
* ``avenues``       -- per-avenue signal->order logic (crypto, stocks, options).
* ``engine``        -- the cycle orchestrator that ties avenues to persistence.
* ``api``           -- local FastAPI control plane (127.0.0.1, X-API-Key).
"""

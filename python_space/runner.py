"""Long-running scheduler for scalpbot.

Nothing previously ran a trading cycle unless something POSTed to
``/run-cycle``, ``/stocks/run-cycle``, or ``/options/run-cycle`` -- there was
no scheduler in the repo at all, so a stop-loss (or, since PR 3, an option's
DTE/premium exit) was only ever evaluated when a human or a cron job
happened to hit the API.

This module drives :class:`engine.engine.TradingEngine` on two independent
intervals via :class:`Runner`:

* exits (``EXIT_INTERVAL_SECONDS``, default 30) -- a stop-loss can't wait for
  the next entry scan;
* entries (``ENTRY_INTERVAL_SECONDS``, default 300) -- new positions are
  opened far less often than existing ones are checked.

``main()`` runs this scheduler alongside the existing FastAPI control plane
in a single process (one systemd unit -- see ``deploy/scalpbot.service``):
uvicorn serves the API on a background thread while the scheduler loop owns
the main thread, both sharing one :class:`TradingEngine` (and so one DB
session factory / Alpaca client).
"""
from __future__ import annotations

import logging
import os
import signal
import time
from typing import Callable, Optional

from engine.engine import MARKET_CRYPTO, MARKET_OPTION, MARKET_STOCK, TradingEngine

logger = logging.getLogger("scalpbot.runner")

MARKETS = (MARKET_CRYPTO, MARKET_STOCK, MARKET_OPTION)


class Runner:
    """Drives TradingEngine.manage_exits / open_entries on independent
    intervals. ``sleep``/``clock`` are injectable so tests run instantly and
    deterministically, with no real time passing."""

    def __init__(
        self,
        engine: Optional[TradingEngine] = None,
        exit_interval: Optional[float] = None,
        entry_interval: Optional[float] = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.engine = engine or TradingEngine()
        self.exit_interval = (
            exit_interval
            if exit_interval is not None
            else float(os.environ.get("EXIT_INTERVAL_SECONDS", "30") or "30")
        )
        self.entry_interval = (
            entry_interval
            if entry_interval is not None
            else float(os.environ.get("ENTRY_INTERVAL_SECONDS", "300") or "300")
        )
        self._sleep = sleep
        self._clock = clock
        self._stop = False

    def request_stop(self, *_args) -> None:
        logger.info("runner: stop requested")
        self._stop = True

    def install_signal_handlers(self) -> None:
        """Shut down cleanly on SIGTERM/SIGINT. Separate from run_forever()
        so tests can drive the loop without touching process-wide signal
        handlers (only main() calls this)."""
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)

    def _run_exits(self) -> None:
        for market in MARKETS:
            try:
                report = self.engine.manage_exits(market)
                if report.errors:
                    logger.warning("%s exits: %s", market, report.errors)
            except Exception:  # pragma: no cover - defensive
                logger.exception("%s exits iteration failed", market)

    def _run_entries(self) -> None:
        for market in MARKETS:
            try:
                report = self.engine.open_entries(market)
                if report.errors:
                    logger.warning("%s entries: %s", market, report.errors)
            except Exception:  # pragma: no cover - defensive
                logger.exception("%s entries iteration failed", market)

    def run_forever(self, max_iterations: Optional[int] = None) -> None:
        """Loop until ``request_stop()`` is called (or, for tests,
        ``max_iterations`` loop passes have run)."""
        next_exit = self._clock()
        next_entry = self._clock()
        logger.info(
            "runner starting: exits every %.0fs, entries every %.0fs",
            self.exit_interval, self.entry_interval,
        )
        iterations = 0
        while not self._stop:
            now = self._clock()
            if now >= next_exit:
                self._run_exits()
                next_exit = now + self.exit_interval
            if now >= next_entry:
                self._run_entries()
                next_entry = now + self.entry_interval
            iterations += 1
            if max_iterations is not None and iterations >= max_iterations:
                break
            self._sleep(min(self.exit_interval, self.entry_interval, 1.0))
        logger.info("runner stopped")


def main() -> None:  # pragma: no cover - process entry point
    logging.basicConfig(level=logging.INFO)

    engine = TradingEngine()
    engine.ensure_schema()
    runner = Runner(engine=engine)
    runner.install_signal_handlers()

    import threading

    import uvicorn

    from engine.api import create_app

    config = uvicorn.Config(
        create_app(engine),
        host="127.0.0.1",
        port=int(os.environ.get("PORT", "8000") or "8000"),
        log_level="info",
    )
    server = uvicorn.Server(config)
    api_thread = threading.Thread(target=server.run, name="scalpbot-api", daemon=True)
    api_thread.start()

    try:
        runner.run_forever()
    finally:
        server.should_exit = True
        api_thread.join(timeout=10)


if __name__ == "__main__":  # pragma: no cover
    main()

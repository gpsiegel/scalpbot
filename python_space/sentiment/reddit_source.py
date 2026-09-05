"""Reddit sentiment source (OAuth via PRAW).

Scans coin-specific + general crypto subreddits for recent hot/new posts and
scores their titles (and self-text) with the local lexicon. Reddit aggressively
throttles generic User-Agents, so a descriptive UA is REQUIRED. The expected
format is::

    platform:app_id:version (by /u/username)

e.g. ``python:scalpbot:1.0 (by /u/your_username)``

Credentials (Doppler secrets):
  * REDDIT_CLIENT_ID
  * REDDIT_CLIENT_SECRET
  * REDDIT_USER_AGENT   (must follow the format above)

Rate limit: 100 QPM on OAuth-authenticated app-only requests, which the few
listing calls per cycle stay well within.

Subreddits scanned: r/solana, r/dogecoin (coin-specific) + r/CryptoCurrency,
r/CryptoMarkets (general).

Weight in the aggregator: SENTIMENT_WEIGHT_REDDIT (default 0.30).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .base import (
    GENERAL_SUBREDDITS,
    BaseSentimentSource,
    SentimentSignal,
    clamp,
)
from .text_scoring import score_many


class RedditSource(BaseSentimentSource):
    name = "reddit"

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        user_agent: Optional[str] = None,
        posts_per_subreddit: int = 25,
        timeout: float = 15.0,
    ):
        self.client_id = client_id if client_id is not None else os.getenv("REDDIT_CLIENT_ID", "")
        self.client_secret = (
            client_secret if client_secret is not None else os.getenv("REDDIT_CLIENT_SECRET", "")
        )
        self.user_agent = (
            user_agent if user_agent is not None else os.getenv("REDDIT_USER_AGENT", "")
        )
        self.posts_per_subreddit = posts_per_subreddit
        self.timeout = timeout
        self._reddit = None

    # ------------------------------------------------------------------
    def _client(self):
        if self._reddit is not None:
            return self._reddit
        if not (self.client_id and self.client_secret and self.user_agent):
            raise RuntimeError(
                "Reddit OAuth not configured: set REDDIT_CLIENT_ID, "
                "REDDIT_CLIENT_SECRET and REDDIT_USER_AGENT"
            )
        import praw  # imported lazily so the package installs even if unused

        self._reddit = praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent,
            timeout=self.timeout,
            check_for_updates=False,
        )
        # App-only (read-only) mode -- no user login required.
        self._reddit.read_only = True
        return self._reddit

    def _subreddits_for(self, coin: str) -> List[str]:
        meta = self.coin_meta(coin)
        # Coin-specific subs weigh more; general crypto subs add breadth.
        return list(meta["subreddits"]) + list(GENERAL_SUBREDDITS)

    def _fetch(self, coin: str) -> Optional[SentimentSignal]:
        reddit = self._client()
        meta = self.coin_meta(coin)
        name = meta["name"].lower()
        symbol = meta["symbol"].lower()
        subs = self._subreddits_for(coin)

        texts: List[str] = []
        raw: Dict[str, Any] = {"subreddits": subs, "matched_posts": 0}

        for i, sub in enumerate(subs):
            is_coin_specific = i < len(meta["subreddits"])
            try:
                submissions = reddit.subreddit(sub).hot(limit=self.posts_per_subreddit)
            except Exception:  # noqa: BLE001 - skip a failing subreddit
                continue
            for post in submissions:
                title = getattr(post, "title", "") or ""
                body = getattr(post, "selftext", "") or ""
                blob = f"{title} {body}"
                # For general subs, only count posts that mention the coin.
                if not is_coin_specific:
                    low = blob.lower()
                    if name not in low and f" {symbol} " not in f" {low} ":
                        continue
                texts.append(blob)
                raw["matched_posts"] += 1

        if not texts:
            return SentimentSignal.unavailable(self.name, coin, "no relevant posts found")

        score, hits = score_many(texts)
        raw["sentiment_hits"] = hits
        # Confidence grows with number of matched posts (capped).
        confidence = clamp(0.2 + min(raw["matched_posts"], 40) / 40.0 * 0.8, 0.0, 1.0)
        return SentimentSignal(
            source=self.name, coin=coin, score=score, confidence=confidence, raw=raw
        )

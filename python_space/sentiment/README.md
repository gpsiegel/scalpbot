# scalpbot Sentiment Stack

`SentimentAggregator` is shared by all three avenues -- crypto, stocks, and
options all call `get_sentiment(symbol)` and get back a single normalized
score in `[-1, 1]`. What's crypto-only today is the **source roster**, not
the aggregator itself: every source below but Fear & Greed only resolves a
symbol through `sentiment.base.COIN_METADATA` (crypto tickers), so a stock
ticker like `PLTR` currently has just one source (a market-wide crypto
index) to draw on. The `coverage`/`actionable` gate below exists precisely
to catch that case, and it's why `ENABLE_STOCKS`/`ENABLE_OPTIONS` are off in
`environments/*.env` until an equity-capable source is added (see
`sentiment.base.EQUITY_CAPABLE_SOURCES`, empty today).

## Sources & weights

| Source                | Tier / key                         | Weight env var                | Default |
|-----------------------|------------------------------------|-------------------------------|---------|
| Reddit (OAuth)        | Free OAuth                          | `SENTIMENT_WEIGHT_REDDIT`     | 0.30    |
| cryptocurrency.cv     | Free, **no key**                   | `SENTIMENT_WEIGHT_CRYPTOCV`   | 0.25    |
| CoinGecko             | Paid **Basic** (`COINGECKO_API_KEY`)| `SENTIMENT_WEIGHT_COINGECKO`  | 0.20    |
| LunarCrush            | Free **Hobby** (`LUNARCRUSH_API_KEY`)| `SENTIMENT_WEIGHT_LUNARCRUSH` | 0.10    |
| Google Trends (trendspyg)| Free                            | `SENTIMENT_WEIGHT_TRENDS`     | 0.08    |
| Fear & Greed Index    | Free, **no key**                   | `SENTIMENT_WEIGHT_FEAR_GREED` | 0.07    |

Weights are **operational config** and live in the plain `.env` file
(`python_space/.env.example`), **never in Doppler**. If a source is
unavailable in a cycle, its weight is dropped and the survivors are
renormalized.

## Coverage and actionability

Every source but Fear & Greed only resolves crypto tickers (calls
`coin_meta()`, which raises for anything not in `COIN_METADATA`) — so for a
stock ticker, Fear & Greed (weight 0.07 by default) is the *only* survivor,
and the old renormalization logic would silently hand it 100% of the score.
To stop a single thin source from driving a trade, `AggregatedSentiment` now
carries:

- `coverage` — the sum of base weights of currently-available sources divided
  by the sum of all positive base weights.
- `actionable` — `coverage >= SENTIMENT_MIN_COVERAGE` (default `0.5`, plain
  `.env`). When not actionable, `score` is forced to `0.0` and `label` to
  `"insufficient_data"` instead of computing a renormalized score from
  whatever thin sliver of data is available.

The engine skips new entries when `not sentiment.actionable`, and exits never
infer a sentiment-reversal from a non-actionable score (price-based
take-profit/stop-loss still apply). This is also why `ENABLE_STOCKS` /
`ENABLE_OPTIONS` are off by default in `environments/*.env` today — see
`sentiment.base.EQUITY_CAPABLE_SOURCES`.

## Per-source result cache

Each source is queried up to twice per symbol per crypto cycle, and some
(Google Trends especially) rate-limit hard. `SentimentAggregator` caches each
source's result per coin for a per-source TTL (plain `.env`, seconds):

| Source                | TTL env var                | Default |
|------------------------|----------------------------|---------|
| Fear & Greed           | `SENTIMENT_TTL_FEAR_GREED` | 3600    |
| Google Trends          | `SENTIMENT_TTL_TRENDS`     | 3600    |
| Reddit                 | `SENTIMENT_TTL_REDDIT`     | 600     |
| cryptocurrency.cv      | `SENTIMENT_TTL_CRYPTOCV`   | 600     |
| LunarCrush             | `SENTIMENT_TTL_LUNARCRUSH` | 900     |
| CoinGecko              | `SENTIMENT_TTL_COINGECKO`  | 300     |

An `unavailable` result (dead API, missing key) is cached for at most ~60
seconds regardless of the source's normal TTL, so a down provider is retried
soon instead of staying dark for its full (possibly hour-long) window. The
cache key is `(source, coin)`; a fresh `SentimentAggregator` (as constructed
once per `TradingEngine`) starts with an empty cache.

### Removed sources
`CryptoPanic` (free tier killed Aug 2026), `Twitter/X` (shelved),
`pytrends` (archived Apr 2025 -> replaced by `trendspyg`), and `NewsAPI`
(localhost-only, 24h delay) have been removed entirely.

## Source details

- **Reddit** — PRAW app-only OAuth. Scans `r/solana`, `r/dogecoin`,
  `r/CryptoCurrency`, `r/CryptoMarkets`. `REDDIT_USER_AGENT` **must** follow
  `platform:app_id:version (by /u/username)` or Reddit throttles the app.
- **cryptocurrency.cv** — keyless HTTP; `/api/sentiment?ticker=` +
  `/api/archive?ticker=` (built-in AI sentiment per article).
- **CoinGecko (Basic)** — `/coins/markets` (momentum + votes),
  `/search/trending` (momentum), `/news` (headline sentiment), filtered to
  `solana` / `dogecoin`. Key sent via `x-cg-pro-api-key` on the pro host.
- **LunarCrush (Hobby)** — `/public/coins/{symbol}/v1`; uses **galaxy_score**
  and **alt_rank** as a social-momentum proxy (social time-series is paywalled).
- **Google Trends** — `trendspyg.download_google_trends_interest_over_time`
  (drop-in successor to `pytrends.interest_over_time`); rising search interest
  = bullish momentum.
- **Fear & Greed** — `https://api.alternative.me/fng/`, market-wide.

## Usage

```python
from sentiment import SentimentAggregator

agg = SentimentAggregator()               # reads weights + keys from env
result = agg.get_sentiment("SOL")
print(result.score, result.label)         # e.g. 0.42 bullish
print(result.as_dict())                   # full per-source breakdown

# All coins at once
for coin, res in agg.get_all(["SOL", "DOGE"]).items():
    print(coin, res.score, res.sources_available)
```

Every source is defensive: a failing provider returns an *unavailable* signal
instead of raising, so one dead API never breaks a trading cycle.

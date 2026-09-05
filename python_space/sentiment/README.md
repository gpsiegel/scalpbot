# scalpbot Sentiment Stack

Aggregates crypto sentiment for the tradeable coins (**SOL/USD**, **DOGE/USD**)
into a single normalized score in `[-1, 1]` per coin.

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

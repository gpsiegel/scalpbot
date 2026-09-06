# Per-environment operational config (PR-controlled)

These files hold **non-secret operational config** for each environment and are
**committed to git on purpose** so that every change is a reviewable pull
request. This is how avenues are turned on/off, how the risk tier is chosen, and
how caps/behaviour are set per environment.

| File          | Environment | Live account? |
|---------------|-------------|---------------|
| `nonprod.env` | nonprod     | Never — always paper |
| `prod.env`    | prod        | Only prod can go live, and only with all three live guards set |

## How to enable/disable an avenue (via PR)

1. Edit the relevant file (`nonprod.env` or `prod.env`).
2. Flip the switch, e.g. `ENABLE_OPTIONS=false`.
3. Open a PR. Once merged and deployed (`git pull` on the droplet), the change
   takes effect on the next engine restart.

The avenue switches are:

```
ENABLE_CRYPTO=true|false
ENABLE_STOCKS=true|false
ENABLE_OPTIONS=true|false
```

When an avenue is **off** it still MANAGES and EXITS existing positions but
opens **no new entries**.

## Rules

- **No secrets here — ever.** API keys, DB credentials, and tokens
  (`GROQ_API_KEY`, `ALPACA_*`, `POSTGRES_*`, `DOPPLER_TOKEN`, ...) live in
  **Doppler**, never in these files. See `doppler.env.example`.
- These values are loaded as **defaults** at startup by `config.load_env_file()`.
  A real environment variable of the same name (injected by Doppler / CI /
  shell) **always overrides** the file value.

## Environment safety model

- **nonprod is always paper.** `Config.is_live()` additionally requires
  `APP_ENV=prod`, so no combination of flags can make nonprod trade live.
- **prod is the only live-capable environment**, and even prod stays on the
  paper endpoint unless **all three** guards are set:
  `APP_ENV=prod` **and** `PAPER_TRADING=false` **and** `LIVE_TRADING=true`.
  Going live is therefore an explicit, reviewable edit to `prod.env`.

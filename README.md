# za-thunder-forecast

Builds a South African thunderstorm forecast from [DWD ICON](https://www.dwd.de/EN/ourservices/nwp_forecast_data/nwp_forecast_data.html)
open data, four times a day, and publishes it as a small static file that
[TrailTether](https://trailtether.app) caches offline.

Output: `https://tiles.hilltrek.co.za/weather/za-thunder.json.gz` (plus a
static `za-thunder-grid.json.gz` carrying the cell coordinates).

## Why this exists

TrailTether's hiking score has a thunderstorm veto — the app's own comment calls
lightning "the #1 mountain killer", because on an exposed Drakensberg ridge there
is nowhere to shelter. **In South Africa that veto was dead, and silently so.**

It keys on `weatherCode >= 95 || thunderProb > 15`, and met.no publishes neither
here. Probed live across 16 points on 2026-08-06: `probability_of_thunder`
appears only inside met.no's Nordic model (53–54 hourly blocks at each of eight
Nordic controls, **zero** at Cathedral Peak, Sani Top, Durban, Cape Town, Congo,
Manaus, Jakarta and Kampala), and its `*andthunder` symbols are derived from that
same field — so outside the Nordics no thunder symbol is emitted either, Miami in
peak August storm season included. Both halves of the test are permanently false
in the target region, and every Berg storm day scored as storm-free.

DWD ICON's `WW` field is "weather interpretation (WMO)" — the same code space the
app already uses, thunder at 95/96/99 — and is genuinely global. On 2026-08-06 it
put a WMO 95 thunderstorm 1.1 km from Cathedral Peak on 10 August, which the app
could not see at all.

## The fail-safe contract

This whole feature exists because an absent reading was being presented as a
confident all-clear. The ingest must not repeat that mistake, so:

* A day is marked **assessed** only if *every* forecast step covering it
  downloaded and decoded. Any gap and the day is emitted as **unknown**.
* `code = 0` means "the model forecast no thunder". `code = 255` means "we could
  not tell". They are different values on purpose, and the consumer must render
  them differently.
* If too few steps survive overall, the script **exits non-zero and writes
  nothing** rather than publishing a partial file that reads as a quiet
  region-wide all-clear. A stale-but-honest artifact beats a fresh lie.

Consumers enforce a 30-hour staleness bound and fall back to "unknown" past it —
never to "clear".

## Why it is public, and why it is separate from the app

GitHub Actions is metered on private repos (2,000 Linux minutes/month), and the
owning account runs a $0 spending budget with "Stop usage" enabled. On
2026-08-15 that allowance ran out and Actions stopped, taking CI and this
four-times-daily safety job down together — the wrong coupling for a feature
about lightning. Actions is free and unlimited on **public** repos, so living
here means the forecast can never again be starved by a CI allowance.

Nothing proprietary is here. The app, the website, the backend and the hiking
score all stay private. The artifact this produces is already world-readable,
because the app fetches it without authentication.

## Running it

```bash
pip install numpy eccodes            # plus the ecCodes C library (libeccodes0)
python scripts/fetch_dwd_thunder.py --out out/za-thunder.json.gz
```

Useful flags:

| Flag | Purpose |
|---|---|
| `--run 2026081500` | build a specific ICON run instead of the latest |
| `--steps 0,3,6` | a short run, for testing |
| `--min-steps N` | refuse to publish below N decoded steps (default 40) |
| `--bbox s,w,n,e` | override the South Africa box |

`--bbox` exists for a specific reason: over South Africa in winter every cell is
legitimately `0`, so a run here proves only that the script *can* emit zeros —
which is indistinguishable from a broken decode. Point it at a convective region
(India in August, say) and the same code path must come back non-zero.

## Secrets

The publish step needs three repository secrets: `R2_ACCESS_KEY_ID`,
`R2_SECRET_ACCESS_KEY`, `R2_ENDPOINT`.

Triggers are `schedule` and `workflow_dispatch` only. There is deliberately no
`pull_request` trigger, so a fork can never run this workflow and can never
reach those secrets.

## Licence

The **code** in this repository is © Hilltrek (Pty) Ltd, all rights reserved.
Public visibility is not a grant of licence.

The **data** it fetches is Deutscher Wetterdienst (DWD) ICON, licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribution is not
optional and is carried inside the published artifact so a consumer cannot ship
the data without it:

> Thunderstorm forecast: Deutscher Wetterdienst (DWD), ICON — CC BY 4.0

We reduce DWD's global hourly `WW` field to a per-day South African summary, so
"modified" is part of the required notice.
